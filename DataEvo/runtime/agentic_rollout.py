"""On-policy async vLLM -> Harbor terminal -> verifier -> verl GRPO adapter."""
import asyncio
import copy
import time

from .agentic_env import HarborEpisode
from .agentic_protocol import action, observation, prompt, token_spans
from .common import read_json, write_json


def pack_episodes(episodes, tokenizer, prompt_length, response_length):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl.protocol import DataProto

    size = len(episodes)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    prompts = torch.full((size, prompt_length), pad, dtype=torch.long)
    responses = torch.full((size, response_length), pad, dtype=torch.long)
    attention = torch.zeros((size, prompt_length + response_length), dtype=torch.long)
    loss = torch.zeros_like(attention)
    rewards = torch.zeros((size, response_length), dtype=torch.float32)
    for index, episode in enumerate(episodes):
        p, r, mask = token_spans(tokenizer, episode['messages'], episode['prompt_count'])
        if not p or not r or len(p) > prompt_length or len(r) > response_length or not any(mask):
            raise ValueError('episode has invalid lengths or no assistant tokens; refusing to truncate')
        prompts[index, -len(p):] = torch.tensor(p)
        responses[index, :len(r)] = torch.tensor(r)
        attention[index, prompt_length - len(p):prompt_length + len(r)] = 1
        loss[index, prompt_length:prompt_length + len(r)] = torch.tensor(mask)
        last = max(i for i, value in enumerate(mask) if value)
        rewards[index, last] = episode['reward']
    inputs = torch.cat((prompts, responses), dim=1)
    tensors = TensorDict({'prompts': prompts, 'responses': responses, 'input_ids': inputs,
        'attention_mask': attention, 'position_ids': (attention.cumsum(-1) - 1).clamp(min=0) * attention,
        'response_mask': loss[:, prompt_length:], 'loss_mask': loss, 'rm_scores': rewards}, batch_size=size)
    return DataProto(batch=tensors, non_tensor_batch={
        'agentic_trace': np.array([e['trace'] for e in episodes], dtype=object),
        '__num_turns__': np.array([e['turns'] for e in episodes], dtype=np.int32)})


class HarborCompletionCallback:
    tool_schemas = None  # JSON text actions; no vLLM tool-parser dependency.
    extra_body = None

    def __init__(self, config, scheduler):
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        self.config, self.scheduler = config, scheduler
        self.rollout = config.actor_rollout_ref.rollout
        self.spec_path = self.rollout.multi_turn.interaction_config_path
        self.spec = read_json(self.spec_path)
        self.tokenizer = hf_tokenizer(copy_to_local(config.actor_rollout_ref.model.path), trust_remote_code=True)
        from .agentic_sft import configure_tokenizer
        expected = copy.deepcopy(self.tokenizer)
        configure_tokenizer(expected)
        if self.tokenizer.chat_template != expected.chat_template:
            raise ValueError('RL must start from a TerminalSFTDataset checkpoint with the fixed ChatML template')

    async def generate_sequences(self, batch):
        started = time.monotonic()
        validate = batch.meta_info.get('validate', False)
        n = 1 if validate else self.rollout.n
        split = 'val' if validate else 'train'
        semaphore = asyncio.Semaphore(self.spec['concurrency'])
        addresses = [address for _, address in self.scheduler.weighted_addresses]

        async def one(index):
            async with semaphore:
                original = index // n
                raw = batch.non_tensor_batch['raw_prompt'][original]
                messages = copy.deepcopy(raw.tolist() if hasattr(raw, 'tolist') else list(raw))
                task = batch.non_tensor_batch['tools_kwargs'][original]
                if task.get('split') != split:
                    raise ValueError('rollout task split does not match training/validation mode')
                async with asyncio.timeout(self.spec['episode_timeout_s']):
                    return await self.episode(messages, task['task_id'], split,
                                              addresses[index % len(addresses)], validate)

        tasks = [asyncio.create_task(one(i)) for i in range(len(batch) * n)]
        try:
            episodes = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        output = pack_episodes(episodes, self.tokenizer, self.rollout.prompt_length, self.rollout.response_length)
        output.meta_info['timing'] = {'generate_sequences': time.monotonic() - started}
        return output

    async def episode(self, messages, task_id, split, address, validate):
        p_count = len(messages)
        initial = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False, return_dict=False)
        if len(initial) > self.rollout.prompt_length:
            raise ValueError('agentic prompt exceeds rollout.prompt_length')
        valid, turns, reason, events = True, 0, 'turn_limit', []
        sampling = self.rollout.val_kwargs if validate else self.rollout
        async with HarborEpisode(self.spec_path, task_id, split) as env:
            if messages != prompt(env.ready['instruction']):
                raise ValueError('RL prompt differs from the allowlisted Harbor task instruction')
            for _ in range(self.spec['max_turns']):
                current = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                # Reserve closing-template tokens; injected observations also consume context.
                remaining = len(initial) + self.rollout.response_length - len(current) - 16
                if remaining <= 0:
                    reason = 'context_limit'
                    break
                completion = await self.scheduler._chat_completions_aiohttp(address,
                    model=self.scheduler.model_name, messages=messages,
                    temperature=sampling.temperature, top_p=sampling.top_p,
                    max_tokens=min(self.spec['max_tokens_per_turn'], remaining),
                    extra_body={'chat_template_kwargs': {'enable_thinking': False}}, extra_headers={})
                choice = completion.choices[0]
                message = choice.message.model_dump(exclude_none=True)
                if message.get('reasoning_content') or message.get('tool_calls'):
                    raise ValueError('adapter requires visible JSON text actions, with thinking disabled')
                messages.append({'role': 'assistant', 'content': message.get('content') or ''})
                turns += 1
                # Validate token alignment immediately, before using a reward.
                _, response, _ = token_spans(self.tokenizer, messages, p_count)
                if len(response) > self.rollout.response_length:
                    raise ValueError('model/template exceeded the reserved response token budget')
                try:
                    kind, value = action(messages[-1]['content'])
                except (ValueError, TypeError):
                    valid, reason = False, 'invalid_action'
                    break
                if choice.finish_reason == 'length':
                    valid, reason = False, 'truncated_action'
                    break
                if kind == 'final':
                    reason = 'final'
                    break
                result = await env.request('exec', command=value)
                events.append({'command': value, 'result': result})
                obs = observation(result, self.spec['max_observation_chars'])
                next_input = self.tokenizer.apply_chat_template(messages + [obs], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                if len(next_input) + 32 > len(initial) + self.rollout.response_length or turns == self.spec['max_turns']:
                    reason = 'context_limit' if turns < self.spec['max_turns'] else 'turn_limit'
                    break  # Keep last observation in trace; it was never fed to the policy.
                messages.append(obs)
            result = await env.request('verify')
            episode = {'task_id': task_id, 'split': split, 'messages': messages, 'prompt_count': p_count,
                'turns': turns, 'reward': result['reward'] if valid else 0.0,
                'verifier_reward': result['reward'], 'termination': reason, 'events': events,
                'trace': str(env.output / 'trajectory.json'), 'harbor_result': result['result']}
            write_json(env.output / 'trajectory.json', episode)
        return episode
