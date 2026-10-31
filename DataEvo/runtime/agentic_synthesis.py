"""Teacher terminal trajectories using the same sandbox/verifier as RL."""
import asyncio
import json
from pathlib import Path

from .agentic_env import HarborEpisode
from .agentic_protocol import action, observation, prompt
from .api import complete
from .common import read_json, write_json
from .data import load_rows


async def synthesize(context, output):
    spec_path = context['agentic_spec']
    spec = read_json(spec_path)
    generation = context['generation']
    with Path(output).open('w') as stream:
        for row in load_rows(context['data']['seed'])[:generation.get('max_tasks', 16)]:
            task_id = row['extra_info']['tools_kwargs']['task_id']
            for sample in range(generation.get('samples_per_task', 1)):
                async with HarborEpisode(spec_path, task_id, 'train', 'synthesis') as env:
                    messages = prompt(env.ready['instruction'])
                    # Teacher-only guidance is stripped from the SFT prompt below.
                    guidance = generation.get('guidance', '')
                    policy_prompt = messages[0]['content']
                    if guidance:
                        messages[0]['content'] += '\n' + guidance
                    valid, turns = True, 0
                    for _ in range(spec['max_turns']):
                        result = await asyncio.to_thread(complete, context, messages,
                            **{'max_tokens': spec['max_tokens_per_turn'], **generation.get('sampling', {})})
                        choice = result['choices'][0]
                        messages.append({'role': 'assistant', 'content': choice['message'].get('content') or ''})
                        turns += 1
                        try:
                            kind, value = action(messages[-1]['content'])
                        except (ValueError, TypeError):
                            valid = False
                            break
                        if choice.get('finish_reason') == 'length':
                            valid = False
                            break
                        if kind == 'final':
                            break
                        result = await env.request('exec', command=value)
                        if turns < spec['max_turns']:
                            messages.append(observation(result, spec['max_observation_chars']))
                    verified = await env.request('verify')
                    messages[0]['content'] = policy_prompt
                    evidence = {'task_id': task_id, 'messages': messages, 'reward': verified['reward'],
                                'valid_actions': valid, 'harbor_result': verified['result']}
                    write_json(env.output / 'trajectory.json', evidence)
                    stream.write(json.dumps({'task_id': task_id, 'messages': messages,
                        'execution_ref': str(env.output / 'trajectory.json'),
                        'verified': valid and verified['reward'] == 1, 'reward': verified['reward'],
                        'turns': turns, 'sample_index': sample, 'enable_thinking': False}, ensure_ascii=False) + '\n')


def postprocess(source, output):
    seen = set()
    with Path(output).open('w') as stream:
        for row in load_rows(source):
            signature = json.dumps(row['messages'], sort_keys=True)
            if row.get('verified') is True and signature not in seen:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                seen.add(signature)
