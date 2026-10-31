"""CPU adapter tests. Model inference/cloud sandboxes are explicit test doubles."""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from DataEvo.runtime.agentic_config import DEFAULTS, grpo_overrides
from DataEvo.runtime.agentic_env import HarborEpisode
from DataEvo.runtime.agentic_protocol import action, observation, prompt, reward_value, token_spans
from DataEvo.runtime.agentic_rollout import HarborCompletionCallback, pack_episodes
from DataEvo.runtime.common import REPO, read_json, write_json


class CharTokenizer:
    """Deterministic prefix-stable template; EOS is an actual dedicated token."""
    pad_token_id = 0
    eos_token_id = ord('~')

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = ''.join(f"<{m['role']}>" + m['content'] + '~\n' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(map(ord, text))


class ProtocolTest(unittest.TestCase):
    @unittest.skipUnless(Path('/home/hrh/models/Qwen3-4B/tokenizer_config.json').exists(), 'local Qwen tokenizer unavailable')
    def test_real_qwen_tokenizer_and_sft_dataset(self):
        import pandas as pd
        from transformers import AutoTokenizer
        from omegaconf import OmegaConf
        from DataEvo.runtime.agentic_sft import TerminalSFTDataset, configure_tokenizer
        tokenizer = AutoTokenizer.from_pretrained('/home/hrh/models/Qwen3-4B', local_files_only=True)
        configure_tokenizer(tokenizer)
        messages = prompt('fixture task') + [{'role': 'assistant', 'content': '{"command":"pwd"}'},
            observation({'stdout': 'TOOL_SECRET'}, 200), {'role': 'assistant', 'content': '{"final":"done"}'}]
        _, ids, mask = token_spans(tokenizer, messages, 2)
        trained = tokenizer.decode([token for token, take in zip(ids, mask) if take])
        self.assertIn('command', trained)
        self.assertIn('final', trained)
        self.assertNotIn('TOOL_SECRET', trained)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'sft.parquet'
            pd.DataFrame([{'messages': messages, 'enable_thinking': False}]).to_parquet(file)
            dataset = TerminalSFTDataset(str(file), tokenizer,
                OmegaConf.create({'max_length': 1024, 'multiturn': {'messages_key': 'messages'}}))
            item = dataset[0]
            actual = tokenizer.decode(item['input_ids'][item['loss_mask'].bool()].tolist())
            self.assertIn('command', actual)
            self.assertNotIn('TOOL_SECRET', actual)

    def test_prepare_profile_and_reject_task_leakage(self):
        import pandas as pd
        from DataEvo.runtime.contracts import validate_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'dev', 'probe'):
                task = root / split / 'task'
                task.mkdir(parents=True)
                (task / 'task.toml').write_text('version = "1.0"\n')
                (task / 'instruction.md').write_text(split + ' task instruction')
                (task / 'environment').mkdir()
                (task / 'environment/Dockerfile').write_text('FROM ubuntu:24.04\n')
                (task / 'tests').mkdir()
                (task / 'tests/test.sh').write_text('#!/bin/sh\nexit 0\n')
            sft = root / 'sft.parquet'
            pd.DataFrame([{'messages': prompt('held out sft task') +
                [{'role': 'assistant', 'content': '{"final":"done"}'}]}]).to_parquet(sft)
            run = subprocess.run([sys.executable, str(REPO / 'DataEvo/scripts/prepare_agentic_session.py'),
                '--train-tasks', str(root / 'train'), '--dev-tasks', str(root / 'dev'),
                '--probe-tasks', str(root / 'probe'), '--sft-val', str(sft), '--base-model', tmp,
                '--harbor-python', sys.executable, '--output', str(root / 'prepared'),
                '--teacher-model', 'fixture'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            cfg = read_json(root / 'prepared/session.json')
            validate_config(cfg)
            harbor_python = Path('/home/hrh/RSIBench-Data-full/.venv/bin/python')
            if harbor_python.exists():
                preflight = [str(harbor_python), '-m', 'DataEvo.runtime.agentic_preflight',
                             '--kind', 'harbor', '--config', str(root / 'prepared/session.json')]
                checked = subprocess.run(preflight, capture_output=True, text=True, timeout=30)
                self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertEqual(json.loads(checked.stdout)['validated_tasks'], 3)
                (root / 'train/task/tests/test.sh').unlink()
                checked = subprocess.run(preflight, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn('invalid Harbor task', checked.stderr)
            (root / 'probe/task/instruction.md').write_text('train task instruction')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                validate_config(cfg)

    def test_action_and_reward_reject_untrusted_values(self):
        self.assertEqual(action('{"command":"pwd"}'), ('command', 'pwd'))
        for value in ('[]', '{"command":"x","reward":1}', '{"final":""}', '```json\n{}\n```'):
            with self.assertRaises((ValueError, TypeError)):
                action(value)
        for value in (None, True, float('nan'), float('inf'), -1, 2, '1'):
            with self.assertRaises(ValueError):
                reward_value({'reward': value})

    def test_mask_excludes_observations_headers_and_separator(self):
        p = prompt('task')
        m = p + [{'role': 'assistant', 'content': '{"command":"pwd"}'},
                 observation({'stdout': 'PRIVATE_TOOL_RESULT'}, 100),
                 {'role': 'assistant', 'content': '{"final":"done"}'}]
        _, tokens, mask = token_spans(CharTokenizer(), m, len(p))
        trained = ''.join(chr(t) for t, selected in zip(tokens, mask) if selected)
        self.assertEqual(trained, '{"command":"pwd"}~{"final":"done"}~')
        packed = pack_episodes([{'messages': m, 'prompt_count': len(p), 'reward': .75,
                                 'trace': '/fixture', 'turns': 2}], CharTokenizer(), 2048, 2048)
        self.assertEqual(float(packed.batch['rm_scores'].sum()), .75)
        self.assertEqual(int(packed.batch['loss_mask'].sum()), sum(mask))
        from verl.workers.reward_manager.naive import NaiveRewardManager
        reward = NaiveRewardManager(CharTokenizer(), 0)(packed)
        self.assertEqual(float(reward.sum()), .75)
        with self.assertRaisesRegex(ValueError, 'refusing to truncate'):
            pack_episodes([{'messages': m, 'prompt_count': len(p), 'reward': .75}], CharTokenizer(), 2, 2)

    def test_prefix_rewrite_is_rejected(self):
        class BadTokenizer(CharTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [len(messages)] + super().apply_chat_template(messages, **kwargs)
        with self.assertRaisesRegex(ValueError, 'prefix-stable'):
            token_spans(BadTokenizer(), prompt('task') + [{'role': 'assistant', 'content': 'x'}], 2)

    def test_observation_cap_is_explicit(self):
        obs = observation({'stdout': 'a' * 1000, 'stderr': 'b' * 1000, 'return_code': 1}, 100)
        data = json.loads(obs['content'].removeprefix('TOOL_RESULT '))
        self.assertTrue(data['truncated'])
        self.assertEqual(len(data['stdout']) + len(data['stderr']), 100)

    def test_fixed_hydra_overrides_compose(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(REPO / 'config'), version_base=None):
            config = compose(config_name='ppo_trainer', overrides=grpo_overrides(Path('/tmp/fixture')))
        self.assertTrue(config.actor_rollout_ref.rollout.multi_turn.enable)
        self.assertEqual(config.actor_rollout_ref.model.lora_rank, 0)
        self.assertEqual(config.reward_model.reward_manager, 'naive')
        self.assertIsNone(config.custom_reward_function.path)


class EpisodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_group_order_and_concurrency_limit(self):
        import numpy as np
        from verl.protocol import DataProto
        callback = HarborCompletionCallback.__new__(HarborCompletionCallback)
        callback.spec = {**DEFAULTS, 'concurrency': 2}
        callback.rollout = SimpleNamespace(n=2, prompt_length=2048, response_length=128)
        callback.scheduler = SimpleNamespace(weighted_addresses=[[0, 'current-policy']])
        callback.tokenizer = CharTokenizer()
        running, peak, order = 0, 0, []

        async def episode(messages, task, split, address, validate):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(.01)
            running -= 1
            order.append(task)
            return {'messages': messages + [{'role': 'assistant', 'content': '{"final":"done"}'}],
                    'prompt_count': 2, 'reward': float(task == 'b'), 'turns': 1, 'trace': task}
        callback.episode = episode
        prompts = np.empty(2, dtype=object)
        prompts[:] = [prompt('a'), prompt('b')]
        batch = DataProto(non_tensor_batch={'raw_prompt': prompts,
            'tools_kwargs': np.array([{'task_id': x, 'split': 'train'} for x in ('a', 'b')], dtype=object)})
        result = await callback.generate_sequences(batch)
        self.assertEqual(result.non_tensor_batch['agentic_trace'].tolist(), ['a', 'a', 'b', 'b'])
        self.assertEqual(result.batch['rm_scores'].sum(-1).tolist(), [0, 0, 1, 1])
        self.assertEqual(peak, 2)
        batch.meta_info['validate'] = True
        with self.assertRaisesRegex(ValueError, 'split'):
            await callback.generate_sequences(batch)

    async def test_real_pipe_state_verification_and_cleanup(self):
        harbor_python = Path('/home/hrh/RSIBench-Data-full/.venv/bin/python')
        if not harbor_python.exists():
            self.skipTest('optional Harbor 0.17.1 interpreter unavailable')
        create = asyncio.create_subprocess_exec

        async def fixture_process(*args, **kwargs):
            output = args[args.index('--output') + 1]
            return await create(str(harbor_python), '-m', 'DataEvo.tests.fixture_harbor_bridge', output, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            spec = {**DEFAULTS, 'harbor_python': str(harbor_python), 'output_dir': tmp,
                    'tasks': {'train': {'x': '/unused'}, 'val': {}}, 'max_episodes': 2}
            path = Path(tmp) / 'spec.json'
            write_json(path, spec)
            with patch('asyncio.create_subprocess_exec', fixture_process):
                async with HarborEpisode(path, 'x', 'train') as env:
                    self.assertEqual(env.ready['instruction'], 'fixture task')
                    await env.request('exec', command='set correct')
                    self.assertEqual((await env.request('exec', command='get'))['stdout'], 'correct')
                    self.assertEqual((await env.request('verify'))['reward'], 1)
                self.assertTrue((env.output / 'released').exists())
                self.assertIsNone(read_json(env.output / 'episode.json')['cleanup_error'])
                with self.assertRaisesRegex(RuntimeError, 'policy request failed'):
                    async with HarborEpisode(path, 'x', 'train') as failed:
                        raise RuntimeError('policy request failed')
                self.assertTrue((failed.output / 'released').exists())
                self.assertIsNone(read_json(failed.output / 'episode.json')['result'])
                with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                    async with HarborEpisode(path, 'x', 'train'):
                        pass
            with self.assertRaisesRegex(ValueError, 'allowlist'):
                HarborEpisode(path, 'not-allowed', 'train')

    async def test_rollout_uses_observation_then_verifier(self):
        closed, requests, replies = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            class Env:
                def __init__(self, *args):
                    self.output = Path(tmp)
                    self.ready = {'instruction': 'fixture task'}
                async def __aenter__(self): return self
                async def __aexit__(self, *exc): closed.append(True)
                async def request(self, op, **kw):
                    requests.append((op, kw))
                    if op == 'verify': return {'reward': 1, 'result': '/fixture'}
                    return {'stdout': 'persistent-state', 'stderr': '', 'return_code': 0}

            async def complete(address, **kwargs):
                replies.append(copy.deepcopy(kwargs['messages']))
                text = '{"command":"pwd"}' if len(replies) == 1 else '{"final":"done"}'
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(model_dump=lambda **kw: {'content': text}))])

            callback = HarborCompletionCallback.__new__(HarborCompletionCallback)
            callback.spec, callback.spec_path = DEFAULTS, '/fixture'
            callback.tokenizer = CharTokenizer()
            callback.rollout = SimpleNamespace(prompt_length=2048, response_length=2048,
                                                temperature=.7, top_p=1)
            callback.scheduler = SimpleNamespace(model_name='current-policy', _chat_completions_aiohttp=complete)
            with patch('DataEvo.runtime.agentic_rollout.HarborEpisode', Env):
                result = await callback.episode(prompt('fixture task'), 'x', 'train', 'local', False)
            self.assertEqual(result['reward'], 1)
            self.assertEqual([x[0] for x in requests], ['exec', 'verify'])
            self.assertIn('persistent-state', replies[1][-1]['content'])
            self.assertEqual(len(closed), 1)
            self.assertTrue(Path(result['trace']).exists())

            async def broken(*args, **kwargs):
                raise ConnectionError('model server unavailable')
            callback.scheduler._chat_completions_aiohttp = broken
            with patch('DataEvo.runtime.agentic_rollout.HarborEpisode', Env):
                with self.assertRaises(ConnectionError):
                    await callback.episode(prompt('fixture task'), 'x', 'train', 'local', False)
            self.assertEqual(len(closed), 2)


if __name__ == '__main__':
    unittest.main()
