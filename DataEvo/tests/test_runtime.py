"""CPU-only integration checks. All GPU training/evaluation is explicitly simulated."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from DataEvo.runtime import backends
from DataEvo.runtime.cli import initialize, select, submit
from DataEvo.runtime.common import REPO, Store, config, digest, read_json, write_json
from DataEvo.runtime.contracts import validate_attempt, validate_config
from DataEvo.runtime.runner import Runner


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.original_run = Runner.run
        self.tmp = tempfile.TemporaryDirectory(prefix='dataevo-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = self.root / 'session'
        self.cfg = read_json(REPO / 'DataEvo/config/session.gsm8k.example.json')
        self.cfg['python'] = sys.executable
        self.cfg['base_model'] = str(self.root / 'base')
        Path(self.cfg['base_model']).mkdir()
        self.cfg['resources'].update(lock_dir=str(self.root / 'locks'), check_gpu_memory=False)
        for key in self.cfg['data']:
            path = self.root / (key + '.jsonl')
            path.write_text(json.dumps({'question': key + ' question', 'answer': '#### 2'}) + '\n')
            self.cfg['data'][key] = str(path)
        self.config_file = self.root / 'config.json'
        write_json(self.config_file, self.cfg)
        initialize(self.session, self.config_file)

    def candidate(self, attempt='a0000'):
        folder = self.session / 'workspace/attempts' / attempt
        if not folder.exists():
            shutil.copytree(self.session / 'workspace/attempts/a0000', folder)
        recipe = read_json(folder / 'recipe.json')
        recipe['attempt_id'] = attempt
        recipe['generation'] = {'backend': 'program', 'entrypoint': 'synthesize.py'}
        recipe['postprocess'] = {}
        write_json(folder / 'recipe.json', recipe)
        # An actual subprocess writes selected candidates; training is mocked below.
        (folder / 'synthesize.py').write_text('''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--context'); p.add_argument('--output'); a=p.parse_args()
Path(a.output).write_text(json.dumps({'question':'new training question','answer':'#### 2'})+'\\n')
''')
        return folder

    @contextlib.contextmanager
    def no_gpu(self, runner):
        runner.gpus = ['1']
        yield

    def simulated_pipeline(self, runner, stage, generation=None, eval_data=None, eval_output=None):
        self.executed.append(stage)
        if stage == 'sft':
            checkpoint = runner.root / 'experiment/sft/checkpoint-last'
            checkpoint.mkdir(parents=True, exist_ok=True)
            write_json(checkpoint / 'config.json', {})
            (checkpoint / 'model.safetensors').write_text('TEST ONLY')
            log = runner.root / 'logs/sft.log'
            log.parent.mkdir(exist_ok=True)
            log.write_text('step:1 - train/loss:1.0\nstep:2 - train/loss:0.5\n')
        elif stage == 'rl':
            checkpoint = runner.root / 'experiment/grpo/global_step_100/actor/huggingface'
            checkpoint.mkdir(parents=True, exist_ok=True)
            write_json(checkpoint / 'config.json', {})
            (checkpoint / 'model.safetensors').write_text('TEST ONLY')
            log = runner.root / 'logs/grpo.log'
            log.write_text('step:1 - critic/rewards/mean:0.2 actor/entropy:0.8\n')
        elif stage in ('sft_eval', 'rl_eval'):
            if self.fail_eval and stage == 'rl_eval':
                self.fail_eval = False
                raise RuntimeError('simulated evaluation service failure')
            output = eval_output or runner.root / 'eval' / ('sft' if stage == 'sft_eval' else 'rl')
            write_json(output / 'summary.json', {'score': .4 if stage == 'sft_eval' else .6, 'n_tasks': 5})

    def execute(self, attempt='a0000'):
        runner = Runner(self.session, attempt)
        with patch.object(Runner, 'resources', lambda r: self.no_gpu(r)), \
             patch.object(Runner, 'pipeline', lambda r, *a, **kw: self.simulated_pipeline(r, *a, **kw)), \
             patch.object(Runner, 'run', lambda r, *a, **kw: self.fake_run_for_base_probe_call(r, *a, **kw)):
            runner.execute()
        return runner

    def fake_run_for_base_probe_call(self, runner, command, name):
        if name == 'probe_base':
            output = Path(command[command.index('--output-dir')+1])
            write_json(output / 'summary.json', {'score': .2, 'n_tasks': 5})
        else:
            self.original_run(runner, command, name)

    def test_full_attempt_feedback_and_resume_without_retraining(self):
        self.candidate()
        submit(self.session, 'a0000', background=False)
        self.executed, self.fail_eval = [], True
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.execute()
        self.assertEqual(Store(self.session).get('a0000')['state'], 'FAILED')
        self.execute()
        store = Store(self.session)
        self.assertEqual(store.get('a0000')['state'], 'COMPLETED')
        self.assertEqual(self.executed.count('sft'), 1)
        self.assertEqual(self.executed.count('rl'), 1)
        feedback = read_json(self.session / 'runner/attempts/a0000/feedback.json')
        self.assertAlmostEqual(feedback['sft_to_rl_gain'], .2)
        self.assertTrue((self.session / 'runner/attempts/a0000/metrics/rl.svg').exists())
        self.assertTrue(feedback['curves']['sft']['available'])
        self.assertEqual(select(self.session, 'a0000')['rl_dev_score'], .6)
        self.assertTrue(submit(self.session, 'a0000', background=False)['existing'])

    def test_freeze_and_input_tampering(self):
        folder = self.candidate()
        submit(self.session, 'a0000', background=False)
        (folder / 'synthesize.py').write_text('not the frozen code')
        self.assertNotEqual(digest(folder / 'synthesize.py'),
            digest(self.session / 'runner/snapshots/a0000/synthesize.py'))
        (self.session / 'runner/snapshots/a0000/synthesize.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'frozen attempt'):
            Runner(self.session, 'a0000').execute()
        self.assertEqual(Store(self.session).get('a0000')['state'], 'FAILED')

    def test_recipe_rejects_training_override_and_path_escape(self):
        folder = self.candidate()
        recipe = read_json(folder / 'recipe.json')
        recipe['generation'] = {'backend': 'pipeline', 'args': {'sft-epochs': 999}}
        write_json(folder / 'recipe.json', recipe)
        with self.assertRaisesRegex(ValueError, 'synthesis flag'):
            validate_attempt(self.session, 'a0000')
        recipe['generation'] = {'backend': 'program', 'entrypoint': '../synthesize.py'}
        write_json(folder / 'recipe.json', recipe)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            validate_attempt(self.session, 'a0000')

    def test_data_leakage_rejected(self):
        from DataEvo.runtime.data import prepare
        source = self.root / 'leaked.jsonl'
        shutil.copyfile(self.cfg['data']['dev'], source)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            prepare(source, self.root / 'train.parquet', self.cfg)

    def test_pipeline_uses_fixed_split_and_existing_repo_entry(self):
        command = backends.pipeline_command(self.cfg, 'a0000', self.root / 'attempt', ['2'], 'rl')
        self.assertIn(str(REPO / 'DataObs/scripts/experiment_pipeline_vllm.py'), command)
        self.assertEqual(command[command.index('--stages')+1], 'grpo')
        self.assertEqual(command[command.index('--rl-val-data')+1], self.cfg['data']['rl_val'])
        self.assertIn('--grpo-arg=trainer.total_training_steps=100', command)
        self.assertEqual(backends.flags({'cot-do-sample': False}), ['--no-cot-do-sample'])

    def test_real_pipeline_accepts_generated_sft_command(self):
        command = backends.pipeline_command(self.cfg, 'a0000', self.root / 'dryrun/a0000', ['1'], 'sft')
        completed = subprocess.run(command + ['--dry-run'], text=True, capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('sft_dataobs.sh', completed.stdout)

    def test_bootstrap_produces_nonoverlapping_splits(self):
        from DataEvo.runtime.data import load_rows, validate_splits
        train = self.root / 'raw-train.jsonl'
        train.write_text(''.join(json.dumps({'question': f'train-{i}', 'answer': '#### 2'}) + '\n'
                                 for i in range(10)))
        output = self.root / 'prepared'
        result = subprocess.run([sys.executable, str(REPO / 'DataEvo/scripts/prepare_gsm8k_session.py'),
            '--train', str(train), '--test', self.cfg['data']['final_test'], '--output', str(output),
            '--base-model', self.cfg['base_model'], '--teacher-model', 'fixture', '--heldout-size', '2'],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = validate_config(read_json(output / 'session.json'))
        validate_splits(cfg)
        self.assertEqual(len(load_rows(cfg['data']['seed'])), 4)
        self.assertEqual(len(load_rows(cfg['data']['probe'])), 2)

    def test_detached_worker_reports_real_subprocess_failure(self):
        from DataEvo.runtime.cli import wait_for
        folder = self.candidate()
        (folder / 'synthesize.py').write_text('raise RuntimeError("intentional test failure")\n')
        submit(self.session, 'a0000')
        result = wait_for(self.session, 'a0000', 20)
        self.assertEqual(result['state'], 'FAILED')
        self.assertIn('generate failed', result['error'])
        lease = read_json(self.session / 'runner/attempts/a0000/lease.json')
        self.assertIn('released', lease)

    def test_shared_gpu_lease_excludes_concurrent_runner(self):
        from DataEvo.runtime.common import lock
        self.candidate()
        submit(self.session, 'a0000', background=False)
        runner = Runner(self.session, 'a0000')
        runner.cfg['resources'].update(gpu_ids=['1'], gpu_count=1, wait_timeout_s=0)
        with lock(self.root / 'locks/gpu-1.lock'):
            with self.assertRaisesRegex(TimeoutError, 'no available GPUs'):
                with runner.resources():
                    self.fail('must not acquire an already leased GPU')
        with runner.resources():
            self.assertEqual(runner.gpus, ['1'])

    def test_probe_resource_timeout_preserves_completed_attempt(self):
        from DataEvo.runtime.cli import main
        self.candidate()
        submit(self.session, 'a0000', background=False)
        Store(self.session).update('a0000', 'COMPLETED', score=.5)
        @contextlib.contextmanager
        def unavailable(runner):
            runner.store.update(runner.attempt, 'WAITING')
            raise TimeoutError('test GPU unavailable')
            yield
        with patch.object(Runner, 'resources', unavailable), \
             patch('sys.argv', ['run_session.py', 'probe', '--session', str(self.session), '--attempt', 'a0000']), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 1)
        self.assertEqual(Store(self.session).get('a0000')['state'], 'COMPLETED')

    def test_missing_evaluation_is_not_success(self):
        from DataEvo.runtime.feedback import eval_summary
        with self.assertRaisesRegex(ValueError, 'no recognized metrics'):
            eval_summary(self.root / 'missing')

    def test_rl_lora_export_uses_sft_weights(self):
        root = self.root / 'export-attempt'
        sft = root / 'experiment/sft/checkpoint-last'
        sft.mkdir(parents=True)
        write_json(sft / 'config.json', {})
        (sft / 'model.safetensors').write_text('SFT TEST WEIGHTS')
        actor = root / 'experiment/grpo/global_step_100/actor'
        (actor / 'lora_adapter').mkdir(parents=True)
        (actor / 'lora_adapter/adapter_model.safetensors').write_text('RL TEST DELTA')
        calls = []
        backends.export_rl(self.cfg, root, lambda command, name: calls.append(command))
        self.assertEqual(calls[0][calls[0].index('--base')+1], str(sft))
        self.assertNotIn(self.cfg['base_model'], calls[0])
        command = backends.pipeline_command(self.cfg, 'a0000', root, ['1'], 'rl_eval')
        self.assertEqual(command[command.index('--base-model')+1], str(sft))

    def test_codex_exec_protocol_without_model_request(self):
        from DataEvo.runtime.codex_harness import decide
        executable = self.root / 'fake-codex'
        executable.write_text('#!' + sys.executable + '\n' + '''import json, sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == 'exec' and '--json' in args
assert args[args.index('--sandbox')+1] == 'workspace-write'
prompt = sys.stdin.read()
assert 'feedback/trajectories' in prompt
Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(
    {'action':'stop', 'attempt_id':'', 'rationale':'protocol test only'}))
print(json.dumps({'type':'thread.started','thread_id':'test-thread'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':3}}))
''')
        executable.chmod(0o700)
        fixed = config(self.session)
        fixed['codex']['executable'] = str(executable)
        write_json(self.session / 'runner/config.json', fixed)
        result = decide(self.session, 0)
        self.assertEqual(result['action'], 'stop')
        usage = Store(self.session).costs()[0]
        self.assertEqual(usage['category'], 'codex')
        self.assertEqual(usage['tokens'], 13)
        saved = read_json(self.session / 'runner/codex/decision-0000/result.json')
        self.assertEqual(saved['thread_id'], 'test-thread')

    def test_reject_unimplemented_agentic_rl(self):
        self.cfg['dataset'] = 'terminal'
        with self.assertRaisesRegex(ValueError, 'multiturn RL adapter'):
            validate_config(self.cfg)

    def test_two_controller_rounds_receive_prior_feedback(self):
        from DataEvo.runtime.cli import run_loop
        self.candidate()
        self.executed, self.fail_eval = [], False
        fixed = read_json(self.session / 'runner/config.json')
        fixed['budget']['max_attempts'] = 2
        write_json(self.session / 'runner/config.json', fixed)
        # Update test fixture's frozen identity, not production session behavior.
        manifest = read_json(self.session / 'runner/fixed_manifest.json')
        manifest[str(self.session / 'runner/config.json')] = digest(self.session / 'runner/config.json')
        write_json(self.session / 'runner/fixed_manifest.json', manifest)
        calls = []
        def decision(session, index):
            calls.append(index)
            if index == 1:
                self.assertEqual(Store(session).rows()[0]['state'], 'COMPLETED')
                feedback = read_json(session / 'runner/attempts/a0000/feedback.json')
                self.assertEqual(feedback['rl']['score'], .6)
                self.candidate('a0001')
            # Mirror the real harness's persisted decision directory.
            (session / 'runner/codex' / f'decision-{index:04d}').mkdir(parents=True)
            return {'action': 'attempt', 'attempt_id': f'a{index:04d}', 'rationale': 'test'}
        def synchronous_launch(session, attempt):
            self.execute(attempt)
        with patch('DataEvo.runtime.codex_harness.decide', decision), \
             patch('DataEvo.runtime.cli.launch', synchronous_launch):
            result = run_loop(self.session)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(len(Store(self.session).rows()), 2)
        self.assertEqual(result['rl_dev_score'], .6)


if __name__ == '__main__':
    unittest.main()
