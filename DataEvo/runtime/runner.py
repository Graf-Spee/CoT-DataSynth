from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import backends
from .common import REPO, STAGES, Store, config, file_manifest, lock, read_json, write_json
from .data import prepare
from .feedback import build_feedback, eval_summary


def terminate(process, grace=10):
    until = time.monotonic() + grace
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    if process.poll() is None:
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    if grace > 10:
        # A shell/Ray parent can exit before Harbor descendants finish deleting
        # remote environments. Allow their SIGTERM cleanup to finish as well.
        while time.monotonic() < until:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.2)
    # Training launchers may exit before their rollout/serving descendants.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Runner:
    def __init__(self, session, attempt):
        self.store = Store(session)
        self.cfg = config(session)
        self.attempt = attempt
        self.snapshot = self.store.session / 'runner/snapshots' / attempt
        self.root = self.store.session / 'runner/attempts' / attempt
        self.root.mkdir(parents=True, exist_ok=True)
        self.recipe = read_json(self.snapshot / 'recipe.json')
        self.env = os.environ.copy()
        self.env['PYTHONPATH'] = str(REPO) + os.pathsep + self.env.get('PYTHONPATH', '')
        self.env['PATH'] = str(Path(self.cfg['python']).parent) + os.pathsep + self.env.get('PATH', '')
        self.env['CONFIG_DIR'] = str(REPO / 'config')
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'
        if self.env.get('STUDENT_API_KEY'):
            self.env['VLLM_API_KEY'] = self.env['STUDENT_API_KEY']
        self.gpus = []
        self.children = []
        self.deadline = time.monotonic() + max(0, read_json(self.store.session / 'runner/session.json')['deadline'] - time.time())

    @contextlib.contextmanager
    def process(self, command, name):
        logs = self.root / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / f'{name}.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=self.snapshot, env=self.env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(process)
            try:
                yield process
            finally:
                grace = 10
                if self.cfg['training'].get('agentic_rl'):
                    grace = self.cfg['training']['agentic_rl'].get('cleanup_timeout_s', 120) + 5
                terminate(process, grace)
                self.children.remove(process)

    def run(self, command, name):
        start = time.time()
        with (self.root / 'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': name, 'argv': command, 'time': start}) + '\n')
        timeout = min(self.cfg['budget']['stage_timeout_s'], self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError('runner deadline exceeded')
        try:
            with self.process(command, name) as process:
                until = time.monotonic() + timeout
                while True:
                    try:
                        code = process.wait(timeout=min(5, max(0.001, until - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        with self.store.db:
                            self.store.db.execute('UPDATE attempts SET updated=? WHERE id=?', (time.time(), self.attempt))
                        if time.monotonic() >= until:
                            raise TimeoutError(f'{name} exceeded its time budget')
            if code:
                raise RuntimeError(f'{name} failed (exit {code}); see {self.root}/logs/{name}.log')
        finally:
            elapsed = time.time() - start
            rate = self.cfg['resources'].get('gpu_usd_per_hour')
            self.store.usage(self.attempt, 'compute', cost=None if rate is None else
                             elapsed * len(self.gpus) * rate / 3600,
                             detail={'stage': name, 'elapsed_s': elapsed, 'gpu_ids': self.gpus})
            if self.cfg['training'].get('agentic_rl'):
                self.record_episodes()

    def record_episodes(self):
        recorded = {json.loads(row['detail']).get('episode') for row in self.store.db.execute(
            "SELECT detail FROM usage WHERE attempt=? AND category='harbor_episode'", (self.attempt,))}
        for path in (self.root / 'agentic').glob('*/*/episode.json'):
            if str(path) not in recorded:
                value = read_json(path)
                self.store.usage(self.attempt, 'harbor_episode', cost=value.get('cost_usd'),
                                 detail={'episode': str(path), **value})

    @contextlib.contextmanager
    def resources(self):
        resource = self.cfg['resources']
        until = min(self.deadline, time.monotonic() + resource.get('wait_timeout_s', 300))
        self.store.update(self.attempt, 'WAITING')
        while True:
            stack = contextlib.ExitStack()
            selected = []
            for gpu in sorted(map(str, resource['gpu_ids'])):
                if not all(c.isalnum() or c in '-_' for c in gpu):
                    raise ValueError('invalid GPU identifier')
                try:
                    candidate = contextlib.ExitStack()
                    candidate.enter_context(lock(Path(resource['lock_dir']) / f'gpu-{gpu}.lock', blocking=False))
                    if resource.get('check_gpu_memory', True):
                        used = subprocess.check_output(['nvidia-smi', '-i', gpu,
                            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                        if int(used.strip()) > resource.get('max_used_memory_mib', 1024):
                            candidate.close()
                            continue
                    stack.enter_context(candidate)
                    selected.append(gpu)
                    if len(selected) == resource['gpu_count']:
                        break
                except BlockingIOError:
                    continue
                except BaseException:
                    candidate.close()
                    stack.close()
                    raise
            if len(selected) == resource['gpu_count']:
                try:
                    self.gpus = selected
                    self.env['CUDA_VISIBLE_DEVICES'] = ','.join(selected)
                    write_json(self.root / 'lease.json', {'gpu_ids': selected, 'pid': os.getpid(), 'time': time.time()})
                    yield
                finally:
                    write_json(self.root / 'lease.json', {'gpu_ids': selected, 'pid': os.getpid(),
                                                         'released': time.time()})
                    stack.close()
                return
            stack.close()
            if time.monotonic() >= until:
                raise TimeoutError('no available GPUs within resource wait timeout')
            time.sleep(1)

    def pipeline(self, stage, generation=None, eval_data=None, eval_output=None):
        self.run(backends.pipeline_command(self.cfg, self.attempt, self.root, self.gpus,
                                          stage, generation, eval_data, eval_output), 'pipeline_' + stage)

    def context(self):
        value = {'schema_version': 1, 'session': str(self.store.session), 'attempt_id': self.attempt,
                 'data': {'seed': self.cfg['data']['seed']},
                 'base_model': self.cfg['base_model'], 'gpu_ids': self.gpus,
                 'budget': self.cfg['budget'], 'generation': self.recipe['generation'],
                 'teacher_api': self.cfg.get('teacher_api'), 'history': self.store.rows(),
                 'repo_root': str(REPO), 'artifact_dir': str(self.root)}
        if self.cfg['training'].get('agentic_rl'):
            from .agentic_config import write_spec
            value['agentic_spec'] = str(write_spec(self.cfg, self.root))
        path = self.root / 'context.json'
        write_json(path, value)
        return path

    def evaluate(self, stage, data=None, output=None, sampling=None):
        output = output or self.root / 'eval' / stage
        if stage == 'rl':
            backends.export_rl(self.cfg, self.root, self.run)
        if self.cfg['evaluation']['backend'] == 'vllm':
            if stage == 'base' or sampling:
                model = Path(self.cfg['base_model']) if stage == 'base' else backends.checkpoint(self.root, stage)
                self.run(backends.evaluate_model(self.cfg, model, data or self.cfg['data']['dev'],
                                                 output, self.gpus, sampling), 'probe_' + stage)
            else:
                self.pipeline('sft_eval' if stage == 'sft' else 'rl_eval', eval_data=data, eval_output=output)
        else:
            model = Path(self.cfg['base_model']) if stage == 'base' else backends.checkpoint(self.root, stage)
            backends.harbor_eval(self.cfg, model, output, self.gpus, self.run, self.process, dataset=data)
        summary = eval_summary(output)
        if summary.get('cost_usd') is not None:
            self.store.usage(self.attempt, 'evaluation_api', cost=summary['cost_usd'],
                             tokens=(summary.get('input_tokens') or 0) + (summary.get('output_tokens') or 0))
        return summary

    def stage(self, name):
        context = self.context()
        generation = self.recipe['generation']
        if name == 'generate':
            if generation['backend'] == 'pipeline':
                self.pipeline('generate', generation)
            else:
                self.run([self.cfg['python'], str(self.snapshot / generation['entrypoint']),
                          '--context', str(context), '--output', str(self.root / 'candidates.jsonl')], name)
        elif name == 'postprocess':
            source = self.root / ('candidates.parquet' if generation['backend'] == 'pipeline' else 'candidates.jsonl')
            if not source.is_file():
                raise ValueError('synthesis produced no candidates')
            spec = self.recipe.get('postprocess', {})
            if spec.get('entrypoint'):
                self.run([self.cfg['python'], str(self.snapshot / spec['entrypoint']), '--context', str(context),
                          '--input', str(source), '--output', str(self.root / 'selected.jsonl')], name)
            else:
                import shutil
                shutil.copyfile(source, self.root / ('selected' + source.suffix))
        elif name == 'validate_data':
            selected = next((self.root / p for p in ('selected.jsonl', 'selected.parquet')
                             if (self.root / p).exists()), None)
            if selected is None:
                raise ValueError('postprocessing produced no selected data')
            prepare(selected, self.root / 'dataset/train.parquet', self.cfg)
        elif name == 'sft':
            self.pipeline(name)
            backends.checkpoint(self.root, 'sft')
        elif name == 'rl':
            command = self.cfg['training'].get('rl_command')
            if command:
                replacements = {'checkpoint': str(backends.checkpoint(self.root, 'sft')),
                                'output': str(self.root / 'experiment/grpo'),
                                'gpu_ids': ','.join(self.gpus), 'context': str(context)}
                self.run([p.format_map(replacements) for p in command], name)
            else:
                self.pipeline(name)
        elif name in ('sft_eval', 'rl_eval'):
            self.evaluate(name.split('_')[0])
        elif name == 'probe':
            result = {}
            for stage in ('base', 'sft', 'rl'):
                source = (self.cfg['evaluation'].get('probe_dataset')
                          if self.cfg['evaluation']['backend'] == 'harbor' else self.cfg['data']['probe'])
                if not source:
                    raise ValueError('Harbor requires a separate probe_dataset')
                base_cache = self.store.session / 'runner/base_probe.json'
                if stage == 'base' and base_cache.exists():
                    result[stage] = read_json(base_cache)
                else:
                    result[stage] = self.evaluate(stage, source, self.root / 'probes' / stage)
                    if stage == 'base':
                        write_json(base_cache, result[stage])
            write_json(self.root / 'probes/summary.json', result)
        elif name == 'feedback':
            feedback = build_feedback(self.store, self.attempt, self.cfg)
            self.store.update(self.attempt, 'FEEDBACK', score=feedback['rl']['score'])

    def execute(self):
        with lock(self.store.session / 'runner' / 'execution.lock', blocking=False):
            state = self.store.get(self.attempt)
            if state['state'] == 'COMPLETED':
                return
            with self.store.db:
                self.store.db.execute('UPDATE attempts SET pid=? WHERE id=?', (os.getpid(), self.attempt))
            try:
                expected = read_json(self.store.session / 'runner/manifests' / f'{self.attempt}.json')
                actual = {k: v for k, v in file_manifest(self.snapshot).items() if '__pycache__' not in Path(k).parts}
                if actual != expected['files']:
                    raise ValueError('frozen attempt was modified; refusing to execute')
                for path, expected_hash in expected['fixed_files'].items():
                    from .common import digest
                    if digest(path) != expected_hash:
                        raise ValueError(f'fixed profile/data/source changed: {path}; start a new session')
                for path, metadata in read_json(self.store.session / 'runner/base_model_manifest.json').items():
                    stat = Path(path).stat()
                    if stat.st_size != metadata['bytes'] or stat.st_mtime_ns != metadata['mtime_ns']:
                        raise ValueError('base model changed during session')
                with self.resources():
                    for name in STAGES:
                        previous = self.store.db.execute('SELECT state FROM stages WHERE attempt=? AND name=?',
                                                         (self.attempt, name)).fetchone()
                        if previous and previous['state'] == 'COMPLETED':
                            artifact_file = self.root / 'stage_artifacts' / f'{name}.json'
                            if artifact_file.exists():
                                from .common import digest
                                for path, expected_hash in read_json(artifact_file).items():
                                    if not Path(path).is_file() or digest(path) != expected_hash:
                                        raise ValueError(f'completed stage artifact changed: {path}')
                            continue
                        self.store.update(self.attempt, name.upper())
                        with self.store.db:
                            self.store.db.execute('INSERT OR REPLACE INTO stages VALUES(?,?,?,?,NULL)',
                                                  (self.attempt, name, 'RUNNING', time.time()))
                        self.stage(name)
                        artifacts = {
                            'generate': [self.root / ('candidates.parquet' if self.recipe['generation']['backend'] == 'pipeline' else 'candidates.jsonl')],
                            'postprocess': list(self.root.glob('selected.*')),
                            'validate_data': [self.root / 'dataset/train.parquet', self.root / 'dataset/summary.json'],
                            'sft_eval': list((self.root / 'eval/sft').rglob('*.json')),
                            'rl_eval': list((self.root / 'eval/rl').rglob('*.json')),
                            'probe': [self.root / 'probes/summary.json'],
                            'feedback': [self.root / 'feedback.json'],
                        }.get(name, [])
                        from .common import digest
                        write_json(self.root / 'stage_artifacts' / f'{name}.json',
                                   {str(p): digest(p) for p in artifacts if p.is_file()})
                        with self.store.db:
                            self.store.db.execute('UPDATE stages SET state=?,ended=? WHERE attempt=? AND name=?',
                                                  ('COMPLETED', time.time(), self.attempt, name))
                    self.store.update(self.attempt, 'COMPLETED')
                    leaderboard = sorted([r for r in self.store.rows() if r['state'] == 'COMPLETED'],
                                         key=lambda r: r['score'], reverse=True)
                    write_json(self.store.session / 'runner/leaderboard.json', leaderboard)
            except BaseException as error:
                self.store.update(self.attempt, 'FAILED', str(error))
                raise
            finally:
                for process in list(self.children):
                    terminate(process)


def worker(session, attempt):
    runner = Runner(session, attempt)
    def interrupted(signum, frame):
        raise InterruptedError(f'worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    runner.execute()
