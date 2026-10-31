"""Async lifecycle client for a per-episode Harbor subprocess."""
import asyncio
import contextlib
import json
import time
from pathlib import Path
from uuid import uuid4

from .common import REPO, lock, read_json, write_json


class HarborEpisode:
    def __init__(self, spec_path, task_id, split, phase='rl'):
        self.spec_path = Path(spec_path)
        self.spec = read_json(self.spec_path)
        if task_id not in self.spec['tasks'].get(split, {}):
            raise ValueError('task is outside the fixed task allowlist')
        self.task_id, self.split = task_id, split
        self.output = Path(self.spec['output_dir']) / phase / uuid4().hex
        self.process, self.log = None, None
        self.started = None
        self.result = None

    async def __aenter__(self):
        ledger = Path(self.spec['output_dir'])
        ledger.mkdir(parents=True, exist_ok=True)
        with lock(ledger / 'budget.lock'):
            counter = ledger / 'budget.json'
            usage = read_json(counter) if counter.exists() else {'episodes_reserved': 0}
            if usage['episodes_reserved'] >= self.spec['max_episodes']:
                raise ValueError('Harbor episode budget exhausted')
            usage['episodes_reserved'] += 1
            write_json(counter, usage)
        self.output.mkdir(parents=True)
        self.started = time.monotonic()
        self.log = (self.output / 'bridge.log').open('w')
        try:
            self.process = await asyncio.create_subprocess_exec(self.spec['harbor_python'], '-m',
                'DataEvo.runtime.harbor_bridge', '--spec', str(self.spec_path), '--task-id', self.task_id,
                '--split', self.split, '--output', str(self.output), cwd=REPO,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log,
                limit=max(65536, self.spec['max_observation_chars'] * 12))
            self.ready = await self._read('ready')
            return self
        except BaseException:
            await self.close()
            raise

    async def _read(self, event):
        remaining = self.spec['episode_timeout_s'] - (time.monotonic() - self.started)
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout=max(.01, remaining))
        if not line:
            raise RuntimeError(f'Harbor bridge exited; inspect {self.output / "bridge.log"}')
        result = json.loads(line)
        if result.get('event') != event:
            raise RuntimeError(f'Harbor protocol expected {event}, got {result}')
        return result

    async def request(self, op, **kwargs):
        self.process.stdin.write((json.dumps({'op': op, **kwargs}) + '\n').encode())
        await self.process.stdin.drain()
        result = await self._read('verified' if op == 'verify' else 'observation')
        if op == 'verify':
            self.result = result
        return result

    async def close(self):
        cleanup_error = None
        try:
            if self.process and self.process.returncode is None:
                if self.result is None:
                    # SIGTERM cancels Trial; Harbor's finally deletes its environment.
                    with contextlib.suppress(ProcessLookupError):
                        self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), self.spec.get('cleanup_timeout_s', 120))
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                    cleanup_error = 'Harbor teardown timed out; inspect bridge.log and provider for leaked sandbox'
            if self.result is not None and self.process.returncode != 0:
                cleanup_error = f'Harbor bridge exit status {self.process.returncode}'
        finally:
            if self.log:
                self.log.close()
            if self.started is not None:
                write_json(self.output / 'episode.json', {'task_id': self.task_id, 'split': self.split,
                    'elapsed_s': time.monotonic() - self.started, 'result': self.result,
                    'cleanup_error': cleanup_error, 'cost_usd': None})
        if cleanup_error:
            raise RuntimeError(cleanup_error)

    async def __aexit__(self, *exc):
        await self.close()
