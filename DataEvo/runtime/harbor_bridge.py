"""Run a real Harbor Trial in Harbor's Python, controlled over JSONL stdin/stdout.

Harbor owns environment setup, verifier execution, artifact download and teardown.
No command is executed on the training host. Requires Harbor 0.17.1.
"""
import argparse
import asyncio
import contextlib
import json
import signal
import sys
from pathlib import Path

from harbor.agents.base import BaseAgent

from DataEvo.runtime.agentic_protocol import reward_value


def emit(value):
    print(json.dumps(value, ensure_ascii=False), file=sys.__stdout__, flush=True)


class BridgeAgent(BaseAgent):
    def __init__(self, *args, command_timeout_s=30, max_observation_chars=12000, max_turns=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_timeout_s = command_timeout_s
        self.max_chars = max_observation_chars
        self.max_turns = max_turns

    @staticmethod
    def name():
        return 'dataevo-bridge'

    def version(self):
        return '1'

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        reader = asyncio.StreamReader()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
        emit({'event': 'ready', 'instruction': instruction})
        commands = 0
        try:
            while True:
                line = await reader.readline()
                if not line:
                    raise RuntimeError('controller disconnected before verification')
                request = json.loads(line)
                if request.get('op') == 'verify':
                    context.metadata = {'commands': commands}
                    return  # Harbor now runs its configured verifier and tears down.
                if request.get('op') != 'exec' or commands >= self.max_turns:
                    raise ValueError('invalid command or episode turn limit reached')
                command = request.get('command')
                if not isinstance(command, str) or not command.strip():
                    raise ValueError('command must be nonempty')
                commands += 1
                result = await environment.exec(command=command, timeout_sec=self.command_timeout_s)
                emit({'event': 'observation', 'stdout': (result.stdout or '')[:self.max_chars],
                      'stderr': (result.stderr or '')[:self.max_chars], 'return_code': result.return_code})
        finally:
            transport.close()


def trial_config(spec, task_id, split, output):
    from harbor.models.trial.config import TrialConfig, TaskConfig, AgentConfig, EnvironmentConfig
    task_path = spec['tasks'][split].get(task_id)
    if not task_path:
        raise ValueError('task is outside the fixed task allowlist')
    return TrialConfig(task=TaskConfig(path=Path(task_path)), trial_name='trial', trials_dir=output,
        agent=AgentConfig(import_path='DataEvo.runtime.harbor_bridge:BridgeAgent',
            override_timeout_sec=spec['episode_timeout_s'], kwargs={k: spec[k] for k in
                ('command_timeout_s', 'max_observation_chars', 'max_turns')}),
        environment=EnvironmentConfig(type=spec['environment'], delete=True))


async def serve(spec, task_id, split, output):
    from harbor.trial.trial import Trial
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    trial = await Trial.create(trial_config(spec, task_id, split, output))
    result = await trial.run()
    if result.exception_info is not None or result.verifier_result is None:
        raise RuntimeError(f'Harbor episode failed; inspect {trial.paths.result_path}')
    rewards = result.verifier_result.rewards
    emit({'event': 'verified', 'reward': reward_value(rewards, spec.get('reward_key', 'reward')),
          'rewards': rewards, 'result': str(trial.paths.result_path)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Harbor and vendor SDK chatter must never corrupt the protocol stream.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            asyncio.run(serve(json.loads(args.spec.read_text()), args.task_id, args.split, args.output))
        except BaseException as error:
            emit({'event': 'error', 'error': type(error).__name__, 'detail': str(error)})
            raise


if __name__ == '__main__':
    main()
