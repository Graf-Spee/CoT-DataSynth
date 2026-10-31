"""TEST ONLY: real Harbor BridgeAgent/protocol, in-memory fake sandbox/verifier."""
import asyncio
import contextlib
import json
import signal
import sys
from types import SimpleNamespace

from harbor.models.agent.context import AgentContext
from DataEvo.runtime.harbor_bridge import BridgeAgent, emit


class FakeEnvironment:
    value = None

    async def exec(self, command, timeout_sec):
        if command.startswith('set '):
            self.value = command[4:]
        elif command != 'get':
            raise ValueError('fixture accepts only set/get; no shell execution')
        return SimpleNamespace(stdout=self.value, stderr='', return_code=0)


async def run(output):
    from pathlib import Path
    output = Path(output)
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    env = FakeEnvironment()
    agent = BridgeAgent(logs_dir=output, max_turns=4)
    try:
        await agent.run('fixture task', env, AgentContext())
        emit({'event': 'verified', 'reward': float(env.value == 'correct'), 'result': 'TEST ONLY'})
    finally:
        (output / 'released').touch()


if __name__ == '__main__':
    with contextlib.redirect_stdout(sys.stderr):
        asyncio.run(run(sys.argv[1]))
