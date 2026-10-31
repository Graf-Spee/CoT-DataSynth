"""Harbor evaluation agent using the identical JSON terminal protocol as GRPO."""
import json
import os

from harbor.agents.base import BaseAgent
from openai import AsyncOpenAI

from .agentic_protocol import action, observation, prompt


class TerminalAgent(BaseAgent):
    def __init__(self, *args, api_base, max_turns=16, max_tokens_per_turn=512,
                 max_observation_chars=12000, command_timeout_s=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_base, self.max_turns = api_base, max_turns
        self.max_tokens, self.max_chars = max_tokens_per_turn, max_observation_chars
        self.command_timeout_s = command_timeout_s

    @staticmethod
    def name():
        return 'dataevo-terminal'

    def version(self):
        return '1'

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        messages, events = prompt(instruction), []
        context.n_input_tokens = context.n_output_tokens = 0
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        client = AsyncOpenAI(base_url=self.api_base, api_key=(os.environ.get('STUDENT_API_KEY')
                            or os.environ.get('OPENAI_API_KEY') or 'local-student'),
                             timeout=180, max_retries=0)
        try:
            for _ in range(self.max_turns):
                completion = await client.chat.completions.create(model=self.model_name.removeprefix('openai/'),
                    messages=messages, temperature=0, max_tokens=self.max_tokens,
                    extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                if completion.usage:
                    context.n_input_tokens += completion.usage.prompt_tokens
                    context.n_output_tokens += completion.usage.completion_tokens
                choice = completion.choices[0]
                messages.append({'role': 'assistant', 'content': choice.message.content or ''})
                try:
                    kind, value = action(messages[-1]['content'])
                except (ValueError, TypeError):
                    context.metadata = {'termination': 'invalid_action'}
                    break
                if kind == 'final' or choice.finish_reason == 'length':
                    context.metadata = {'termination': 'final' if kind == 'final' else 'truncated_action'}
                    break
                result = await environment.exec(command=value, timeout_sec=self.command_timeout_s)
                obs = observation({'stdout': result.stdout, 'stderr': result.stderr,
                                   'return_code': result.return_code}, self.max_chars)
                events.append({'command': value, 'observation': obs})
                messages.append(obs)
        finally:
            await client.close()
            (self.logs_dir / 'trajectory.json').write_text(json.dumps({'messages': messages, 'events': events},
                                                                       ensure_ascii=False, indent=2))
