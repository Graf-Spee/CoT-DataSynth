"""Isolated optional-dependency Harbor entry point; matches RSIBench's Job API."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path


def build_config(spec):
    from harbor import JobConfig
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig

    output = Path(spec['output'])
    source = Path(spec['dataset'])
    name, separator, version = spec['dataset'].partition('@')
    dataset = (DatasetConfig(path=source, n_tasks=spec['n_tasks']) if source.exists() else
               DatasetConfig(name=name, version=version if separator else None, n_tasks=spec['n_tasks']))
    agent = AgentConfig(name=spec['agent'], model_name=spec['model'], kwargs={
        'api_base': spec['api_base'], 'parser_name': 'json', 'max_turns': spec['max_turns'],
        'temperature': 0.0,
        'model_info': {'max_input_tokens': spec.get('context_window', 32768),
                       'max_output_tokens': spec['max_tokens'],
                       'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0},
        'llm_call_kwargs': {'max_tokens': spec['max_tokens']},
    })
    if spec['agent'] == 'dataevo-terminal':
        agent = AgentConfig(import_path='DataEvo.runtime.harbor_terminal_agent:TerminalAgent',
                            model_name=spec['model'], kwargs={'api_base': spec['api_base'], **spec['terminal']})
    return JobConfig(job_name='evaluation', jobs_dir=output / 'jobs', n_attempts=1,
                           n_concurrent_trials=spec['concurrency'], quiet=True,
                           agents=[agent], datasets=[dataset],
                           environment=EnvironmentConfig(type=EnvironmentType(spec['environment']), delete=True))


def summarize(raw, output, reward_key='reward'):
    from DataEvo.runtime.agentic_protocol import reward_value
    stats = raw['stats']
    total = raw.get('n_total_trials', stats.get('n_completed_trials', stats.get('n_trials', 0)))
    pending = sum(stats.get(k, 0) for k in ('n_pending_trials', 'n_running_trials', 'n_cancelled_trials'))
    errors = stats.get('n_errored_trials', stats.get('n_errors', 0))
    weighted, counted = 0.0, 0
    for group in stats.get('evals', {}).values():
        rewards = group.get('reward_stats', {}).get(reward_key, {})
        for reward, trials in rewards.items():
            weighted += reward_value({reward_key: float(reward)}, reward_key) * len(trials)
            counted += len(trials)
    if not total or counted != total or pending or errors:
        raise ValueError(f'incomplete Harbor evaluation: total={total}, scored={counted}, errors={errors}, pending={pending}')
    return {'score': weighted / total, 'n_tasks': total, 'n_scored': counted,
            'n_errors': errors, 'raw': str(output / 'harbor_result.json'),
            'trajectories': str(output / 'jobs'), 'cost_usd': stats.get('cost_usd'),
            'input_tokens': stats.get('n_input_tokens'), 'output_tokens': stats.get('n_output_tokens')}


async def execute(spec):
    from harbor import Job
    # LiteLLM reads the OpenAI provider key from the environment. Keeping it out
    # of AgentConfig prevents Harbor's persisted job config from containing it.
    os.environ['OPENAI_API_KEY'] = os.environ.get('STUDENT_API_KEY', 'local-student')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=True)
    result = await Job(build_config(spec)).run()
    raw = result.model_dump(mode='json')
    (output / 'harbor_result.json').write_text(json.dumps(raw, indent=2))
    summary = summarize(raw, output, spec.get('reward_key', 'reward'))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    asyncio.run(execute(json.loads(Path(args.spec).read_text())))
