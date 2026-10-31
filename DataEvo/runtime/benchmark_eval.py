"""Standalone, evaluation-only Harbor jobs for the two selected agent benchmarks."""
from __future__ import annotations

import csv
import importlib.metadata
import ipaddress
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .agentic_protocol import reward_value
from .common import digest, file_manifest, write_json
from .harbor_job import summarize

BENCHMARKS = ('terminal_bench_2', 'swe_bench_verified')


def make_plan(args):
    """Resolve an exact task list. This does not instantiate environments or agents."""
    from harbor.models.task.task import Task
    from DataEvo.scripts.preview_agentic_datasets import inventory

    url = urlsplit(args.api_base)
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError('api-base must be an HTTP(S) URL without credentials, query or fragment')
    model = args.model.removeprefix('openai/')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:@+-]*', model):
        raise ValueError('model must be an OpenAI-compatible served model identifier')
    selected = BENCHMARKS if args.benchmark == 'both' else (args.benchmark,)
    if args.task and len(selected) != 1:
        raise ValueError('--task requires one --benchmark')
    datasets = []
    for key in selected:
        override = getattr(args, key + '_dataset', None)
        if override:
            source = {'dataset': str(override.resolve()), 'name': key, 'expected_tasks': None}
        else:
            source = inventory(args.rsibench_root.resolve(), key)['benchmarks'][0]
        dataset = Path(source['dataset']).resolve()
        paths = sorted(p.parent for p in dataset.rglob('task.toml'))
        if not paths:
            raise ValueError(f'{key}: no tasks in {dataset}')
        if source['expected_tasks'] is not None and len(paths) != source['expected_tasks']:
            raise ValueError(f'{key}: local task count does not match RSIBench spec')
        if args.task:
            names = set(args.task)
            paths = [p for p in paths if p.relative_to(dataset).as_posix() in names]
            if len(paths) != len(names):
                raise ValueError(f'{key}: unknown --task identifier')
        elif not args.all_tasks:
            paths = paths[:args.n_tasks or 1]
            if args.n_tasks and len(paths) < args.n_tasks:
                raise ValueError(f'{key}: --n-tasks exceeds available task count')
        tasks = []
        for path in paths:
            if not Task.is_valid_dir(path):
                raise ValueError(f'invalid Harbor task: {path}')
            task = Task(path)
            if task.has_steps:
                raise ValueError(f'multi-step task layout not supported by this entry point: {path}')
            # Hash the verifier/solution too, but never put their content in model prompts.
            tasks.append({'id': path.relative_to(dataset).as_posix(), 'name': task.name,
                          'path': str(path), 'files': file_manifest(path)})
        if len({t['name'] for t in tasks}) != len(tasks):
            raise ValueError(f'{key}: duplicate Harbor task names')
        agent = ('dataevo-terminal' if args.agent_profile == 'dataevo' else
                 'terminus-2' if key == 'terminal_bench_2' else 'mini-swe-agent')
        datasets.append({'benchmark': key, 'dataset': str(dataset), 'agent': agent,
                         'available_tasks': len(list(dataset.rglob('task.toml'))),
                         'n_tasks': len(tasks), 'tasks': tasks,
                         'scope': 'local RSIBench copy' if not override else 'user-supplied local Harbor tasks'})
    return {'schema_version': 1, 'mode': 'evaluation_only',
            'harbor_version': importlib.metadata.version('harbor'),
            'model': model, 'api_base': args.api_base.rstrip('/'),
            'agent_profile': args.agent_profile, 'environment': args.environment,
            'concurrency': args.concurrency, 'n_attempts': 1, 'max_retries': 0,
            'max_turns': args.max_turns, 'max_tokens': args.max_tokens,
            'context_window': args.context_window, 'command_timeout_s': args.command_timeout,
            'max_observation_chars': args.max_observation_chars,
            'mini_swe_agent_version': '2.4.5', 'temperature': 0.0,
            'source_sha256': {p.name: digest(p) for p in
                (Path(__file__), Path(__file__).with_name('harbor_terminal_agent.py'),
                 Path(__file__).with_name('agentic_protocol.py'))},
            'benchmarks': datasets}


def endpoint_check(plan):
    key = os.environ.get('STUDENT_API_KEY') or os.environ.get('OPENAI_API_KEY') or 'local-student'
    request = urllib.request.Request(plan['api_base'] + '/models',
                                     headers={'Authorization': f'Bearer {key}'})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
        names = {row['id'] for row in payload['data']}
        if plan['model'] not in names:
            return {'ok': False, 'detail': 'requested model is not listed by GET /models'}
        return {'ok': True, 'detail': 'GET /models lists requested model; generation not tested'}
    except urllib.error.HTTPError as error:
        return {'ok': False, 'detail': f'GET /models returned HTTP {error.code}'}
    except Exception as error:
        # Avoid printing server response bodies or credentials embedded in exceptions.
        return {'ok': False, 'detail': f'GET /models failed ({type(error).__name__})'}


def preflight(plan):
    """Read-only checks only: no sandbox creation, model generation or paid calls."""
    checks = [{'name': 'task_layouts', 'ok': True,
               'detail': f"validated {sum(b['n_tasks'] for b in plan['benchmarks'])} Harbor tasks"}]
    try:
        from harbor.agents.factory import AgentFactory
        from harbor.models.agent.name import AgentName
        for agent in {b['agent'] for b in plan['benchmarks']}:
            if agent == 'dataevo-terminal':
                from .harbor_terminal_agent import TerminalAgent  # noqa: F401
            else:
                AgentFactory.get_agent_class(AgentName(agent))
        checks.append({'name': 'agent_imports', 'ok': True})
    except Exception as error:
        checks.append({'name': 'agent_imports', 'ok': False, 'detail': type(error).__name__})
    if plan['environment'] == 'e2b':
        checks.append({'name': 'e2b_credentials', 'ok': bool(os.environ.get('E2B_API_KEY')),
                       'detail': 'E2B_API_KEY required; presence check only, quota not checked'})
    else:
        try:
            proc = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                                  capture_output=True, timeout=10, check=False)
            checks.append({'name': 'docker_daemon', 'ok': proc.returncode == 0,
                           'detail': 'docker info must succeed for the current user'})
        except (OSError, subprocess.TimeoutExpired) as error:
            checks.append({'name': 'docker_daemon', 'ok': False, 'detail': type(error).__name__})
    if any(b['agent'] == 'mini-swe-agent' for b in plan['benchmarks']):
        host = urlsplit(plan['api_base']).hostname
        try:
            local = ipaddress.ip_address(host).is_loopback or ipaddress.ip_address(host).is_unspecified
        except ValueError:
            local = host.lower() == 'localhost' or host.lower().endswith('.localhost')
        checks.append({'name': 'sandbox_model_endpoint', 'ok': not local,
                       'detail': 'mini-swe-agent runs inside the sandbox: API URL must be reachable there; '
                                 'host preflight cannot verify sandbox networking'})
    checks.append({'name': 'model_endpoint', **endpoint_check(plan)})
    return {'ready': all(c['ok'] for c in checks), 'checks': checks,
            'note': 'Builds, agent installation and verifier execution are tested only by a real smoke run.'}


def mini_config(plan):
    # JSON is valid YAML, and contains no credentials. Harbor layers this over -c mini.
    return {'agent': {'step_limit': plan['max_turns']},
            'model': {'model_kwargs': {'api_base': plan['api_base'], 'temperature': plan['temperature']}}}


def build_config(plan, benchmark, output):
    from harbor import JobConfig
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.job.config import RetryConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    agent = benchmark['agent']
    if agent == 'dataevo-terminal':
        config = AgentConfig(import_path='DataEvo.runtime.harbor_terminal_agent:TerminalAgent',
            model_name=plan['model'], kwargs={'api_base': plan['api_base'],
                'max_turns': plan['max_turns'], 'max_tokens_per_turn': plan['max_tokens'],
                'max_observation_chars': plan['max_observation_chars'],
                'command_timeout_s': plan['command_timeout_s']})
    elif agent == 'mini-swe-agent':
        config = AgentConfig(name=agent, model_name='openai/' + plan['model'], kwargs={
            'config_file': str(output / 'mini_config.json'),
            'version': plan['mini_swe_agent_version'], 'max_tokens': plan['max_tokens']})
    else:
        config = AgentConfig(name=agent, model_name='openai/' + plan['model'], kwargs={
            'api_base': plan['api_base'], 'parser_name': 'json', 'max_turns': plan['max_turns'],
            'temperature': plan['temperature'], 'suppress_max_turns_warning': True,
            'model_info': {'max_input_tokens': plan['context_window'],
                           'max_output_tokens': plan['max_tokens'],
                           'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0},
            'llm_call_kwargs': {'max_tokens': plan['max_tokens']}})
    return JobConfig(job_name='evaluation', jobs_dir=output / 'jobs', quiet=True,
        n_attempts=1, n_concurrent_trials=plan['concurrency'], retry=RetryConfig(max_retries=0),
        agents=[config], tasks=[TaskConfig(path=Path(t['path']), source=benchmark['benchmark'])
                              for t in benchmark['tasks']],
        environment=EnvironmentConfig(type=EnvironmentType(plan['environment']), delete=True))


def collect_trials(output):
    rows = []
    for path in sorted((output / 'jobs').rglob('result.json')):
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            rows.append({'task_name': None, 'trial_name': None, 'status': 'error',
                         'score': None, 'result': str(path), 'exception': 'unreadable result file'})
            continue
        if 'task_name' not in raw or 'trial_name' not in raw:
            continue  # skip Harbor job-level result.json
        error = raw.get('exception_info')
        score = None
        try:
            if error or not raw.get('finished_at'):
                raise ValueError('trial errored or unfinished')
            score = reward_value((raw.get('verifier_result') or {}).get('rewards'))
        except ValueError:
            pass
        agent = raw.get('agent_result') or {}
        rows.append({'task_name': raw['task_name'], 'trial_name': raw['trial_name'],
                     'status': 'completed' if score is not None else 'error', 'score': score,
                     'exception': error, 'started_at': raw.get('started_at'),
                     'finished_at': raw.get('finished_at'), 'result': str(path),
                     'trial_directory': str(path.parent),
                     'input_tokens': agent.get('n_input_tokens'),
                     'output_tokens': agent.get('n_output_tokens'),
                     'reported_model_cost_usd': agent.get('cost_usd')})
    return rows


def summarize_benchmark(raw, benchmark, output, trials):
    expected = sorted(t['name'] for t in benchmark['tasks'])
    result = {'benchmark': benchmark['benchmark'], 'agent': benchmark['agent'],
              'status': 'incomplete', 'score': None, 'n_tasks': len(expected),
              'n_scored': sum(t['score'] is not None for t in trials),
              'raw': str(output / 'harbor_result.json'), 'trajectories': str(output / 'jobs'),
              'cost_usd': None,
              'cost_note': 'Total cost unknown. Harbor model estimates exclude sandbox/GPU charges; '
                           'unknown model prices may be reported as zero.'}
    try:
        summary = summarize(raw, output)
        if summary['n_tasks'] != len(expected):
            raise ValueError('Harbor total differs from frozen task count')
        if sorted(str(t['task_name']) for t in trials) != expected:
            raise ValueError('missing, duplicate or unexpected per-task results')
        if result['n_scored'] != len(expected):
            raise ValueError('some task results errored or have no valid reward')
        score = sum(t['score'] for t in trials) / len(expected)
        if abs(summary['score'] - score) > 1e-9:
            raise ValueError('aggregate and per-task scores disagree')
        result.update(status='completed', score=score, input_tokens=summary['input_tokens'],
                      output_tokens=summary['output_tokens'],
                      reported_model_cost_usd=summary['cost_usd'])
    except (ValueError, KeyError, TypeError) as error:
        result['error'] = str(error)
    return result


async def execute(plan, output, checks):
    from harbor import Job

    # mkdir is exclusive: never resume/overwrite another evaluation accidentally.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', plan)
    write_json(output / 'preflight.json', checks)
    os.environ['OPENAI_API_KEY'] = (os.environ.get('STUDENT_API_KEY')
                                    or os.environ.get('OPENAI_API_KEY') or 'local-student')
    os.environ['OPENAI_BASE_URL'] = plan['api_base']
    os.environ['OPENAI_API_BASE'] = plan['api_base']
    summaries = []
    for benchmark in plan['benchmarks']:
        started = time.monotonic()
        folder = output / benchmark['benchmark']
        folder.mkdir()
        write_json(folder / 'mini_config.json', mini_config(plan))
        config = build_config(plan, benchmark, folder)
        write_json(folder / 'job_config.json', config.model_dump(mode='json'))
        try:
            raw = (await Job(config).run()).model_dump(mode='json')
            write_json(folder / 'harbor_result.json', raw)
            trials = collect_trials(folder)
            result = summarize_benchmark(raw, benchmark, folder, trials)
        except Exception as error:
            # Partial trial artifacts remain available even if Job.run itself fails.
            trials = collect_trials(folder)
            result = {'benchmark': benchmark['benchmark'], 'agent': benchmark['agent'],
                      'status': 'error', 'score': None, 'n_tasks': benchmark['n_tasks'],
                      'n_scored': sum(t['score'] is not None for t in trials),
                      'error_type': type(error).__name__, 'cost_usd': None}
        result['wall_time_s'] = time.monotonic() - started
        write_json(folder / 'trials.json', trials)
        write_json(folder / 'summary.json', result)
        summaries.append(result)
        write_report(output, plan, summaries)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all(s['status'] == 'completed' for s in summaries) else 1


def write_report(output, plan, summaries):
    write_json(output / 'summary.json', {'model': plan['model'], 'agent_profile': plan['agent_profile'],
                                        'benchmarks': summaries})
    fields = ['benchmark', 'agent', 'status', 'score', 'n_tasks', 'n_scored', 'wall_time_s']
    with (output / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    lines = [f"Model: {plan['model']} | agent profile: {plan['agent_profile']}", '',
             '| Benchmark | Agent | Status | Pass@1 / resolved rate | Scored / selected |',
             '| --- | --- | --- | --- | --- |']
    for row in summaries:
        score = f"{100 * row['score']:.2f}%" if row['score'] is not None else 'N/A'
        lines.append(f"| {row['benchmark']} | {row['agent']} | {row['status']} | {score} | "
                     f"{row['n_scored']} / {row['n_tasks']} |")
    lines += ['', 'One attempt per task. See request.json for exact task IDs and file hashes.',
              'Incomplete jobs have no aggregate score. Total API/sandbox/GPU cost is unknown.',
              'Different agent profiles or task subsets are not directly comparable.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
