from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import REPO, STAGES, Store, config, digest, file_manifest, identifier, lock, read_json, write_json
from .contracts import validate_attempt, validate_config


def initialize(session, config_path):
    cfg = validate_config(read_json(config_path))
    from .data import validate_splits
    validate_splits(cfg)
    if cfg['training'].get('agentic_rl'):
        for kind, interpreter in (('training', cfg['python']), ('harbor', cfg['harbor_python'])):
            check = subprocess.run([interpreter, '-m', 'DataEvo.runtime.agentic_preflight', '--kind', kind,
                                    '--model', cfg['base_model'], '--config', str(Path(config_path).resolve())],
                                   cwd=REPO, capture_output=True, text=True, timeout=120)
            if check.returncode:
                raise ValueError(f'{kind} preflight failed before allocating API/GPU:\n{check.stderr}')
    if session.exists() and any(session.iterdir()):
        raise ValueError('init requires an empty session directory')
    (session / 'runner').mkdir(parents=True, exist_ok=True)
    (session / 'workspace/attempts').mkdir(parents=True)
    write_json(session / 'runner/config.json', cfg)
    write_json(session / 'runner/session.json', {'created': time.time(),
               'deadline': time.time() + cfg['budget']['session_timeout_s']})
    template = REPO / 'DataEvo/workspace_template'
    shutil.copyfile(template / 'RESEARCH.md', session / 'workspace/RESEARCH.md')
    kind = 'agentic_attempt' if cfg['training'].get('agentic_rl') else 'attempt'
    shutil.copytree(template / kind, session / 'workspace/attempts/a0000')
    write_json(session / 'workspace/session.json', {
        'session': str(session), 'dataset': cfg['dataset'], 'base_model': cfg['base_model'],
        'data': {k: v for k, v in cfg['data'].items() if k != 'final_test'},
        'teacher_api': cfg.get('teacher_api'), 'budget': cfg['budget']})
    # Freeze fixed source/data identities at session initialization.
    source = {}
    for top in ('DataObs', 'DataEvo/runtime', 'DataEvo/config', 'verl', 'config'):
        for path in (REPO / top).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh', '.yaml', '.json', '.env', '.jinja') and '__pycache__' not in path.parts:
                source[str(path)] = digest(path)
                copy = session / 'runner/source_snapshot' / path.relative_to(REPO)
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, copy)
    for path in cfg['data'].values():
        if isinstance(path, str) and Path(path).is_file():
            source[str(Path(path).resolve())] = digest(path)
    if cfg['training'].get('agentic_rl'):
        manifest = Path(cfg['training']['agentic_rl']['task_manifest'])
        source[str(manifest.resolve())] = digest(manifest)
        for mapping in read_json(manifest).values():
            for directory in mapping.values():
                for path in Path(directory).rglob('*'):
                    if path.is_file():
                        source[str(path.resolve())] = digest(path)
    if cfg['evaluation']['backend'] == 'harbor':
        for key in ('dataset', 'probe_dataset', 'final_dataset'):
            value = cfg['evaluation'].get(key)
            if value and Path(value).is_dir():
                for path in Path(value).rglob('*'):
                    if path.is_file():
                        source[str(path.resolve())] = digest(path)
    source[str(session / 'runner/config.json')] = digest(session / 'runner/config.json')
    write_json(session / 'runner/fixed_manifest.json', source)
    try:
        dependencies = subprocess.run([cfg['python'], '-m', 'pip', 'freeze'], capture_output=True,
                                      text=True, timeout=30)
        (session / 'runner/dependencies.txt').write_text(dependencies.stdout or dependencies.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        (session / 'runner/dependencies.txt').write_text(f'Unavailable: {error}\n')
    model_files = {}
    for path in Path(cfg['base_model']).rglob('*'):
        if path.is_file():
            stat = path.stat()
            model_files[str(path)] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if path.suffix in ('.json', '.jinja', '.txt'):
                model_files[str(path)]['sha256'] = digest(path)
    write_json(session / 'runner/base_model_manifest.json', model_files)
    Store(session)
    return {'session': str(session), 'state': 'INITIALIZED', 'config_sha256': digest(session / 'runner/config.json')}


def submit(session, attempt, background=True):
    store = Store(session)
    cfg = config(session)
    identifier(attempt)
    with lock(session / 'runner/submission.lock'):
        if (session / 'runner/final_test_started.json').exists():
            raise ValueError('session is closed after final-test; no further optimization')
        if time.time() >= read_json(session / 'runner/session.json')['deadline']:
            raise ValueError('session deadline exceeded')
        rows = store.rows()
        if any(r['id'] == attempt for r in rows):
            return {'job_id': attempt, 'state': store.get(attempt)['state'], 'existing': True}
        if any(r['state'] not in ('COMPLETED', 'FAILED', 'CANCELLED') for r in rows):
            raise ValueError('a session can have only one active attempt')
        if len(rows) >= cfg['budget']['max_attempts']:
            raise ValueError('attempt budget exhausted')
        recipe = validate_attempt(session, attempt)
        if recipe.get('parent_attempt_id'):
            store.get(recipe['parent_attempt_id'])
        source = session / 'workspace/attempts' / attempt
        target = session / 'runner/snapshots' / attempt
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        write_json(session / 'runner/manifests' / f'{attempt}.json', {
            'files': file_manifest(target), 'fixed_files': read_json(session / 'runner/fixed_manifest.json'),
            'created': time.time(), 'recipe': recipe})
        now = time.time()
        with store.db:
            store.db.execute('INSERT INTO attempts(id,state,created,updated) VALUES(?,?,?,?)',
                              (attempt, 'QUEUED', now, now))
    if background:
        launch(session, attempt)
    return {'job_id': attempt, 'state': 'QUEUED', 'snapshot': str(target)}


def launch(session, attempt):
    root = session / 'runner/attempts' / attempt
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(REPO / 'DataEvo/scripts/run_session.py'),
                                    'worker', '--session', str(session), '--attempt', attempt],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=REPO)
    # Reap child workers in a long-lived controller; daemon thread does not keep
    # a submit-only CLI alive. The worker has its own session and survives exit.
    threading.Thread(target=process.wait, daemon=True).start()
    store = Store(session)
    with store.db:
        store.db.execute('UPDATE attempts SET pid=? WHERE id=?', (process.pid, attempt))
    return process.pid


def select(session, attempt):
    store = Store(session)
    row = store.get(attempt)
    if row['state'] != 'COMPLETED':
        raise ValueError('only a completed, evaluated attempt may be selected')
    result = {'attempt_id': attempt, 'rl_dev_score': row['score'], 'selected_at': time.time(),
              'feedback': str(session / 'runner/attempts' / attempt / 'feedback.json')}
    write_json(session / 'runner/selected.json', result)
    return result


def run_loop(session):
    from .codex_harness import decide
    store = Store(session)
    cfg = config(session)
    deadline = time.monotonic() + max(0, read_json(session / 'runner/session.json')['deadline'] - time.time())
    with lock(session / 'runner/controller.lock', blocking=False):
        if (session / 'runner/final_test_started.json').exists():
            raise ValueError('session is closed after final-test')
        while True:
            # Recover a job submitted before the previous controller disconnected.
            pending = [r for r in store.rows() if r['state'] not in ('COMPLETED', 'FAILED', 'CANCELLED')]
            if pending:
                job = pending[0]['id']
            else:
                if len(store.rows()) >= cfg['budget']['max_attempts']:
                    break
                if time.monotonic() >= deadline:
                    break
                events = session / 'runner/codex'
                index = max([int(p.name.split('-')[-1]) for p in events.glob('decision-*')] + [-1]) + 1
                decision = decide(session, index)
                if decision['action'] == 'stop':
                    break
                job = decision['attempt_id']
                if any(r['id'] == job for r in store.rows()):
                    raise ValueError('Codex must return a new attempt id for a new experiment')
                submit(session, job)
            wait_for(session, job, max(0, deadline - time.monotonic()))
            print(json.dumps(store.get(job), ensure_ascii=False), flush=True)
            if time.monotonic() >= deadline:
                break
        complete = [r for r in store.rows() if r['state'] == 'COMPLETED']
        if complete:
            return select(session, max(complete, key=lambda r: r['score'])['id'])
        return {'state': 'NO_COMPLETED_ATTEMPTS', 'history': store.rows()}


def wait_for(session, attempt, timeout):
    store = Store(session)
    until = time.monotonic() + timeout
    while True:
        row = store.get(attempt)
        if row['state'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
            return row
        pid = row['pid']
        if pid:
            try:
                os.kill(pid, 0)
                stat = Path(f'/proc/{pid}/stat').read_text()
                dead = stat.split(') ', 1)[1].startswith('Z')
            except (ProcessLookupError, FileNotFoundError):
                dead = True
            if dead:
                store.update(attempt, 'FAILED', 'worker exited unexpectedly; resume to retry the incomplete stage')
                return store.get(attempt)
        if time.monotonic() >= until:
            raise TimeoutError(f'wait timeout; job remains managed: {attempt}')
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Fixed attempt interface for Codex data research')
    parser.add_argument('command', choices=('init', 'validate', 'submit', 'worker', 'resume', 'status',
                                            'history', 'feedback', 'select', 'run', 'wait', 'probe', 'final-test', 'cancel'))
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--attempt')
    parser.add_argument('--checkpoint', choices=('base', 'sft', 'rl'), default='rl')
    parser.add_argument('--probe-spec', type=Path, help='JSON sampling overrides for probe only')
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    session = args.session.resolve()
    try:
        if args.command == 'init':
            if not args.config:
                parser.error('init requires --config')
            result = initialize(session, args.config)
        elif args.command == 'run':
            result = run_loop(session)
        elif args.command in ('status', 'history'):
            store = Store(session)
            result = {'attempts': store.rows(), 'usage': store.costs(),
                      'stages': [dict(r) for r in store.db.execute('SELECT * FROM stages')]}
        else:
            if not args.attempt:
                parser.error(f'{args.command} requires --attempt')
            identifier(args.attempt)
            if args.command == 'validate':
                result = {'valid': True, 'recipe': validate_attempt(session, args.attempt)}
            elif args.command == 'submit':
                result = submit(session, args.attempt)
            elif args.command == 'worker':
                from .runner import worker
                worker(session, args.attempt)
                result = Store(session).get(args.attempt)
            elif args.command == 'resume':
                store = Store(session)
                with lock(session / 'runner/submission.lock'):
                    if store.get(args.attempt)['state'] != 'FAILED':
                        raise ValueError('resume requires FAILED state; do not duplicate active jobs')
                    if any(r['state'] not in ('COMPLETED', 'FAILED', 'CANCELLED') for r in store.rows()):
                        raise ValueError('another attempt is already active')
                    with lock(session / 'runner/execution.lock', blocking=False):
                        store.update(args.attempt, 'QUEUED')
                pid = launch(session, args.attempt)
                result = {'job_id': args.attempt, 'pid': pid}
            elif args.command == 'cancel':
                import signal
                store = Store(session)
                row = store.get(args.attempt)
                if row['state'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    raise ValueError('attempt is not active')
                pid = row['pid']
                if pid:
                    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                    expected = [b'worker', str(session).encode(), args.attempt.encode()]
                    if not all(item in command for item in expected):
                        raise ValueError('worker PID no longer matches this job')
                    os.kill(pid, signal.SIGTERM)
                    wait_for(session, args.attempt, 30)
                store.update(args.attempt, 'CANCELLED', 'cancelled by operator')
                result = store.get(args.attempt)
            elif args.command == 'wait':
                result = wait_for(session, args.attempt, args.timeout)
            elif args.command == 'feedback':
                result = read_json(session / 'runner/attempts' / args.attempt / 'feedback.json')
            elif args.command == 'select':
                result = select(session, args.attempt)
            elif args.command in ('probe', 'final-test'):
                from .runner import Runner
                runner = Runner(session, args.attempt)
                if runner.store.get(args.attempt)['state'] != 'COMPLETED':
                    raise ValueError('probe/final-test requires a completed attempt')
                cfg = runner.cfg
                if args.probe_spec and args.command != 'probe':
                    raise ValueError('final-test sampling is fixed')
                sampling = read_json(args.probe_spec) if args.probe_spec else None
                if args.command == 'final-test':
                    if args.checkpoint != 'rl':
                        raise ValueError('final-test evaluates the selected RL checkpoint')
                    selected = read_json(session / 'runner/selected.json')
                    if selected['attempt_id'] != args.attempt:
                        raise ValueError('final-test must use selected attempt')
                    marker = session / 'runner/final_test_started.json'
                    with lock(session / 'runner/submission.lock'):
                        if marker.exists():
                            raise ValueError('final-test already consumed for this session')
                        source = (cfg['evaluation'].get('final_dataset') if cfg['evaluation']['backend'] == 'harbor'
                                  else cfg['data'].get('final_test'))
                        if not source:
                            raise ValueError('configure evaluation.final_dataset (Harbor) or data.final_test (vLLM) first')
                        write_json(marker, {'attempt_id': args.attempt, 'time': time.time()})
                else:
                    source = (cfg['evaluation'].get('probe_dataset') if cfg['evaluation']['backend'] == 'harbor'
                              else cfg['data']['probe'])
                    if sampling and cfg['evaluation']['backend'] == 'harbor':
                        raise ValueError('Harbor probe sampling is fixed in this adapter')
                with lock(session / 'runner/execution.lock', blocking=False):
                    try:
                        with runner.resources():
                            output = runner.root / 'extra' / f'{args.command}-{time.time_ns()}'
                            result = runner.evaluate(args.checkpoint, source, output, sampling)
                            write_json(output / 'probe_spec.json', {'checkpoint': args.checkpoint,
                                       'dataset': source, 'sampling': sampling, 'result': result})
                    finally:
                        runner.store.update(args.attempt, 'COMPLETED')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
