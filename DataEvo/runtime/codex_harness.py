"""One durable research decision per Codex exec; training belongs to the runner."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .common import REPO, Store, config, read_json, write_json
from .runner import terminate

DECISION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'action': {'type': 'string', 'enum': ['attempt', 'stop']},
                   'attempt_id': {'type': 'string'}, 'rationale': {'type': 'string'}},
    'required': ['action', 'attempt_id', 'rationale'],
}


def decide(session, index):
    session = Path(session).resolve()
    cfg = config(session)
    store = Store(session)
    root = session / 'runner' / 'codex' / f'decision-{index:04d}'
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'schema.json', DECISION_SCHEMA)
    history = store.rows()
    feedback = []
    for row in history:
        path = session / 'runner/attempts' / row['id'] / 'feedback.json'
        feedback.append({'attempt': row, 'feedback_path': str(path) if path.exists() else None,
                         'artifact_dir': str(path.parent), 'logs': str(path.parent / 'logs')})
    context = {'history': feedback, 'usage': store.costs(),
               'base_model': cfg['base_model'], 'dataset': cfg['dataset'],
               'budget': cfg['budget'], 'teacher_api': cfg.get('teacher_api'),
               'next_attempt_id': f'a{len(history):04d}'}
    write_json(root / 'context.json', context)
    # The entire prompt is retained as reviewable research evidence.
    prompt = f'''You are the data synthesis researcher for this experiment.
Read RESEARCH.md in your workspace. Fixed runtime: {REPO / 'DataEvo/scripts/run_session.py'}.
Session: {session}. Read context {root / 'context.json'} and the feedback/trajectories referenced there.
Choose ONE new data experiment based on evidence. Write attempts/<id>/recipe.json,
hypothesis.md, and any synthesis/postprocessing Python code and templates needed.
Use absolute {cfg['python']} when checking your scripts. Python plugins run with the repository on PYTHONPATH.
Validate via: {cfg['python']} {REPO / 'DataEvo/scripts/run_session.py'} validate --session {session} --attempt <id>
You may inspect artifacts, but do not modify runner-owned files, fixed profiles, training code, or evaluation data.
Do not start training or submit jobs in this decision turn: the controller submits the frozen attempt after you return.
The controller will supply real training feedback on the next decision. Do not invent results.
Return action=attempt and its id, or action=stop with an empty id and reason if no useful feasible experiment remains.
'''
    (root / 'prompt.txt').write_text(prompt)
    codex = cfg.get('codex', {})
    command = [codex.get('executable', 'codex'), 'exec', '--json', '--skip-git-repo-check',
               '--sandbox', codex.get('sandbox', 'workspace-write'),
               '--output-schema', str(root / 'schema.json'),
               '--output-last-message', str(root / 'decision.json'),
               '-c', 'approval_policy="never"', '-C', str(session / 'workspace')]
    if codex.get('model'):
        command += ['--model', codex['model']]
    if codex.get('profile'):
        command += ['--profile', codex['profile']]
    command.append('-')
    start = time.time()
    remaining = read_json(session / 'runner/session.json')['deadline'] - start
    if remaining <= 0:
        raise TimeoutError('session deadline exceeded before Codex decision')
    failure = None
    with (root / 'events.jsonl').open('w') as events, (root / 'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=stderr,
                                text=True, start_new_session=True, cwd=session / 'workspace')
        try:
            proc.communicate(prompt, timeout=min(codex.get('timeout_s', 1800), remaining))
        except subprocess.TimeoutExpired as error:
            failure = error
        finally:
            terminate(proc)
    tokens = 0
    thread_id = None
    for line in (root / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed process may leave a partial final event.
        if event.get('type') == 'thread.started':
            thread_id = event.get('thread_id')
        usage = event.get('usage', {})
        tokens += usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
    store.usage('session', 'codex', tokens=tokens,
                 detail={'thread_id': thread_id, 'elapsed_s': time.time() - start,
                         'events': str(root / 'events.jsonl')})
    if failure:
        raise TimeoutError(f'Codex decision timed out: see {root}/events.jsonl') from failure
    if proc.returncode:
        raise RuntimeError(f'Codex decision failed: see {root}/stderr.log')
    decision = read_json(root / 'decision.json')
    if (decision.get('action') not in ('attempt', 'stop') or
            not isinstance(decision.get('attempt_id'), str) or not isinstance(decision.get('rationale'), str)):
        raise ValueError('Codex returned an invalid decision')
    write_json(root / 'result.json', {'decision': decision, 'thread_id': thread_id, 'tokens': tokens})
    return decision
