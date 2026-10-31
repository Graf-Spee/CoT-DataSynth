"""Canonical JSONL -> existing repository SFT parquet adapter."""
from __future__ import annotations

import json
from pathlib import Path

from .common import digest, write_json


def load_rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pandas as pd
        def native(value):
            if hasattr(value, 'tolist'):
                return native(value.tolist())
            if isinstance(value, dict):
                return {k: native(v) for k, v in value.items()}
            if isinstance(value, list):
                return [native(v) for v in value]
            return value
        return native(pd.read_parquet(path).to_dict('records'))
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(row):
    value = row.get('question')
    if value is None:
        messages = row.get('messages', row.get('prompt', []))
        from .agentic_protocol import SYSTEM
        if len(messages) >= 2 and messages[0].get('content') == SYSTEM:
            # Terminal observations use user roles; they are not new task identities.
            messages = messages[:2]
        value = [m for m in messages if m.get('role') in ('user', 'system')]
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)).strip()


def validate_splits(cfg):
    """Reject exact train/held-out overlap before any API or GPU expenditure."""
    train = set()
    for key in ('seed', 'rl_train'):
        train.update(fingerprint(row) for row in load_rows(cfg['data'][key]))
    for key in ('sft_val', 'rl_val', 'dev', 'probe', 'final_test'):
        source = cfg['data'].get(key)
        if source and train.intersection(fingerprint(row) for row in load_rows(source)):
            raise ValueError(f'train/held-out prompt overlap: {key}')


def prepare(selected, output, cfg):
    import pandas as pd
    rows = load_rows(selected)
    if not rows or len(rows) > cfg['budget']['max_rows']:
        raise ValueError('selected dataset empty or exceeds max_rows')
    held = set()
    for key in ('sft_val', 'rl_val', 'dev', 'probe', 'final_test'):
        if cfg['data'].get(key):
            held.update(fingerprint(r) for r in load_rows(cfg['data'][key]))
    seen, converted = set(), []
    for i, row in enumerate(rows):
        key = fingerprint(row)
        if key in held:
            raise ValueError(f'held-out prompt leakage at selected row {i}')
        if key in seen:
            # Multiple reasoning paths for the same task are legitimate.
            pass
        seen.add(key)
        if cfg['dataset'] == 'gsm8k':
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('question', 'answer')):
                raise ValueError(f'GSM8K row {i} requires nonempty question/answer strings')
            converted.append(row)
        else:
            messages = row.get('messages', [])
            if not isinstance(messages, list) or not messages:
                raise ValueError(f'row {i}: messages missing')
            calls = set()
            for message in messages:
                if message.get('role') not in ('system', 'user', 'assistant', 'tool'):
                    raise ValueError('invalid message role')
                for call in message.get('tool_calls') or []:
                    calls.add(call['id'])
                if message['role'] == 'tool' and message.get('tool_call_id') not in calls:
                    raise ValueError('tool response has no matching action')
            if not any(m['role'] == 'assistant' for m in messages):
                raise ValueError('no assistant actions to train')
            if not row.get('task_id') or not row.get('execution_ref') or row.get('verified') is not True:
                raise ValueError('agentic examples require task_id, execution_ref and verified=true')
            evidence = Path(row['execution_ref'])
            if not evidence.is_absolute():
                evidence = Path(selected).parent / evidence
            if not evidence.is_file():
                raise ValueError(f'missing execution evidence: {evidence}')
            if cfg['training'].get('agentic_rl'):
                from .agentic_config import instruction
                from .agentic_protocol import prompt, reward_value
                from .common import read_json
                tasks = read_json(cfg['training']['agentic_rl']['task_manifest'])['train']
                if row['task_id'] not in tasks or messages[:2] != prompt(instruction(tasks[row['task_id']])):
                    raise ValueError('selected terminal trajectory is outside the training task allowlist')
                record = read_json(evidence)
                if record.get('messages') != messages or record.get('task_id') != row['task_id']:
                    raise ValueError('selected trajectory does not match execution evidence')
                if record.get('valid_actions') is not True or record.get('reward') != 1:
                    raise ValueError('selected trajectory did not pass execution verification')
                result = read_json(record['harbor_result'])
                if result.get('exception_info') or reward_value((result.get('verifier_result') or {}).get('rewards'),
                        cfg['training']['agentic_rl'].get('reward_key', 'reward')) != 1:
                    raise ValueError('Harbor verifier result does not prove success')
                row['enable_thinking'] = False
            converted.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(converted).to_parquet(output, index=False)
    summary = {'rows': len(rows), 'unique_prompts': len(seen), 'sha256': digest(output),
               'selected_sha256': digest(selected), 'format': 'qa' if cfg['dataset'] == 'gsm8k' else 'messages'}
    write_json(output.parent / 'summary.json', summary)
    return summary
