#!/usr/bin/env python3
"""Read-only RSIBench dataset inventory; never starts models or task environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inventory(root, benchmark=None, samples=0, max_chars=800):
    specs = sorted((root / 'benchmarks').glob('*/spec.json'))
    if not specs:
        raise ValueError(f'no benchmarks/*/spec.json under {root}')
    rows = []
    for path in specs:
        spec = json.loads(path.read_text())
        key = spec['benchmark_key']
        if benchmark and key != benchmark:
            continue
        configured = root / spec['harbor_dataset']
        if configured.is_dir():
            dataset = configured
        else:
            # Terminal-Bench spec uses a registry name; inspect its vendored copy.
            local = sorted(p for p in (path.parent / 'datasets').glob('*') if p.is_dir())
            if len(local) != 1:
                raise ValueError(f'{key}: expected one local dataset, found {len(local)}')
            dataset = local[0]
        tasks = sorted(p.parent for p in dataset.rglob('task.toml'))
        if not tasks:
            raise ValueError(f'{key}: no local Harbor tasks under {dataset}')
        expected = spec['official_eval']['n_tasks']
        examples = []
        for task in tasks[:samples]:
            instruction = task / 'instruction.md'
            text = instruction.read_text()
            examples.append({'task_id': task.relative_to(dataset).as_posix(),
                'instruction_path': str(instruction), 'instruction_preview': text[:max_chars],
                'preview_truncated': len(text) > max_chars})
        rows.append({'benchmark': key, 'name': spec['display_name'],
            'task_family': spec['task_family'], 'spec': str(path), 'dataset': str(dataset),
            'role': 'evaluation_only', 'local_tasks': len(tasks), 'expected_tasks': expected,
            'task_count_matches_spec': len(tasks) == expected,
            'rsibench_agent': spec['harbor_agent'],
            'attempts_per_task': spec['official_eval'].get('n_attempts', 1),
            'primary_metric': spec['primary_metric'],
            'seed_factory_references': spec.get('seed_factories', []), 'samples': examples})
    if not rows:
        raise ValueError(f'unknown benchmark: {benchmark}; use benchmark_key from the unfiltered inventory')
    return {'rsibench_root': str(root), 'benchmarks': rows,
        'note': 'Evaluation tasks only. This inventory does not validate runtime compatibility, '
                'execute verifiers, or authorize use of these tasks for RL training.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rsibench-root', type=Path, default=Path('/home/hrh/RSIBench-Data-main'))
    parser.add_argument('--benchmark', help='benchmark_key, e.g. terminal_bench_2 or swe_bench_verified')
    parser.add_argument('--samples', type=int, default=0, help='Number of task instruction previews per benchmark')
    parser.add_argument('--max-chars', type=int, default=800)
    args = parser.parse_args()
    if args.samples < 0 or args.max_chars < 1:
        parser.error('--samples must be nonnegative and --max-chars must be positive')
    try:
        result = inventory(args.rsibench_root.resolve(), args.benchmark, args.samples, args.max_chars)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
