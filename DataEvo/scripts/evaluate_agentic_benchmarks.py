#!/usr/bin/env python3
"""Evaluate a served model on Terminal-Bench 2.0 / SWE-bench Verified with Harbor."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from DataEvo.runtime.benchmark_eval import BENCHMARKS, execute, make_plan, preflight


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rsibench-root', type=Path, default=Path('/home/hrh/RSIBench-Data-main'))
    p.add_argument('--benchmark', choices=(*BENCHMARKS, 'both'), default='both')
    p.add_argument('--terminal-dataset', dest='terminal_bench_2_dataset', type=Path)
    p.add_argument('--swe-dataset', dest='swe_bench_verified_dataset', type=Path)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--n-tasks', type=positive, help='First N sorted task IDs per dataset; default 1')
    selection.add_argument('--all-tasks', action='store_true')
    selection.add_argument('--task', action='append', help='Exact local task ID; repeatable; one benchmark only')
    p.add_argument('--model', required=True, help='Served model ID, optionally prefixed with openai/')
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--agent-profile', choices=('dataevo', 'native'), default='dataevo')
    p.add_argument('--environment', choices=('e2b', 'docker'), default='e2b')
    p.add_argument('--concurrency', type=positive, default=1)
    p.add_argument('--max-turns', type=positive, default=200)
    p.add_argument('--max-tokens', type=positive, default=4096)
    p.add_argument('--context-window', type=positive, default=32768, help='Native Terminus input token limit')
    p.add_argument('--command-timeout', type=positive, default=120, help='DataEvo shell command timeout, seconds')
    p.add_argument('--max-observation-chars', type=positive, default=24000)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Validate and print exact plan; no network or output writes')
    mode.add_argument('--preflight', action='store_true', help='Check task/agent/env/API prerequisites; no sandbox created')
    p.add_argument('--output', type=Path, help='New directory required for a real run; existing directories rejected')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.dry_run and not args.preflight and not args.output:
        p.error('--output is required for a real run')
    if args.output and args.output.exists() and not (args.dry_run or args.preflight):
        p.error('--output already exists; choose a new evaluation directory')
    try:
        plan = make_plan(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        checks = preflight(plan)
        print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
        if not checks['ready']:
            return 2
        if args.preflight:
            return 0
        return asyncio.run(execute(plan, args.output.resolve(), checks))
    except (ValueError, OSError, ImportError) as error:
        p.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
