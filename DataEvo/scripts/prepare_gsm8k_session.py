#!/usr/bin/env python3
"""Create reproducible train/dev/probe splits and a concrete local training profile."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from DataEvo.runtime.common import REPO, digest, read_json, write_json
from DataEvo.runtime.data import fingerprint, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Original GSM8K train parquet/jsonl')
    parser.add_argument('--test', type=Path, required=True, help='Original GSM8K final test (not used for optimization)')
    parser.add_argument('--output', type=Path, required=True, help='New directory for split artifacts/config')
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--teacher-model', required=True, help='API model identifier')
    parser.add_argument('--gpu-ids', default='0')
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--heldout-size', type=int, default=256, help='Size of EACH of SFT val, dev, probe')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import pandas as pd
    rows = load_rows(args.train)
    if args.heldout_size < 1 or len(rows) <= 3 * args.heldout_size:
        parser.error('need positive heldout-size and more than 3*heldout-size train rows')
    keys = [fingerprint(row) for row in rows]
    if len(set(keys)) != len(keys):
        parser.error('deduplicate source questions before splitting')
    if set(keys) & {fingerprint(row) for row in load_rows(args.test)}:
        parser.error('original train and final-test prompts overlap')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    random.Random(args.seed).shuffle(rows)
    n = args.heldout_size
    splits = {'sft_val': rows[:n], 'dev': rows[n:2*n], 'probe': rows[2*n:3*n], 'train_seed': rows[3*n:]}
    for name, split in splits.items():
        pd.DataFrame(split).to_parquet(output / f'{name}.parquet', index=False)
    cfg = read_json(REPO / 'DataEvo/config/session.gsm8k.example.json')
    cfg.update(base_model=str(args.base_model.resolve()), python=str(args.python.absolute()))
    cfg['teacher_api']['model'] = args.teacher_model
    cfg['resources'].update(gpu_ids=args.gpu_ids.split(','), gpu_count=args.gpu_count)
    cfg['data'] = {'seed': str(output / 'train_seed.parquet'), 'rl_train': str(output / 'train_seed.parquet'),
                   'rl_val': str(output / 'dev.parquet'), 'dev': str(output / 'dev.parquet'),
                   'sft_val': str(output / 'sft_val.parquet'), 'probe': str(output / 'probe.parquet'),
                   'final_test': str(args.test.resolve())}
    write_json(output / 'session.json', cfg)
    write_json(output / 'split_manifest.json', {'seed': args.seed, 'train_sha256': digest(args.train),
               'test_sha256': digest(args.test), 'rows': {k: len(v) for k, v in splits.items()}})
    print(output / 'session.json')


if __name__ == '__main__':
    main()
