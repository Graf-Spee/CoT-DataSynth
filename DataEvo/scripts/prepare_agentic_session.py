#!/usr/bin/env python3
"""Materialize a terminal-task session; never starts training or cloud resources."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from DataEvo.runtime.agentic_config import DEFAULTS, instruction
from DataEvo.runtime.agentic_protocol import prompt
from DataEvo.runtime.common import REPO, read_json, write_json
from DataEvo.runtime.contracts import validate_config
from DataEvo.runtime.data import load_rows, validate_splits


def task_rows(root, split):
    mapping, rows = {}, []
    for file in sorted(root.rglob('task.toml')):
        path = file.parent.resolve()
        task_id = path.relative_to(root.resolve()).as_posix()
        mapping[task_id] = str(path)
        rows.append({'prompt': prompt(instruction(path)), 'data_source': 'harbor_terminal',
            'reward_model': {'style': 'rule', 'ground_truth': ''},
            'extra_info': {'index': len(rows), 'tools_kwargs': {'task_id': task_id, 'split': split}}})
    if not rows:
        raise ValueError(f'no Harbor tasks under {root}')
    return mapping, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train-tasks', 'dev-tasks', 'probe-tasks', 'sft-val', 'base-model', 'harbor-python', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--final-tasks', type=Path)
    p.add_argument('--teacher-model', required=True)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--environment', choices=('e2b', 'docker'), default='e2b')
    p.add_argument('--gpu-ids', default='0')
    p.add_argument('--gpu-count', type=int, default=1)
    p.add_argument('--max-episodes', type=int, default=1000)
    args = p.parse_args()
    import pandas as pd
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train, train_rows = task_rows(args.train_tasks, 'train')
    val, val_rows = task_rows(args.dev_tasks, 'val')
    _, probe_rows = task_rows(args.probe_tasks, 'probe')
    datasets = {'train': train_rows, 'dev': val_rows, 'probe': probe_rows}
    if args.final_tasks:
        _, datasets['final'] = task_rows(args.final_tasks, 'final')
    sft_rows = load_rows(args.sft_val)
    if not sft_rows or any(not r.get('messages') or not any(m['role'] == 'assistant' for m in r['messages']) for r in sft_rows):
        raise ValueError('--sft-val requires held-out assistant trajectories in messages format')
    for row in sft_rows:
        row['enable_thinking'] = False
    datasets['sft_val'] = sft_rows
    for name, rows in datasets.items():
        pd.DataFrame(rows).to_parquet(output / f'{name}.parquet', index=False)
    write_json(output / 'tasks.json', {'train': train, 'val': val})
    cfg = read_json(REPO / 'DataEvo/config/session.gsm8k.example.json')
    cfg.update(dataset='agentic_terminal', base_model=str(args.base_model.resolve()),
               python=str(args.python.absolute()), harbor_python=str(args.harbor_python.absolute()))
    cfg['data'] = {key: str(output / f'{name}.parquet') for key, name in
        {'seed': 'train', 'rl_train': 'train', 'rl_val': 'dev', 'dev': 'dev',
         'probe': 'probe', 'sft_val': 'sft_val'}.items()}
    if args.final_tasks:
        cfg['data']['final_test'] = str(output / 'final.parquet')
    cfg['resources'].update(gpu_ids=args.gpu_ids.split(','), gpu_count=args.gpu_count)
    cfg['teacher_api']['model'] = args.teacher_model
    cfg['training']['agentic_rl'] = {**DEFAULTS, 'environment': args.environment,
        'task_manifest': str(output / 'tasks.json'), 'max_episodes': args.max_episodes}
    cfg['training']['pipeline_args'] = {'sft-epochs': 1,
        'sft-arg': ['data.max_length=8192', 'trainer.save_freq=5'],
        'grpo-arg': ['trainer.total_training_steps=10', 'trainer.save_freq=5', 'trainer.test_freq=5',
            'data.train_batch_size=8', 'data.val_batch_size=8',
            'data.max_prompt_length=2048', 'data.max_response_length=6144',
            'actor_rollout_ref.rollout.n=4', 'actor_rollout_ref.actor.ppo_mini_batch_size=8',
            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
            'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
            'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1',
            'actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'trainer.logger=[console]']}
    cfg['evaluation'] = {'backend': 'harbor', 'agent': 'dataevo-terminal', 'environment': args.environment,
        'dataset': str(args.dev_tasks.resolve()), 'probe_dataset': str(args.probe_tasks.resolve()),
        'public_base_url_env': 'STUDENT_PUBLIC_BASE_URL', 'port': 8000,
        'n_tasks': len(val_rows), 'concurrency': 2, 'max_turns': DEFAULTS['max_turns'],
        'max_tokens': DEFAULTS['max_tokens_per_turn'], 'context_window': 8192}
    if args.final_tasks:
        cfg['evaluation']['final_dataset'] = str(args.final_tasks.resolve())
    validate_config(cfg)
    validate_splits(cfg)
    write_json(output / 'session.json', cfg)
    print(output / 'session.json')


if __name__ == '__main__':
    main()
