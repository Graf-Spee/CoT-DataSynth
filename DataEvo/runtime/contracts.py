"""Validate fixed session profiles and researcher-owned attempt inputs."""
from __future__ import annotations

import os
from pathlib import Path

from .common import STAGES, identifier, read_json, within


def validate_config(cfg):
    for key in ('base_model', 'dataset', 'python', 'data', 'resources', 'training', 'evaluation', 'budget'):
        if key not in cfg:
            raise ValueError(f'missing session setting: {key}')
    for key in ('seed', 'sft_val', 'rl_train', 'rl_val', 'dev', 'probe'):
        path = Path(cfg['data'].get(key, ''))
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'data.{key} must be an existing absolute file')
    # Files may share dev/rl_val, but train and held-out data must be distinct.
    train = {Path(cfg['data'][k]).resolve() for k in ('seed', 'rl_train')}
    held = {Path(cfg['data'][k]).resolve() for k in ('sft_val', 'rl_val', 'dev', 'probe')}
    if train & held:
        raise ValueError('train and held-out paths overlap; split data before initializing')
    for key in ('python', 'base_model'):
        if not Path(cfg[key]).is_absolute() or not Path(cfg[key]).exists():
            raise ValueError(f'{key} must exist and be absolute')
    resources = cfg['resources']
    if not resources.get('gpu_ids') or not resources.get('gpu_count', 0) > 0:
        raise ValueError('configure a nonempty GPU pool and gpu_count > 0')
    if len(set(map(str, resources['gpu_ids']))) != len(resources['gpu_ids']):
        raise ValueError('duplicate GPU identifiers')
    if resources['gpu_count'] > len(resources['gpu_ids']):
        raise ValueError('gpu_count exceeds pool')
    if not Path(resources.get('lock_dir', '')).is_absolute():
        raise ValueError('resources.lock_dir must be absolute and shared across sessions')
    for key in ('max_attempts', 'stage_timeout_s', 'session_timeout_s', 'max_rows'):
        if cfg['budget'].get(key, 0) <= 0:
            raise ValueError(f'budget.{key} must be positive')
    evaluation = cfg['evaluation']
    if evaluation.get('backend') not in ('vllm', 'harbor'):
        raise ValueError('evaluation.backend must be vllm or harbor')
    if evaluation['backend'] == 'harbor':
        for key in ('dataset', 'probe_dataset', 'agent', 'public_base_url_env', 'port'):
            if not evaluation.get(key):
                raise ValueError(f'Harbor evaluation requires {key}')
        if evaluation['dataset'] == evaluation['probe_dataset']:
            raise ValueError('Harbor dev and probe datasets must be distinct')
        if evaluation.get('final_dataset') in (evaluation['dataset'], evaluation['probe_dataset']):
            raise ValueError('Harbor final dataset must be distinct from dev and probe')
        if evaluation['agent'] not in ('terminus-2', 'dataevo-terminal'):
            raise ValueError('Harbor agent must be terminus-2 or dataevo-terminal')
        if evaluation.get('environment', 'e2b') not in ('e2b', 'docker'):
            raise ValueError('Harbor environment must be e2b or docker')
    if cfg['training'].get('agentic_rl'):
        from .agentic_config import validate_profile
        validate_profile(cfg)
    elif cfg['dataset'] != 'gsm8k' and not cfg['training'].get('agentic_rl_verified', False):
        raise ValueError('non-GSM8K profiles require an explicitly verified multiturn RL adapter; '
                         'configure training.rl_command and agentic_rl_verified')
    if cfg['dataset'] != 'gsm8k' and not cfg['training'].get('rl_command') and not cfg['training'].get('agentic_rl'):
        raise ValueError('agentic RL needs a fixed runner-owned rl_command')
    return cfg


def validate_attempt(session, attempt):
    attempt = identifier(attempt)
    root = Path(session) / 'workspace' / 'attempts' / attempt
    recipe = read_json(root / 'recipe.json')
    allowed = {'schema_version', 'attempt_id', 'parent_attempt_id', 'hypothesis', 'generation', 'postprocess'}
    unknown = set(recipe) - allowed
    if unknown:
        raise ValueError(f'unknown recipe fields: {sorted(unknown)}')
    if recipe.get('schema_version') != 1 or recipe.get('attempt_id') != attempt:
        raise ValueError('recipe schema_version/attempt_id mismatch')
    if not isinstance(recipe.get('hypothesis'), str) or len(recipe['hypothesis'].strip()) < 10:
        raise ValueError('provide an evidence-based hypothesis (at least 10 characters)')
    if not (root / 'hypothesis.md').is_file():
        raise ValueError('missing hypothesis.md')
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'attempt must not contain symlinks: {p}')
    generation = recipe.get('generation', {})
    if generation.get('backend') not in ('program', 'pipeline'):
        raise ValueError('generation.backend must be program or pipeline')
    for spec in (generation, recipe.get('postprocess', {})):
        if spec.get('entrypoint'):
            path = within(root, spec['entrypoint'])
            if not path.is_file() or path.suffix != '.py':
                raise ValueError('entrypoint must be an existing Python file inside attempt')
    if generation['backend'] == 'program' and not generation.get('entrypoint'):
        raise ValueError('program generation requires entrypoint')
    if generation['backend'] == 'pipeline':
        # Restrict researcher-supplied pipeline flags to synthesis only.
        for flag in generation.get('args', {}):
            if not (flag.startswith(('teacher-', 'cot-', 'instruction-', 'rip-', 'code-',
                                     'consistency-', 'rephrase-', 'backward-question-')) or
                    flag in ('distill-method', 'disable-teacher-filter', 'smoke-num-rows')):
                raise ValueError(f'not an allowed synthesis flag: {flag}')
    if recipe.get('parent_attempt_id'):
        identifier(recipe['parent_attempt_id'])
    return recipe
