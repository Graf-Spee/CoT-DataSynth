"""Fixed Harbor task allowlist and GRPO profile, outside Codex's edit surface."""
import re
import tomllib
from pathlib import Path

from .agentic_protocol import prompt
from .common import read_json, write_json
from .data import load_rows

DEFAULTS = {'environment': 'e2b', 'concurrency': 2, 'max_turns': 16,
            'max_tokens_per_turn': 512, 'max_observation_chars': 12000,
            'episode_timeout_s': 900, 'command_timeout_s': 30,
            'cleanup_timeout_s': 120, 'max_episodes': 1000, 'reward_key': 'reward'}


def instruction(path):
    lines = (Path(path) / 'instruction.md').read_text().split('\n')
    while lines and re.match(r'^(<!--.*canary.*-->|#.*canary.*)$', lines[0].strip(), re.I):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return '\n'.join(lines)


def settings(cfg):
    return {**DEFAULTS, **cfg['training']['agentic_rl'], 'harbor_python': cfg['harbor_python']}


def validate_profile(cfg):
    if cfg['training'].get('rl_command'):
        raise ValueError('choose the built-in agentic adapter or a custom rl_command, not both')
    if cfg['evaluation'].get('agent') != 'dataevo-terminal' or cfg['evaluation']['backend'] != 'harbor':
        raise ValueError('built-in agentic RL requires Harbor dataevo-terminal evaluation for protocol parity')
    if not Path(cfg.get('harbor_python', '')).is_file():
        raise ValueError('agentic RL requires harbor_python (Harbor 0.17.1 environment)')
    spec = settings(cfg)
    if not isinstance(spec['reward_key'], str) or not spec['reward_key'].strip():
        raise ValueError('agentic_rl.reward_key must be a nonempty string')
    if spec['environment'] not in ('docker', 'e2b'):
        raise ValueError('agentic environment must be docker or e2b')
    if cfg['evaluation'].get('environment', 'e2b') != spec['environment']:
        raise ValueError('training and evaluation Harbor environments must match')
    for key in DEFAULTS.keys() - {'environment', 'reward_key'}:
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f'agentic_rl.{key} must be a positive integer')
    if spec['max_observation_chars'] < 64 or spec['max_tokens_per_turn'] < 32:
        raise ValueError('observation/action limits are too small')
    if spec['command_timeout_s'] >= spec['episode_timeout_s']:
        raise ValueError('command timeout must fit inside the episode timeout')
    manifest = Path(spec['task_manifest'])
    if not manifest.is_absolute():
        raise ValueError('task_manifest must be an absolute path')
    tasks = read_json(manifest)
    if set(tasks) != {'train', 'val'} or not all(isinstance(v, dict) and v for v in tasks.values()):
        raise ValueError('task_manifest needs nonempty train and val task-id/path mappings')
    identities = {}
    for split, mapping in tasks.items():
        identities[split] = set()
        for task_id, directory in mapping.items():
            path = Path(directory)
            if not task_id or not path.is_absolute() or not (path / 'task.toml').is_file():
                raise ValueError('task ids and absolute Harbor task directories are required')
            task_config = tomllib.loads((path / 'task.toml').read_text())
            if task_config.get('steps') or task_config.get('environment', {}).get('os', 'linux') != 'linux':
                raise ValueError('built-in adapter supports single-step Linux terminal tasks only')
            identities[split].add(instruction(path).strip())
        rows = load_rows(cfg['data']['rl_train' if split == 'train' else 'rl_val'])
        if not rows:
            raise ValueError('RL task dataset is empty')
        for row in rows:
            meta = row.get('extra_info', {}).get('tools_kwargs', {})
            if meta.get('split') != split or meta.get('task_id') not in mapping:
                raise ValueError('RL parquet task metadata is not in the corresponding allowlist')
            if row.get('prompt') != prompt(instruction(mapping[meta['task_id']])):
                raise ValueError('RL parquet prompt does not match its Harbor task')
    if identities['train'] & identities['val']:
        raise ValueError('Harbor train/val task instructions overlap')
    # Resolve local evaluation sets before running; registry names cannot be audited here.
    previous_held = set()
    for key in ('dataset', 'probe_dataset', 'final_dataset'):
        source = cfg['evaluation'].get(key)
        if not source:
            continue
        root = Path(source)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError('agentic sessions require materialized local Harbor evaluation task directories')
        configs = list(root.rglob('task.toml'))
        if not configs:
            raise ValueError(f'empty Harbor evaluation dataset: {root}')
        held = {instruction(p.parent).strip() for p in configs}
        if held & previous_held:
            raise ValueError('Harbor dev/probe/final task instructions overlap')
        previous_held.update(held)
        if held & identities['train']:
            raise ValueError(f'Harbor train/evaluation task overlap: {key}')
        if key != 'dataset' and held & identities['val']:
            raise ValueError(f'Harbor RL validation overlaps {key}')
    return spec


def write_spec(cfg, root):
    spec = settings(cfg)
    spec['tasks'] = read_json(spec['task_manifest'])
    spec['output_dir'] = str(root / 'agentic')
    target = root / 'agentic_spec.json'
    write_json(target, spec)
    return target


def grpo_overrides(root):
    return ['actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=async',
        'actor_rollout_ref.rollout.multi_turn.enable=true',
        'actor_rollout_ref.rollout.multi_turn.tool_config_path=null',
        'actor_rollout_ref.rollout.multi_turn.completion_callback=DataEvo.runtime.agentic_rollout.HarborCompletionCallback',
        'actor_rollout_ref.rollout.multi_turn.interaction_config_path=' + str(root / 'agentic_spec.json'),
        'data.return_raw_chat=true', 'reward_model.enable=false', 'reward_model.reward_manager=naive',
        'custom_reward_function.path=null', 'algorithm.adv_estimator=grpo',
        # This repository's async server does not load/update PEFT adapters.
        # SFT may use LoRA (merged by the existing pipeline); RL uses dense weights.
        'actor_rollout_ref.model.lora_rank=0']
