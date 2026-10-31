"""Read-only dependency/tokenizer checks, before any paid synthesis or GPU work."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path


def check_training(model):
    from packaging.version import Version
    version = importlib.metadata.version('vllm')
    if not Version('0.8.4') <= Version(version) <= Version('0.8.5'):
        raise ValueError(f'this repository async rollout requires vLLM 0.8.4–0.8.5; found {version}. '
                         'Use a separate compatible training environment; do not replace a running experiment environment.')
    # Importing the server checks the actual legacy APIs, not just package presence.
    importlib.import_module('verl.workers.rollout.vllm_rollout.vllm_async_server')
    from transformers import AutoTokenizer
    from .agentic_sft import configure_tokenizer
    from .agentic_protocol import prompt, token_spans, observation
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True))
    messages = prompt('Check the working directory.')
    messages += [{'role': 'assistant', 'content': '{"command":"pwd"}'},
                 observation({'stdout': '/app', 'return_code': 0}, 100),
                 {'role': 'assistant', 'content': '{"final":"done"}'}]
    _, response, mask = token_spans(tokenizer, messages, 2)
    if not response or not any(mask):
        raise ValueError('tokenizer alignment check failed')
    return {'vllm': version, 'tokenizer_alignment': 'passed'}


def check_harbor(config=None):
    version = importlib.metadata.version('harbor')
    if version != '0.17.1':
        raise ValueError(f'Harbor adapter tested against 0.17.1; found {version}')
    from .harbor_bridge import BridgeAgent
    from .harbor_terminal_agent import TerminalAgent
    if BridgeAgent.__abstractmethods__ or TerminalAgent.__abstractmethods__:
        raise ValueError('Harbor agent API mismatch')
    result = {'harbor': version, 'agent_interfaces': 'passed'}
    if config is not None:
        from harbor.models.task.task import Task
        cfg = json.loads(Path(config).read_text())
        manifest = json.loads(Path(cfg['training']['agentic_rl']['task_manifest']).read_text())
        paths = {Path(path) for mapping in manifest.values() for path in mapping.values()}
        for key in ('dataset', 'probe_dataset', 'final_dataset'):
            if cfg['evaluation'].get(key):
                paths.update(p.parent for p in Path(cfg['evaluation'][key]).rglob('task.toml'))
        for path in sorted(paths):
            if not Task.is_valid_dir(path):
                raise ValueError(f'invalid Harbor task (config/environment/instruction/verifier): {path}')
            task = Task(path)
            if task.has_steps or task.config.environment.os.value != 'linux':
                raise ValueError(f'only single-step Linux terminal tasks are supported: {path}')
        result['validated_tasks'] = len(paths)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('training', 'harbor'), required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--config', type=Path, help='validate local Harbor tasks without creating environments')
    args = p.parse_args()
    print(json.dumps(check_training(args.model) if args.kind == 'training' else check_harbor(args.config)))


if __name__ == '__main__':
    main()
