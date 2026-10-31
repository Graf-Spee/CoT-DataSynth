"""Repository pipeline and Harbor adapters (runner-owned configuration only)."""
from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from .common import REPO, read_json, write_json


def flags(mapping):
    negative_booleans = {'teacher-enforce-eager', 'teacher-disable-custom-all-reduce',
                         'consistency-do-sample', 'cot-do-sample', 'cot-enforce-eager',
                         'cot-disable-custom-all-reduce', 'instruction-do-sample',
                         'rip-response-do-sample', 'code-do-sample'}
    result = []
    for key, value in mapping.items():
        key = '--' + key
        if value is True:
            result.append(key)
        elif value is False:
            if key[2:] in negative_booleans:
                result.append('--no-' + key[2:])
            continue
        elif value is None:
            continue
        else:
            for item in value if isinstance(value, list) else [value]:
                result.append(f'{key}={item}')
    return result


def pipeline_command(cfg, attempt, root, gpu_ids, stage, generation=None, eval_data=None, eval_output=None):
    mapped = {'generate': 'distill', 'rl': 'grpo', 'rl_eval': 'grpo_eval'}.get(stage, stage)
    base = str(rl_base(root)) if stage == 'rl_eval' else cfg['base_model']
    cmd = [cfg['python'], str(REPO / 'DataObs/scripts/experiment_pipeline_vllm.py'),
           '--experiment-id', attempt, '--dataset', cfg['dataset'], '--base-model', base,
           '--output-dir', str(root.parent), '--stages', mapped, '--gpu-ids', ','.join(gpu_ids),
           '--sft-output-dir', str(root / 'experiment/sft'),
           '--grpo-output-dir', str(root / 'experiment/grpo'),
           '--sft-data', str(root / 'dataset/train.parquet'),
           '--sft-val-data', cfg['data']['sft_val'],
           '--rl-train-data', cfg['data']['rl_train'], '--rl-val-data', cfg['data']['rl_val'],
           '--distill-input', cfg['data']['seed'], '--distill-output', str(root / 'candidates.parquet'),
           '--sft-eval-output-dir', str(eval_output or root / 'eval/sft'),
           '--grpo-eval-output-dir', str(eval_output or root / 'eval/rl'),
           '--eval-data', eval_data or cfg['data']['dev']]
    cmd += flags(cfg['training'].get('pipeline_args', {}))
    cmd += flags(cfg['evaluation'].get('pipeline_args', {}))
    if generation:
        cmd += flags(generation.get('args', {}))
    if cfg['dataset'] != 'gsm8k':
        cmd += ['--sft-arg=data.multiturn.enable=true', '--sft-arg=data.multiturn.messages_key=messages']
    if cfg['training'].get('agentic_rl'):
        cmd += ['--sft-arg=data.custom_cls.path=' + str(REPO / 'DataEvo/runtime/agentic_sft.py'),
                '--sft-arg=data.custom_cls.name=TerminalSFTDataset']
    if stage == 'rl' and cfg['training'].get('agentic_rl'):
        from .agentic_config import grpo_overrides, write_spec
        write_spec(cfg, root)
        cmd += ['--grpo-arg=' + item for item in grpo_overrides(root)]
    return cmd


def checkpoint(root, stage):
    if stage == 'base':
        raise ValueError('base checkpoint is supplied directly by config')
    if stage == 'sft':
        from DataObs.scripts.experiment_pipeline import resolve_sft_checkpoint
        candidate = resolve_sft_checkpoint(root / 'experiment/sft')
    else:
        grpo = root / 'experiment/grpo'
        steps = sorted((p for p in grpo.glob('global_step_*') if p.name[12:].isdigit()),
                       key=lambda p: int(p.name[12:]))
        if not steps:
            raise ValueError('no RL checkpoint saved')
        actor = steps[-1] / 'actor'
        # Existing grpo_eval exports/merges its actor into one of these paths.
        candidate = next((p for p in (actor / 'merged_model', actor / 'huggingface')
                          if model_ready(p)), actor / 'huggingface')
    if not model_ready(candidate):
        raise ValueError(f'checkpoint is not an exported HF/LoRA model: {candidate}')
    return candidate


def model_ready(path):
    return path.is_dir() and ((path / 'adapter_model.safetensors').is_file() or
                             ((path / 'config.json').is_file() and
                              (any(path.glob('*.safetensors')) or any(path.glob('*.bin')))))


def rl_base(root):
    """RL LoRA deltas are relative to the SFT model, not the original student."""
    model = checkpoint(root, 'sft')
    if (model / 'adapter_model.safetensors').is_file():
        model = root / 'experiment/grpo/merged_sft_model'
        if not model_ready(model):
            raise ValueError('RL LoRA export needs the merged SFT initialization model')
    return model


def evaluate_model(cfg, model, data, output, gpu_ids, sampling=None):
    arguments = cfg['evaluation'].get('pipeline_args', {})
    settings = {'num-samples': arguments.get('eval-num-samples', 1),
                'temperature': arguments.get('eval-temperature', 0),
                'top-p': arguments.get('eval-top-p', 1),
                'seed': arguments.get('eval-seed', 42),
                'max-response-len': arguments.get('eval-max-response-len', 4096),
                'batch-size': arguments.get('eval-batch-size', 8),
                'tensor-parallel-size': arguments.get('eval-tensor-parallel-size', 1),
                'gpu-memory-utilization': arguments.get('eval-gpu-memory-utilization', .8),
                'prompt-template-method': arguments.get('eval-prompt-template-method', 'zeroshot')}
    allowed = {'num-samples', 'temperature', 'top-p', 'max-response-len', 'max-samples', 'seed'}
    if sampling and set(sampling) - allowed:
        raise ValueError('unknown probe sampling field')
    settings.update(sampling or {})
    return [cfg['python'], str(REPO / 'DataObs/lib/evaluation/eval_vllm.py'),
            '--model-path', str(model), '--base-model', cfg['base_model'], '--dataset', cfg['dataset'],
            '--eval-data', data, '--output-dir', str(output), '--gpu-ids', ','.join(gpu_ids)] + flags(settings)


def export_rl(cfg, root, run):
    """Use the same exports as the repository vLLM eval, without running test eval."""
    grpo = root / 'experiment/grpo'
    steps = sorted((p for p in grpo.glob('global_step_*') if p.name[12:].isdigit()), key=lambda p: int(p.name[12:]))
    if not steps:
        raise ValueError('RL did not produce a checkpoint')
    actor = steps[-1] / 'actor'
    if model_ready(actor / 'merged_model'):
        return
    if (actor / 'lora_adapter/adapter_model.safetensors').is_file():
        run([cfg['python'], str(REPO / 'DataObs/lib/model_ops/merge_lora_qwen.py'),
             '--base', str(rl_base(root)), '--lora', str(actor / 'lora_adapter'),
             '--tokenizer', str(actor / 'huggingface'),
             '--output', str(actor / 'merged_model')], 'export_rl')
    elif not model_ready(actor / 'huggingface'):
        run([cfg['python'], '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
             '--local_dir', str(actor), '--target_dir', str(actor / 'huggingface')], 'export_rl')


def harbor_eval(cfg, model, output, gpu_ids, run, managed_process, dataset=None):
    evaluation = cfg['evaluation']
    public_url = (f"http://127.0.0.1:{evaluation['port']}/v1" if evaluation['agent'] == 'dataevo-terminal'
                  else os.environ.get(evaluation['public_base_url_env']))
    if not public_url:
        raise ValueError(f"set {evaluation['public_base_url_env']} to the E2B-reachable /v1 URL")
    if (model / 'adapter_model.safetensors').exists():
        # Harbor serves a merged model; it must not silently evaluate the base weights.
        merged = output.parent / (output.name + '_merged')
        run([cfg['python'], str(REPO / 'DataObs/lib/model_ops/merge_lora_qwen.py'),
             '--base', cfg['base_model'], '--lora', str(model), '--tokenizer', str(model),
             '--output', str(merged)], 'merge_for_harbor')
        model = merged
    port = evaluation['port']
    with socket.socket() as probe:
        probe.bind(('0.0.0.0', port))  # refuse to connect to an unrelated model on an occupied port
    command = evaluation.get('serve_command', [cfg['python'], '-m', 'vllm.entrypoints.openai.api_server',
        '--model', '{checkpoint}', '--served-model-name', 'student', '--host', '0.0.0.0', '--port', '{port}',
        '--tensor-parallel-size', str(len(gpu_ids))])
    values = {'checkpoint': str(model), 'port': str(port), 'gpu_ids': ','.join(gpu_ids)}
    command = [part.format_map(values) for part in command]
    if cfg['training'].get('agentic_rl') and not evaluation.get('serve_command'):
        command += ['--chat-template', str(REPO / 'DataEvo/config/terminal_chatml.jinja')]
    service_key = os.environ.get('STUDENT_API_KEY')
    output.mkdir(parents=True, exist_ok=True)
    with managed_process(command, 'serving') as server:
        local_url = f'http://127.0.0.1:{port}/v1/models'
        until = time.monotonic() + evaluation.get('startup_timeout_s', 600)
        while True:
            if server.poll() is not None:
                raise ValueError('model server exited before becoming ready')
            try:
                request = urllib.request.Request(local_url, headers={
                    'Authorization': 'Bearer ' + (service_key or 'local-student')})
                with urllib.request.urlopen(request, timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= until:
                raise TimeoutError('model server readiness timeout')
            time.sleep(1)
        spec = {'output': str(output), 'api_base': public_url, 'model': 'openai/student',
                'dataset': dataset or evaluation['dataset'], 'agent': evaluation['agent'],
                'environment': evaluation.get('environment', 'e2b'),
                'n_tasks': evaluation.get('n_tasks', 20), 'concurrency': evaluation.get('concurrency', 2),
                'max_turns': evaluation.get('max_turns', 100),
                'max_tokens': evaluation.get('max_tokens', 4096),
                'context_window': evaluation.get('context_window', 32768)}
        if cfg['training'].get('agentic_rl'):
            from .agentic_config import settings
            limits = settings(cfg)
            spec['reward_key'] = limits['reward_key']
            spec['terminal'] = {k: limits[k] for k in
                ('max_turns', 'max_tokens_per_turn', 'max_observation_chars', 'command_timeout_s')}
        write_json(output / 'request.json', spec)
        run([cfg.get('harbor_python', cfg['python']), str(REPO / 'DataEvo/runtime/harbor_job.py'),
             '--spec', str(output / 'request.json')], 'harbor')
