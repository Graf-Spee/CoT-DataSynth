# DataEvo

The autonomous Codex experiment runtime is available at
[`scripts/run_session.py`](scripts/run_session.py). It freezes synthesis code,
runs the existing **DataObs SFT → evaluation → GRPO → evaluation** pipeline,
collects probes/curves/costs, and gives feedback to the next `codex exec` decision.
Built-in profiles cover GSM8K and Linux terminal agents. The terminal profile
connects on-policy async vLLM rollout to persistent Harbor/E2B environments,
verifier rewards and the existing GRPO trainer. See the
[agentic adapter guide (中文)](docs/agentic_runtime.md) for protocol, setup,
RSIBench comparison and the current dependency/GPU verification limits.

See [implementation and startup guide (中文)](docs/autonomous_runtime.md) for
Codex authentication, API/GPU configuration, fixed interfaces, recovery and
the current verification limits.

The sections below describe the **legacy rule-based DataEvo loop**, which is a
separate entry point from the autonomous runtime.

For the completed two-round GSM8K GPU experiment, see the
[actual code, configuration, commands and results (中文)](docs/gsm8k_two_rounds_runbook.md).

For available RSIBench task sets, examples and the proposed agentic RL scope,
see the [dataset review (中文)](docs/agentic_datasets_review.md) and
[`scripts/preview_agentic_datasets.py`](scripts/preview_agentic_datasets.py).

For standalone **Terminal-Bench 2.0 (89 tasks) + SWE-bench Verified (RSIBench's
100-task subset)** evaluation, use
[`scripts/evaluate_agentic_benchmarks.py`](scripts/evaluate_agentic_benchmarks.py).
See the [evaluation runbook (中文)](docs/agentic_benchmark_eval.md) for model serving,
E2B/Docker preflight, smoke/full runs, native/DataEvo agent profiles and result files.
This entry point does not launch training.

`DataEvo` implements Training Dynamics Guided Data Optimization: a closed loop over
data generation, model training, training dynamics analysis, and data generation
strategy updates.

The legacy code was copied and adapted from `DataObs`. The new autonomous
runtime calls `DataObs/scripts/experiment_pipeline_vllm.py` directly to avoid
depending on a second copied training pipeline.

## Closed Loop

```text
strategy_round_t
  -> distill cold-start CoT data with adaptive controls
  -> SFT / GRPO training via the DataEvo experiment pipeline
  -> extract loss, reward, entropy, KL, clip ratio, grad norm, convergence speed
  -> Data Controller updates strategy_round_{t+1}
```

The controller does not predict a single abstract data-quality score. It consumes
training dynamics directly and applies rule-based updates:

- Fast early convergence: increase hard and longer-reasoning samples.
- Entropy collapse: increase prompt/style diversity, sampling temperature, and CoT samples.
- KL/clip/grad instability or response clipping: reduce hard/long pressure and lower temperature.
- Weak learning signal: increase learning value and move to a stronger teacher when a registry match is available.
- Low teacher pass rate or empty answers: simplify generation and enforce valid completion.

After the rule-based controller, an optional prompt agent rewrites the next-round
distillation prompt suffix from the same dynamics plus DataObs observation priors
such as SC gain/pass@K coverage, entropy headroom, SDS, template collapse, and
recovery/turning-point diagnostics.

## Main Entry

```bash
cd /home/hrh/CoT-DataSynth
python DataEvo/scripts/dataevo_pipeline.py \
  --evo-id dataevo_gsm8k_smoke \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3.5-0.8B \
  --teacher-model /data/pretrain_models/Qwen3.5-9B \
  --output-dir /data/hrh/COT/dataevo_experiments \
  --rounds 3 \
  --round-stages distill,metrics,sft,grpo \
  --gpu-ids 0 \
  --smoke-num-rows 64 \
  --teacher-num-samples 2 \
  --teacher-do-sample \
  --prompt-agent-mode heuristic \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --grpo-arg trainer.total_epochs=1
```

Use `--dry-run` to write strategies, feedback placeholders, manifests, and print
the exact inner commands without launching training.

To use a real LLM prompt agent, pass a command that reads the JSON context from
stdin and prints `{"prompt_suffix": "...", "rationale": "..."}`:

```bash
python DataEvo/scripts/dataevo_pipeline.py \
  ... \
  --prompt-agent-mode llm \
  --prompt-agent-cmd "python my_llm_prompt_agent.py"
```

## Outputs

```text
<output-dir>/<evo-id>/
  manifest.json
  history.jsonl
  rounds/<evo-id>_r000/
    distill/
    logs/
    sft/
    grpo/
  feedback/
    strategy_round_000.json
    dynamics_round_000.json
    prompt_agent_context_round_000.json
    prompt_agent_round_000.json
    update_round_000_to_001.json
    final_strategy.json
```

Each round strategy controls distillation through:

- `--evo-difficulty`: `easy`, `medium`, `hard`, or `mixed`
- `--evo-reason-length`: `short`, `medium`, `long`, or `mixed`
- `--evo-teacher-type`: `base`, `instruct`, `reasoning`, `specialist`, or `mixed`
- `--evo-prompt-suffix`: adaptive free-form prompt control text
- teacher model, temperature, top-p, number of CoTs, max new tokens, tensor parallel size

## Inner Pipeline

For one fixed strategy round, use:

```bash
python DataEvo/scripts/experiment_pipeline_vllm.py \
  --experiment-id dataevo_manual_r000 \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3.5-0.8B \
  --teacher-model /data/pretrain_models/Qwen3.5-9B \
  --output-dir /data/hrh/COT/dataevo_manual \
  --stages distill,metrics,sft,grpo \
  --gpu-ids 0 \
  --evo-difficulty hard \
  --evo-reason-length long \
  --evo-teacher-type reasoning
```
