# DataEvo

`DataEvo` implements Training Dynamics Guided Data Optimization: a closed loop over
data generation, model training, training dynamics analysis, and data generation
strategy updates.

The code is copied and adapted from `DataObs` without modifying the original
directory. The inner per-round pipeline stays aligned with
`DataObs/scripts/experiment_pipeline_vllm.py`; the outer loop is new.

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
