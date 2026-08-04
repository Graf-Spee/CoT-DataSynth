# DataObs Heuristic Augmentation Baselines

This directory contains local-teacher heuristic data augmentation baselines for DataObs-format parquet distillation data.

All scripts read parquet files with `prompt`, `reward_model`, and `data_source`, generate candidates with the local vLLM teacher, and write:

- `--output-file`: kept training rows
- `<output stem>.candidates.parquet`: all generated candidates and filter metadata
- `<output stem>.summary.json`: run summary

Common required arguments:

```bash
python DataObs/baselines/heuristics/<script>.py \
  --dataset <dataset> \
  --input-file <prepared_input.parquet> \
  --output-file <augmented_output.parquet> \
  --model-id <teacher_model_path> \
  --gpu-ids 0 \
  --trust-remote-code
```

Use `--smoke-num-rows N` for quick checks. If generating more than one sample per prompt, pass `--do-sample`.

Pipelines that generate answers apply the corresponding `verl.utils.reward_score` reward through DataObs helpers. `question_augmentation.py` is different: it generates backward questions and only drops empty generations.

## Pipelines

### Question Rephrasing

File: `question_rephrasing.py`

Generates semantically equivalent rewrites of each seed question, then asks the teacher to solve each rewrite. A rewritten sample is kept only if the generated CoT answer passes the DataObs reward check against the original gold answer.

Example:

```bash
python DataObs/baselines/heuristics/question_rephrasing.py \
  --dataset gsm8k \
  --input-file /path/to/prepared_gsm8k.parquet \
  --output-file /path/to/gsm8k_question_rephrasing.parquet \
  --model-id /path/to/teacher \
  --gpu-ids 0 \
  --num-rephrases 1 \
  --num-cots 1
```

### Answer Augmentation

File: `answer_augmentation.py`

Keeps the original question fixed and samples multiple teacher reasoning paths. Each candidate is kept only if its final answer passes the DataObs reward check against the original gold answer.

Example:

```bash
python DataObs/baselines/heuristics/answer_augmentation.py \
  --dataset gsm8k \
  --input-file /path/to/prepared_gsm8k.parquet \
  --output-file /path/to/gsm8k_answer_augmentation.parquet \
  --model-id /path/to/teacher \
  --gpu-ids 0 \
  --do-sample \
  --num-augmented-answers 4
```

`--use-original-metamath-prompt` switches from DataObs dataset-specific reasoning prompts to the short two-shot MetaMath-style prompt used by the original answer augmentation baseline.

### Reverse Thinking Augmentation

File: `reverse_thinking_augmentation.py`

Runs the RevThink-style pipeline: generate a backward/inverse question, generate forward reasoning for the original question, generate backward reasoning for the backward question, then run a teacher consistency check. A kept quadruplet is expanded into three rows: forward reasoning, backward-question generation, and backward reasoning.

Coding datasets are not supported for this method because the backward-question prompts do not cover code generation tasks.

Example:

```bash
python DataObs/baselines/heuristics/reverse_thinking_augmentation.py \
  --dataset gsm8k \
  --input-file /path/to/prepared_gsm8k.parquet \
  --output-file /path/to/gsm8k_reverse_thinking.parquet \
  --model-id /path/to/teacher \
  --gpu-ids 0
```

### Question Augmentation

File: `question_augmentation.py`

Implements the original reference code's practical Question Augmentation setting: instead of generating truly novel questions, it trains the backward-question generation task. For each seed `(question, gold_answer)`, the teacher generates an inverse/backward question using the RevThink backward-question prompt. The kept training row asks the student to generate the inverse question, with the generated backward question as the target answer.

This covers the datasets that have backward-question ICL prompts in this directory: `arc-challenge`, `commonsenseqa`, `gsm8k`, `math`, `math-500`, and `strategyqa`.
Coding datasets are disabled for this method because the backward-question prompts do not cover code generation tasks.

Example:

```bash
python DataObs/baselines/heuristics/question_augmentation.py \
  --dataset gsm8k \
  --input-file /path/to/prepared_gsm8k.parquet \
  --output-file /path/to/gsm8k_question_augmentation.parquet \
  --model-id /path/to/teacher \
  --gpu-ids 0 \
  --do-sample \
  --num-backward-questions 1
```
