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

When these scripts are launched through `DataObs/scripts/experiment_pipeline.py` or `experiment_pipeline_vllm.py`, `--teacher-max-new-tokens` is the default override for every distillation generation budget whose specific option is not set. Specific options still take precedence:

- `--forward-reasoning-max-new-tokens`: teacher filter, answer augmentation, question-rephrasing answer/solve generation, and reverse-thinking forward reasoning.
- `--backward-reasoning-max-new-tokens`: reverse-thinking backward reasoning.
- `--rephrase-max-new-tokens`: question-rephrasing rewrite generation.
- `--backward-question-max-new-tokens`: question augmentation and reverse-thinking backward-question generation.
- `--consistency-max-new-tokens`: reverse-thinking consistency checks.

Pipelines that generate answers apply the corresponding `verl.utils.reward_score` reward through DataObs helpers. `question_augmentation.py` is different: it generates backward questions and keeps only candidates whose cleaned backward question can be extracted.

## Teacher Filter Toggle

`--disable-teacher-filter` only disables reward-based teacher correctness filtering. It does not disable structural filters needed to build valid training rows.

- `teacher_filter`, `answer_augmentation`, and `question_rephrasing`: reward scores are still computed and recorded in `teacher_score` / `teacher_filter_passed`, but all answer candidates that reached the reward stage are kept. For `question_rephrasing`, malformed or unextractable rephrases are still dropped before answer generation.
- `reverse_thinking`: the forward-answer reward check is bypassed, but the backward-question extraction, backward-answer extraction, and consistency check still gate kept rows.
- `question_augmentation`: the flag has no effect on kept rows because there is no answer correctness reward filter. Candidates are kept only when `clean_backward_question` is nonempty.

## Prompt Support Matrix

Legend:

- Native baseline: task-specific prompt/ICL from the baseline prompt set.
- Borrowed: prompt reused outside its native dataset family.
- DataObs fallback: DataObs dataset-specific reasoning suffix and reward-compatible final-answer format.
- Unsupported: pipeline rejects or has no backward-question prompt.

| Dataset | teacher_filter | answer_augmentation | question_rephrasing | question_augmentation | reverse_thinking |
| --- | --- | --- | --- | --- | --- |
| `gsm8k` | DataObs fallback, boxed final answer | DataObs fallback by default; original MetaMath is native with `--use-original-metamath-prompt` | MetaMath rephrase prompt + DataObs fallback answer prompt | math backward-question prompt, no MCQ wording | math backward-question prompt + DataObs/baseline reasoning prompts |
| `math`, `math-500` | DataObs fallback, boxed final answer | DataObs fallback by default; original MetaMath is borrowed | MetaMath rephrase prompt + DataObs fallback answer prompt | math backward-question prompt, no MCQ wording | math backward-question prompt + DataObs/baseline reasoning prompts |
| `numinamath` | DataObs fallback, boxed final answer | DataObs fallback by default; original MetaMath is borrowed | MetaMath rephrase prompt + DataObs fallback answer prompt | unsupported | unsupported |
| `arc-challenge` | DataObs fallback MCQ prompt | DataObs fallback only; original MetaMath unsupported | MetaMath rephrase prompt + DataObs fallback answer prompt | native baseline MCQ backward prompt | native baseline MCQ backward prompt |
| `commonsenseqa` | DataObs fallback MCQ prompt | DataObs fallback only; original MetaMath unsupported | MetaMath rephrase prompt + DataObs fallback answer prompt | native baseline MCQ backward prompt | native baseline MCQ backward prompt |
| `aqua_rat` | DataObs fallback MCQ prompt | DataObs fallback only; original MetaMath unsupported | MetaMath rephrase prompt + DataObs fallback answer prompt | unsupported | unsupported |
| `strategyqa` | DataObs fallback with `Yes or no:` student cue | DataObs fallback with `Yes or no:` student cue; original MetaMath unsupported | MetaMath rephrase prompt + DataObs yes/no answer prompt | yes/no backward-question prompt | yes/no backward-question prompt |
| `mbpp`, `mbppplus`, `humaneval`, `humanevalplus` | DataObs code fallback prompt | DataObs code fallback only; original MetaMath unsupported | MetaMath rephrase prompt + DataObs code fallback answer prompt | unsupported | unsupported |

`answer_augmentation.py` keeps the DataObs prompt as the default. The original MetaMath prompt is exactly the two math few-shot examples from `/home/hrh/Baseline-Heuristics-CoT/rephrase_questions.py` followed by `Q: <question>\nA: Let's think step by step.` It is native for GSM8K and borrowed for math-style datasets; it is not a native prompt for MCQ, StrategyQA, or code tasks. The pipeline warns when the DataObs default is used for answer augmentation, and rejects original MetaMath prompt use on unsupported dataset families.

## Output Cleaning Policy

- `teacher_filter` and `answer_augmentation`: SFT answers are not cleaned, so Qwen `<think>` blocks are preserved as CoT training material.
- `question_rephrasing`: the rephrase generation is cleaned before it is used as SFT input or fed back to the teacher. The raw rephrase is retained in metadata, but `question` is the clean rephrased question. Answer generations are not cleaned.
- `question_augmentation`: the SFT input is the backward-question generation task. The SFT answer is the raw backward-question generation with only a trailing backward-answer annotation removed, so complete teacher `<think>` traces are preserved. A candidate is kept only when `clean_backward_question` is nonempty. Reusable `backward_question`/`clean_backward_question` is retained separately with thinking, parser labels, and answer annotations removed.
- `reverse_thinking`: forward and backward reasoning answers are not cleaned. The backward-question generation SFT row follows `question_augmentation` and keeps the raw generation target with only trailing answer annotation removed. The backward question is cleaned before it is used for backward reasoning or student input.

Student inputs should never contain `<think>`, generated reasoning, or unintended final-answer annotations. `<think>` is allowed only in SFT output fields for answer/reasoning generation tasks.

## Final-Answer Format

GSM8K teacher filtering now asks for `\boxed{}` final answers, matching GSM8K eval and the heuristic GSM8K reasoning prompt. This avoids relying on `#### <number>` while the active DataObs GSM8K reward extracts boxed answers first and falls back to the last visible number only when no box exists.

The vLLM eval path defaults to `enable_thinking=False`; training outputs may still contain `<think>` when the teacher produced reasoning. This is the current DataObs strategy: train on teacher CoT traces, evaluate with thinking disabled unless `eval_vllm.py --enable-thinking` is explicitly added outside the pipeline.

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

`--use-original-metamath-prompt` switches from DataObs dataset-specific reasoning prompts to the short two-shot MetaMath-style prompt used by the original answer augmentation baseline. This prompt follows the original MetaMath answer-augmentation format and does not ask the teacher to end with `\boxed{}`; for GSM8K, the DataObs reward still scores it through the GSM8K extractor, which can fall back to the final visible numeric answer when no boxed answer is present.

### Reverse Thinking Augmentation

File: `reverse_thinking_augmentation.py`

Runs the RevThink-style pipeline: generate a backward/inverse question, generate forward reasoning for the original question, generate backward reasoning for the backward question, then run a teacher consistency check. A kept quadruplet is expanded into three rows: forward reasoning, backward-question generation, and backward reasoning.

Coding datasets, `aqua_rat`, and `numinamath` are not supported for this method because there is no backward-question prompt/ICL coverage.

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

Implements the original reference code's practical Question Augmentation setting: instead of generating truly novel questions, it trains the backward-question generation task. For each seed `(question, gold_answer)`, the teacher generates an inverse/backward question using the RevThink backward-question prompt. The kept training row asks the student to generate the inverse question, with the generated backward question as the target answer. Candidates whose raw generation cannot be cleaned into a nonempty backward question are dropped.

This covers the datasets that have backward-question ICL prompts in this directory: `arc-challenge`, `commonsenseqa`, `gsm8k`, `math`, `math-500`, and `strategyqa`.
Coding datasets, `aqua_rat`, and `numinamath` are disabled for this method because the backward-question prompts do not cover them.

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
