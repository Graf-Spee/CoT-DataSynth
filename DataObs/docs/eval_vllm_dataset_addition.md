# vLLM Eval Flow and Dataset Onboarding

本文说明当前 DataObs 纯 vLLM eval 的执行链路、已支持数据集、新增数据集需要补哪些位置，并用一个具体数据集 `svamp` 举例。最后汇总本机已有 GSM8K vLLM 评测结果。

## 1. 当前 vLLM eval 流程

纯 vLLM eval 的主入口是：

```bash
python DataObs/lib/evaluation/eval_vllm.py \
  --model-path /path/to/checkpoint_or_hf_model \
  --base-model /path/to/base_model \
  --dataset gsm8k \
  --output-dir /path/to/eval_output \
  --gpu-ids 0 \
  --num-samples 1 \
  --temperature 0.0 \
  --max-response-len 16384
```

在实验 pipeline 中，`DataObs/scripts/experiment_pipeline_vllm.py` 的 `sft_eval` 和 `grpo_eval` stage 会调用这个脚本。它和老的 `scripts/eval.sh` 不同：这里不走 `verl.trainer.main_generation` / `verl.trainer.main_eval`，而是在一个 Python 脚本里直接用 vLLM 生成并调用 reward function 打分。

流程如下：

1. 解析 `--dataset`，通过 `_normalize_dataset()` 做别名归一化。
2. 用 `DATASET_CONFIG` 找默认 eval parquet 和 reward function。
3. 如果 `--model-path` 是 LoRA adapter，即包含 `adapter_model.safetensors`，先 merge 到 `output_dir/merged_model`。
4. 调 `DataObs/lib/evaluation/prepare_eval_data.py` 生成评测 parquet。
5. `prepare_eval_data.py` 如果发现源 parquet 已经包含 `prompt`、`reward_model`、`data_source` 三列，就直接复制；否则调用 `verl/utils/eval/apply_prompt_template.py` 按数据集套模板。
6. `eval_vllm.py` 读取 `prompt` 列；如果 prompt 是 chat message list，就用 tokenizer 的 `apply_chat_template()` 转成 vLLM 输入字符串。
7. 用 `vllm.LLM.generate()` 生成 responses。
8. 对每条题目的第一个 response 调 `compute_score(response, ground_truth)` 得到分数。
9. 写出结果：
   - `generated/responses.parquet`
   - `generated/results.json`
   - `generated/responses_labeled.metrics.json`
   - `generated/responses_labeled.json`
   - `logs/evaluation.log`

需要注意：

- `--num-samples > 1` 会让 vLLM 每题生成多条，但 `eval_vllm.py` 当前 `accuracy` 只按第一个 response 算。
- `extract_pred` 会从 reward 文件里加载，但当前纯 vLLM 脚本没有实际用它计算主指标。
- `--batch-size` 现在只写进 config 记录，`llm.generate(prompts, sampling_params)` 没有显式用这个 batch size 切批。

## 2. 纯 vLLM 当前支持的数据集

`DataObs/lib/evaluation/eval_vllm.py` 当前 `DATASET_CONFIG` 支持：

| Dataset | Reward function | 默认 eval data |
|---|---|---|
| `arc-challenge` | `verl/utils/reward_score/multiple_choice.py` | `/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet` |
| `aqua_rat` | `verl/utils/reward_score/multiple_choice.py` | `/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet` |
| `commonsenseQA` | `verl/utils/reward_score/multiple_choice.py` | `/data/open_datasets/CommonsenseQA/data/validation-00000-of-00001.parquet` |
| `gsm8k` | `verl/utils/reward_score/gsm8k.py` | `/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet` |
| `humaneval` | `verl/utils/reward_score/mbpp.py` | `/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet` |
| `humanevalplus` | `verl/utils/reward_score/mbpp.py` | `/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet` |
| `math-500` | `verl/utils/reward_score/math_verify.py` | `/data/open_datasets/MATH-500/test.parquet` |
| `mbpp` | `verl/utils/reward_score/mbpp.py` | `/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet` |
| `mbppplus` | `verl/utils/reward_score/mbpp.py` | `/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet` |
| `numinamath` | `verl/utils/reward_score/math_verify.py` | `/data/open_datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet` |
| `strategyQA` | `verl/utils/reward_score/truefalse.py` | `/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet` |

容易混淆的点：

- `_normalize_dataset()` 里有 `livecodebench` 和 `math` 的别名，但它们没有注册到 `DATASET_CONFIG`，所以纯 vLLM eval 当前不能直接跑这两个。
- 老入口 `scripts/eval.sh` 支持更多：`livecodebench`、`math`、`bfcl` 等；其中 BFCL 是单独评测链路，不是普通 parquet + reward_model。
- `DataObs/tools/run_all_eval.py` 有自己的默认 `DATASETS` 列表；如果新数据集要进入批量评测，也要同步加入。

## 3. 新增数据集要补哪些部分

最少需要确认四件事：

1. eval parquet 路径：默认文件在哪里，是否是 `.parquet`。
2. prompt 构造：源 parquet 是否已经包含 `prompt/reward_model/data_source` 三列。
3. reward function：如何从模型输出和 ground truth 计算 0/1 或 float score。
4. 注册入口：`eval_vllm.py` 的 `DATASET_CONFIG` 和别名映射。

如果源 parquet 已经是统一格式：

```text
prompt: chat message list or plain string
reward_model: dict, must contain ground_truth
data_source: dataset name
```

那么只需要补 `DATASET_CONFIG` 和 reward function。

如果源 parquet 是原始格式，比如只有 `question` / `answer`，还需要补 prompt template preprocessing。

## 4. 例子：新增 `svamp`

假设我们要加 SVAMP，原始 parquet 路径为：

```text
/data/open_datasets/SVAMP/test.parquet
```

假设原始 schema 是：

```text
Body: problem background
Question: question text
Answer: numeric answer
```

### 4.1 新增 prompt template 文件

新增文件：

```text
verl/utils/eval/prompt_templates/svamp.py
```

内容示例：

```python
svamp_plain = [
    {
        "role": "user",
        "content": "{question}\nPlease put your final answer within \\boxed{{}}.",
    },
]

svamp_zeroshot = [
    {
        "role": "user",
        "content": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    },
]

svamp_4_shot = [
    {
        "role": "user",
        "content": "{question}\nLet's think step by step\nAnswer:",
    },
]
```

如果不需要 few-shot，可以不提供 `svamp_4_shot`，然后在 `_select_template()` 调用时只传 plain/zeroshot。这样用户传 `--prompt-template-method fewshot` 会明确报错。

### 4.2 在 `apply_prompt_template.py` 里注册预处理

文件：

```text
verl/utils/eval/apply_prompt_template.py
```

先在 import 区加：

```python
from verl.utils.eval.prompt_templates.svamp import svamp_plain, svamp_zeroshot
```

fallback import 区也要加：

```python
from prompt_templates.svamp import svamp_plain, svamp_zeroshot
```

然后新增处理函数：

```python
def _svamp(df: pd.DataFrame, method: str) -> pd.DataFrame:
    template = _select_template(method, svamp_plain, svamp_zeroshot)
    df = df.copy()

    def _fill_template(row: pd.Series) -> List[Dict]:
        body = str(_get_column(row, "Body")).strip()
        question = str(_get_column(row, "Question")).strip()
        full_question = f"{body}\n{question}".strip()
        return _deepcopy_template_with_last_message(template, question=full_question)

    def _reward_model(row: pd.Series) -> Dict[str, str]:
        return {"ground_truth": str(_get_column(row, "Answer")).strip()}

    df["reward_model"] = df.apply(_reward_model, axis=1)
    df["prompt"] = df.apply(_fill_template, axis=1)
    df["data_source"] = "svamp"
    return df[["prompt", "reward_model", "data_source"]]
```

最后在 `PROMPT_TEMPLATE` 注册：

```python
PROMPT_TEMPLATE: Dict[str, PreprocessFn] = {
    ...
    "svamp": _svamp,
}
```

### 4.3 新增 reward function

新增文件：

```text
verl/utils/reward_score/svamp.py
```

如果 SVAMP 是数字答案，可以复用 GSM8K 的 boxed/number 抽取逻辑。示例：

```python
from verl.utils.reward_score.gsm8k import extract_solution


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_pred(solution_str: str) -> str:
    answer = extract_solution(solution_str, method="OC-boxed")
    return answer if answer is not None else ""


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    pred = _to_float(extract_pred(solution_str))
    gt = _to_float(ground_truth)
    if pred is None or gt is None:
        return 0.0
    return score if pred == gt else format_score
```

如果答案是选择题、true/false、代码测试，不要照搬这个文件；应复用或新写对应任务类型的判分器。

### 4.4 在 `eval_vllm.py` 注册数据集

文件：

```text
DataObs/lib/evaluation/eval_vllm.py
```

在 `DATASET_CONFIG` 加：

```python
DATASET_CONFIG: dict[str, dict[str, Any]] = {
    ...
    "svamp": {
        "reward": "verl/utils/reward_score/svamp.py",
        "eval_data": "/data/open_datasets/SVAMP/test.parquet",
    },
}
```

在 `_normalize_dataset()` 里加别名：

```python
mapping = {
    ...
    "svamp": "svamp",
}
```

如果有常见别名，比如 `svampmath`，也可以一起加。

### 4.5 如果要接入其他入口

纯 vLLM eval 只需要上面几步。但如果你还希望这些工具也能跑 `svamp`，需要额外同步：

- `DataObs/tools/run_all_eval.py`：把 `svamp` 加入 `DATASETS`，用于批量模型评测。
- `DataObs/lib/evaluation/dataobs_eval_runner.py`：如果你还使用 Python-native verl/Ray eval runner，要在 `_dataset_config()` 加 `svamp`。
- `scripts/eval.sh`：如果你还使用老 shell eval，要加 case 分支、reward path、eval data path。
- `DataObs/scripts/experiment_pipeline.py`：如果要做完整 distill/SFT/GRPO pipeline，还要在 `DATASET_DEFAULTS` 加 `distill_input`、`rl_train`、`rl_val`。

## 5. 最小验证命令

先验证预处理是否能跑：

```bash
python DataObs/lib/evaluation/prepare_eval_data.py \
  --dataset svamp \
  --input /data/open_datasets/SVAMP/test.parquet \
  --output /tmp/svamp_prepared.parquet \
  --method zeroshot
```

然后跑小样本 vLLM smoke test：

```bash
python DataObs/lib/evaluation/eval_vllm.py \
  --model-path /data/pretrain_models/Qwen3-4B \
  --base-model /data/pretrain_models/Qwen3-4B \
  --dataset svamp \
  --output-dir /tmp/eval_svamp_qwen3_4b \
  --gpu-ids 0 \
  --max-samples 16 \
  --num-samples 1 \
  --temperature 0.0 \
  --max-response-len 2048
```

检查输出：

```bash
cat /tmp/eval_svamp_qwen3_4b/generated/results.json
cat /tmp/eval_svamp_qwen3_4b/logs/evaluation.log
```

## 6. GSM8K 当前表现汇总

下面结果来自本机已有输出目录：

```text
outputs/evals/*/generated/results.json
outputs/evals/*/passk/passk_metrics.json
```

### 6.1 `eval_vllm.py` 主 accuracy 口径

这个 accuracy 是 `eval_vllm.py` 在 `generated/results.json` 中写出的主指标。即使 `num_samples_per_prompt=16`，当前脚本也只用每题第一个 response 计算 accuracy。

| Run | Model path 摘要 | Samples | Temp | Max len | Correct / Total | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `qwen3_4b_gsm8k_greedy_pass1` | `/data/pretrain_models/Qwen3-4B` | 1 | 0.0 | 2048 | 1182 / 1319 | 89.61% |
| `qwen3_4b_gsm8k_greedy_pass1_len16k` | `/data/pretrain_models/Qwen3-4B` | 1 | 0.0 | 16384 | 1178 / 1319 | 89.31% |
| `qwen3_4b_gsm8k_passk_k16` | `/data/pretrain_models/Qwen3-4B` | 16 | 0.7 | 2048 | 1187 / 1319 | 89.99% |
| `qwen3_4b_gsm8k_passk_k16_len16k` | `/data/pretrain_models/Qwen3-4B` | 16 | 0.7 | 16384 | 1188 / 1319 | 90.07% |
| `exp1200_qwen3_14b_distill_sft_gsm8k_passk_k16_len16k` | Qwen3-14B teacher distill SFT merged model | 16 | 0.7 | 16384 | 1182 / 1319 | 89.61% |
| `exp1200_qwen3_14b_distill_sft_grpo_gsm8k_passk_k16_len16k` | Qwen3-14B teacher distill + GRPO merged model | 16 | 0.7 | 16384 | 1191 / 1319 | 90.30% |
| `exp5000_easy_distill_sft_gsm8k_passk_k16_len16k_run2` | easy subset distill SFT merged model | 16 | 0.7 | 16384 | 1187 / 1319 | 89.99% |

### 6.2 pass@k / self-consistency 口径

这些指标来自 `passk/passk_metrics.json`，不是 `eval_vllm.py` 主脚本直接产出的 `accuracy`。它们基于每题 16 个 samples 重新统计：

| Run | Problems | Samples | pass@1 baseline | pass@16 | self-consistency@16 |
|---|---:|---:|---:|---:|---:|
| `qwen3_4b_gsm8k_passk_k16` | 1319 | 16 | 89.75% | 96.13% | 92.42% |
| `qwen3_4b_gsm8k_passk_k16_len16k` | 1319 | 16 | 89.80% | 96.44% | 92.12% |
| `exp1200_qwen3_14b_distill_sft_gsm8k_passk_k16_len16k` | 1319 | 16 | 89.62% | 96.44% | 92.42% |
| `exp1200_qwen3_14b_distill_sft_grpo_gsm8k_passk_k16_len16k` | 1319 | 16 | 89.73% | 96.36% | 92.42% |
| `exp5000_easy_distill_sft_gsm8k_passk_k16_len16k_run2` | 1319 | 16 | 90.18% | 96.06% | 92.49% |

### 6.3 简要结论

- Qwen3-4B base 在 GSM8K 上 greedy pass@1 约 89.3%-89.6%。
- 用 16 samples、temperature 0.7 时，第一个样本的 accuracy 约 90.0%，pass@16 约 96.1%-96.4%。
- self-consistency@16 约 92.1%-92.5%，比单样本主 accuracy 高约 2 个点左右。
- 当前几组 SFT/GRPO 结果和 base Qwen3-4B 差距不大；`exp1200_qwen3_14b_distill_sft_grpo...` 的主 accuracy 是 90.30%，比对应 SFT 的 89.61% 高 0.68 个百分点。
- 因为 `eval_vllm.py` 主 accuracy 只看第一个 response，若目标是比较采样潜力，应同时看 `passk/passk_metrics.json`。
