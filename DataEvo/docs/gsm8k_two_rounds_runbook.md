# GSM8K 两轮 GPU 实验：实际代码、配置与运行方法

本文记录 2026-10-19 已完成的实验，并给出在本机新建目录复跑的方法。查看结果不需要重新训练；第 5 节以后才会启动新的 GPU 实验。

## 1. 实际跑了什么

本次使用 **Codex 决策 → 冻结合成代码 → 本地教师合成/筛选 → 全参数 SFT → 评估 → 在线 GRPO → 评估 → probe/反馈 → 下一轮**。

学生是 `/home/hrh/models/Qwen3-4B`，教师是 `/home/hrh/models/Qwen3-14B`。教师以 OpenAI 兼容 API 服务运行，训练使用仓库已有的 DataObs/verl 流水线。**本次没有使用 Tinker、LoRA、Harbor/E2B，也没有运行工具交互 RL**；GRPO rollout 使用同步 HF，模型评估使用 vLLM。

每轮独立从同一个原始学生模型开始：512 条合成数据 → 64 个 optimizer step SFT → 32 step GRPO，每题 4 个 rollout。第二轮继承第一轮的实验反馈，不继承其权重。RL 使用 6705 道训练题，不只是 512 条 SFT 数据。

| 模型 | Dev 正确数 / 256 | 准确率 |
|---|---:|---:|
| 原模型 | 231 | 90.23% |
| 第一轮 SFT | 229 | 89.45% |
| 第一轮 RL | 227 | 88.67% |
| 第二轮 SFT | 232 | 90.63% |
| 第二轮 RL | 236 | 92.19% |

两轮结束后按 RL dev 选择第二轮，再评估官方测试集：原模型 **1191/1319 = 90.30%**，第二轮 RL **1178/1319 = 89.31%**，差值 **−0.99 个百分点**。本次没有验证出测试集提升，不建议据此替换原模型。只有一次训练随机种子；即使 temperature=0、seed=42，GPU 推理仍有小幅差异。

第二轮曾因候选上限不足失败，之后在独立 session 恢复；RL 权重导出也做了进程级兼容处理。因此这是有真实 Codex 反馈决策、并包含人工恢复的实验，不是已经验证过的全程无人值守运行。

## 2. 去哪里找“真正使用的代码”

原始产物根目录：

```text
/data/hrh/dataevo/gsm8k-two-rounds-20261019
```

下文简称 `SOURCE_RUN`。合成代码以 **runner/snapshots** 的冻结版本为准；固定训练代码以 **runner/source_snapshot** 和 `fixed_manifest.json` 为准。当前工作区后续可能变化，不能仅靠 Git HEAD 还原本次实验。

| 内容 | 相对于 SOURCE_RUN 的路径 |
|---|---|
| 第一轮实际合成代码 | `session/runner/snapshots/a0000/` |
| 第二轮最初 Codex 提交，候选不足失败 | `session/runner/snapshots/a0001/` |
| 第二轮实际完成训练所用合成代码 | `round2-recovery/runner/snapshots/a0001/` |
| 第一轮运行产物 | `session/runner/attempts/a0000/` |
| 第二轮成功运行产物 | `round2-recovery/runner/attempts/a0001/` |
| 原始部署配置 / 切分记录 | `session.json` / `split_manifest.json` |
| 各 session 的实际固定配置 | `session/runner/config.json`、`round2-recovery/runner/config.json` |
| 训练代码快照 | 各 session 的 `runner/source_snapshot/` |
| 固定文件哈希 / 环境 / 模型身份 | `runner/fixed_manifest.json`、`dependencies.txt`、`base_model_manifest.json` |
| 两轮逻辑映射 / 最终选择 | `rounds.json` / `selected-round.json` |
| Codex 两次原始决策 | `session/runner/codex/decision-0000/`、`decision-0001/` |

每份合成代码目录都有 `recipe.json`（参数）、`hypothesis.md`（假设与依据）、`teacher.txt`（提示模板）、`synthesize.py`（合成入口）、`synthesis_utils.py`（筛选工具）、`postprocess.py`（后处理）。第二轮还包括 `ledger.py`、`audit.txt`、`check_offline.py`。

仓库里具体负责执行的文件如下，链接指向当前源码：

| 代码 | 职责 |
|---|---|
| [scripts/run_session.py](../scripts/run_session.py) | 固定 CLI 入口 |
| [scripts/prepare_gsm8k_session.py](../scripts/prepare_gsm8k_session.py) | 创建可复现切分与初始配置 |
| [runtime/cli.py](../runtime/cli.py) | init、提交、等待、恢复、选择和 final-test |
| [runtime/codex_harness.py](../runtime/codex_harness.py) | 组装历史反馈，调用 `codex exec --json`，接收结构化决策 |
| [runtime/contracts.py](../runtime/contracts.py) | 配置与 attempt 接口校验 |
| [runtime/runner.py](../runtime/runner.py) | 冻结身份检查、GPU 租约、阶段执行与恢复 |
| [runtime/api.py](../runtime/api.py) | 教师 API 请求、token 预留与计量 |
| [runtime/data.py](../runtime/data.py) | 数据校验、防交叉污染、生成 SFT parquet |
| [runtime/backends.py](../runtime/backends.py) | 拼接 DataObs 训练/评估命令、导出权重 |
| [runtime/feedback.py](../runtime/feedback.py) | 汇总分数、轨迹、训练曲线和成本引用 |
| [DataObs/scripts/experiment_pipeline_vllm.py](../../DataObs/scripts/experiment_pipeline_vllm.py) | 本次实际调用的内层训练流水线 |
| [DataObs/lib/training/sft_dataobs.sh](../../DataObs/lib/training/sft_dataobs.sh) | SFT 启动脚本 |
| [verl/trainer/fsdp_sft_trainer.py](../../verl/trainer/fsdp_sft_trainer.py) | FSDP SFT trainer |
| [DataObs/lib/training/grpo_from_sft.sh](../../DataObs/lib/training/grpo_from_sft.sh) | GRPO 启动脚本 |
| [verl/trainer/main_ppo.py](../../verl/trainer/main_ppo.py) | RL trainer 入口 |
| [verl/workers/rollout/hf_rollout.py](../../verl/workers/rollout/hf_rollout.py) | 本次使用的同步 HF rollout |
| [DataObs/lib/evaluation/eval_vllm.py](../../DataObs/lib/evaluation/eval_vllm.py) | dev、probe 和正式测试生成/打分 |

不要使用 README 下半部分的旧 `dataevo_pipeline.py` 命令复现本实验；那是另一套基于规则的循环入口。

### 原样查看实际执行命令

每个成功 attempt 的 `commands.jsonl` 保存命令和环境；`logs/sft.log`、`logs/grpo.log` 保存 trainer 展开的最终参数。比下面的摘要更完整的实际参数，以这些记录和固定配置为准。

```bash
export SOURCE_RUN=/data/hrh/dataevo/gsm8k-two-rounds-20261019
export PY=/home/hrh/verl-cu130/bin/python
less "$SOURCE_RUN/session/runner/attempts/a0000/commands.jsonl"
less "$SOURCE_RUN/round2-recovery/runner/attempts/a0001/commands.jsonl"
```

## 3. 两轮合成方法及数据切分

第一轮：让教师生成简短、关闭 thinking、以 boxed 答案结束的解题过程。通过数值答案正确性、长度、格式、去重、训练来源检查后，按原始训练解答的计算步骤数分为四层，每层选 128 条，共 512 条。教师提示只提供题目；gold 用于筛选，不把原始 gold 解答补入训练数据。

第二轮：让教师输出 JSON 数量账本，`ledger.py` 以受限 AST 和精确 Fraction 执行计算，检查数量标签、状态与反向等式，再渲染成 boxed 解答。算术检查通过后，再由教师独立审核语义。账本允许 2–10 步，限制为数字、先前变量、算术及 ceil/floor；不使用 Python `eval`。

| 合成参数 | 第一轮 | 第二轮 |
|---|---|---|
| seed | 20261019 | 20261019 |
| temperature / top_p / max_tokens | 0.7 / 0.95 / 768 | 同左 |
| thinking | false | false |
| 并发 | 8 | 8 |
| 初始候选 / 每层 | 160 | 160 |
| 补充批次 | 32 | 32 |
| 目标 / 每层 | 128 | 128 |
| 序列 token 上限 | 1000 | 1000 |
| 语义审核 | 无 | temperature=0，top_p=1，max_tokens=256 |
| max_tasks | 1280 | 初始 1280；恢复后 2304 |

第一轮实际生成 699 个候选、529 个合格、选出 512 个。第二轮初次达到 1280 上限时，各层合格数为 `[130,128,112,85]`，没有进入训练。恢复时保留失败现场，校验候选文件 SHA256、候选顺序、问题和分层，复用原 1280 个候选，只对新候选发请求；最终 1490 个候选、516 个合格、选出 512 个。

恢复说明在 `round2-recovery/runner/recovery_provenance.json`，操作脚本在 `prepare_recovery.py`。恢复 session 内的 recipe parent 为 null，跨 session 的研究谱系由上述记录和 `rounds.json` 保存。

输入数据：

```text
/home/hrh/open_datasets/GSM8K/main/train-00000-of-00001.parquet  # 7473 题
/home/hrh/open_datasets/GSM8K/main/test-00000-of-00001.parquet   # 1319 题
```

训练集用 seed=42 打乱后，依次取 256 题 SFT val、256 题 dev、256 题 probe，剩下 6705 题作为 train_seed 和 RL train。合成仅使用 train_seed。dev 用于策略选择，probe 用于诊断；原实验在两轮结束并选定模型后才打开官方 test。现在 test 结果已经公开，复跑属于复现，不能把它当作新的未见测试证据。

## 4. 固定训练配置与本机环境

Python：`/home/hrh/verl-cu130/bin/python`。本次环境包含 torch 2.13.0、transformers 5.17.0、vLLM 0.29.0、Ray 2.56.1、flash_attn 2.8.3.post1；完整版本见 `runner/dependencies.txt`。

本机为 RTX PRO 6000 Blackwell 96GB；实验用 GPU 1、2 训练/评估，GPU 3 跑教师。重新运行前先确认这些卡当前空闲，文中的编号只表示上次的分配。

| 阶段 | 实际设置 |
|---|---|
| SFT | 全参数 FSDP；512 条；global batch 16；micro batch 2/GPU；max_length 1024 |
| SFT 优化 | lr=5e-6；warmup ratio=0.05；64 optimizer steps；save=64；validation=32 |
| SFT 其他 | gradient checkpointing；remove_padding=false；sequence parallel=1；nonthinking |
| GRPO | 全参数；32 steps；train batch / PPO minibatch=16；micro batch=1/GPU |
| GRPO rollout | HF sync；n=4；response length=768；nonthinking |
| GRPO 优化 | actor lr=1e-6；reference KL 系数 0.001；save=32；不做训练中的周期验证 |
| 评估 | vLLM；temperature=0；top_p=1；n=1；seed=42；max response=1024 |
| 评估其他 | batch=8；TP=1；GPU memory utilization=0.6；zeroshot；nonthinking |
| 预算 | max_attempts=2；max_rows=512；API tokens=3,000,000；阶段 4 小时；session 48 小时 |

配置同时指定 SFT epochs=4 和 total_training_steps=64，本次以实际完成的 64 optimizer steps 计。双卡 trainer 可能打印归一化后的每 rank batch=8，应与全局 batch=16 区分。

本机需要的环境处理：关闭 FlashInfer sampler；NCCL P2P/IB 关闭；Ray 使用 local 和独立临时目录。旧训练侧 vLLM 接口与 vLLM 0.29 不兼容，所以本次 GRPO 选 HF rollout，评估仍用 vLLM。不要为此停止机器上其他人的 Ray 或 GPU 进程。

## 5. 复跑公共准备：新目录、教师与固定配置

以下命令在仓库根目录执行，适用于本机已有环境。**第 6 节固定策略复跑与第 7 节自主研究二选一，每种模式都使用独立的新 session。** 不要直接复用已经结束的历史 session；其中已消费 final-test 标记，且存在冻结身份和预算记录。

### 5.1 设置新实验目录和环境

```bash
cd /home/hrh/CoT-DataSynth
export SOURCE_RUN=/data/hrh/dataevo/gsm8k-two-rounds-20261019
export PY=/home/hrh/verl-cu130/bin/python
# 改为自己这次的新目录；prepare 脚本拒绝覆盖已有目录。
export RUN_DIR=/data/hrh/dataevo/gsm8k-replay-20261020-01
export SESSION="$RUN_DIR/session"
export PATH=/home/hrh/verl-cu130/bin:$PATH
export PYTHONPATH=/home/hrh/CoT-DataSynth
export SYNTHESIS_API_BASE=http://127.0.0.1:8137/v1
export SYNTHESIS_API_KEY=local-teacher
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ULYSSES_SEQUENCE_PARALLEL_SIZE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export RAY_ADDRESS=local
export DATAEVO_RAY_TEMP_DIR="/tmp/dataevo-$(basename "$RUN_DIR")-ray"
export PYTHONUNBUFFERED=1
nvidia-smi
```

`local-teacher` 是本地服务占位 key。改用外部 API 时通过环境注入真实 key，不写入 recipe。这里直接调用 CLI，以便使用本次独立 Ray 目录；历史 `run_controller.sh` 会覆盖部分环境，并默认指向旧 session。

### 5.2 在另一个终端启动教师

```bash
CUDA_VISIBLE_DEVICES=3 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/hrh/verl-cu130/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/hrh/models/Qwen3-14B \
  --served-model-name local-qwen3-14b \
  --host 127.0.0.1 --port 8137 \
  --max-model-len 4096 --gpu-memory-utilization 0.65 \
  --enforce-eager --no-enable-log-requests
```

主终端检查服务已就绪：

```bash
curl --fail http://127.0.0.1:8137/v1/models
```

### 5.3 生成相同切分，套用本次真实训练配置，再初始化

`prepare_gsm8k_session.py` 生成的是通用配置，不能直接认为它就是本次 64/32 步配置。下面保留新数据路径，再套用历史真实 profile。

```bash
"$PY" DataEvo/scripts/prepare_gsm8k_session.py \
  --train /home/hrh/open_datasets/GSM8K/main/train-00000-of-00001.parquet \
  --test /home/hrh/open_datasets/GSM8K/main/test-00000-of-00001.parquet \
  --output "$RUN_DIR" \
  --base-model /home/hrh/models/Qwen3-4B \
  --python "$PY" --teacher-model local-qwen3-14b \
  --gpu-ids 1,2 --gpu-count 2 --heldout-size 256 --seed 42

"$PY" - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ['SOURCE_RUN'])
run = Path(os.environ['RUN_DIR'])
new = json.loads((run / 'session.json').read_text())
actual = json.loads((source / 'session.json').read_text())
actual['data'] = new['data']
# 如果更换 GPU、模型或教师，应在 init 前修改 actual 中对应字段。
(run / 'session.json').write_text(json.dumps(actual, indent=2) + '\n')
assert json.loads((run / 'split_manifest.json').read_text()) == json.loads(
    (source / 'split_manifest.json').read_text()), '原始数据或切分发生变化'
PY

"$PY" DataEvo/scripts/run_session.py init \
  --session "$SESSION" --config "$RUN_DIR/session.json"
```

init 会冻结训练代码、配置、数据和模型身份。之后不要原地修改固定源码或升级依赖；需要修改时创建新 session。这个命令冻结的是**当前**仓库与环境；若要审计级复现历史版本，应先在独立 checkout 对照历史 `source_snapshot`、manifest 和依赖清单准备环境，不能覆盖现有脏工作区。

## 6. 模式 A：复跑这两轮固定合成策略

这适合检查同一方法能否再得到相近结果。它使用历史 Codex 写出的代码，不重新调用 Codex 决策；会重新请求教师，不能保证得到完全相同的 512 条数据或逐位一致的权重。

### 6.1 准备两份工作区源码

```bash
"$PY" - <<'PY'
import json, os, shutil
from pathlib import Path
source = Path(os.environ['SOURCE_RUN'])
workspace = Path(os.environ['SESSION']) / 'workspace/attempts'
sources = {
    'a0000': source / 'session/runner/snapshots/a0000',
    'a0001': source / 'round2-recovery/runner/snapshots/a0001',
}
for aid, frozen in sources.items():
    target = workspace / aid
    shutil.copytree(frozen, target, dirs_exist_ok=True)
    recipe = json.loads((target / 'recipe.json').read_text())
    if aid == 'a0001':
        # 全新生成，不消费历史失败 session 的候选缓存。
        recipe['generation'].pop('cached_candidates', None)
        recipe['generation'].pop('cached_candidates_sha256', None)
        recipe['generation']['max_tasks'] = 2304
        recipe['parent_attempt_id'] = 'a0000'
    (target / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
PY
```

第一轮仍保留历史上限 1280；教师重新采样也可能出现候选不足。失败时不能改已冻结 attempt，应以新 attempt/session 记录预算或方法变更。不要复制 gold 解答补足。

### 6.2 提交第一轮并等待

```bash
"$PY" DataEvo/scripts/run_session.py validate --session "$SESSION" --attempt a0000
"$PY" DataEvo/scripts/run_session.py submit --session "$SESSION" --attempt a0000
"$PY" DataEvo/scripts/run_session.py wait --session "$SESSION" --attempt a0000 --timeout 14400
"$PY" DataEvo/scripts/run_session.py status --session "$SESSION"
```

submit 启动后台 worker。wait 返回终态信息；即使 CLI 成功退出，里面的状态也可能是 FAILED，必须看清是否 COMPLETED。遇到第 9 节的导出错误，先处理并 resume，再继续。所有阶段完成后查看反馈：

```bash
"$PY" DataEvo/scripts/run_session.py feedback --session "$SESSION" --attempt a0000
```

### 6.3 第一轮 COMPLETED 后提交第二轮

```bash
"$PY" DataEvo/scripts/run_session.py validate --session "$SESSION" --attempt a0001
"$PY" DataEvo/scripts/run_session.py submit --session "$SESSION" --attempt a0001
"$PY" DataEvo/scripts/run_session.py wait --session "$SESSION" --attempt a0001 --timeout 14400
"$PY" DataEvo/scripts/run_session.py history --session "$SESSION"
"$PY" DataEvo/scripts/run_session.py feedback --session "$SESSION" --attempt a0001
```

每轮自动运行 generate → postprocess → validate_data → SFT → SFT eval → RL → RL eval → base/SFT/RL probe → feedback。这个模式中的第二轮方法已经固定，不会因为这次第一轮结果不同而重新设计。

## 7. 模式 B：让 Codex 根据新反馈自主做两轮

使用第 5 节**刚初始化且尚未提交任何 attempt** 的新 session，不执行第 6 节。已有可用 Codex CLI 和认证的本机环境下，接入点就是 `runtime/codex_harness.py`；认证和一般配置见 [自主 runtime 使用说明](autonomous_runtime.md)。

历史 `RESEARCH.md` 有本次 512 行、64/32 步和不得因首轮退步停止等研究约束。复制时去掉旧 baseline 结果路径，避免将历史实验结果当成本次观测：

```bash
"$PY" - <<'PY'
import os
from pathlib import Path
source = Path(os.environ['SOURCE_RUN'])
session = Path(os.environ['SESSION'])
text = (source / 'session/workspace/RESEARCH.md').read_text()
start = text.index('Real baseline dev evidence')
end = text.index('First round: choose', start)
text = text[:start] + (
    'Use only feedback and probe artifacts from this new session as current evidence.\n'
    'Never synthesize from held-out questions; the final test stays sealed until selection.\n\n'
) + text[end:]
(session / 'workspace/RESEARCH.md').write_text(text)
PY

"$PY" DataEvo/scripts/run_session.py run --session "$SESSION"
```

控制器将历史、曲线、probe、成本和 workspace 交给 Codex，启动大致如下的调用（实际参数见源码及事件记录）：

```text
codex exec --json --skip-git-repo-check --sandbox workspace-write
  --output-schema <schema> --output-last-message <decision.json>
  -c 'approval_policy="never"' -C <workspace> -
```

Codex 编写 `attempts/<id>/` 下的 recipe、模板、合成及筛选程序，并返回 action=attempt 或 stop。固定 runner 再校验、冻结和训练，Codex 不直接发起训练。下一轮读取实际反馈再提出新假设。

合成程序接口是：

```text
synthesize.py --context CONTEXT_JSON --output CANDIDATES_JSONL
postprocess.py --context CONTEXT_JSON --input CANDIDATES_JSONL --output SELECTED_JSONL
```

教师调用必须走 `DataEvo.runtime.api.complete(context, messages, **sampling)` 计量。Codex 可以调整方法、rollout sampling、模板和筛选，固定训练/评估配置由 runner 控制。这里的 workspace 约束和预算计量不等于强网络隔离。

每轮决策保存到 `runner/codex/decision-*/`，包含 `context.json`、`prompt.txt`、`events.jsonl`、`decision.json`、`result.json`。API token 预算仅针对合成 API，不包含 Codex 自身账户消耗。

`max_attempts=2` 计算的是提交次数，失败也占次数，并非保证两个成功实验。本次第二轮因此另建 recovery session。当前环境的权重导出问题也可能打断自主循环；若还没修复通用导出适配，模式 A 更方便逐轮检查恢复。新策略的效果未知，不保证复现历史策略或获得提升。

## 8. 监控、基线、模型选择与正式测试

### 查看当前进度

```bash
"$PY" DataEvo/scripts/run_session.py status --session "$SESSION"
tail -n 60 "$SESSION/runner/attempts/a0000/worker.log"
tail -n 40 "$SESSION/runner/attempts/a0000/logs/sft.log"
tail -n 40 "$SESSION/runner/attempts/a0000/logs/grpo.log"
```

把 a0000 换为当前 attempt。历史只读监控脚本也可指定 session：

```bash
"$PY" "$SOURCE_RUN/monitor.py" "$SESSION"
```

### 原模型基线评估

下面命令使用与本次相同的评估参数，不经过 runner GPU 租约；仅在 GPU 1 空闲、没有运行中的 attempt 时执行。训练运行中不要并发执行它。

```bash
"$PY" DataObs/lib/evaluation/eval_vllm.py \
  --model-path /home/hrh/models/Qwen3-4B \
  --base-model /home/hrh/models/Qwen3-4B --dataset gsm8k \
  --eval-data "$RUN_DIR/dev.parquet" --output-dir "$RUN_DIR/base-dev" \
  --gpu-ids 1 --num-samples 1 --temperature 0 --top-p 1 --seed 42 \
  --max-response-len 1024 --batch-size 8 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.6 --prompt-template-method zeroshot
```

### 两轮都 COMPLETED 后，按 RL dev 自动选择，然后 final-test

```bash
export BEST_ATTEMPT="$("$PY" - <<'PY'
import os
from pathlib import Path
from DataEvo.runtime.common import Store
rows = Store(Path(os.environ['SESSION'])).rows()
assert len(rows) == 2 and all(r['state'] == 'COMPLETED' for r in rows), rows
print(max(rows, key=lambda r: r['score'])['id'])
PY
)"
test -n "$BEST_ATTEMPT"
"$PY" DataEvo/scripts/run_session.py select --session "$SESSION" --attempt "$BEST_ATTEMPT"
"$PY" DataEvo/scripts/run_session.py final-test --session "$SESSION" --attempt "$BEST_ATTEMPT"
```

final-test 使用固定评估 profile，写入一次性标记并关闭该 session 的后续优化。原模型官方测试用上面的基线命令，将 `--eval-data` 改为原始 test 路径、`--output-dir` 改为 `$RUN_DIR/base-final-test`，仍在 GPU 空闲时执行。先选择模型再看 test，不根据 test 挑另一轮。

## 9. 本次遇到的失败与恢复

### Transformers 5 权重导出错误

本次 RL 的 32 步已经训练完成，但旧 merger 引用不存在的 `AutoModelForVision2Seq` 导致导出失败。历史 `export_checkpoint.py` 仅在独立 Python 进程补一个兼容别名，再调用 `verl.model_merger`，没有修改冻结训练源码。

仅当 attempt 已 FAILED，确认错误发生于导出、且 actor 分片已完整保存时执行：

```bash
# 改成当前失败的 attempt。不要对正在训练的 checkpoint 操作。
export ATTEMPT=a0000
export ACTOR="$SESSION/runner/attempts/$ATTEMPT/experiment/grpo/global_step_32/actor"
CUDA_VISIBLE_DEVICES='' "$PY" "$SOURCE_RUN/export_checkpoint.py" \
  merge --backend fsdp --local_dir "$ACTOR" --target_dir "$ACTOR/huggingface"
CUDA_VISIBLE_DEVICES='' "$PY" "$SOURCE_RUN/verify_export.py" "$ACTOR"
# 本次双卡模型验证结果：798 个 rank tensors，exact_bfloat16_match。
"$PY" DataEvo/scripts/run_session.py resume --session "$SESSION" --attempt "$ATTEMPT"
"$PY" DataEvo/scripts/run_session.py wait --session "$SESSION" --attempt "$ATTEMPT" --timeout 14400
```

检查 `exact_bfloat16_match` 后再 resume。runner 会跳过已完成阶段，继续评估和反馈。verify 脚本面向本次 FSDP/模型布局，不能把上述验证数量泛化到其他模型。自主控制器可能已经将 FAILED 计入预算并继续或退出，恢复时还需核对控制器及其他 attempt 状态，不能盲目重启多个 worker。

### 候选不足

本次第二轮并非训练失败，而是筛选后的难题层不足。历史处理是保留原 FAILED attempt，创建独立 recovery session，候选 cap 从 1280 增至 2304，复用经身份校验的候选；训练参数、模板、筛选规则保持一致。

`prepare_recovery.py` 是针对旧路径的历史一次性操作脚本，**不要在已完成的旧目录直接重跑**。复跑第 6 节已移除缓存引用并采用第二轮恢复后的 cap。若仍不足，应记录新失败和新预算决定；resume 只适合源码、数据和配置不变的恢复。

## 10. 产物、模型、曲线与报告

每个成功 attempt 根目录下：

```text
recipe/context 与命令记录：commands.jsonl
候选与 API：candidates.jsonl、api_calls.jsonl、generation_stats.json
筛选：selected.jsonl、selection.jsonl、selection_stats.json
训练输入：dataset/train.parquet、dataset/summary.json
SFT 模型：experiment/sft/global_step_64/
RL 模型：experiment/grpo/global_step_32/actor/huggingface/
开发集：eval/sft/generated/、eval/rl/generated/
probe：probes/base/generated/、probes/sft/generated/、probes/rl/generated/
反馈：feedback.json、metrics/、logs/
```

评估目录的 `results.json` 保存分数，`responses.parquet` 保存逐题生成、答案及打分。额外 probe/final-test 位于 `extra/<操作>-<时间戳>/`。模型路径应以 `runtime/backends.py` 的 checkpoint 解析结果为准，以上为本次配置的实际布局。

本次已经生成的报告与清洗后曲线：

- [实验报告](/data/hrh/dataevo/gsm8k-two-rounds-20261019/REPORT.md)
- [比较统计](/data/hrh/dataevo/gsm8k-two-rounds-20261019/comparison.json)
- [训练曲线](/data/hrh/dataevo/gsm8k-two-rounds-20261019/training-curves.png)
- 第一轮 `a0000-sft-training.csv`、`a0000-grpo-training.csv`；第二轮 `a0001-sft-training.csv`、`a0001-grpo-training.csv`，均在 SOURCE_RUN 下，只包含实际 optimizer step 记录。

通用日志解析器可能混入配置/主机数字，检查此次曲线优先使用上述清洗 CSV：每轮 SFT 64 个 step、GRPO 32 个 step。

本次第二轮正式测试结果：

```text
round2-recovery/runner/attempts/a0001/extra/final-test-1792408682941825911/generated/
base-final-test/generated/
```

仅重新生成历史汇总报告（不训练、不评估）：

```bash
"$SOURCE_RUN/report-env/bin/python" "$SOURCE_RUN/summarize.py"
```

`report-env` 是继承训练环境的独立绘图环境，主训练 Python 没有安装 matplotlib。`finish_evaluation.py` 已执行过，不能用它在旧 session 重跑正式测试；它和 `prepare_recovery.py` 都绑定历史目录。`summarize.py` 也绑定自身所在目录，新的实验应按新路径汇总。

两轮教师共 2748 次调用、1,499,092 token，含第二轮语义审核。使用本地教师，没有外部付费教师 API；这里记录请求数和 token，不虚构美元成本。教师服务不归训练 worker 自动关闭，实验结束后在其独立终端 Ctrl-C，只停止自己本次启动的进程。
