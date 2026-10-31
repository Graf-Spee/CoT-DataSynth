# Terminal-Bench 2.0 / SWE-bench Verified 测试入口

本入口只做评测：已有模型 API → Harbor 创建环境 → agent 调用工具 → 原任务 verifier → 成绩和轨迹。无需初始化训练 session，不运行 SFT、RL 或数据合成，也不调用旧 RL rollout 的依赖预检。

代码：[`evaluate_agentic_benchmarks.py`](../scripts/evaluate_agentic_benchmarks.py) 是命令行入口；[`benchmark_eval.py`](../runtime/benchmark_eval.py) 负责选题、预检、Harbor 配置和汇总；[`harbor_terminal_agent.py`](../runtime/harbor_terminal_agent.py) 是已有的 DataEvo 工具交互 agent。

## 数据范围与 agent

| benchmark 参数 | 默认本地数据 | 完整本地范围 | native profile |
| --- | --- | --- | --- |
| `terminal_bench_2` | `/home/hrh/RSIBench-Data-main/benchmarks/terminal_bench_2/datasets/full_89` | Terminal-Bench 2.0，89 题 | Harbor `terminus-2`，JSON parser |
| `swe_bench_verified` | `/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_verified/datasets/seed23_random_100` | RSIBench 固定 seed=23 的 100 题子集 | Harbor `mini-swe-agent==2.4.5` |

SWE-bench Verified 官方全集为 500 题；这里的 `--all-tasks` 默认是本地 100 题，不能当成 500 题全集成绩。若准备好其他 Harbor 格式数据，可用 `--swe-dataset /absolute/path` 或 `--terminal-dataset /absolute/path` 显式覆盖；不会自动下载、修改或扩充任务。

默认 `--agent-profile dataevo`：两套数据都使用本仓库的 host-side JSON terminal agent，与已有训练协议一致。模型返回 `{"command":"..."}`，工具在沙箱里执行，结果送回模型；`{"final":"..."}` 结束，随后由任务自己的 tests/verifier 判分。它不读取参考 solution，且模型 API 可以是本机 `127.0.0.1`。

`--agent-profile native`：采用表中的 Harbor agent，方便测这些 agent 下的模型能力。mini-swe-agent 本身在沙箱中运行，模型 API 必须能从沙箱访问。Terminus 在宿主侧访问模型。native 与 RSIBench 并非完全相同设置：这里 temperature=0，RSIBench Terminus 为 1；这里使用 Harbor 自带 mini-swe 安装器，没有复制 RSIBench 额外的 LiteLLM 固定版本适配。因此都应记录为本框架对应配置下的结果，不直接声称官方榜单成绩。

## 当前机器已验证与阻塞

2026-10-19：Harbor Python 为 `/home/hrh/RSIBench-Data-full/.venv/bin/python`，Harbor 0.17.1。两套本地任务共 189 题通过真实 Harbor Task 格式加载；新入口和已有相关测试共 7 项通过，覆盖真实 HTTP 工具交互、三种 agent 的配置、选题和严格汇总。HTTP 的模型响应与沙箱执行是明确的测试替身，不代表这些 benchmark 的实跑。

尚无真实分数：当前进程未设置 `E2B_API_KEY`，当前用户 `docker info` 被 daemon 权限拒绝，默认 `http://127.0.0.1:8000/v1` 也未提供模型服务。预检会以退出码 2 报告这些阻塞，不创建沙箱、不生成零分报告。需要可用 E2B 凭据或者当前用户可用的 Docker，以及运行中的模型 API，才能继续真实 smoke。

## 1. 无费用检查任务

所有命令从仓库根目录运行：

```bash
cd /home/hrh/CoT-DataSynth
HARBOR_PY=/home/hrh/RSIBench-Data-full/.venv/bin/python

# 默认每套选排序后的第一题；输出包含 task ID、设置、文件 SHA256。
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --model student --dry-run

# 完整本地范围：89 + 100。只读校验，无网络、无 GPU、无文件写入。
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --model student --all-tasks --dry-run
```

选题支持 `--n-tasks N`（每套按任务目录 ID 排序取前 N）、`--all-tasks` 或重复 `--task ID`，三者互斥。`--task` 要同时指定单个 `--benchmark`，例如：

```bash
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --benchmark swe_bench_verified --task django__django-11265 \
  --model student --dry-run
```

只把 evaluation 数据用于测评，不将这些题目或轨迹混入训练数据。

## 2. 启动待测模型 API

如果已经有 OpenAI-compatible 服务，直接使用其 `--api-base` 和 `/models` 中列出的模型 ID。以下是当前本地 Qwen3-4B 的独立 vLLM 服务示例，在另一个终端前台启动。先检查 GPU，选择空闲卡；`EVAL_GPU` 由操作者按实时占用填写，不复用其他任务的卡。

```bash
nvidia-smi
EVAL_GPU=3
CUDA_VISIBLE_DEVICES="$EVAL_GPU" /home/hrh/verl-cu130/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /home/hrh/models/Qwen3-4B \
  --served-model-name student \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85
```

此 vLLM 启动模块在本机版本仍可用，但已发出弃用提示。评测只依赖 HTTP 协议，不限定 vLLM 版本，也不要求训练 rollout 的旧版本。命令行帮助已检查，尚未为这次测试实际启动 GPU 服务。

测 SFT / RL 后的模型时，将 `--model` 后的模型目录替换为已导出的 Hugging Face checkpoint，`--served-model-name` 可保持 `student`。原始分布式训练分片不能直接当 HF checkpoint 使用。已有 DataEvo terminal 协议训练的 checkpoint 如训练时使用了专用模板，服务也要加 `--chat-template DataEvo/config/terminal_chatml.jinja`，保持训练与评测一致；普通 Qwen instruct 模型使用其自带模板。

模型服务鉴权可通过环境变量 `STUDENT_API_KEY`（优先）或 `OPENAI_API_KEY` 提供。不需要鉴权的本地 vLLM 会使用占位 key。不要将 key 放入 URL、命令行参数或提交进仓库。

## 3. 预检与两套数据 smoke

在启动评测的 shell 安全设置 `E2B_API_KEY`。入口不自动读取其他项目 `.env`。选 Docker 时不需要 E2B key，但 `docker info` 必须能成功。

```bash
HARBOR_PY=/home/hrh/RSIBench-Data-full/.venv/bin/python

# 只检查依赖、任务、key 是否存在，以及 GET /models；不会创建环境。
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --model student --api-base http://127.0.0.1:8000/v1 \
  --environment e2b --preflight

# 真正创建环境，各跑一题；目录必须此前不存在。
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --model student --api-base http://127.0.0.1:8000/v1 \
  --environment e2b --n-tasks 1 --concurrency 1 \
  --output artifacts/agentic_eval/student_smoke_001
```

使用 Docker 时将 `--environment e2b` 替换为 `--environment docker`。预检不验证 E2B 额度，也不构建镜像、不安装 agent、不运行 verifier，首次真实 smoke 才能确认这些环节。默认任务时限来自各题 `task.toml`；入口固定每题一次尝试、零重试，避免因自动重跑改变分母或额外消耗。

## 4. 完整本地评测

```bash
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --benchmark both --all-tasks \
  --model student --api-base http://127.0.0.1:8000/v1 \
  --agent-profile dataevo --environment e2b \
  --max-turns 200 --max-tokens 4096 --concurrency 4 \
  --output artifacts/agentic_eval/student_full_001
```

并发数同时影响 GPU 服务压力和沙箱费用。默认并发为 1。DataEvo agent 每条命令默认 120 秒，观察截断上限为 24000 字符，可用 `--command-timeout` / `--max-observation-chars` 调整。该 agent 未实现历史摘要或自动压缩，上下文溢出会记录成错误；小上下文模型宜先降低步数做 smoke。`--context-window` 是 native Terminus 的输入 token 限制，不是 DataEvo 的历史裁剪开关。

native 模式例子：

```bash
# MODEL_API 必须是 mini-swe-agent 从 E2B 内能访问到的可信模型服务地址。
"$HARBOR_PY" DataEvo/scripts/evaluate_agentic_benchmarks.py \
  --model student --api-base "$MODEL_API" \
  --agent-profile native --environment e2b --n-tasks 1 \
  --output artifacts/agentic_eval/student_native_smoke_001
```

对 native SWE，`localhost`、loopback 和 unspecified 地址会预检失败。E2B 无法访问宿主私有局域网地址，Docker 容器也不能通过 `127.0.0.1` 访问宿主模型。入口不会自动开放端口或建立代理；需使用沙箱可达地址。宿主侧预检无法证明沙箱网络可达，仍需 smoke 确认。

## 5. 结果与比较

```text
<output>/
  request.json                  # 模型、agent、任务 ID/文件哈希、版本和参数
  preflight.json
  summary.json                  # 每套数据单独成绩，不混合两套分母
  scores.csv
  report.md
  terminal_bench_2/              # 或 swe_bench_verified/
    job_config.json
    mini_config.json
    harbor_result.json          # Harbor 正常返回时的原始结果
    summary.json
    trials.json                 # 单题 reward、异常、token、原始结果位置
    jobs/evaluation/...         # Harbor 环境/agent/verifier 日志、轨迹
```

任务 manifest 保存文件哈希和路径，不复制数据集；评测过程中应保持源任务目录不变。Harbor 自身也保存任务 checksum。DataEvo 的轨迹为 `agent/trajectory.json` 中的 messages/events，native 轨迹格式由 Harbor agent 提供。

完整且每题均有有效 reward 时才输出平均 reward，即这些二元任务的 pass@1 / resolved rate。verifier 真实判零保留为 0；环境失败、模型 API 错误、未完成、漏题、重复题、异常 reward 均使该套聚合 `score=null`，同时保留已有单题结果，真实运行返回非零退出码。两个数据集依次执行，其中一套失败仍尝试另一套。被强制终止时可查 Harbor 已落盘日志，但不承诺生成最终汇总。中断重试使用新输出目录，避免混入旧结果。

耗时和可获得的 token 使用量会保存；`cost_usd=null` 表示总费用未知。Harbor 模型费用估计另存为 `reported_model_cost_usd`，不包含 E2B/GPU 费用，未知模型价格有时会报告为 0，不代表免费。

比较 base、SFT、RL 时分别启动模型 API，使用相同任务 ID、agent profile、步数/token/时限等设置和不同输出目录。修改参数后不能仅凭差值认定模型提升。这里没有新增训练阶段，也没有把这两套评测题接为 RL 训练环境。

## 开发验证

```bash
"$HARBOR_PY" -m unittest \
  DataEvo.tests.test_benchmark_eval DataEvo.tests.test_api_harbor -v
```
