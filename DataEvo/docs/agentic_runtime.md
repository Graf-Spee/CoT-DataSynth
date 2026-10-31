# Terminal agent 合成、SFT 与工具交互 RL

这次补上的是实际训练适配器，不是用 Harbor 的训练后评测代替 RL rollout。训练入口继续调用仓库的 `DataObs/scripts/experiment_pipeline_vllm.py` → `grpo_from_sft.sh` → verl GRPO。

## 已实现的训练路径

```text
固定 train task + 当前策略权重
  → async vLLM 输出 JSON command
  → Harbor Trial 启动独立 E2B/Docker 环境
  → 执行 command，得到 stdout/stderr/exit code
  → 将 observation 加入历史，当前策略继续生成
  → final / turn limit / context limit
  → Harbor 上传 tests、运行 verifier、保存结果、删除环境
  → reward 放入最后一个 assistant token
  → verl 按任务分组计算 GRPO advantage、更新策略
  → 下一批使用更新后的策略权重
```

每条采样拥有一个独立环境；同一条采样内文件状态持续存在。每个 command 是一个新 shell，工作目录和 shell 变量不会自动继承，因此提示要求使用显式目录。命令只送到 Harbor 环境，不在训练主机执行。

使用 `{"command":"..."}` / `{"final":"..."}` 两种文本动作。观察以 `TOOL_RESULT` user 消息注入，不是模型生成 token。`loss_mask` 对 prompt、观察、assistant header、padding 和 EOS 后模板分隔符为零，SFT 和 RL 使用相同 mask 构造函数。不能只返回 `response_mask`，因为当前 trainer 会重建它；实际 actor/GRPO 使用的是完整 `loss_mask`。

Qwen3 默认模板会根据后续 user 消息重写历史 thinking 段，因此内置 profile 固定使用 `config/terminal_chatml.jinja`，在 SFT dataset 初始化时安装并随 checkpoint 保存。RL 检查 checkpoint 的模板，评估显式使用相同模板；不支持该模板的 tokenizer 会被拒绝。目前支持具有 ChatML special tokens 的模型、Linux 单阶段终端任务。浏览器、GUI、MCP 和 Harbor multi-step 任务不是本次已实现范围。

奖励只来自 Harbor verifier 的有限 `[0,1]` 值，不读取模型自报分数。无效/截断动作的训练奖励为零；终止前已经完成任务、到达轮数上限的有效轨迹可获得 verifier 分数。正式评估报告原始任务成功率，不加这个训练用格式惩罚。环境/API/verifier 故障使本轮失败，不作为零分加入 GRPO。完整轨迹不做静默 token 截断，观察长度有明确上限。

## 与你给出的 RSIBench 流程的区别

| 环节 | 你给出的流程 | 当前实现 |
| --- | --- | --- |
| 研究控制 | Codex 调用 `run_attempt.sh` 后继续会话 | 每轮 `codex exec` 读磁盘历史，返回结构化候选；控制器提交并等待固定 worker |
| 优化空间 | 候选数据与训练配置 | Codex 修改合成代码、提示和筛选；SFT/RL 超参、任务、verifier 固定，便于比较数据效果 |
| 训练 | Tinker LoRA SFT | 本地仓库 SFT → dev eval → GRPO → dev eval；没有接入 Tinker |
| 权重 | Tinker sampler checkpoint | SFT/HF/verl checkpoint，经现有 merger 导出；agentic RL 使用全参数权重 |
| 交互与评测 | Harbor 调度 E2B 评测 | Harbor/E2B 同时用于 teacher 交互、在线 RL rollout 和独立 dev/probe/final 评测 |
| Agent 协议 | 通常是 Terminus 的终端协议 | 内置 `dataevo-terminal`，合成/训练/评测一致；仍保留独立 `terminus-2` 评测配置 |
| 反馈 | 分数、轨迹、费用、耗时 | 另外含 SFT→RL 增益、训练曲线、base/SFT/RL probe、终止原因和环境用时 |
| 最终提交 | `final_submit.sh`、审计、可选正式重评 | `select` / `final-test`；有冻结 hash、来源核验和一次性最终测试，没有独立完整的提交包审计器 |
| 预算与预检 | benchmark spec、费用估算、预算状态 | session 配置、固定 task manifest、token/episode/时间/GPU 检查；没有覆盖所有服务的精确美元预估器 |

两者都实现“候选 → 固定执行入口 → 反馈 → 下一轮研究”。这里额外优化的是数据对 **SFT 后再 RL 的表现** 的影响；不是原样复制 Tinker 的 SFT-only 实验。

## 准备与启动

需要四类输入：训练 Harbor tasks、dev tasks、probe tasks，以及独立的 SFT validation trajectories（parquet/jsonl 的 `messages`）。正式测试 tasks 可选。task 目录需符合 Harbor 格式，包含 instruction、environment 和 tests；SFT val 不能只放任务指令，必须包含 assistant 动作。train/dev/probe/final 的任务指令会检查重叠；建议使用数据集自身的语义隔离划分，精确匹配不能识别改写后的近重复。

```bash
/path/to/training/python DataEvo/scripts/prepare_agentic_session.py \
  --train-tasks /data/terminal/train \
  --dev-tasks /data/terminal/dev \
  --probe-tasks /data/terminal/probe \
  --final-tasks /data/terminal/test \
  --sft-val /data/terminal/sft_val.parquet \
  --base-model /home/hrh/models/Qwen3-4B \
  --python /path/to/training/python \
  --harbor-python /home/hrh/RSIBench-Data-full/.venv/bin/python \
  --teacher-model YOUR_TEACHER_API_MODEL \
  --environment e2b --gpu-ids 0,1 --gpu-count 2 \
  --max-episodes 1000 --output /data/dataevo/terminal-inputs-001
```

准备命令只写数据/config，不启动 API/GPU。输出 `session.json`、`tasks.json`、RL parquet 和规范化 SFT val。请按任务数、显存调整固定训练 batch、步数、上下文和预算；示例不是已实测的显存配置。默认每任务采样 4 条，GRPO 必须保留多样本组。

`training.agentic_rl` 提供环境、并发、轮数、每轮 token、观察长度、命令/episode/清理超时、`max_episodes` 和 verifier 的 reward key。`max_episodes` 是**每个 attempt 合成与 RL（含 RL validation）合计**的硬预留上限，失败启动也占一次，resume 不重置；独立 dev/probe/final Job 的额度由各自固定任务数及 stage/session 超时限制。`episode_timeout_s` 包含初始化和验证等端到端等待；应大于命令和环境启动所需时间。

SFT 可以通过固定 `sft-arg` 设置 LoRA，随后由原 pipeline 合并再进入 RL。仓库旧 async vLLM server 未实现动态 PEFT 更新，所以内置 RL 固定 `lora_rank=0`，不会把未更新的 LoRA/base 当作在线策略。它与 Tinker LoRA 训练的显存和费用模型不同。

设置 teacher 的 `SYNTHESIS_API_BASE`、`SYNTHESIS_API_KEY`，E2B 的 `E2B_API_KEY`，并按原文档完成 Codex 认证。`dataevo-terminal` 的模型调用在 Harbor 主机执行，可以访问本机 vLLM，无需给 E2B 暴露模型公网端口；旧 `terminus-2` 配置仍要求 `STUDENT_PUBLIC_BASE_URL`。

```bash
/path/to/training/python DataEvo/scripts/run_session.py init \
  --session /data/dataevo/terminal-session-001 \
  --config /data/dataevo/terminal-inputs-001/session.json
/path/to/training/python DataEvo/scripts/run_session.py run \
  --session /data/dataevo/terminal-session-001
```

初始化自动选择 `workspace_template/agentic_attempt`。默认 teacher 在训练任务中实际执行命令并验证，后处理保留成功且不重复的轨迹。Codex 可以改 `generation.guidance`、采样和完整合成/筛选代码；`agentic_synthesis.py` 提供可复用的基线。候选进入 SFT 前会核对训练 allowlist、轨迹证据中的 messages 和真实 Harbor result 中的 verifier reward。任意插件与 runner 仍不是操作系统级权限隔离；这些检查不是防恶意代码的安全边界。

## 环境检查和本次验证边界

Harbor 适配器针对实际安装的 **0.17.1** API 编写，使用 `Trial.create` / `Trial.run`，由 Harbor 负责环境、权限、verifier 和删除。Harbor 使用单独 Python 3.12，不要求装入训练 Python。`init` 还通过真实 Harbor `Task` 类校验所有固定任务的配置、environment、instruction 和 verifier 文件，不创建环境；结构错误提前失败。

当前仓库 async vLLM 服务仍依赖 **vLLM 0.8.4–0.8.5**，与 `setup.py` 的旧版本约束一致。2026-10-19 检查发现 `/home/hrh/verl-cu130` 实际是 **vLLM 0.29.0**，旧 `serving_chat` 模块已不存在。因此当前环境不能直接启动这条异步训练路径。`init` 会先检查版本、真实模块导入和 tokenizer，对不兼容环境提前失败；不会先花 API 费用再失败。使用隔离的兼容训练环境，或者另做一次仓库 async server 到 vLLM 0.29 的迁移。此次未安装/降级任何现有环境。

可单独运行只读预检：

```bash
/path/to/training/python -m DataEvo.runtime.agentic_preflight \
  --kind training --model /home/hrh/models/Qwen3-4B
/home/hrh/RSIBench-Data-full/.venv/bin/python -m DataEvo.runtime.agentic_preflight --kind harbor
/home/hrh/verl-cu130/bin/python -m unittest discover -s DataEvo/tests -v
```

CPU 测试包括真实 Qwen3 tokenizer、SFT dataset、loss mask、reward manager、分组重复顺序、并发上限、Hydra 配置、task split 校验、工具观察驱动下一次生成，以及真实子进程 JSONL 管道里的状态/验证/取消清理/预算检查。管道测试使用真实 Harbor `BridgeAgent` 类，环境与 verifier 明确为内存测试替身；不是云环境实跑。原闭环的合成子进程、恢复和反馈测试也保留。

未启动真实 GPU 优化、E2B 收费任务或 Codex 研究回合。当前 Docker daemon 也返回权限拒绝，因此没有宣称 Docker/E2B 端到端通过。完整上线验收仍需在兼容环境做至少一个真实 task 的 SFT→RL→eval，并检查优化前后权重/策略确实变化。

## 可审阅的实现与产物

主要代码：`agentic_rollout.py`（在线 rollout/GRPO tensor）、`harbor_bridge.py`（Harbor Trial）、`agentic_env.py`（管道/预算/清理）、`agentic_protocol.py`（动作与 mask）、`agentic_sft.py`（固定模板/多轮 SFT）、`harbor_terminal_agent.py`（相同协议评测）、`agentic_config.py`（固定配置/任务边界）、`agentic_synthesis.py`（teacher 实际交互）。verl 只新增 scheduler 的可选完整 episode hook；默认数学 rollout 不改变。

`runner/attempts/<id>/agentic/{synthesis,rl}/<episode>/` 保存 `trajectory.json`、`episode.json`、桥接日志、Harbor 原始 trial/verifier 结果。轨迹含动作、观察、任务、终止原因、verifier reward 和训练 reward。`feedback.json.agentic` 汇总 train/val 奖励、失败类型、环境用时；SQLite 的 `harbor_episode` 记录每个环境 episode 的用量引用。美元成本没有供应商账单就保留 unknown，不把环境墙钟时间冒充账单。

取消时先 SIGTERM，让 Harbor 的 finally 删除环境；超出清理时限才强制结束并记录清理异常。机器掉电或 SIGKILL 不能保证云资源即时删除，需按轨迹中的 Harbor trial 信息核查提供商资源。
