# Codex 数据实验闭环：实现与运行

入口是 `DataEvo/scripts/run_session.py`。本实现借鉴 RSIBench 的固定入口、候选提交、训练评估和反馈接口；训练直接调用本仓库的 `DataObs/scripts/experiment_pipeline_vllm.py`，没有引入 Tinker 训练依赖。

## 已实现的链路

```text
SQLite 历史 + feedback.json + 轨迹/曲线
  → codex exec：提出假设，修改合成程序/模板/筛选程序
  → validate + submit：冻结 attempt，领取 GPU，注入 API 环境
  → generate → postprocess → validate_data
  → 本仓库 SFT → dev 评估 → 本仓库 GRPO → dev 评估
  → base/SFT/RL probe → 曲线、轨迹、用量、反馈落盘
  → 下一次 codex exec → 新 attempt，直到预算耗尽或 Codex 停止
```

每个候选从同一个 base 开始 SFT，再从该候选的 SFT checkpoint 开始 RL。不同候选共享固定的训练预算、RL 数据和 dev 评估规则，优化目标为 `rl_dev_score`。父候选 ID 表示研究沿袭，不表示继续训练父模型。

目前自带可运行的 GSM8K API 蒸馏模板，含已知答案正确性、截断和重复筛选。Codex 可以完全重写合成和后处理 Python，也可以调用现有 distill 方法；diversity、难度筛选是可编写的策略，不是已经自动实现了所有算法。

Harbor/E2B 评估支持 `terminus-2` 和 `dataevo-terminal`。新增的 terminal profile 已实现有状态工具交互 RL：当前训练策略生成动作，Harbor 环境执行，verifier 评分，再交给现有 GRPO 更新。合成、SFT、RL 和评估共享协议与 token 模板。代码和 CPU 验证已完成，真实 GPU/E2B 联调尚未完成；本机 vLLM 0.29 与旧异步接口不兼容，启动前检查会阻止消耗资源。详见 [agentic 实现与流程对比](agentic_runtime.md)。GSM8K 仍用原 vLLM 评估。

## Codex 如何接入

`runtime/codex_harness.py` 启动官方非交互 CLI，核心命令为：

```bash
codex exec --json --skip-git-repo-check \
  --sandbox workspace-write \
  --output-schema /SESSION/runner/codex/decision-0000/schema.json \
  --output-last-message /SESSION/runner/codex/decision-0000/decision.json \
  -c 'approval_policy="never"' -C /SESSION/workspace -
```

控制器把本轮研究指令写入 stdin。Codex 读取历史、反馈和轨迹，在 `workspace/attempts/<id>/` 写程序，最后返回：

```json
{"action":"attempt","attempt_id":"a0001","rationale":"根据上轮错误类型调整合成与筛选"}
```

也可以返回 `action=stop`、空 ID 和停止理由。每轮为新的 `codex exec`，记忆来自磁盘历史，并不依赖上一段聊天上下文。事件日志、thread ID、最终决策、token 用量都会保存；不存在用规则控制器假装 Codex 做决策的步骤。

运行机需已安装并完成 Codex 登录，或已配置可用的 provider/profile。先用 `codex login status` 检查。session 配置可以填写 `codex.executable`、`codex.profile`、`codex.model`、`codex.timeout_s`。未指定模型时使用本机 Codex 配置，不写死模型名。

Codex 的认证与生成数据的 teacher API 是两套独立配置。`SYNTHESIS_API_KEY` 供蒸馏脚本使用，不会替代 Codex 登录。

接口依据 [官方 Codex 非交互执行文档](https://developers.openai.com/codex/noninteractive)，并核对了本机 CLI 参数。

## GSM8K 启动

使用已有训练环境，要求 pandas、pyarrow、PyTorch、verl/vLLM 等与当前仓库训练流水线兼容。先从原始 `question/answer` train 划分独立的 SFT val、dev、probe；官方 test 仅留给最终测试。以下路径是示例，替换模型和 API model ID 后执行：

```bash
cd /home/hrh/CoT-DataSynth
/home/hrh/verl-cu130/bin/python DataEvo/scripts/prepare_gsm8k_session.py \
  --train /home/hrh/open_datasets/GSM8K/main/train-00000-of-00001.parquet \
  --test /home/hrh/open_datasets/GSM8K/main/test-00000-of-00001.parquet \
  --output /data/hrh/dataevo/gsm8k-split-001 \
  --base-model /absolute/path/to/student-base \
  --python /home/hrh/verl-cu130/bin/python \
  --teacher-model YOUR_TEACHER_API_MODEL \
  --gpu-ids 1,2,3 --gpu-count 1
```

该命令生成 parquet、`session.json` 和 split manifest，不启动 API 或训练。默认从原 train 留出三份各 256 行，剩余数据作为合成 seed 和固定 RL train，RL val 使用 dev。需要新输出目录。

编辑生成的 `session.json`，确定 GPU 数量、训练步数、上下文长度和预算。默认每次 SFT/RL 各 100 steps、最多 3 个 attempt；SFT epochs 设为 100 是迭代上限，实际由 100 steps 截止。不同模型的显存需求不同，示例配置不是已完成实跑的显存配置。SFT 数据至少要满足现有 dataloader 的 batch 要求（默认全局 batch 8）。LoRA 可通过固定 `training.pipeline_args.sft-arg` 加入 `model.lora_rank=32`、`model.lora_alpha=64`，默认沿用本仓库全参数 SFT。

设置 `SYNTHESIS_API_BASE`（OpenAI-compatible `/v1` URL）和 `SYNTHESIS_API_KEY` 环境变量，然后：

```bash
codex login status
/home/hrh/verl-cu130/bin/python DataEvo/scripts/run_session.py init \
  --session /data/hrh/dataevo/session-001 \
  --config /data/hrh/dataevo/gsm8k-split-001/session.json

/home/hrh/verl-cu130/bin/python DataEvo/scripts/run_session.py run \
  --session /data/hrh/dataevo/session-001
```

`init` 不调用模型，复制一个可编辑的 `a0000` 模板并记录固定配置、数据/source hash、依赖清单和 base 文件元信息。`run` 才开始调用 Codex 和实验。建议在 tmux 或已有任务管理器中运行控制器；worker 独立于 Codex/控制器进程，控制器中断后再次 `run` 会等待已提交任务，再继续决策。

初始化后不要编辑固定配置、训练源代码和数据文件；改这些内容应新建 session。大权重只记录大小和 mtime，配置等小文件有 SHA256；没有为所有模型权重计算内容 hash。代码保留快照，但当前 worker 执行工作树代码并在启动时核对 hash，因此执行期间也应保持工作树稳定。

## 固定接口与人工查看

所有命令都带 `--session /data/hrh/dataevo/session-001`。除 `init/run/status/history` 外，还需 `--attempt a0000`。

| 命令 | 行为 |
| --- | --- |
| `validate` | 校验 recipe、程序路径和研究者可修改范围，不训练 |
| `submit` | 快照冻结并启动独立 worker，相同 ID 幂等 |
| `wait --timeout 3600` | 等待该任务，超时不会取消任务 |
| `status` / `history` | 查看 SQLite 中任务、阶段和用量 |
| `feedback` | 返回该候选的反馈 JSON |
| `resume` | 仅恢复 FAILED 候选，跳过已完成且可核验的阶段 |
| `probe --checkpoint sft` | 对已完成候选的 base/sft/rl 做额外 probe |
| `select` | 选定一个已完成候选；自动 loop 默认选 RL dev 最高者 |
| `final-test` | 对选中的 RL 模型做一次最终测试，关闭该 session 后续优化 |
| `cancel` | 检查 worker 身份后终止任务 |

自动 loop 遇到失败会把失败历史反馈给下次 Codex 决策。若想保留同一个冻结候选，使用 `resume`；改代码请使用新 ID。恢复粒度为阶段，失败在某个训练阶段中间不承诺从最后一个 optimizer step 自动续训。worker 被强制 SIGKILL 后，需要先 `wait` 检测失联并标记 FAILED；本实现不是系统级进程监管器。

vLLM 的额外 probe 可以传 `--probe-spec sampling.json`，允许 `temperature/top-p/num-samples/max-response-len/max-samples/seed`。正式 dev 和 final-test 采样不随候选变化。额外 probe 保存在 `extra/`；自动循环的每轮 base/SFT/RL probe 固定，base 结果在同一 session 缓存。

研究程序接口：

```text
synthesize.py --context CONTEXT_JSON --output CANDIDATES_JSONL
postprocess.py --context CONTEXT_JSON --input CANDIDATES --output SELECTED_JSONL
```

程序从冻结目录执行，额外输出写入 `context.artifact_dir`。调用 `DataEvo.runtime.api.complete(context, messages, **sampling)` 可以使用计量的 teacher API；源数据路径、GPU lease 和生成参数在 context 中。研究者不能通过 recipe 覆盖训练预算；`generation.backend=pipeline` 只接受合成相关参数。

## Harbor / E2B 接入

将 `config/evaluation.harbor.example.json` 的内容作为 session 的 `evaluation` 对象，填写三套互不相交的任务目录 `dataset/probe_dataset/final_dataset`。也接受 Harbor registry 的 `name@version`；可复现实验优先用固定的本地任务副本。Harbor 任务需有环境、instruction 和 verifier，而不是只含问题答案的 parquet。

训练继续用 `python` 指向的本仓库环境；可另设顶层 `harbor_python` 为安装 Harbor/E2B 的解释器。本机验证使用 Harbor 0.17.1。需要配置：

- `E2B_API_KEY`：Harbor 创建评估环境使用。
- `STUDENT_PUBLIC_BASE_URL`：`terminus-2` 使用的 E2B 可达模型地址，必须包含 `/v1`；内置 `dataevo-terminal` 在主机调用模型，无需公网地址。
- `STUDENT_API_KEY`：本地 vLLM 服务及 Harbor 请求共同使用的 key。
- `evaluation.port`：本地服务端口，默认 18080。
- `evaluation.context_window/max_tokens`：与模型支持的上下文和输出长度一致。

runner 在租用的 GPU 上启动训练后模型的 vLLM OpenAI 服务，检查本地 readiness，再启动 Harbor Job；评估结束关闭服务。SFT LoRA 先合并，RL actor 使用仓库已有 merger 导出。`terminus-2` 需要用户预先配置将公网 `/v1` 转发到本机端口的隧道/网关；这里没有自动创建公网域名或隧道。内置 `dataevo-terminal` 不需要该转发。

RL LoRA 的合并基底是本轮 SFT 模型（SFT 本身用 LoRA 时为其 merged model），不是最初的 base；vLLM 和 Harbor 两条评估路径都遵守这一点。

每次评估保存 `request.json`、`harbor_result.json`、标准 `summary.json` 和 `jobs/` 轨迹。必须所有选定任务有 reward 且无未完成/异常任务，才接受分数；基础设施失败不会被悄悄从分母去掉。当前归一化要求 reward 为 `[0,1]`。

## agentic 数据与 RL 扩展

GSM8K 用 `question/answer`。非 GSM8K 候选使用 `messages`，并要求 `task_id`、`execution_ref`、`verified=true`。`execution_ref` 必须指向实际存在的运行证据文件，tool 结果必须对应此前的 tool call ID。该校验只能检查格式和证据存在，不能证明任意自定义 verifier 的可信性，也不等于自动执行环境重放。

多轮 SFT 自动启用现有 `data.multiturn.enable=true`，训练 assistant token；SFT val 也需是相同消息格式，且模型 chat template 必须支持相应工具消息。

推荐使用内置 `training.agentic_rl` profile，通过 `scripts/prepare_agentic_session.py` 生成。它包含真实 Harbor Trial 桥接、有状态 rollout、verifier reward、assistant-only loss mask 和轨迹反馈，不需要自写 `rl_command`。使用方法和当前环境限制见 [agentic_runtime.md](agentic_runtime.md)。

其他工具协议仍保留自定义适配器入口；例如浏览器或非 ChatML 模型需要单独适配，此时可以配置：

```json
{
  "training": {
    "agentic_rl_verified": true,
    "rl_command": ["/absolute/python", "/absolute/train_agentic.py",
                   "--checkpoint", "{checkpoint}", "--output", "{output}",
                   "--gpu-ids", "{gpu_ids}", "--context", "{context}"],
    "pipeline_args": {"sft-arg": ["data.max_length=8192"]}
  }
}
```

上述 command 仅为自定义入口示例；内置 terminal profile 不走该分支。自定义训练须导出 `{output}/global_step_N/actor/huggingface/`（或符合当前 merger 的 actor/LoRA 格式）。`agentic_rl_verified` 是操作者声明，不代替测试；不要与内置 `agentic_rl` 同时配置。

## 产物、成本与边界

```text
SESSION/
  workspace/attempts/<id>/          可编辑 recipe、假设、脚本、模板
  runner/session.sqlite             任务、阶段、用量
  runner/config.json                固定实验配置
  runner/source_snapshot/           初始化时固定代码副本
  runner/snapshots/<id>/            已提交候选代码
  runner/manifests/<id>.json        代码与固定输入身份
  runner/codex/decision-NNNN/       prompt/context/events/decision/result
  runner/attempts/<id>/
    dataset/                        训练 parquet、统计/hash
    experiment/sft/                 SFT checkpoint
    experiment/grpo/                RL checkpoint
    eval/{sft,rl}/                  dev 指标、生成结果或 Harbor 轨迹
    probes/{base,sft,rl}/            固定诊断集表现与 rollout
    metrics/                        实际日志提取的 CSV/JSONL/SVG
    logs/                           阶段 stdout/stderr
    feedback.json                   setting → 结果 → 下一轮上下文
  runner/leaderboard.json
  runner/selected.json
```

GPU pool 由共享目录下的文件锁协调，并检查显存占用；不会终止已有外部任务。当前一个 session 同时只运行一个候选，候选期间持有 GPU lease。API 从环境注入一个固定 teacher 配置，不包含多供应商账号池/自动开户功能。

teacher helper 对 token 先预留后按实际 usage 结算，未知 usage 保留预留。可设置 teacher 的 `input_usd_per_million/output_usd_per_million` 和资源的 `gpu_usd_per_hour` 来计价。GPU 成本为命令执行时长估算；Codex、E2B 或其他未定价部分保留 unknown，已知金额不是完整总账。Harbor 中本地 student 的 per-token 价格为零，仅表示不重复计算本地推理 API 费用，不代表 E2B 免费。

attempt 数量、阶段超时、session 总时限和 helper 的 synthesis token 预算有执行检查。Python 插件仍可访问当前进程权限范围；文件冻结校验、Codex workspace sandbox 和 helper 计量不是完整的文件/网络隔离，直接绕过 helper 的外部请求不受 token 配额约束。

## 验证

```bash
/home/hrh/verl-cu130/bin/python -m unittest discover -s DataEvo/tests -v
```

测试包含真实合成子进程、独立 worker、失败恢复、两轮反馈、GPU 锁、输入冻结、评估缺失拒绝、Codex CLI 协议、HTTP API 计量和 Harbor 汇总。训练/模型推理使用明确的测试替身；Harbor JobConfig 另在实际 0.17.1 环境验证。尚未启动真实 GPU 训练、真实 Codex 决策调用或 E2B 付费任务，因此不宣称模型效果或完整实跑已验证。
