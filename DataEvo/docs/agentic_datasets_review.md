# Agentic RL 数据集候选与接入范围（待选择）

核对时间：2026-10-19。来源是本机 `/home/hrh/RSIBench-Data-main/benchmarks/*/spec.json` 与实际 task 目录，不是推测的支持列表。本轮先整理数据供选择，并补充只读预览入口；尚未启动新的合成、训练或 E2B 任务。

## 推荐先做哪两个

建议第一批选 **Terminal-Bench 2.0 + SWE-bench Verified**。前者检验终端状态、文件操作、多步命令和 verifier 奖励，最接近我们已有适配器；后者检验真实代码库检索、修改、运行测试和修复能力。Multilingual 与 Pro 作为后续扩展，避免第一次就同时排查多个语言构建环境与稀疏奖励问题。

**RSIBench 这些目录是评测集。SFT 合成和在线 RL 都需要独立的训练任务，不能把下面的评测题直接拿去训练。** RSIBench 本身主要做 Tinker LoRA SFT 后的 Harbor 评估，不提供现成的本地在线 GRPO 训练集。

## 本地已有的六套评测数据

| 数据集 | 本地题数 / 评测方式 | 能力与奖励 | RSIBench agent | 建议 |
|---|---|---|---|---|
| Terminal-Bench 2.0 | 89；每题 1 次 | 终端操作、调试、数据处理；任务 verifier | terminus-2 | 首批 |
| SWE-bench Verified | seed=23 抽样 100；每题 1 次 | 以 Python 项目为主的 issue 修复；测试通过率/解决率 | mini-swe-agent | 首批 |
| SWE-bench Multilingual | seed=23 抽样 100；每题 1 次 | 多语言仓库修复；语言相关测试 | mini-swe-agent | 第二批 |
| SWE-bench Pro | seed=23 抽样 100；每题 1 次 | 较复杂的真实仓库任务；issue-specific tests | mini-swe-agent | 第二批 |
| GPQA Diamond | 198 题中 seed=23 抽样 100；每题 1 次 | 科学选择题；写 answer.txt 后精确判分 | terminus-2 | 可选，主要检验推理 |
| AIME 2026 | 完整 30；每题 4 次取平均 | 数学整数答案；写 answer.txt 后精确判分 | terminus-2 | 可选，主要检验推理 |

GPQA/AIME 虽然通过终端提交文件、也能做工具 rollout，但不能仅凭这个就认为它们代表丰富的工具交互任务。优先验证 terminal/SWE 更符合本次目标。

上述数量是这个 RSIBench checkout 的评测范围，不是各上游数据集的总题数。三套 SWE 的 100 题也不是官方完整测试集。

| 数据集 | 本地数据目录（点击查看） | 上游 / 来源 |
|---|---|---|
| Terminal-Bench 2.0 | [full_89](/home/hrh/RSIBench-Data-main/benchmarks/terminal_bench_2/datasets/full_89) | Harbor registry：`terminal-bench/terminal-bench-2` |
| SWE-bench Verified | [seed23_random_100](/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_verified/datasets/seed23_random_100) | [SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)，test |
| SWE-bench Multilingual | [seed23_random_100](/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_multilingual/datasets/seed23_random_100) | [SWE-bench/SWE-bench_Multilingual](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual)，test |
| SWE-bench Pro | [seed23_random_100](/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_pro/datasets/seed23_random_100) | [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)，test |
| GPQA Diamond | [seed23_random_100](/home/hrh/RSIBench-Data-main/benchmarks/gpqa_diamond/datasets/seed23_random_100) | Harbor GPQA Diamond，具体子集见其 spec |
| AIME 2026 | [full_30_2026](/home/hrh/RSIBench-Data-main/benchmarks/aime/datasets/full_30_2026) | MathArena AIME 2026，具体版本见其 spec |

每个任务已经包含 `instruction.md`、`task.toml`、`environment/`、`tests/`。因此不需要先下载一批评测数据。预览只读取指令，不读取 solution 或测试答案。

### 可直接查看的真实题目

- Terminal-Bench：[sqlite-db-truncate](/home/hrh/RSIBench-Data-main/benchmarks/terminal_bench_2/datasets/full_89/sqlite-db-truncate/instruction.md)。从二进制截断损坏的 SQLite 数据库恢复记录，输出指定 JSON 文件。
- SWE Verified：[django__django-11265](/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_verified/datasets/seed23_random_100/django__django-11265/instruction.md)。修复 Django 对带 FilteredRelation 注解的 queryset 调用 exclude 时产生的错误。
- SWE Multilingual：[apache__druid-14092](/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_multilingual/datasets/seed23_random_100/apache__druid-14092/instruction.md)。处理非 200 响应后 leader cache 不刷新的问题。
- SWE Pro：[swepro_000](/home/hrh/RSIBench-Data-main/benchmarks/swe_bench_pro/datasets/seed23_random_100/swepro_000/instruction.md)。修改邀请注册流程，使合法 token 在缺少 email 时仍能完成注册。
- GPQA：[gpqa_000](/home/hrh/RSIBench-Data-main/benchmarks/gpqa_diamond/datasets/seed23_random_100/gpqa_000/instruction.md)。根据有机物结构和光谱条件选择 NMR 数据，写入答案字母。
- AIME：[aime2026_1](/home/hrh/RSIBench-Data-main/benchmarks/aime/datasets/full_30_2026/aime2026_1/instruction.md)。用三人不同出发时间和速度关系求路程，写入整数答案。

这些是评测示例，仅帮助选择任务类型；不要把题目、答案、解题轨迹或改写版本混入训练。

## 已补充的只读预览代码

[preview_agentic_datasets.py](../scripts/preview_agentic_datasets.py) 读取 RSIBench spec，定位本地 tasks，核对任务数量，并可输出题目片段。只依赖 Python 标准库，不启动 GPU、API 或沙箱。

```bash
cd /home/hrh/CoT-DataSynth
# 查看全部数据集及数量、RSIBench agent、评测次数、数据来源参考。
/home/hrh/verl-cu130/bin/python DataEvo/scripts/preview_agentic_datasets.py

# 查看某一套的前 3 道题及完整指令文件路径。
/home/hrh/verl-cu130/bin/python DataEvo/scripts/preview_agentic_datasets.py \
  --benchmark terminal_bench_2 --samples 3 --max-chars 1200
/home/hrh/verl-cu130/bin/python DataEvo/scripts/preview_agentic_datasets.py \
  --benchmark swe_bench_verified --samples 3 --max-chars 1200
```

可用 `--rsibench-root` 切换 checkout；`--benchmark` 使用上面表格对应的 key：`terminal_bench_2`、`swe_bench_verified`、`swe_bench_multilingual`、`swe_bench_pro`、`gpqa_diamond`、`aime`。数量不匹配会显示 `task_count_matches_spec=false`，不能当作完整评测结果。

## RL 训练数据从哪里来

建议分开管理 `train / sft_val / dev / probe / final`，固定题目身份、环境、verifier、版本和预算。基准评测目录保留作 final，dev/probe 从独立来源准备；SFT val 需要实际 assistant 交互轨迹，只有任务指令还不够。

| 目标能力 | 训练任务来源候选 | 接入前要做的事 |
|---|---|---|
| Terminal | 参考 [endless-terminals](https://github.com/kanishkg/endless-terminals)、[tmax](https://github.com/hamishivi/tmax) 的生成方法，基于公开资料创建新任务 | 固定初始环境、真实可执行 verifier；运行验证；与评测任务去重 |
| SWE | [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)、[SWE-smith](https://github.com/SWE-bench/SWE-smith) 或公开非基准 issue/PR | 检查实际可用的训练 split、仓库/commit/issue 重叠；转换为 Harbor tasks；验证依赖与测试 |
| 数学/科学工具推理 | 非 AIME/GPQA 的公开推理材料与新生成任务 | 隔离评测题，构建答案验证器与工具环境 |

这些是候选来源与生成框架，**不代表本机已经具备可直接训练、无重叠的任务集**。RSIBench 的 `seed_factories/README.md` 也将这些项目作为方法参考，并要求训练来源独立于基准。不能直接把任何 factory 附带的 benchmark 题当训练题。

## 选定之后补哪些运行代码

仓库已有 [prepare_agentic_session.py](../scripts/prepare_agentic_session.py) 和 [agentic runtime](agentic_runtime.md)：teacher 在真实环境合成轨迹 → 成功轨迹筛选 → SFT → 在线工具交互 GRPO → Harbor eval → probe/反馈。

选定数据集之后，需要将其补成可运行的 benchmark profile：

1. 固定 spec：数据路径与任务身份、verifier、agent 协议、每题次数、上下文、轮数、超时和环境资源。
2. 准备独立 train/dev/probe 与 SFT val 轨迹，接入现有 task manifest 和数据校验。
3. 打通当前 GPU 环境的在线 rollout/权重更新，再做真实单任务验收。
4. 用同一 profile 比较 base / SFT / RL，记录逐任务成功率、完整轨迹、曲线、token、环境耗时和故障；基础设施失败单独报告。
5. 两轮按 dev 选择后，才运行固定 final；Codex 根据训练与诊断反馈迭代合成代码。

评估建议同时保留两种清楚命名的口径：训练协议一致的 `dataevo-terminal` 用于比较本地 RL 收益；RSIBench 使用的 `terminus-2` / `mini-swe-agent` 用于对齐外部流程。不同 agent、上下文、轮数、模型与次数的分数不能直接混在同一列。AIME 必须补上每题 4 次平均：当前 `harbor_job.py` 固定 `n_attempts=1`，不能把现有单次结果标成 RSIBench avg@4。

## 当前实际验证状态

已用本机 Harbor **0.17.1** 的 `Task.is_valid_dir` 和 `Task` 类逐个加载这六套，共 **519 个任务**；均能通过结构解析，都是 Linux、没有 Harbor 多阶段 steps。这只证明数据格式可被该版本加载，尚未拉取镜像、构建环境或运行测试。

训练环境 `/home/hrh/verl-cu130` 当前是 **vLLM 0.29.0**，而已有在线 agentic adapter 的异步服务依赖旧 **0.8.4–0.8.5** API；当前 preflight 会拒绝启动。因此“数据已经在本地、目录校验通过”和“完整 agentic RL 已在现有 GPU 上跑通”是两个不同状态。接下来的实现需要解决 rollout 服务适配，不能只换数据集名字就宣称已完成。

SWE 的镜像构建、语言工具链、测试耗时，以及 Terminal 的个别任务资源需求，也需逐任务检查。结构校验不代表当前 4B 模型能取得正奖励；首轮宜先做少量任务的真实环境/奖励/权重更新验收，再扩大规模。
