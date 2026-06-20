# Experiment Recipes

这些目录按 observation 实验编号组织。每个目录里的 `commands.sh` 默认跑 Qwen2.5 版本；同目录下还有三个显式 family 入口：

- `commands_qwen2_5.sh`
- `commands_qwen3.sh`
- `commands_qwen3_5.sh`

默认所有脚本都带 `--dry-run`，只打印命令和写 manifest，不启动训练/蒸馏。确认无误后这样真跑：

```bash
DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands_qwen2_5.sh
DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands_qwen3.sh
DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands_qwen3_5.sh
```

常用覆盖参数：

```bash
PYTHON_BIN=/home/hrh/anaconda3/envs/verl-cot/bin/python
DATASET=gsm8k
OUTPUT_DIR=/data/hrh/COT/experiments
GPU_IDS=0
BIG_GPU_IDS=0,1
XL_GPU_IDS=0,1,2,3
```

相关文档：

- `DataObs/docs/runbook.md`: 每组实验要回答的问题、运行顺序和完整参数说明。
- `DataObs/docs/review.md`: 当前 `DataObs` 每个文件的用途、冗余点和重构建议。

目录说明：

- `exp1000_smoke`: 蒸馏链路 smoke test。
- `exp1100_teacher_size`: Qwen2.5 / Qwen3 / Qwen3.5 同系列 teacher size curve。
- `exp1200_teacher_type`: reasoning / cross-family / task-specialist teacher 对照。
- `exp2000_data_quality`: 不同质量蒸馏数据和 SFT/RL 表现。
- `exp3000_seed_reasoning`: prompt-only teacher reasoning vs human reasoning vs answer-only。
- `exp3100_sft_rl_prompt_overlap`: SFT train prompt 和 RL train prompt 相同/不相同的对照。
- `exp4000_sft_rl_correlation`: 固定 SFT checkpoint 后启动 RL，比较 SFT/RL 相关性。
- `exp5000_difficulty`: easy / medium / hard 数据桶。
- `exp6000_diversity`: low / mid / high diversity 数据桶。
- `exp7000_size`: 不同数据规模。
- `exp8000_temperature_sampling`: temperature / num_samples / filter sweep。
