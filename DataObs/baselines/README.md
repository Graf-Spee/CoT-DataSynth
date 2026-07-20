# Baselines

此处记录使用的所有 baseline 方法。特别注意：现在我们做的测试中，蒸馏环节 **使用了** teacher correctness filter

ZeroShot: 使用 `run_all_eval.py` + plain + no thinking 评测。命令：

```bash
python DataObs/tools/run_all_eval.py --gpu-ids 2 --prompt-template-method plain --output-root /data/hrh/COT/baselines/ZeroShot
```

ZeroShot CoT: 使用 `run_all_eval.py` + zeroshot + thinking 评测。命令：

```bash
python DataObs/tools/run_all_eval.py --gpu-ids 2 --prompt-template-method zeroshot --enable-thinking
```

Vanilla Distill: 只对蒸馏结果加 Teacher Correctness Filter

Heuristics: Data Augmentation 方法，参考 The Quest for Efficient Reasoning: A Data-Centric Benchmark to CoT Distillation
- question rephrasing
- question augmentation
    - 此方法的实现也参考了 Common 7B Language Models Already Possess Strong Math Capabilities，因为 Quest 中的实现是错误的
- answer augmentation
- reverse thinking augmentation

Self Instruct

CoT-Self-Instruct