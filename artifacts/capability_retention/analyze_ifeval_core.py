#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
IFEVAL_ROOT = ROOT / "ifeval_core"
MODELS = [
    ("base", "base_qwen3_4b"),
    ("sft", "easy_sft"),
    ("grpo", "easy_grpo"),
]
GSM8K_RESULTS = {
    "base": Path("/data/hrh/COT/evals/Qwen3-4B/gsm8k/zeroshot/generated/results.json"),
    "sft": Path("/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_easy_distill/eval/sft/generated/results.json"),
    "grpo": Path("/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_easy_distill/eval/grpo/generated/results.json"),
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def instruction_family(instruction_id: str) -> str:
    return instruction_id.split(":", 1)[0]


def load_ifeval_model(model_dir: str):
    result = load_json(IFEVAL_ROOT / model_dir / "generated" / "results.json")
    labeled = load_json(IFEVAL_ROOT / model_dir / "generated" / "responses_labeled.json")
    return result, labeled


def summarize_instruction_rates(labeled):
    by_inst = defaultdict(lambda: {"strict": [], "loose": []})
    by_family = defaultdict(lambda: {"strict": [], "loose": []})
    for row in labeled:
        ids = row["reference"]["instruction_id_list"]
        strict = row["strict_follow_instruction_list"]
        loose = row["loose_follow_instruction_list"]
        for inst_id, s, l in zip(ids, strict, loose):
            by_inst[inst_id]["strict"].append(bool(s))
            by_inst[inst_id]["loose"].append(bool(l))
            fam = instruction_family(inst_id)
            by_family[fam]["strict"].append(bool(s))
            by_family[fam]["loose"].append(bool(l))

    def finish(bucket):
        return {
            key: {
                "n": len(vals["strict"]),
                "strict": mean(vals["strict"]) * 100,
                "loose": mean(vals["loose"]) * 100,
            }
            for key, vals in sorted(bucket.items())
        }

    return finish(by_inst), finish(by_family)


def response_len(row):
    return len(str(row.get("scored_response", "")).split())


def main() -> int:
    model_data = {}
    summary = {
        "config": {
            "ifeval_root": str(IFEVAL_ROOT),
            "primary_metric": "Prompt-level-strict-accuracy",
            "models": {},
        },
        "ifeval": {},
        "gsm8k": {},
        "deltas": {},
        "instruction_family_rates": {},
        "largest_grpo_family_drops_vs_sft": [],
        "prompt_flips": {},
        "response_length_words": {},
    }

    for short, model_dir in MODELS:
        result, labeled = load_ifeval_model(model_dir)
        model_data[short] = {"result": result, "labeled": labeled}
        summary["config"]["models"][short] = result["model_path"]
        metrics = result["opencompass_metrics"]
        summary["ifeval"][short] = {
            "prompt_strict": metrics["Prompt-level-strict-accuracy"],
            "inst_strict": metrics["Inst-level-strict-accuracy"],
            "prompt_loose": metrics["Prompt-level-loose-accuracy"],
            "inst_loose": metrics["Inst-level-loose-accuracy"],
            "num_samples": result["num_samples"],
        }
        if GSM8K_RESULTS[short].exists():
            gsm = load_json(GSM8K_RESULTS[short])
            summary["gsm8k"][short] = {
                "accuracy": gsm["accuracy"] * 100,
                "num_correct": gsm["num_correct"],
                "num_samples": gsm["num_samples"],
            }

        _, by_family = summarize_instruction_rates(labeled)
        summary["instruction_family_rates"][short] = by_family
        summary["response_length_words"][short] = {
            "mean": mean(response_len(row) for row in labeled),
            "median_approx": sorted(response_len(row) for row in labeled)[len(labeled) // 2],
        }

    for metric_group in ("ifeval", "gsm8k"):
        base = summary[metric_group].get("base", {})
        sft = summary[metric_group].get("sft", {})
        grpo = summary[metric_group].get("grpo", {})
        summary["deltas"][metric_group] = {}
        for key, base_value in base.items():
            if isinstance(base_value, (int, float)) and key in sft and key in grpo:
                summary["deltas"][metric_group][key] = {
                    "sft_minus_base": sft[key] - base_value,
                    "grpo_minus_base": grpo[key] - base_value,
                    "grpo_minus_sft": grpo[key] - sft[key],
                }

    sft_family = summary["instruction_family_rates"]["sft"]
    grpo_family = summary["instruction_family_rates"]["grpo"]
    drops = []
    for fam, sft_stats in sft_family.items():
        if fam in grpo_family:
            drops.append({
                "family": fam,
                "n": sft_stats["n"],
                "strict_sft": sft_stats["strict"],
                "strict_grpo": grpo_family[fam]["strict"],
                "strict_grpo_minus_sft": grpo_family[fam]["strict"] - sft_stats["strict"],
                "loose_sft": sft_stats["loose"],
                "loose_grpo": grpo_family[fam]["loose"],
                "loose_grpo_minus_sft": grpo_family[fam]["loose"] - sft_stats["loose"],
            })
    summary["largest_grpo_family_drops_vs_sft"] = sorted(
        drops, key=lambda x: (x["strict_grpo_minus_sft"], x["n"])
    )

    labels = {
        short: [bool(row["is_strict_correct"]) for row in data["labeled"]]
        for short, data in model_data.items()
    }
    flip_counts = Counter()
    examples = defaultdict(list)
    for i in range(len(labels["base"])):
        pattern = f"{int(labels['base'][i])}{int(labels['sft'][i])}{int(labels['grpo'][i])}"
        flip_counts[pattern] += 1
        if len(examples[pattern]) < 5:
            row = model_data["grpo"]["labeled"][i]
            examples[pattern].append({
                "index": i,
                "instruction_id_list": row["reference"]["instruction_id_list"],
                "prompt": row["prompt"][:500],
                "base_correct": labels["base"][i],
                "sft_correct": labels["sft"][i],
                "grpo_correct": labels["grpo"][i],
                "grpo_response_prefix": str(row["scored_response"])[:500],
            })
    summary["prompt_flips"] = {
        "counts": dict(sorted(flip_counts.items())),
        "legend": "pattern order is base/sft/grpo strict prompt correctness; 1=correct, 0=wrong",
        "examples": dict(examples),
    }

    out_json = ROOT / "ifeval_core_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_md = ROOT / "ifeval_core_report.md"
    lines = [
        "# SFT+GRPO Capability Retention Probe",
        "",
        "## Setup",
        "",
        "- Base model: `/data/pretrain_models/Qwen3-4B`",
        "- SFT/GRPO chain: `exp5000_gsm8k_qwen3_difficulty_easy_distill`",
        "- IFEval data: `/data/open_datasets/IFEval/input_data.jsonl`",
        "- Decoding: `temperature=0`, `top_p=1`, `top_k=-1`, `max_response_len=4096`, `enable_thinking=false`",
        "- Backend: vLLM V0, OpenCompass-aligned IFEval scorer",
        "",
        "## Main Results",
        "",
        "| model | IFEval prompt strict | IFEval inst strict | IFEval prompt loose | GSM8K acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for short, _ in MODELS:
        ife = summary["ifeval"][short]
        gsm = summary["gsm8k"].get(short, {})
        lines.append(
            f"| {short} | {ife['prompt_strict']:.2f} | {ife['inst_strict']:.2f} | "
            f"{ife['prompt_loose']:.2f} | {gsm.get('accuracy', float('nan')):.2f} |"
        )
    lines.extend([
        "",
        "## Deltas",
        "",
        "| metric | SFT-Base | GRPO-Base | GRPO-SFT |",
        "|---|---:|---:|---:|",
    ])
    for metric in ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose"):
        d = summary["deltas"]["ifeval"][metric]
        lines.append(f"| IFEval {metric} | {d['sft_minus_base']:+.2f} | {d['grpo_minus_base']:+.2f} | {d['grpo_minus_sft']:+.2f} |")
    d = summary["deltas"]["gsm8k"]["accuracy"]
    lines.append(f"| GSM8K accuracy | {d['sft_minus_base']:+.2f} | {d['grpo_minus_base']:+.2f} | {d['grpo_minus_sft']:+.2f} |")
    lines.extend([
        "",
        "## Largest GRPO Drops Vs SFT By IFEval Family",
        "",
        "| family | n | SFT strict | GRPO strict | delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["largest_grpo_family_drops_vs_sft"][:8]:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['strict_sft']:.2f} | "
            f"{row['strict_grpo']:.2f} | {row['strict_grpo_minus_sft']:+.2f} |"
        )
    lines.extend([
        "",
        "## Prompt-Level Flip Counts",
        "",
        "`pattern` order is base/sft/grpo strict correctness; `1` means correct.",
        "",
        "| pattern | count |",
        "|---|---:|",
    ])
    for pattern, count in sorted(summary["prompt_flips"]["counts"].items()):
        lines.append(f"| {pattern} | {count} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "On this GSM8K easy-distill chain, SFT does not show IFEval degradation; it is slightly above the base model. The GRPO checkpoint loses IFEval relative to both Base and SFT, while also giving up the SFT GSM8K gain in the existing eval. This is evidence of a small but measurable instruction-following regression after GRPO for this run, not a broad proof for every SFT+GRPO configuration.",
        "",
        f"Full machine-readable analysis: `{out_json}`",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
