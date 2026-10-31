from __future__ import annotations

try:
    from .common import extract_instruction, extract_reward_ground_truth
except ImportError:
    from common import extract_instruction, extract_reward_ground_truth


def generate_code_tasks(seed_tasks: list[dict], args: object) -> list[dict]:
    """Prepare code tasks directly from the seed parquet for code_tests filtering."""
    generated_items: list[dict] = []
    for idx, task in enumerate(seed_tasks):
        question = extract_instruction(task)
        ground_truth = extract_reward_ground_truth(task)
        if not question or ground_truth is None:
            continue

        generated_items.append(
            {
                "source_index": idx,
                "question": question.strip(),
                "instruction": question.strip(),
                "target_answer": "",
                "reasoning_text": "",
                "code_prompt": question.strip(),
                "code_ground_truth": ground_truth,
                "generation_method": "cot_self_instruct_code",
                "augmentation_method": "cot_self_instruct",
                "task_type": "code",
            }
        )
    return generated_items
