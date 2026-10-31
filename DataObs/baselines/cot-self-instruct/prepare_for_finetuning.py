from __future__ import annotations


def _build_completion(item: dict) -> str:
    task_type = str(item.get("task_type", "") or "").strip()
    if task_type == "instruction":
        best_response = str(item.get("rip_best_response", "") or "").strip()
        if best_response:
            return f"{best_response}<|endoftext|>"
    if task_type == "code":
        best_response = str(item.get("code_best_response", "") or "").strip()
        if best_response:
            return f"{best_response}<|endoftext|>"

    reasoning_text = str(item.get("reasoning_text", "") or "").strip()
    target_answer = str(item.get("target_answer", "") or "").strip()
    raw_generation = str(item.get("cot_raw_generation", "") or "").strip()

    if reasoning_text and target_answer:
        return f"{reasoning_text}\n\nFinal Answer: \\boxed{{{target_answer}}}<|endoftext|>"
    if raw_generation:
        return f"{raw_generation}<|endoftext|>"
    if target_answer:
        return f"Final Answer: \\boxed{{{target_answer}}}<|endoftext|>"
    return "<|endoftext|>"


def prepare_for_finetuning(generated_items: list[dict], dataset: str) -> list[dict]:
    rows = []
    for item in generated_items:
        task_type = str(item.get("task_type", "reasoning") or "reasoning").strip()
        passed = (
            bool(item.get("consistency_passed", False))
            if task_type == "reasoning"
            else bool(item.get("rip_passed", False))
            if task_type == "instruction"
            else bool(item.get("code_passed", False))
        )
        if not passed:
            continue

        question = str(item.get("question", item.get("instruction", "")) or "").strip()
        answer = (
            str(item.get("target_answer", "") or "").strip()
            if task_type == "reasoning"
            else str(item.get("rip_best_response", "") or "").strip()
            if task_type == "instruction"
            else str(item.get("code_best_response", "") or "").strip()
        )
        if not question or not answer:
            continue

        gold_answer = (
            item.get("code_ground_truth")
            if task_type == "code"
            else answer
        )

        completion = _build_completion(item)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "gold_answer": gold_answer,
                "dataset": dataset,
                "data_source": "cot_self_instruct",
                "source_index": item.get("source_index"),
                "augmentation_method": "cot_self_instruct",
                "generation_method": item.get("generation_method", "cot_self_instruct_reasoning"),
                "teacher_filter_passed": True,
                "prompt": question,
                "completion": completion,
                "instruction": question,
                "instance_input": "",
                "instance_output": completion.removesuffix("<|endoftext|>"),
                "task_type": task_type,
                "target_answer": str(item.get("target_answer", "") or "").strip(),
                "consistency_majority_answer": item.get("consistency_majority_answer", ""),
                "consistency_vote_ratio": item.get("consistency_vote_ratio", 0.0),
                "rip_min_score": item.get("rip_min_score", 0),
                "rip_best_response_score": item.get("rip_best_response_score", 0),
                "code_best_response_score": item.get("code_best_response_score", 0.0),
            }
        )
    return rows
