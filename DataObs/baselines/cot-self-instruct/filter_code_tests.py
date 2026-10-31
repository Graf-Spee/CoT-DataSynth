from __future__ import annotations

try:
    from .common import batched_generate_full, make_teacher
except ImportError:
    from common import batched_generate_full, make_teacher

from verl.utils.reward_score import mbpp as reward_mod


def filter_code_tests(generated_items: list[dict], args: object) -> list[dict]:
    """Generate code answers and keep samples that pass code_tests scoring."""
    if not generated_items:
        return []

    teacher = make_teacher(args)
    prompts = [str(item.get("code_prompt", item.get("question", ""))).strip() for item in generated_items]
    num_samples = int(getattr(args, "code_num_samples", 1))
    batch_size = int(getattr(args, "gen_batch_size", 1))
    threshold = float(getattr(args, "code_correct_threshold", 1.0))

    batch_outputs = batched_generate_full(
        teacher,
        prompts,
        n=num_samples,
        max_new_tokens=int(getattr(args, "code_max_new_tokens", 2048)),
        do_sample=bool(getattr(args, "code_do_sample", True)),
        temperature=float(getattr(args, "code_temperature", 0.7)),
        top_p=float(getattr(args, "code_top_p", 0.95)),
        gen_batch_size=batch_size,
        desc="Generate code solutions",
        enable_thinking=False,
        use_chat_template=True,
    )

    filtered_items: list[dict] = []
    for item, outputs in zip(generated_items, batch_outputs):
        responses = [str(output.get("text", "") or "").strip() for output in outputs]
        scores = [
            float(reward_mod.compute_score(response, item.get("code_ground_truth")))
            for response in responses
        ]
        best_index = max(range(len(scores)), key=lambda idx: scores[idx]) if scores else -1
        best_score = scores[best_index] if best_index >= 0 else 0.0
        best_response = responses[best_index] if best_index >= 0 else ""
        code_passed = bool(best_response and best_score >= threshold)
        filtered_items.append(
            {
                **item,
                "code_candidate_responses": responses,
                "code_candidate_metadata": outputs,
                "code_scores": scores,
                "code_best_response_index": best_index,
                "code_best_response_score": best_score,
                "code_best_response": best_response,
                "code_passed": code_passed,
                "teacher_filter_passed": code_passed,
            }
        )
    return filtered_items
