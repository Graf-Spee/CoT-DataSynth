from __future__ import annotations

import re

try:
    from .common import batched_generate_full, make_teacher, normalize_text
    from .templates import RIP_RESPONSE_TEMPLATE, RIP_SCORE_TEMPLATE
except ImportError:
    from common import batched_generate_full, make_teacher, normalize_text
    from templates import RIP_RESPONSE_TEMPLATE, RIP_SCORE_TEMPLATE


SCORE_RE = re.compile(r"score\s*[:：]\s*([1-5])", re.IGNORECASE)


def _parse_score(text: str) -> int:
    match = SCORE_RE.search(str(text or ""))
    if match:
        return int(match.group(1))
    digits = re.findall(r"\b([1-5])\b", str(text or ""))
    if digits:
        return int(digits[-1])
    return 0


def filter_rip(generated_items: list[dict], args: object) -> list[dict]:
    """Apply RIP-style filtering for non-verifiable instruction-following tasks."""
    if not generated_items:
        return []

    teacher = make_teacher(args)
    response_batch_size = int(getattr(args, "gen_batch_size", 1))
    num_responses = int(getattr(args, "rip_num_responses", 3))
    min_score = int(getattr(args, "rip_min_score", 4))

    response_prompts = [
        RIP_RESPONSE_TEMPLATE.format(instruction=str(item.get("instruction", item.get("question", ""))).strip())
        for item in generated_items
    ]

    all_responses: list[list[str]] = [[] for _ in generated_items]
    all_response_metadata: list[list[dict[str, object]]] = [[] for _ in generated_items]
    for sample_idx in range(num_responses):
        sample_outputs = batched_generate_full(
            teacher,
            response_prompts,
            n=1,
            max_new_tokens=int(getattr(args, "rip_response_max_new_tokens", 1024)),
            do_sample=bool(getattr(args, "rip_response_do_sample", True)),
            temperature=float(getattr(args, "rip_response_temperature", 0.7)),
            top_p=float(getattr(args, "rip_response_top_p", 0.95)),
            gen_batch_size=response_batch_size,
            desc=f"RIP response sample {sample_idx + 1}/{num_responses}",
            enable_thinking=False,
            use_chat_template=True,
        )
        for item_idx, outputs in enumerate(sample_outputs):
            first_output = outputs[0] if outputs else {}
            text = str(first_output.get("text", "") or "").strip()
            all_responses[item_idx].append(text)
            all_response_metadata[item_idx].append(first_output)

    score_prompts: list[str] = []
    score_lookup: list[tuple[int, int]] = []
    for item_idx, item in enumerate(generated_items):
        instruction = str(item.get("instruction", item.get("question", ""))).strip()
        for response_idx, response in enumerate(all_responses[item_idx]):
            score_prompts.append(
                RIP_SCORE_TEMPLATE.format(
                    instruction=instruction,
                    response=response,
                )
            )
            score_lookup.append((item_idx, response_idx))

    score_outputs = batched_generate_full(
        teacher,
        score_prompts,
        n=1,
        max_new_tokens=int(getattr(args, "rip_score_max_new_tokens", 32)),
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        gen_batch_size=response_batch_size,
        desc="RIP score responses",
        enable_thinking=False,
        use_chat_template=True,
    ) if score_prompts else []

    all_scores: list[list[int]] = [[] for _ in generated_items]
    all_score_texts: list[list[str]] = [[] for _ in generated_items]
    for (item_idx, _response_idx), outputs in zip(score_lookup, score_outputs):
        first_output = outputs[0] if outputs else {}
        score_text = str(first_output.get("text", "") or "")
        all_score_texts[item_idx].append(score_text)
        all_scores[item_idx].append(_parse_score(score_text))

    filtered_items: list[dict] = []
    for item_idx, item in enumerate(generated_items):
        response_scores = all_scores[item_idx]
        response_texts = all_responses[item_idx]
        response_metadata = all_response_metadata[item_idx]
        if response_scores:
            min_response_score = min(response_scores)
            best_response_idx = max(range(len(response_scores)), key=lambda idx: response_scores[idx])
            best_response_score = response_scores[best_response_idx]
            best_response = response_texts[best_response_idx]
        else:
            min_response_score = 0
            best_response_idx = -1
            best_response_score = 0
            best_response = ""

        rip_passed = bool(best_response and min_response_score >= min_score)
        enriched = {
            **item,
            "rip_response_prompt": response_prompts[item_idx],
            "rip_candidate_responses": response_texts,
            "rip_candidate_response_metadata": response_metadata,
            "rip_score_texts": all_score_texts[item_idx],
            "rip_scores": response_scores,
            "rip_min_score": min_response_score,
            "rip_best_response_index": best_response_idx,
            "rip_best_response_score": best_response_score,
            "rip_best_response": normalize_text(best_response),
            "rip_passed": rip_passed,
            "teacher_filter_passed": rip_passed,
        }
        filtered_items.append(enriched)

    return filtered_items
