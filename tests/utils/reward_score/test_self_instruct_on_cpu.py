from verl.utils.reward_score import self_instruct
from verl.utils.reward_score.reward_fn_router import compute_score_router


def test_exact_code_ignores_thinking_and_fences():
    solution = "<think>work it out</think>\n```python\ndef add(a, b):\n    return a + b\n```"
    reference = "def add(a, b):\n    return a + b"

    assert self_instruct.compute_score(solution, reference) == 1.0


def test_partial_answer_gets_bounded_token_f1():
    score = self_instruct.compute_score("The answer is 42.", "The answer is 41.")

    assert 0.0 < score < 1.0


def test_empty_answer_gets_zero():
    assert self_instruct.compute_score("", "reference") == 0.0


def test_router_supports_self_instruct():
    assert compute_score_router("self_instruct", "blue", "blue") == 1.0
