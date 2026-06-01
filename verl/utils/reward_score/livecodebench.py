"""
LiveCodeBench Code Generation Evaluation

Based on OpenCompass LiveCodeBench implementation:
https://github.com/open-compass/opencompass/tree/main/opencompass/datasets/livecodebench

This module provides a compute_score function for evaluating code generation solutions
from language models against test cases from the LiveCodeBench dataset.

Dataset: https://huggingface.co/datasets/livecodebench/code_generation
"""

import re
import json
import signal
import io
import contextlib
from unittest.mock import patch


class TimeoutException(Exception):
    """Raised when code execution exceeds the time limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException("Code execution timed out")


def extract_code_generation(model_output: str):
    """Extract code from markdown code blocks (first code block).

    Adapted from OpenCompass LiveCodeBench extract_utils.py.
    Finds the first pair of ``` markers and returns the content between them.

    Args:
        model_output: Raw text output from the language model

    Returns:
        Extracted code string, or empty string if no code block found
    """
    outputlines = model_output.split('\n')
    indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    if len(indexlines) < 2:
        return ''
    return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])


def extract_code_generation_v2(model_output: str):
    """Extract code from markdown code blocks (keep only the last code block).

    Adapted from OpenCompass LiveCodeBench extract_utils.py.
    When multiple code blocks exist, only keeps the last one (more strict).

    Args:
        model_output: Raw text output from the language model

    Returns:
        Extracted code string, or empty string if no code block found
    """
    outputlines = model_output.split('\n')
    indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    if len(indexlines) < 2:
        return ''
    elif len(indexlines) > 2:
        # Only keep the last code block
        indexlines = indexlines[-2:]
    return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])


def parse_test_cases(ground_truth):
    """Parse test cases from various ground truth formats.

    Supports multiple formats:
    1. LiveCodeBench HuggingFace format: dict with 'public_test_cases'/'private_test_cases'
    2. OpenCompass internal format: dict with 'input_output' containing 'inputs'/'outputs'
    3. OpenCompass evaluation_sample format: dict with 'evaluation_sample' field
    4. Direct list of test case dicts with 'input' and 'output' keys

    Args:
        ground_truth: Problem sample containing test case information

    Returns:
        List of test case dicts with 'input' and 'output' keys, or empty list
    """
    test_cases = []

    # Parquet may deserialize nested list-like values as numpy.ndarray.
    # Normalize any array-like object first so downstream type checks work.
    if hasattr(ground_truth, "tolist") and not isinstance(ground_truth, (str, dict, list)):
        try:
            ground_truth = ground_truth.tolist()
        except Exception:
            pass

    # Some pipelines may persist full ground_truth as a JSON string.
    if isinstance(ground_truth, str):
        s = ground_truth.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                ground_truth = json.loads(s)
            except (json.JSONDecodeError, TypeError):
                return []
        else:
            return []

    if isinstance(ground_truth, list):
        # Already a list of test case dicts
        return ground_truth

    if not isinstance(ground_truth, dict):
        return []

    # Format 1: public_test_cases / private_test_cases (LiveCodeBench HuggingFace format)
    for key in ['public_test_cases', 'private_test_cases']:
        if key in ground_truth:
            try:
                value = ground_truth[key]
                if isinstance(value, str):
                    tests = json.loads(value)
                elif isinstance(value, list):
                    tests = value
                else:
                    continue
                if isinstance(tests, list):
                    for tc in tests:
                        if isinstance(tc, dict) and 'input' in tc and 'output' in tc:
                            test_cases.append({
                                'input': tc['input'],
                                'output': tc['output']
                            })
            except (json.JSONDecodeError, TypeError):
                pass

    # Format 2: input_output (OpenCompass internal format)
    if not test_cases and 'input_output' in ground_truth:
        try:
            io_data = ground_truth['input_output']
            if isinstance(io_data, str):
                io_data = json.loads(io_data)
            inputs = io_data.get('inputs', [])
            outputs = io_data.get('outputs', [])
            for inp, out in zip(inputs, outputs):
                test_cases.append({'input': inp, 'output': out})
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Format 3: evaluation_sample (OpenCompass format)
    if not test_cases and 'evaluation_sample' in ground_truth:
        try:
            eval_sample = ground_truth['evaluation_sample']
            if isinstance(eval_sample, str):
                eval_sample = json.loads(eval_sample)
            if isinstance(eval_sample, str):
                io_data = json.loads(eval_sample)
            else:
                io_data = eval_sample
            inputs = io_data.get('inputs', [])
            outputs = io_data.get('outputs', [])
            for inp, out in zip(inputs, outputs):
                test_cases.append({'input': inp, 'output': out})
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    return test_cases


def execute_code_against_tests(code_str, test_cases, timeout=6):
    """Execute code against test cases with timeout protection.

    Mocks stdin to provide test inputs and captures stdout for comparison.
    Uses signal-based timeout to prevent infinite loops.

    Args:
        code_str: Python code to execute
        test_cases: List of test case dicts with 'input' and 'output' keys
        timeout: Maximum execution time per test case in seconds

    Returns:
        Tuple of (results_list, execution_success)
        - results_list: List of booleans indicating pass/fail for each test
        - execution_success: Boolean indicating if code executed without fatal errors
    """
    if not code_str or not code_str.strip():
        return [False] * len(test_cases), False

    # First, try to compile the code to catch syntax errors early
    try:
        compiled_code = compile(code_str, '<string>', 'exec')
    except SyntaxError:
        return [False] * len(test_cases), False
    except Exception:
        return [False] * len(test_cases), False

    results = []
    execution_success = True

    for test_case in test_cases:
        test_input = test_case.get('input', '')
        expected_output = test_case.get('output', '')

        stdin_data = io.StringIO(test_input)
        stdout_capture = io.StringIO()

        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            with contextlib.redirect_stdout(stdout_capture):
                # Mock input() to read from test input
                def mock_input(prompt=''):
                    line = stdin_data.readline()
                    return line.rstrip('\n')

                with patch('builtins.input', mock_input):
                    namespace = {'__builtins__': __builtins__}
                    exec(compiled_code, namespace)

            actual_output = stdout_capture.getvalue()

            # Compare outputs (normalize by stripping trailing newlines)
            actual_stripped = actual_output.rstrip('\n')
            expected_stripped = expected_output.rstrip('\n')
            passed = actual_stripped == expected_stripped
            results.append(passed)

        except TimeoutException:
            results.append(False)
            execution_success = False
        except Exception:
            # Runtime error (e.g., wrong answer type, index error, etc.)
            results.append(False)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return results, execution_success


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    Evaluate code generation solution using LiveCodeBench-style testing.

    Based on OpenCompass LiveCodeBench implementation. The evaluation process:
    1. Extract code from model output (markdown code blocks)
    2. Compile and execute code against test cases
    3. Score based on test pass rate

    Code extraction (from OpenCompass extract_utils.py):
    - "strict" method: Uses extract_code_generation_v2 which keeps only the 
      last code block. More strict, filters out explanatory code examples.
    - "flexible" method: Uses extract_code_generation which keeps the first
      code block. More lenient, accepts the first found solution.

    Test execution (from OpenCompass testing_util.py / evaluator.py):
    - Compiles code first to catch syntax errors
    - Runs code with mocked stdin/stdout for each test case
    - Uses signal-based timeout to prevent infinite loops
    - Compares actual output with expected output

    Args:
        solution_str: Text generated by the model (may contain markdown code blocks)
        ground_truth: Problem sample containing test cases. Supported formats:
                      - Dict with 'public_test_cases'/'private_test_cases' (JSON strings or lists)
                      - Dict with 'input_output' key (JSON string with 'inputs'/'outputs')
                      - Dict with 'evaluation_sample' key (OpenCompass format)
                      - List of test case dicts [{'input': ..., 'output': ...}, ...]
        method: "strict" uses extract_code_generation_v2 (only last code block);
                "flexible" uses extract_code_generation (first code block)
        format_score: Partial credit (0.0~1.0) for extracting valid, compilable code
                     that fails some or all tests (rewards correct formatting)
        score: Full score for passing all test cases (default 1.0)

    Returns:
        float: score if all tests pass,
               format_score if code is valid but some tests fail,
               0.0 if code extraction fails or all tests fail
    """
    assert method in ["strict", "flexible"]

    if not solution_str or ground_truth is None:
        return 0.0

    # Step 1: Extract code from model output (OpenCompass extract_utils.py)
    if method == "strict":
        extracted_code = extract_code_generation_v2(solution_str)
    else:  # flexible
        extracted_code = extract_code_generation(solution_str)

    if not extracted_code or not extracted_code.strip():
        return 0.0

    # Step 2: Parse test cases from ground_truth
    test_cases = parse_test_cases(ground_truth)

    if not test_cases:
        return 0.0

    # Step 3: Execute code against test cases (OpenCompass evaluator.py style)
    results, execution_success = execute_code_against_tests(extracted_code, test_cases)

    if not results:
        return 0.0

    # Step 4: Calculate score based on results
    pass_rate = sum(results) / len(results)

    if pass_rate == 1.0:
        # All tests passed - full score
        return score
    elif execution_success:
        # Code was extracted and executed (compiled successfully) but some/all tests failed
        return format_score
    else:
        # Code extraction or execution completely failed
        return 0.0


def compute_score_batch(solutions, ground_truths, method="strict", format_score=0.0, score=1.0):
    """Evaluate multiple code generation solutions in batch.

    Args:
        solutions: List of model-generated text strings
        ground_truths: List of problem samples with test cases
        method: "strict" or "flexible" code extraction
        format_score: Partial credit for compilable but incorrect code
        score: Full score for all tests passed

    Returns:
        List of scores (one per solution)
    """
    scores = []
    for sol, gt in zip(solutions, ground_truths):
        s = compute_score(sol, gt, method=method, format_score=format_score, score=score)
        scores.append(s)
    return scores


def extract_pred(solution_str):
    """Extract code from model output without evaluating.

    Useful for inspecting what code would be evaluated.

    Args:
        solution_str: Model-generated text

    Returns:
        Extracted code string
    """
    return extract_code_generation_v2(solution_str)


# Debug Code

if __name__ == "__main__":
    import os

    # Create test data in JSON file to avoid escaping issues
    test_data = {
        "test_cases": [
            {"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n"}
        ]
    }

    test_file = "/tmp/lcb_test_data.json"
    with open(test_file, "w") as f:
        json.dump(test_data, f)

    # Load test data
    with open(test_file, "r") as f:
        loaded = json.load(f)

    gt = loaded["test_cases"]

    # Test
    solution = """```python\nt = int(input())\nfor _ in range(t):\n    s = input().strip()\n    diff = sum(1 for i in range(3) if s[i] != 'abc'[i])\n    print("YES" if diff in (0, 2) else "NO")\n```"""
    score1 = compute_score(solution, gt, method="strict")
    print(f"Test: score = {score1}")

    # Test 1: Correct solution
    solution_correct = """
Here's my solution:

```python
n = int(input())
print(n * 2)
```

This code reads an integer and prints its double.
"""

    score1 = compute_score(solution_correct, gt, method="strict")
    print(f"Example 1 (correct): score = {score1}")  # Expected: 1.0

    # Test 2: Wrong logic but valid code
    solution_wrong = """
```python
n = int(input())
print(n * 3)  # Wrong: should be * 2
```
"""
    score2 = compute_score(solution_wrong, gt, method="strict", format_score=0.1)
    print(f"Example 2 (wrong logic): score = {score2}")  # Expected: 0.1

    # Test 3: No code block
    solution_no_code = "The answer is to multiply by 2."
    score3 = compute_score(solution_no_code, gt)
    print(f"Example 3 (no code): score = {score3}")  # Expected: 0.0

    # Test 4: Multiple code blocks (strict vs flexible)
    solution_multi = """
Here's an example:
```python
print("hello")  # This is just an example
```

And here's the actual solution:
```python
n = int(input())
print(n * 2)
```
"""
    score4_strict = compute_score(solution_multi, gt, method="strict")
    score4_flexible = compute_score(solution_multi, gt, method="flexible")
    print(f"Example 4 (multi-block, strict): score = {score4_strict}")    # Expected: 1.0
    print(f"Example 4 (multi-block, flexible): score = {score4_flexible}")  # Expected: 0.0

    # Cleanup
    os.remove(test_file)
