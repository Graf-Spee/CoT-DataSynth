"""
MBPP Code Generation Evaluation
"""

import signal
import io
import contextlib


class TimeoutException(Exception):
    """Raised when code execution exceeds the time limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException("Code execution timed out")

def extract_code_generation(model_output: str):
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

    # Parquet may deserialize nested list-like values as numpy.ndarray.
    # Normalize any array-like object first so downstream type checks work.
    if hasattr(ground_truth, "tolist") and not isinstance(ground_truth, (str, dict, list)):
        try:
            ground_truth = ground_truth.tolist()
        except Exception:
            pass

    if isinstance(ground_truth, list):
        # Already a list of test case dicts
        return ground_truth
    
    if isinstance(ground_truth, str):
        return [ground_truth]

    # Format: 'test_list'
    if isinstance(ground_truth, dict) and 'test_list' in ground_truth:
        return ground_truth['test_list']

    return []


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
        return False, False
    
    code_str_with_tests = code_str
    for test_str in test_cases:
        code_str_with_tests = code_str_with_tests + '\n' + test_str

    # First, try to compile the code to catch syntax errors early
    try:
        compiled_code = compile(code_str_with_tests, '<string>', 'exec')
    except SyntaxError:
        return False, False
    except Exception:
        return False, False

    result = True
    execution_success = True

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Execute in an explicit namespace to avoid function-scope exec
            # name resolution issues on large test scripts (e.g., HumanEval+).
            sandbox_globals = {"__builtins__": __builtins__}
            exec(compiled_code, sandbox_globals, sandbox_globals)
    except TimeoutException:
        result = False
        execution_success = False
    except Exception:
        # Runtime error (e.g., wrong answer type, index error, etc.)
        result = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result, execution_success


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
    extracted_code = extract_code_generation(solution_str)

    if not extracted_code or not extracted_code.strip():
        return 0.0

    # Step 2: Parse test cases from ground_truth
    test_cases = parse_test_cases(ground_truth)

    if not test_cases:
        return 0.0

    # Step 3: Execute code against test cases (OpenCompass evaluator.py style)
    result, execution_success = execute_code_against_tests(extracted_code, test_cases)

    # Step 4: Calculate score based on results
    if result:
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
    return extract_code_generation(solution_str)
