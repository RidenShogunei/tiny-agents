"""HumanEval multi-agent benchmark with code generation + execution + review loop.

Pipeline:
    Router -> Coder -> Python Executor -> Critic (if fail, loop back to Coder)
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend
from tiny_agents.core.orchestrator import Orchestrator
from tiny_agents.agents.coder import CoderAgent
from tiny_agents.agents.critic import CriticAgent
from tiny_agents.tools.python_executor import PythonExecutor


def extract_code_block(text: str) -> str:
    """Extract only the first python code block, stripping any natural language outside."""
    # Try ```python ... ```
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try plain ``` ... ```
    match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, stop at conversation markers
    lines = text.strip().splitlines()
    code_lines = []
    for line in lines:
        if line.strip().startswith("User:") or line.strip().startswith("Assistant:"):
            break
        code_lines.append(line)
    return "\n".join(code_lines).strip()


# Built-in HumanEval samples
SAMPLES = [
    {
        "task_id": "HumanEval/0",
        "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "test": """
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
assert has_close_elements([1.0, 2.0, 5.0, 6.0], 0.5) == False
assert has_close_elements([1.0, 2.0, 2.1, 6.0], 0.2) == True
""",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "test": """
assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
assert separate_paren_groups('()') == ['()']
assert separate_paren_groups('(())(())') == ['(())', '(())']
""",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
        "test": """
assert truncate_number(3.5) == 0.5
assert truncate_number(10.75) == 0.75
assert truncate_number(0.123) == 0.123
assert truncate_number(100.0) == 0.0
""",
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
        "test": """
assert below_zero([1, 2, 3]) == False
assert below_zero([1, 2, -4, 5]) == True
assert below_zero([]) == False
assert below_zero([10, -5, -3, -2]) == True
""",
        "entry_point": "below_zero",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
        "test": """
assert mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == 1.0
assert mean_absolute_deviation([1.0, 1.0, 1.0, 1.0]) == 0.0
assert mean_absolute_deviation([1.0, 2.0]) == 0.5
""",
        "entry_point": "mean_absolute_deviation",
    },
]


def run_tests(code: str, test_code: str, entry_point: str) -> dict:
    """Execute generated code with tests. Auto-add missing common imports."""
    # Auto-add missing imports if code uses typing but lacks the import
    if "List[" in code or "Dict[" in code or "Tuple[" in code or "Optional[" in code:
        if "from typing import" not in code and "import typing" not in code:
            code = "from typing import List, Dict, Tuple, Optional\n\n" + code
    if "math." in code and "import math" not in code:
        code = "import math\n\n" + code

    full_code = code + "\n\n# Tests\n" + test_code
    executor = PythonExecutor(timeout=10)
    result = executor.run(full_code)
    passed = result["success"] and result["returncode"] == 0
    return {
        "passed": passed,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


async def main():
    print("=" * 70)
    print("HumanEval Multi-Agent Benchmark")
    print("=" * 70)

    backend = VLLMBackend(default_gpu=2)
    backend.load_model("coder", "Qwen/Qwen2.5-3B-Instruct", gpu=2, max_model_len=4096)
    backend.load_model("critic", "Qwen/Qwen2.5-0.5B-Instruct", gpu=3, max_model_len=4096)

    coder = CoderAgent(backend=backend)
    critic = CriticAgent(backend=backend)

    orch = Orchestrator(max_iterations=1, enable_review=False)
    orch.register_agent(coder)
    orch.register_agent(critic)
    orch.critic_agent = "critic"

    results = []
    total_passed = 0

    for sample in SAMPLES:
        task_id = sample["task_id"]
        prompt = sample["prompt"]
        test_code = sample["test"]
        entry_point = sample["entry_point"]

        print(f"\n[{task_id}] {entry_point}")

        # Directly invoke coder with the prompt (no reset needed — context is session-scoped)
        result = await orch.execute(
            {"task": f"Complete this function:\n\n{prompt}", "code": "", "review_feedback": ""},
            entry_agent="coder",
        )

        # Extract code from result
        payload = result.get("result", {})
        generated = payload.get("code", "")
        code = extract_code_block(generated)

        # Run tests
        test_result = run_tests(code, test_code, entry_point)
        passed = test_result["passed"]
        total_passed += int(passed)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}")
        if not passed:
            print(f"  stderr: {test_result['stderr'][:200]}")

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": passed,
            "code": code,
            "stderr": test_result["stderr"],
            "steps": len(result.get("steps", [])),
        })

    print(f"\n{'=' * 70}")
    print(f"Total: {total_passed}/{len(SAMPLES)} = {total_passed/len(SAMPLES)*100:.0f}%")
    print(f"{'=' * 70}")

    out_path = Path("benchmark_results/humaneval_multiagent_results.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to: {out_path}")

    backend.unload_all()


if __name__ == "__main__":
    asyncio.run(main())
