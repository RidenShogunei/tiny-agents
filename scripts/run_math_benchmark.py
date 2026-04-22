"""Quick MATH benchmark runner using built-in samples (no datasets lib required).

This validates the end-to-end inference pipeline:
    1. Load reasoner agent with 3B model
    2. Run on a few competition math problems
    3. Save results to JSONL
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend
from tiny_agents.agents.router import RouterAgent
from tiny_agents.core.orchestrator import Orchestrator


# A few representative MATH problems (competition level)
SAMPLE_PROBLEMS = [
    {
        "problem": (
            "Find the value of $x$ that satisfies $\\frac{\\sqrt{3x+5}}{\\sqrt{6x+5}}=\\frac{\\sqrt{5}}{3}$."
        ),
        "answer": "\\frac{20}{3}",
        "level": "Level 3",
        "type": "Algebra",
    },
    {
        "problem": (
            "What is the least positive integer $n$ such that $\\frac{n}{2010}$ is a terminating decimal?"
        ),
        "answer": "67",
        "level": "Level 2",
        "type": "Number Theory",
    },
    {
        "problem": (
            "The sum of two numbers is 50 and their difference is 10. What is the product of the two numbers?"
        ),
        "answer": "600",
        "level": "Level 1",
        "type": "Algebra",
    },
    {
        "problem": (
            "A circle is inscribed in a right triangle with legs of length 6 and 8. What is the radius of the circle?"
        ),
        "answer": "2",
        "level": "Level 2",
        "type": "Geometry",
    },
    {
        "problem": (
            "How many positive integers $n$ satisfy $\\frac{1}{n} + \\frac{1}{n+1} + \\frac{1}{n+2} < \\frac{1}{2}$ ?"
        ),
        "answer": "4",
        "level": "Level 3",
        "type": "Algebra",
    },
]


REASONER_PROMPT = """You are a mathematical reasoning expert. Solve the given problem step by step.

Rules:
1. Show your reasoning clearly
2. End your response with the final answer inside \\boxed{}
3. Be concise but thorough"""


class QuickReasonerAgent:
    """Minimal wrapper for math reasoning with vLLM backend."""

    def __init__(self, backend, model_key="reasoner"):
        self.backend = backend
        self.model_key = model_key
        self.name = "reasoner"

    async def solve(self, problem: str, max_tokens: int = 1024) -> str:
        messages = [
            {"role": "system", "content": REASONER_PROMPT},
            {"role": "user", "content": f"Solve this problem:\n\n{problem}\n"},
        ]
        return self.backend.generate(
            model_key=self.model_key,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )


async def main():
    print("=" * 70)
    print("Tiny Agents MATH Benchmark (Quick Validation)")
    print("=" * 70)

    backend = VLLMBackend(default_gpu=0)

    # Load reasoner model (3B) on GPU 0
    print("\n[Loading model] Qwen2.5-3B-Instruct on cuda:0 ...")
    backend.load_model(
        model_key="reasoner",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        gpu=0,
        max_model_len=4096,
    )

    reasoner = QuickReasonerAgent(backend)

    results = []
    start_time = time.time()

    for idx, item in enumerate(SAMPLE_PROBLEMS, 1):
        print(f"\n[{idx}/{len(SAMPLE_PROBLEMS)}] Problem: {item['problem'][:80]}...")
        pred = await reasoner.solve(item["problem"])
        print(f"       Predicted: {pred[:200]}...")

        results.append({
            "problem": item["problem"],
            "ground_truth": item["answer"],
            "predicted": pred,
            "level": item["level"],
            "type": item["type"],
        })

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Benchmark complete: {len(results)} problems in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Save results
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "math_quick_results.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to: {out_path}")

    backend.unload_all()
    print("Models unloaded.")


if __name__ == "__main__":
    asyncio.run(main())
