"""MathVista multi-agent benchmark with VLM + LLM collaboration.

Pipeline:
    VLM (visual perception) -> Reasoner (mathematical reasoning) -> Python Executor (if needed)

Since MathVista dataset is not accessible via modelscope/HF from this environment,
this script uses built-in sample problems with text descriptions (mocking VLM output).
To run with real images, download Qwen2.5-VL-3B-Instruct and replace MockVLAgent.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend
from tiny_agents.core.orchestrator import Orchestrator
from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.agents.tool_reasoner import ToolReasonerAgent


class MockVLAgent(BaseAgent):
    """Mock VLM agent that returns pre-defined visual descriptions.
    
    In production, replace with a real VLM (e.g., Qwen2.5-VL-3B-Instruct)
    that takes an image path and returns a description.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="vlm",
            model_name="mock",
            role_prompt="You are a visual perception model.",
            **kwargs,
        )

    async def run(self, input_data: dict) -> AgentOutput:
        """Return the pre-defined visual description."""
        description = input_data.get("visual_description", "")
        return AgentOutput(
            thought="Analyzed image visually",
            action="respond",
            payload={"visual_description": description},
            finished=True,
        )


# Built-in MathVista-style samples (text-based, mocking VLM output)
SAMPLES = [
    {
        "task_id": "MathVista/0",
        "question": "What is the area of the rectangle shown in the image?",
        "visual_description": "The image shows a rectangle with width 8 cm and height 5 cm.",
        "ground_truth": "40",
    },
    {
        "task_id": "MathVista/1",
        "question": "How many triangles are in the figure?",
        "visual_description": "The image shows a large triangle divided into 4 smaller triangles by lines connecting the midpoints of each side.",
        "ground_truth": "5",
    },
    {
        "task_id": "MathVista/2",
        "question": "What is the angle x in the diagram?",
        "visual_description": "The image shows a straight line with two adjacent angles: one is 120 degrees, and the other is labeled x.",
        "ground_truth": "60",
    },
    {
        "task_id": "MathVista/3",
        "question": "If the circle has radius 3, what is its area?",
        "visual_description": "The image shows a circle with center O and radius labeled as 3 units.",
        "ground_truth": "28.27",
    },
    {
        "task_id": "MathVista/4",
        "question": "What is the perimeter of the square?",
        "visual_description": "The image shows a square with each side labeled 7 cm.",
        "ground_truth": "28",
    },
]


def grade(pred: str, truth: str) -> bool:
    """Compare predicted and truth answers, handling numeric equivalence."""
    # Extract answer after #### if present
    if "####" in pred:
        parts = pred.split("####")
        pred = parts[-1].strip().split()[0] if parts[-1].strip() else ""
    # Try to extract boxed answer
    elif "\\boxed{" in pred:
        import re
        m = re.search(r"\\boxed\{(.*?)\}", pred)
        if m:
            pred = m.group(1)
    # Fallback: take last non-empty line
    else:
        lines = pred.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                pred = line
                break

    p = pred.strip().replace(",", "").replace("$", "").lower()
    t = truth.strip().replace(",", "").replace("$", "").lower()
    if p == t:
        return True
    try:
        pf = float(p)
        tf = float(t)
        return abs(pf - tf) < 1e-2
    except ValueError:
        pass
    return False


async def main():
    print("=" * 70)
    print("MathVista Multi-Agent Benchmark (VLM + LLM Collaboration)")
    print("=" * 70)

    backend = VLLMBackend(default_gpu=2)
    backend.load_model(
        "tool_reasoner",
        "Qwen/Qwen2.5-3B-Instruct",
        gpu=2,
        max_model_len=8192,
    )

    vlm = MockVLAgent()
    reasoner = ToolReasonerAgent(backend=backend)

    orch = Orchestrator(max_iterations=1, enable_review=False)
    orch.register_agent(vlm)
    orch.register_agent(reasoner)

    results = []
    total_correct = 0

    for sample in SAMPLES:
        task_id = sample["task_id"]
        question = sample["question"]
        visual_description = sample["visual_description"]
        ground_truth = sample["ground_truth"]

        print(f"\n[{task_id}] {question}")

        # Reset agent state for each problem
        reasoner.reset()

        # Step 1: VLM analyzes the image (mock)
        vlm_result = await orch.execute(
            {"task": question, "visual_description": visual_description},
            entry_agent="vlm",
        )
        vlm_payload = vlm_result.get("result", {})
        description = vlm_payload.get("visual_description", "")
        print(f"  [VLM] {description}")

        # Step 2: Reasoner solves based on VLM description
        combined_task = (
            f"Question: {question}\n"
            f"Visual description from image: {description}\n"
            f"Solve this problem step by step."
        )
        reasoner_result = await orch.execute(
            {"task": combined_task},
            entry_agent="tool_reasoner",
        )
        reasoner_payload = reasoner_result.get("result", {})
        answer = reasoner_payload.get("answer", "")
        print(f"  [Reasoner] Answer: {answer}")

        correct = grade(answer, ground_truth)
        total_correct += int(correct)
        status = "✅ CORRECT" if correct else "❌ WRONG"
        print(f"  {status} (truth: {ground_truth})")

        results.append({
            "task_id": task_id,
            "question": question,
            "visual_description": description,
            "ground_truth": ground_truth,
            "predicted": answer,
            "correct": correct,
        })

    print(f"\n{'=' * 70}")
    print(f"Total: {total_correct}/{len(SAMPLES)} = {total_correct/len(SAMPLES)*100:.0f}%")
    print(f"{'=' * 70}")

    out_path = Path("benchmark_results/mathvista_multiagent_results.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to: {out_path}")

    backend.unload_all()


if __name__ == "__main__":
    asyncio.run(main())
