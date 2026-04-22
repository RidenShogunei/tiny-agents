"""GSM8K benchmark with tool-augmented reasoner (Python executor)."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend
from tiny_agents.agents.tool_reasoner import ToolReasonerAgent
from tiny_agents.core.orchestrator import Orchestrator


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        parts = text.split("####")
        last = parts[-1].strip()
        if last:
            return last.split()[0]
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            nums = [w for w in line.replace(",", "").split() if w.replace(".", "", 1).replace("-", "", 1).isdigit()]
            if nums:
                return nums[-1]
    return ""


def grade(pred: str, truth: str) -> bool:
    """Compare predicted and truth answers, handling numeric equivalence."""
    p = pred.strip().replace(",", "").replace("$", "").lower()
    t = truth.strip().replace(",", "").replace("$", "").lower()
    if p == t:
        return True
    # Try numeric comparison (handles 60.0 == 60)
    try:
        return float(p) == float(t)
    except (ValueError, TypeError):
        pass
    return False


async def main(limit: int = 20):
    from modelscope.msdatasets import MsDataset

    print("=" * 70)
    print("GSM8K Tool-Augmented Benchmark")
    print("=" * 70)

    dataset = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test", trust_remote_code=True)
    subset = list(dataset)[:limit]
    print(f"Running {len(subset)} problems with tool-augmented reasoner\n")

    backend = VLLMBackend(default_gpu=1)
    backend.load_model(
        model_key="tool_reasoner",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        gpu=1,
        max_model_len=8192,
    )

    agent = ToolReasonerAgent(backend=backend)
    # max_iterations must be ≥2 so agent gets tool result + generates final answer
    # enable_review=False so it returns immediately after "respond"
    orch = Orchestrator(max_iterations=3, enable_review=False)
    orch.register_agent(agent)

    results = []
    correct = 0
    start_time = time.time()

    for i, item in enumerate(subset):
        question = item["question"]
        answer_text = item["answer"]
        truth_num = answer_text.split("####")[-1].strip().split()[0] if "####" in answer_text else ""

        result = await orch.execute({"task": question}, entry_agent="tool_reasoner")
        payload = result.get("result", {})
        pred_text = payload.get("answer", "")
        pred_num = extract_gsm8k_answer(pred_text)
        is_correct = grade(pred_num, truth_num)
        correct += int(is_correct)

        results.append({
            "idx": i,
            "question": question,
            "ground_truth": truth_num,
            "predicted": pred_text,
            "predicted_num": pred_num,
            "correct": is_correct,
            "code": payload.get("code", ""),
        })

        status = "✅" if is_correct else "❌"
        print(f"[{i+1}/{len(subset)}] {status} pred={pred_num} truth={truth_num}")
        if payload.get("code"):
            code_lines = payload["code"].splitlines()
            print(f"        Used Python: {code_lines[0][:60]}...")

    elapsed = time.time() - start_time
    acc = correct / len(subset) * 100 if subset else 0
    print(f"\n{'=' * 70}")
    print(f"Accuracy: {correct}/{len(subset)} = {acc:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(subset):.1f}s/iter)")
    print(f"{'=' * 70}")

    out_path = Path("benchmark_results/gsm8k_tool_results.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to: {out_path}")

    backend.unload_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.limit))
