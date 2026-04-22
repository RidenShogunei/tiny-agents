"""GSM8K benchmark runner via modelscope (fallback when HF MATH is unreachable).

Usage:
    python scripts/run_gsm8k_benchmark.py --limit 100
    python scripts/run_gsm8k_benchmark.py --limit 0   # all 1319
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend


def parse_args():
    parser = argparse.ArgumentParser(description="Run GSM8K benchmark")
    parser.add_argument("--limit", type=int, default=100, help="Number of problems (0=all)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_key", type=str, default="reasoner")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="benchmark_results/gsm8k_results.jsonl")
    parser.add_argument("--start_idx", type=int, default=0)
    return parser.parse_args()


REASONER_PROMPT = """You are a mathematical reasoning expert. Solve the word problem step by step.

Rules:
1. Show your reasoning clearly
2. Put your final numerical answer on the last line after ####
3. Do not include units in the final answer, only the number"""


def extract_gsm8k_answer(text: str) -> str:
    """Extract answer after ####."""
    if "####" in text:
        parts = text.split("####")
        last = parts[-1].strip()
        if last:
            return last.split()[0]
    # Fallback: last line number
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            nums = [w for w in line.replace(",", "").split() if w.replace(".", "", 1).replace("-", "", 1).isdigit()]
            if nums:
                return nums[-1]
    return ""


def normalize(ans: str) -> str:
    return ans.strip().replace(",", "").replace("$", "").lower()


def grade(pred: str, truth: str) -> bool:
    """Compare predicted and truth answers, handling numeric equivalence."""
    p = pred.strip().replace(",", "").replace("$", "").lower()
    t = truth.strip().replace(",", "").replace("$", "").lower()
    if p == t:
        return True
    try:
        return float(p) == float(t)
    except (ValueError, TypeError):
        pass
    return False


async def run_benchmark(args):
    from modelscope.msdatasets import MsDataset

    print("=" * 70)
    print("Tiny Agents GSM8K Benchmark")
    print("=" * 70)

    print("\n[Loading dataset] modelscope/gsm8k ...")
    dataset = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test", trust_remote_code=True)
    total_available = len(dataset)
    print(f"Total available: {total_available}")

    end_idx = args.start_idx + args.limit if args.limit > 0 else total_available
    end_idx = min(end_idx, total_available)
    subset = list(dataset)[args.start_idx:end_idx]
    print(f"Running problems [{args.start_idx}, {end_idx}) = {len(subset)} problems")

    backend = VLLMBackend(default_gpu=args.gpu)
    print(f"\n[Loading model] {args.model_name} on cuda:{args.gpu} ...")
    backend.load_model(
        model_key=args.model_key,
        model_name=args.model_name,
        gpu=args.gpu,
        max_model_len=args.max_model_len,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    processed_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                processed_ids.add(item.get("idx", ""))
        print(f"Resuming: {len(processed_ids)} already done")

    start_time = time.time()
    correct_count = 0

    for i, item in enumerate(subset):
        global_idx = args.start_idx + i
        if global_idx in processed_ids:
            continue

        question = item["question"]
        answer_text = item["answer"]
        # Extract truth number from answer_text (format: ...\n#### 18)
        truth_num = answer_text.split("####")[-1].strip().split()[0] if "####" in answer_text else ""

        messages = [
            {"role": "system", "content": REASONER_PROMPT},
            {"role": "user", "content": f"{question}\n"},
        ]

        t0 = time.time()
        pred = backend.generate(
            model_key=args.model_key,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        latency = time.time() - t0

        pred_num = extract_gsm8k_answer(pred)
        is_correct = grade(pred_num, truth_num)
        correct_count += int(is_correct)

        result = {
            "idx": global_idx,
            "question": question,
            "ground_truth": truth_num,
            "predicted": pred,
            "predicted_num": pred_num,
            "correct": is_correct,
            "latency_sec": round(latency, 2),
        }

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0 or i == len(subset) - 1:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            acc = correct_count / (i + 1) * 100
            remaining = (len(subset) - (i + 1)) * avg
            print(f"  [{i+1}/{len(subset)}] acc={acc:.1f}% avg={avg:.1f}s/iter elapsed={elapsed:.0f}s remaining={remaining:.0f}s")

    total_elapsed = time.time() - start_time
    final_acc = correct_count / len(subset) * 100 if subset else 0
    print(f"\n{'=' * 70}")
    print(f"Benchmark complete: {len(subset)} problems in {total_elapsed:.1f}s")
    print(f"Accuracy: {correct_count}/{len(subset)} = {final_acc:.1f}%")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    backend.unload_all()
    print("Models unloaded.")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args))
