"""Full MATH benchmark runner using the hendrycks/competition_math dataset.

Usage:
    python scripts/run_full_math_benchmark.py --limit 100 --model_key reasoner --gpu 0
    python scripts/run_full_math_benchmark.py --limit 0   # run all 5000
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
    parser = argparse.ArgumentParser(description="Run MATH benchmark")
    parser.add_argument("--limit", type=int, default=100, help="Number of problems to run (0 = all)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--model_key", type=str, default="reasoner", help="Backend model key")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HF model id or local path")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="benchmark_results/math_full_results.jsonl")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in dataset")
    return parser.parse_args()


REASONER_PROMPT = """You are a mathematical reasoning expert. Solve the given problem step by step.

Rules:
1. Show your reasoning clearly
2. End your response with the final answer inside \\boxed{}
3. Be concise but thorough"""


async def run_benchmark(args):
    from datasets import load_dataset

    print("=" * 70)
    print("Tiny Agents MATH Benchmark (Full)")
    print("=" * 70)

    # Load dataset
    print("\n[Loading dataset] hendrycks/competition_math ...")
    dataset = load_dataset("hendrycks/competition_math", split="test")
    total_available = len(dataset)
    print(f"Total available problems: {total_available}")

    end_idx = args.start_idx + args.limit if args.limit > 0 else total_available
    end_idx = min(end_idx, total_available)
    subset = dataset.select(range(args.start_idx, end_idx))
    print(f"Running problems [{args.start_idx}, {end_idx}) = {len(subset)} problems")

    # Load model
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

    # Resume support
    processed_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                processed_ids.add(item.get("idx", item.get("problem", "")))
        print(f"Resuming from existing results: {len(processed_ids)} already done")

    results = []
    start_time = time.time()
    correct_count = 0

    for i, item in enumerate(subset):
        global_idx = args.start_idx + i
        problem = item["problem"]
        answer = item["solution"]
        level = item.get("level", "unknown")
        _type = item.get("type", "unknown")

        key = f"{global_idx}_{problem[:50]}"
        if key in processed_ids:
            continue

        messages = [
            {"role": "system", "content": REASONER_PROMPT},
            {"role": "user", "content": f"Solve this problem:\n\n{problem}\n"},
        ]

        t0 = time.time()
        pred = backend.generate(
            model_key=args.model_key,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        latency = time.time() - t0

        result = {
            "idx": global_idx,
            "problem": problem,
            "ground_truth": answer,
            "predicted": pred,
            "level": level,
            "type": _type,
            "latency_sec": round(latency, 2),
        }
        results.append(result)

        # Append immediately for crash safety
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0 or i == len(subset) - 1:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = (len(subset) - (i + 1)) * avg
            print(f"  [{i+1}/{len(subset)}] avg={avg:.1f}s/iter, elapsed={elapsed:.0f}s, remaining={remaining:.0f}s")

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Benchmark complete: {len(results)} problems in {total_elapsed:.1f}s")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 70}")

    backend.unload_all()
    print("Models unloaded.")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args))
