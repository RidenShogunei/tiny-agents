#!/usr/bin/env python3
"""
Delegation Value Benchmark — When Does Subagent Help vs Hurt?
==============================================================

Core question: Does delegating to a subagent improve or degrade task accuracy?
And does the answer depend on the PARENT model's capability?

3 tasks × 3 models × 2 strategies:
  A. Parent alone (no delegation) — pure single-model baseline
  B. Parent → SAME-model subagent (two-step sequential)
     Measures: "does TWO steps of the same model beat ONE step?"

Hypotheses:
  - For weak models (0.5B):  A > B (self-delegation hurts — extra step adds noise)
  - For medium models (1.5B): A ≈ B (neutral — may help or hurt per case)
  - For capable models (3B):  A < B may be possible (chain-of-thought-style benefit)

GPU allocation (sequential, one model at a time):
  GPU 0: existing VLLM process — DO NOT USE
  GPU 1: 0.5B model
  GPU 2: 3B model
  GPU 3: 1.5B model  (9B available but too large for single-GPU here)
"""

import os
import re
import sys
import json
import asyncio
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── Dataset ───────────────────────────────────────────────────────────────────
# 30 tasks: designed so that parent CAN answer alone, but delegation
# may help or hurt depending on model capability.

TASKS = [
    # ── Math (straightforward calculation) ──────────────────────────────────
    {"id":"math01","question":"A store sells 3 apples for $5. How much do 12 apples cost? Answer with the number only (no $).","answer":"20","strategies":["A","B"]},
    {"id":"math02","question":"What is 15% of 240? Answer with the number only.","answer":"36","strategies":["A","B"]},
    {"id":"math03","question":"If a train travels 300 km in 4 hours, what is its speed in km/h? Answer with the number only.","answer":"75","strategies":["A","B"]},
    {"id":"math04","question":"A rectangle is 12 cm by 8 cm. What is its perimeter in cm? Answer with the number only.","answer":"40","strategies":["A","B"]},
    {"id":"math05","question":"What is the square root of 144? Answer with the number only.","answer":"12","strategies":["A","B"]},
    {"id":"math06","question":"Which is larger: 3/7 or 4/9? Answer '4/9' or '3/7'.","answer":"4/9","strategies":["A","B"]},
    {"id":"math07","question":"What is 25% of 80 plus 10% of 50? Answer with the number only.","answer":"25","strategies":["A","B"]},
    {"id":"math08","question":"A circle has radius 5. What is its area using π≈3.14? Answer with the number only.","answer":"78.5","strategies":["A","B"]},
    {"id":"math09","question":"What is the next prime number after 17? Answer with the number only.","answer":"19","strategies":["A","B"]},
    {"id":"math10","question":"If x + 7 = 20, what is x? Answer with the number only.","answer":"13","strategies":["A","B"]},
    # ── Logic / Reasoning ───────────────────────────────────────────────────
    {"id":"logic01","question":"All cats are animals. Some animals are black. Can we conclude some cats are black? Answer 'yes' or 'no'.","answer":"no","strategies":["A","B"]},
    {"id":"logic02","question":"Tom is taller than Jim. Jim is taller than Sam. Who is the shortest? Answer 'Sam', 'Jim', or 'Tom'.","answer":"Sam","strategies":["A","B"]},
    {"id":"logic03","question":"If it rains, the ground gets wet. The ground is wet. Did it rain? Answer 'yes', 'no', or 'maybe'.","answer":"maybe","strategies":["A","B"]},
    {"id":"logic04","question":"Which is heavier: a kilogram of bricks or a kilogram of feathers? Answer 'same', 'bricks', or 'feathers'.","answer":"same","strategies":["A","B"]},
    {"id":"logic05","question":"Is the fraction 16/24 greater than 0.5? Answer 'yes' or 'no'.","answer":"yes","strategies":["A","B"]},
    # ── Comparison / Two-step calculation ────────────────────────────────────
    {"id":"comp01","question":"Mount Everest is 8848 meters. K2 is 8611 meters. Which is taller and by how many meters? Answer 'Everest X' where X is the difference.","answer":"Everest 237","strategies":["A","B"]},
    {"id":"comp02","question":"The Amazon River is about 6400 km. The Nile River is about 6650 km. Which is longer and by how much? Answer 'Nile X' where X is the difference in km.","answer":"Nile 250","strategies":["A","B"]},
    {"id":"comp03","question":"A rectangle has area 48 cm² and width 6 cm. What is its perimeter? Answer with the number and unit.","answer":"28 cm","strategies":["A","B"]},
    {"id":"comp04","question":"How many hours are there between 9:30 AM and 2:45 PM? Answer with the number only (can be decimal).","answer":"5.25","strategies":["A","B"]},
    {"id":"comp05","question":"A book has 350 pages. I read 30 pages per day. After 9 days, how many pages remain? Answer with the number only.","answer":"80","strategies":["A","B"]},
    # ── Domain knowledge ─────────────────────────────────────────────────────
    {"id":"dk01","question":"What is the capital of Australia? Answer with the city name only.","answer":"Canberra","strategies":["A","B"]},
    {"id":"dk02","question":"What is the chemical symbol for gold? Answer with the symbol only.","answer":"Au","strategies":["A","B"]},
    {"id":"dk03","question":"How many sides does a hexagon have? Answer with the number only.","answer":"6","strategies":["A","B"]},
    {"id":"dk04","question":"What is the largest planet in our solar system? Answer with the planet name only.","answer":"Jupiter","strategies":["A","B"]},
    {"id":"dk05","question":"In what year did World War II end? Answer with the year as a number.","answer":"1945","strategies":["A","B"]},
    # ── Chain-of-thought style multi-step ─────────────────────────────────────
    {"id":"step01","question":"If you buy 4 books at $12 each and get a $10 discount, how much do you pay? Answer with the number only.","answer":"38","strategies":["A","B"]},
    {"id":"step02","question":"A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball in dollars? Answer with the number only.","answer":"0.05","strategies":["A","B"]},
    {"id":"step03","question":"How many minutes are there in 2.5 hours? Answer with the number only.","answer":"150","strategies":["A","B"]},
    {"id":"step04","question":"What is 20% of 50% of 200? Answer with the number only.","answer":"20","strategies":["A","B"]},
    {"id":"step05","question":"If a square's area is 81 cm², what is its perimeter? Answer with the number and unit.","answer":"36 cm","strategies":["A","B"]},
]

# ── Model Config ──────────────────────────────────────────────────────────────

MODEL_BASE = "/home/jinxu/.cache/tiny-agents/models/Qwen"
MODEL_PATH = {
    "0.5B": f"{MODEL_BASE}/Qwen2.5-0.5B-Instruct",
    "1.5B": f"{MODEL_BASE}/Qwen2.5-1.5B-Instruct",
    "3B":   f"{MODEL_BASE}/Qwen2.5-3B-Instruct",
}

# Sequential GPU allocation (one model at a time per GPU)
GPU_MAP = {"0.5B": 1, "1.5B": 3, "3B": 2}
MEM_MAP = {"0.5B": 0.40, "1.5B": 0.55, "3B": 0.68}

# ── LLM Backend ───────────────────────────────────────────────────────────────

def make_llm(model: str):
    gpu = GPU_MAP[model]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    torch.cuda.empty_cache()
    from vllm import LLM
    return LLM(
        model=MODEL_PATH[model],
        max_num_seqs=16,
        max_model_len=512,
        gpu_memory_utilization=MEM_MAP[model],
        enable_prefix_caching=True,
    )

def extract_answer(text: str) -> str:
    """Extract the core answer from model output."""
    text = text.strip()
    text = re.sub(r'^(The answer is|Answer:|Final answer:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(True|False|Yes|No)\s*', lambda m: m.group(1).lower() + ' ', text, flags=re.IGNORECASE)
    return text.strip()

def is_correct(model_output: str, expected: str) -> bool:
    """Flexible answer matching."""
    out = extract_answer(model_output).lower().strip()
    exp = expected.lower().strip()

    # Direct match
    if out == exp:
        return True

    # Normalize spaces
    out_n = re.sub(r'\s+', ' ', out)
    exp_n = re.sub(r'\s+', ' ', exp)
    if out_n == exp_n:
        return True

    # Number extraction (for math answers)
    nums_out = re.findall(r'[\d.]+', out)
    nums_exp = re.findall(r'[\d.]+', exp)
    if nums_out and nums_exp:
        try:
            # Primary number close enough
            if abs(float(nums_out[0]) - float(nums_exp[0])) < 0.02:
                return True
        except ValueError:
            pass

    # Keyword containment
    exp_words = [w for w in exp.split() if len(w) > 2]
    for word in exp_words:
        if word in out:
            return True

    return False

# ── Strategy Execution ─────────────────────────────────────────────────────────

def run_strategy_A(llm, task: dict) -> str:
    """Strategy A: parent answers directly, no delegation."""
    prompt = f"Question: {task['question']}\nAnswer:"
    outputs = llm.chat([{"role":"user","content":prompt}])
    return outputs[0].outputs[0].text

def run_strategy_B(llm, task: dict) -> str:
    """Strategy B: parent generates subagent instruction, then synthesizes."""
    parent_prompt = f"""Question: {task['question']}

Think about what question you would ask a subagent to help answer this.
Then answer the question yourself.

Your response format:
Thought: [your reasoning]
Subagent question: [what you'd ask the subagent]
Subagent answer: [what the subagent would say, based on your reasoning]
Final answer: [your answer]"""

    parent_response = llm.chat([{"role":"user","content":parent_prompt}])[0].outputs[0].text

    # Extract subagent output from parent's internal simulation
    sa_match = re.search(r'Subagent answer:\s*(.+?)(?:\n|Final)', parent_response, re.DOTALL)
    subagent_simulated = sa_match.group(1).strip() if sa_match else ""

    # Parent synthesizes with the simulated subagent answer
    synthesis_prompt = f"""Question: {task['question']}
Subagent (simulated) says: {subagent_simulated}

Based on the above, what is the final answer? Be precise."""
    final = llm.chat([{"role":"user","content":synthesis_prompt}])
    return final[0].outputs[0].text

# ── Benchmark Runner ──────────────────────────────────────────────────────────

async def run_model(model: str, tasks: list, result_dir: Path):
    """Run strategies A and B for one model."""
    import torch
    print(f"\n{'='*60}")
    print(f"  Model: {model}  (GPU {GPU_MAP[model]})")
    print(f"{'='*60}")

    llm = make_llm(model)
    results_A = []
    results_B = []

    for task in tasks:
        tid = task["id"]
        exp = task["answer"]

        # Strategy A
        if "A" in task["strategies"]:
            try:
                out_a = run_strategy_A(llm, task)
                ok_a = is_correct(out_a, exp)
                results_A.append({"task_id": tid, "output": out_a, "expected": exp, "correct": ok_a})
                mark = "✓" if ok_a else "✗"
                print(f"  [{model}] A {mark} | {tid:<8} | got: {extract_answer(out_a)[:25]!r:<28} | exp: {exp!r}")
            except Exception as e:
                results_A.append({"task_id": tid, "error": str(e)})
                print(f"  [{model}] A ER | {tid} | {e}")

        await asyncio.sleep(0.05)

        # Strategy B
        if "B" in task["strategies"]:
            try:
                out_b = run_strategy_B(llm, task)
                ok_b = is_correct(out_b, exp)
                results_B.append({"task_id": tid, "output": out_b, "expected": exp, "correct": ok_b})
                mark = "✓" if ok_b else "✗"
                print(f"  [{model}] B {mark} | {tid:<8} | got: {extract_answer(out_b)[:25]!r:<28} | exp: {exp!r}")
            except Exception as e:
                results_B.append({"task_id": tid, "error": str(e)})
                print(f"  [{model}] B ER | {tid} | {e}")

        await asyncio.sleep(0.05)

    # Cleanup
    del llm
    gc = __import__('gc')
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    out_file = result_dir / f"{model}_results.json"
    data = {"A": results_A, "B": results_B}
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    n_a = len(results_A)
    n_b = len(results_B)
    acc_a = 100 * sum(1 for r in results_A if r.get("correct")) / n_a if n_a else 0
    acc_b = 100 * sum(1 for r in results_B if r.get("correct")) / n_b if n_b else 0
    delta = acc_b - acc_a

    print(f"\n  [{model}] A: {acc_a:.1f}%  B: {acc_b:.1f}%  Δ: {delta:+.1f}pp")
    print(f"  Saved: {out_file}")

    return data

# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/home/jinxu/tiny-agents/benchmarks/delegation_value_results")
    parser.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B"])
    args = parser.parse_args()

    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDelegation Value Benchmark")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Models: {args.models}")
    print(f"  Strategies: A (alone) vs B (same-model delegation)")
    print(f"  Output: {result_dir}")

    all_results = {}
    for model in args.models:
        data = await run_model(model, TASKS, result_dir)
        all_results[model] = data
        import torch
        torch.cuda.empty_cache()
        await asyncio.sleep(2)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL RESULTS: Does Self-Delegation Help or Hurt?")
    print("="*70)
    print(f"\n{'Model':<10} {'A (alone)':>10} {'B (delegated)':>14} {'Δ':>8} {'Verdict':<20}")
    print("-" * 65)

    verdicts = []
    for model in args.models:
        data = all_results[model]
        runs_a = data.get("A", [])
        runs_b = data.get("B", [])
        n = len(runs_a)
        acc_a = 100 * sum(1 for r in runs_a if r.get("correct")) / n if n else 0
        acc_b = 100 * sum(1 for r in runs_b if r.get("correct")) / n if n else 0
        delta = acc_b - acc_a

        if delta >= 5:
            verdict = "DELEGATION HELPS"
        elif delta <= -5:
            verdict = "DELEGATION HURTS"
        else:
            verdict = "NEUTRAL"
        verdicts.append((model, acc_a, acc_b, delta, verdict))
        marker = "✓✓" if delta > 0 else ("✗✗" if delta < 0 else "  ")
        print(f"  {model:<8} {acc_a:>9.1f}% {acc_b:>13.1f}% {delta:>+7.1f}pp  {verdict} {marker}")

    print("\n  Verdict meaning:")
    print("    DELEGATION HELPS: B > A by ≥5pp — multi-step reasoning benefits this model")
    print("    DELEGATION HURTS: B < A by ≥5pp — extra step introduces noise/errors")
    print("    NEUTRAL:          |Δ| < 5pp — delegation makes no difference")

    # Per-category breakdown
    print("\n" + "-"*65)
    print("  Per-Category Breakdown (which task types does delegation help/hurt?)")
    print("-"*65)

    categories = {"math": [t for t in TASKS if t["id"].startswith("math")],
                  "logic": [t for t in TASKS if t["id"].startswith("logic")],
                  "comp": [t for t in TASKS if t["id"].startswith("comp")],
                  "domain": [t for t in TASKS if t["id"].startswith("dk")],
                  "chain": [t for t in TASKS if t["id"].startswith("step")]}

    for cat, cat_tasks in categories.items():
        cat_ids = {t["id"] for t in cat_tasks}
        print(f"\n  [{cat.upper()}]")
        for model in args.models:
            data = all_results[model]
            runs_a = [r for r in data.get("A", []) if r["task_id"] in cat_ids]
            runs_b = [r for r in data.get("B", []) if r["task_id"] in cat_ids]
            n = len(runs_a)
            if n == 0:
                continue
            acc_a = 100 * sum(1 for r in runs_a if r.get("correct")) / n
            acc_b = 100 * sum(1 for r in runs_b if r.get("correct")) / n
            delta = acc_b - acc_a
            print(f"    {model}: A={acc_a:.0f}% B={acc_b:.0f}% Δ={delta:+.0f}pp")

    # Save aggregate
    agg_file = result_dir / "aggregate.json"
    with open(agg_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {agg_file}")

    # Write markdown report
    report = result_dir / "REPORT.md"
    with open(report, "w") as f:
        f.write("# Delegation Value Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("Does delegating to a same-model subagent help or hurt accuracy?\n\n")
        f.write("| Model | A (alone) | B (delegated) | Δ | Verdict |\n")
        f.write("|-------|-----------|---------------|---|--------|\n")
        for model, acc_a, acc_b, delta, verdict in verdicts:
            f.write(f"| {model} | {acc_a:.1f}% | {acc_b:.1f}% | {delta:+.1f}pp | {verdict} |\n\n")
        f.write("## Key Findings\n\n")
        for model, acc_a, acc_b, delta, verdict in verdicts:
            if delta > 5:
                f.write(f"- **{model}**: Delegation HELPS (+{delta:.1f}pp). Multi-step reasoning benefits this model.\n")
            elif delta < -5:
                f.write(f"- **{model}**: Delegation HURTS ({delta:.1f}pp). Extra step introduces noise.\n")
            else:
                f.write(f"- **{model}**: Delegation is NEUTRAL ({delta:+.1f}pp). No significant effect.\n")
        f.write("\n## Detailed Per-Category Results\n\n")
        for cat, cat_tasks in categories.items():
            f.write(f"### {cat.upper()}\n\n")
            f.write("| Model | A | B | Δ |\n|-------|---|---|---|\n")
            cat_ids = {t["id"] for t in cat_tasks}
            for model in args.models:
                data = all_results[model]
                runs_a = [r for r in data.get("A", []) if r["task_id"] in cat_ids]
                runs_b = [r for r in data.get("B", []) if r["task_id"] in cat_ids]
                n = len(runs_a)
                if n == 0:
                    continue
                acc_a = 100 * sum(1 for r in runs_a if r.get("correct")) / n
                acc_b = 100 * sum(1 for r in runs_b if r.get("correct")) / n
                delta = acc_b - acc_a
                f.write(f"| {model} | {acc_a:.0f}% | {acc_b:.0f}% | {delta:+.0f}pp |\n")
            f.write("\n")
    print(f"  Report: {report}")


if __name__ == "__main__":
    asyncio.run(main())
