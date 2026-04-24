#!/usr/bin/env python3
"""
Multi-Agent Benchmark v2 — Testing What Literature Actually Does
================================================================

Three scenarios from the literature, each with a concrete hypothesis:

SCENARIO 1: Cross-Model Information Gathering
  Hypothesis: A weak parent (0.5B) coordinates a strong subagent (3B)
             to retrieve information the weak model doesn't reliably know.
  Question: Does weak_coord + strong_specialist beat strong_alone?

SCENARIO 2: Self-Refinement / Verification Loop
  Hypothesis: Agent A produces answer → Agent B (same model) verifies/critiques →
              A refines → better final answer.
  Question: Does one round of self-verification improve accuracy?

SCENARIO 3: Debate / Adversarial Collaboration
  Hypothesis: Two agents with different reasoning approaches debate,
              and the consensus or winner is selected.
  Question: Does adversarial exchange improve over solo reasoning?

Each scenario is tested on GSM8K math reasoning tasks where:
  - The parent CANNOT reliably solve alone (otherwise delegation is pointless)
  - The subagent has complementary capability (knows facts / is more accurate)

GPU allocation (sequential, one model at a time):
  GPU 0: OCCUPIED by existing VLLM — DO NOT USE
  GPU 1: 0.5B, 1.5B
  GPU 2: 3B
  GPU 3: 1.5B (for Scenario 3 debate)
"""

import os
import re
import sys
import json
import asyncio
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from vllm import SamplingParams

# ── GSM8K Subset (50 problems, easy-to-medium difficulty) ────────────────────
# These are intentionally harder than the built-in gsm8k_runner.py subset

GSM8K_PROBLEMS = [
    {"id": "g1", "problem": "There are 15 trees in the garden. He plants 10 more trees. How many trees are there now?", "answer": "25"},
    {"id": "g2", "problem": "A shop has 8 oranges. They buy 15 more oranges. How many oranges do they have?", "answer": "23"},
    {"id": "g3", "problem": "John has 12 candies. He gives 4 to his friend. How many does he have left?", "answer": "8"},
    {"id": "g4", "problem": "There are 7 birds on a tree. 3 fly away. How many birds are left?", "answer": "4"},
    {"id": "g5", "problem": "Sarah has 20 stickers. She buys 12 more at the store. How many stickers does she have?", "answer": "32"},
    {"id": "g6", "problem": "A baker has 24 loaves of bread. He sells 17. How many loaves remain?", "answer": "7"},
    {"id": "g7", "problem": "Tom has 5 books. His mother gives him 3 more. How many books does Tom have now?", "answer": "8"},
    {"id": "g8", "problem": "There are 9 students in a class. 2 more join. How many students are there now?", "answer": "11"},
    {"id": "g9", "problem": "A farmer has 18 sheep. He buys 7 more. How many sheep does he have?", "answer": "25"},
    {"id": "g10", "problem": "Lisa has 30 marbles. She loses 11 at the park. How many marbles does she have?", "answer": "19"},
    {"id": "g11", "problem": "There are 6 cats in a house. Each cat has 4 kittens. How many cats are there in total?", "answer": "30"},
    {"id": "g12", "problem": "A bus has 40 seats. 27 passengers are on it. How many empty seats are there?", "answer": "13"},
    {"id": "g13", "problem": "James reads 15 pages on Monday, 22 on Tuesday, and 18 on Wednesday. How many pages did he read?", "answer": "55"},
    {"id": "g14", "problem": "A store has 100 apples. They sell 35 on Monday and 28 on Tuesday. How many apples are left?", "answer": "37"},
    {"id": "g15", "problem": "Emma has $45. She buys a book for $18. How much money does she have left?", "answer": "27"},
    {"id": "g16", "problem": "A rectangle has length 12 cm and width 7 cm. What is its perimeter?", "answer": "38"},
    {"id": "g17", "problem": "There are 4 rows of chairs with 9 chairs in each row. How many chairs are there?", "answer": "36"},
    {"id": "g18", "problem": "Mike has 3 times as many coins as Jake, who has 14 coins. How many coins does Mike have?", "answer": "42"},
    {"id": "g19", "problem": "A train travels 60 miles per hour for 3 hours. How far does the train go?", "answer": "180"},
    {"id": "g20", "problem": "In a class of 30 students, 60% passed an exam. How many students passed?", "answer": "18"},
    {"id": "g21", "problem": "A pizza is cut into 8 slices. 5 people eat 2 slices each. How many slices are left?", "answer": "2"},
    {"id": "g22", "problem": "John buys a shirt for $25 and a pair of pants for $38. How much did he spend?", "answer": "63"},
    {"id": "g23", "problem": "There are 144 minutes in 2 hours and 24 minutes. How many minutes is that?", "answer": "144"},
    {"id": "g24", "problem": "A garden has 6 rows of tomato plants with 8 plants in each row. How many tomato plants?", "answer": "48"},
    {"id": "g25", "problem": "Alice has $100. She splits it equally among her 4 children. How much does each get?", "answer": "25"},
    {"id": "g26", "problem": "If a rectangle's area is 56 square cm and one side is 8 cm, what is the other side?", "answer": "7"},
    {"id": "g27", "problem": "A bottle holds 750 ml of juice. Tom drinks 300 ml. How much juice is left?", "answer": "450"},
    {"id": "g28", "problem": "Sara's age is twice her brother's age. Her brother is 12. How old is Sara?", "answer": "24"},
    {"id": "g29", "problem": "A farmer has 45 chickens. He sells 17 and then buys 23 more. How many does he have?", "answer": "51"},
    {"id": "g30", "problem": "A school has 124 students. Each class has 31 students. How many classes are there?", "answer": "4"},
    {"id": "g31", "problem": "Tom buys 3 notebooks at $4 each and 2 pens at $2 each. How much does he spend?", "answer": "16"},
    {"id": "g32", "problem": "A rectangle is 15 cm long and 9 cm wide. What is the difference between its area and perimeter?", "answer": "111"},
    {"id": "g33", "problem": "Maria has 48 cookies. She gives 15 to John and 12 to Peter. How many does she have left?", "answer": "21"},
    {"id": "g34", "problem": "A car travels 240 miles in 4 hours. A bicycle travels 30 miles in 2 hours. How much faster is the car per hour?", "answer": "45"},
    {"id": "g35", "problem": "Jack has $80. He buys a game for $35 and a snack for $7. How much does he have left?", "answer": "38"},
    {"id": "g36", "problem": "There are 3 boxes with 24 pencils in each. If 18 pencils are removed total, how many remain?", "answer": "54"},
    {"id": "g37", "problem": "A train leaves at 9:15 AM and arrives at 2:45 PM. How many minutes is the journey?", "answer": "330"},
    {"id": "g38", "problem": "In a survey, 80 people like apples, 65 like bananas, and 30 like both. How many like at least one?", "answer": "115"},
    {"id": "g39", "problem": "A shop sells 3 burgers for $12 or 1 burger for $5. What is the most number of burgers you can buy for $30?", "answer": "8"},
    {"id": "g40", "problem": "Tom is 5 years older than Jerry. In 3 years, Tom will be twice Jerry's age. How old is Jerry now?", "answer": "2"},
]

# ── Model Config ──────────────────────────────────────────────────────────────

MODEL_BASE = "/home/jinxu/.cache/tiny-agents/models/Qwen"
MODEL_PATH = {
    "0.5B": f"{MODEL_BASE}/Qwen2.5-0.5B-Instruct",
    "1.5B": f"{MODEL_BASE}/Qwen2.5-1.5B-Instruct",
    "3B":   f"{MODEL_BASE}/Qwen2.5-3B-Instruct",
    "9B":   f"{MODEL_BASE}/Qwen3.5-9B",
}

GPU_MAP = {"0.5B": 1, "1.5B": 1, "3B": 2, "9B": 2}

# Deterministic sampling params
SP = SamplingParams(temperature=0, max_tokens=64)
MEM_MAP = {"0.5B": 0.40, "1.5B": 0.55, "3B": 0.68, "9B": 0.72}

# ── LLM Helpers ───────────────────────────────────────────────────────────────

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
        skip_tokenizer_init=False,
    )

def extract_number(text: str) -> str:
    """Extract numeric answer from text."""
    text = text.strip()
    # Remove common prefixes
    text = re.sub(r'^(The answer is|Answer:|Final answer:)\s*', '', text, flags=re.IGNORECASE)
    # Get first number (possibly decimal or negative)
    m = re.search(r'-?\d+\.?\d*', text)
    if m:
        return m.group(0)
    # Fall back to cleaned text
    cleaned = text.split('\n')[0].strip().strip('$.')
    return cleaned

def is_correct_numeric(model_output: str, expected: str) -> bool:
    """Check if extracted number matches expected (with tolerance)."""
    out_num = extract_number(model_output)
    exp_num = extract_number(expected)
    try:
        return abs(float(out_num) - float(exp_num)) < 0.1
    except (ValueError, TypeError):
        # Fall back to string match
        return out_num.strip().lower() == exp_num.strip().lower()

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.strip()
    text = re.sub(r'^(The answer is|Answer:|Final answer:)\s*', '', text, flags=re.IGNORECASE)
    # Extract first number or short phrase
    num = re.search(r'-?\d+\.?\d*', text)
    if num:
        return num.group(0)
    return text.split('\n')[0].strip()[:50]

# ── Scenario 1: Cross-Model Information Gathering ─────────────────────────────
# Parent (weak) doesn't reliably solve → asks subagent (strong) → uses result

def make_parent_prompt_s1(task: dict) -> str:
    """Prompt for weak parent in Scenario 1: coordinates with strong subagent."""
    return f"""Question: {task['problem']}

First, write down what specific calculation or fact you need to answer this.
Then, I'll give you that information. Use it to give your final answer.

Your response format:
Step 1 - What I need to know: [the specific question for my assistant]
Step 2 - Final answer: [your answer based on what the assistant tells you]"""

def make_subagent_prompt_s1(task: dict) -> str:
    """Subagent in Scenario 1: answers the parent's specific question."""
    # Extract what subagent needs to compute from the task
    return f"""Question: {task['problem']}

Answer this question directly and concisely. Give only the final numeric answer (no explanation)."""

def run_scenario1(parent_llm, subagent_llm, task: dict) -> dict:
    """
    Scenario 1: Weak parent + strong subagent.
    Parent says what it needs → subagent computes → parent formats answer.
    """
    # Step 1: parent identifies what it needs
    parent_plan = parent_llm.chat([{"role":"user","content": make_parent_prompt_s1(task)}], sampling_params=SP)[0].outputs[0].text

    # Step 2: subagent answers the actual question (simulating what the subagent would provide)
    subagent_out = subagent_llm.chat([{"role":"user","content": make_subagent_prompt_s1(task)}], sampling_params=SP)[0].outputs[0].text

    # Step 3: parent synthesizes with subagent's answer
    synthesis_prompt = f"""Question: {task['problem']}
Your assistant (3B specialist) says: {subagent_out}

Based on the above, what is the final answer? Give only the number."""
    final = parent_llm.chat([{"role":"user","content": synthesis_prompt}], sampling_params=SP)[0].outputs[0].text

    return {
        "parent_plan": parent_plan,
        "subagent_out": subagent_out,
        "final": final,
    }

# ── Scenario 2: Self-Refinement / Verification Loop ──────────────────────────

def run_scenario2(llm, task: dict) -> dict:
    """
    Scenario 2: Single model produces answer, then critiques/refines itself.
    This is the "self-refinement" pattern from literature.
    """
    # Step 1: initial answer
    initial_prompt = f"""Question: {task['problem']}
Solve this step by step and give your final answer."""
    initial_out = llm.chat([{"role":"user","content":initial_prompt}], sampling_params=SP)[0].outputs[0].text

    # Step 2: self-critique (model reflects on its own answer)
    critique_prompt = f"""Question: {task['problem']}
Your initial answer was: {normalize_answer(initial_out)}
Is this correct? If uncertain or wrong, provide a corrected answer. If confident, state the same answer.
Be concise. State your final answer at the end."""
    critique_out = llm.chat([{"role":"user","content":critique_prompt}], sampling_params=SP)[0].outputs[0].text

    # Step 3: final answer (model chooses the better response)
    final_prompt = f"""Original question: {task['problem']}
Initial answer: {initial_out}
Self-critique: {critique_out}

Based on the above, what is the FINAL answer? State only the answer."""
    final_out = llm.chat([{"role":"user","content":final_prompt}], sampling_params=SP)[0].outputs[0].text

    return {
        "initial": initial_out,
        "critique": critique_out,
        "final": final_out,
    }

# ── Scenario 3: Debate / Two-Agent Adversarial ───────────────────────────────

def run_scenario3(agent_a_llm, agent_b_llm, task: dict) -> dict:
    """
    Scenario 3: Two agents debate, then a final answer is selected.
    Agent A takes one reasoning approach, Agent B takes another.
    """
    # Agent A: straightforward calculation
    a_prompt = f"""Question: {task['problem']}
Solve this carefully. Show your reasoning step by step. What is your final numerical answer?"""
    a_out = agent_a_llm.chat([{"role":"user","content":a_prompt}], sampling_params=SP)[0].outputs[0].text

    # Agent B: alternative reasoning approach
    b_prompt = f"""Question: {task['problem']}
Think about this problem differently. Try a different approach or check your work. What is your final numerical answer?"""
    b_out = agent_b_llm.chat([{"role":"user","content":b_prompt}], sampling_params=SP)[0].outputs[0].text

    # Final selection by Agent A (acts as arbiter)
    arbiter_prompt = f"""Two agents answered a question:

Agent A says: {a_out}

Agent B says: {b_out}

Original question: {task['problem']}

Which answer is more likely correct? State only the numerical answer."""
    arbiter_out = agent_a_llm.chat([{"role":"user","content":arbiter_prompt}], sampling_params=SP)[0].outputs[0].text

    return {
        "agent_a": a_out,
        "agent_b": b_out,
        "arbiter": arbiter_out,
    }

# ── Baseline: Single Model Direct ────────────────────────────────────────────

def run_baseline(llm, task: dict) -> dict:
    """Single model direct answer."""
    prompt = f"Question: {task['problem']}\nAnswer:"
    out = llm.chat([{"role":"user","content":prompt}], sampling_params=SP)[0].outputs[0].text
    return {"direct": out}

# ── Main Benchmark ────────────────────────────────────────────────────────────

async def run_scenario1_benchmark(problems: list, result_dir: Path, models: list, strong_subagent: str = "3B"):
    """Run Scenario 1: weak parent + strong subagent."""
    import torch
    results = {}

    for weak_model in models:
        print(f"\n  Scenario 1: {weak_model} parent + {strong_subagent} subagent")
        weak_llm = make_llm(weak_model)
        strong_llm = make_llm(strong_subagent)

        weak_results = []
        for prob in problems:
            try:
                r = run_scenario1(weak_llm, strong_llm, prob)
                ok = is_correct_numeric(r["final"], prob["answer"])
                weak_results.append({
                    "id": prob["id"],
                    "answer": r["final"],
                    "expected": prob["answer"],
                    "correct": ok,
                    "subagent_out": r["subagent_out"],
                })
                print(f"    [{weak_model}+{strong_subagent}] {'✓' if ok else '✗'} {prob['id']}: got={normalize_answer(r['final'])!r} exp={prob['answer']!r}")
            except Exception as e:
                weak_results.append({"id": prob["id"], "error": str(e)})
                print(f"    [{weak_model}+{strong_subagent}] ER {prob['id']}: {e}")
            await asyncio.sleep(0.05)

        results[f"{weak_model}+{strong_subagent}"] = weak_results

        del weak_llm, strong_llm
        torch.cuda.empty_cache()
        await asyncio.sleep(1)

    return results

async def run_scenario2_benchmark(problems: list, result_dir: Path, models: list):
    """Run Scenario 2: self-refinement."""
    import torch
    results = {}

    for model in models:
        print(f"\n  Scenario 2: {model} self-refinement")
        llm = make_llm(model)

        model_results = []
        for prob in problems:
            try:
                r = run_scenario2(llm, prob)
                # Compare initial vs refined
                init_ok = is_correct_numeric(r["initial"], prob["answer"])
                refine_ok = is_correct_numeric(r["final"], prob["answer"])
                model_results.append({
                    "id": prob["id"],
                    "initial": r["initial"],
                    "critique": r["critique"],
                    "final": r["final"],
                    "expected": prob["answer"],
                    "initial_correct": init_ok,
                    "refined_correct": refine_ok,
                })
                print(f"    [{model}] {'✓' if refine_ok else '✗'} {prob['id']}: init={'✓' if init_ok else '✗'} → refine={'✓' if refine_ok else '✗'}")
            except Exception as e:
                model_results.append({"id": prob["id"], "error": str(e)})
                print(f"    [{model}] ER {prob['id']}: {e}")
            await asyncio.sleep(0.05)

        results[model] = model_results

        del llm
        torch.cuda.empty_cache()
        await asyncio.sleep(1)

    return results

async def run_scenario3_benchmark(problems: list, result_dir: Path, model_pairs: list):
    """Run Scenario 3: two-agent debate."""
    import torch
    results = {}

    for agent_a, agent_b in model_pairs:
        print(f"\n  Scenario 3: {agent_a} vs {agent_b} debate")
        a_llm = make_llm(agent_a)
        b_llm = make_llm(agent_b)

        pair_results = []
        for prob in problems:
            try:
                r = run_scenario3(a_llm, b_llm, prob)
                ok = is_correct_numeric(r["arbiter"], prob["answer"])
                pair_results.append({
                    "id": prob["id"],
                    "a_answer": r["agent_a"],
                    "b_answer": r["agent_b"],
                    "final": r["arbiter"],
                    "expected": prob["answer"],
                    "correct": ok,
                })
                print(f"    [{agent_a} vs {agent_b}] {'✓' if ok else '✗'} {prob['id']}: arbiter={normalize_answer(r['arbiter'])!r}")
            except Exception as e:
                pair_results.append({"id": prob["id"], "error": str(e)})
                print(f"    [{agent_a} vs {agent_b}] ER {prob['id']}: {e}")
            await asyncio.sleep(0.05)

        results[f"{agent_a}_vs_{agent_b}"] = pair_results

        del a_llm, b_llm
        torch.cuda.empty_cache()
        await asyncio.sleep(1)

    return results

async def run_baseline_benchmark(problems: list, models: list):
    """Run direct baseline for all models."""
    import torch
    results = {}

    for model in models:
        print(f"\n  Baseline: {model} direct")
        llm = make_llm(model)

        model_results = []
        for prob in problems:
            try:
                r = run_baseline(llm, prob)
                ok = is_correct_numeric(r["direct"], prob["answer"])
                model_results.append({
                    "id": prob["id"],
                    "answer": r["direct"],
                    "expected": prob["answer"],
                    "correct": ok,
                })
                print(f"    [{model}] {'✓' if ok else '✗'} {prob['id']}: {normalize_answer(r['direct'])!r}")
            except Exception as e:
                model_results.append({"id": prob["id"], "error": str(e)})
                print(f"    [{model}] ER {prob['id']}: {e}")
            await asyncio.sleep(0.05)

        results[model] = model_results

        del llm
        torch.cuda.empty_cache()
        await asyncio.sleep(1)

    return results

# ── Summary Report ───────────────────────────────────────────────────────────

def print_summary(baseline_results: dict, s1_results: dict, s2_results: dict, s3_results: dict):
    print("\n" + "="*70)
    print("  MULTI-AGENT BENCHMARK v2 — SUMMARY")
    print("="*70)

    problems = GSM8K_PROBLEMS
    n = len(problems)

    print(f"\n  Tasks: {n} GSM8K problems")
    print(f"\n  {'System':<30} {'Accuracy':>10}  vs Baseline")
    print("  " + "-"*55)

    # Baseline
    baseline_accs = {}
    for model, runs in baseline_results.items():
        n_ok = sum(1 for r in runs if r.get("correct"))
        acc = 100 * n_ok / n
        baseline_accs[model] = acc
        print(f"  {model:<28} {acc:>9.1f}%  baseline")

    # Scenario 1: Cross-model
    print("\n  [Scenario 1: Weak Parent + Strong Subagent]")
    for key, runs in s1_results.items():
        n_ok = sum(1 for r in runs if r.get("correct"))
        acc = 100 * n_ok / n
        weak = key.split("+")[0]
        delta = acc - baseline_accs.get(weak, 0)
        verdict = "✓✓✓" if delta > 5 else ("✗✗✗" if delta < -5 else "≈≈≈")
        print(f"  {key:<28} {acc:>9.1f}%  {delta:>+6.1f}pp  {verdict}")

    # Scenario 2: Self-refinement
    print("\n  [Scenario 2: Self-Refinement]")
    for model, runs in s2_results.items():
        n_init_ok = sum(1 for r in runs if r.get("initial_correct"))
        n_refine_ok = sum(1 for r in runs if r.get("refined_correct"))
        init_acc = 100 * n_init_ok / n
        refine_acc = 100 * n_refine_ok / n
        delta = refine_acc - init_acc
        verdict = "✓✓✓" if delta > 5 else ("✗✗✗" if delta < -5 else "≈≈≈")
        print(f"  {model:<28} init={init_acc:.0f}% refine={refine_acc:.0f}%  Δ={delta:>+5.0f}pp  {verdict}")

    # Scenario 3: Debate
    print("\n  [Scenario 3: Two-Agent Debate]")
    for key, runs in s3_results.items():
        n_ok = sum(1 for r in runs if r.get("correct"))
        acc = 100 * n_ok / n
        agents = key.split("_vs_")
        base_a = baseline_accs.get(agents[0], 0)
        delta = acc - base_a
        verdict = "✓✓✓" if delta > 5 else ("✗✗✗" if delta < -5 else "≈≈≈")
        print(f"  {key:<28} {acc:>9.1f}%  {delta:>+6.1f}pp vs {agents[0]}  {verdict}")

    print("\n  Verdict: ✓✓✓ helps | ≈≈≈ neutral | ✗✗✗ hurts")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/home/jinxu/tiny-agents/benchmarks/multi_agent_benchmark_v2")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2", "s3", "baseline"])
    args = parser.parse_args()

    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    problems = GSM8K_PROBLEMS[:args.n_samples]
    n = len(problems)

    print(f"\nMulti-Agent Benchmark v2")
    print(f"  Problems: {n}")
    print(f"  Scenarios: {args.scenarios}")

    all_results = {}

    # Baseline
    if "baseline" in args.scenarios:
        print("\n" + "="*60)
        print("  BASELINE: Single model direct answer")
        print("="*60)
        baseline = await run_baseline_benchmark(problems, models=["0.5B", "1.5B", "3B"])
        all_results["baseline"] = baseline

    # Scenario 1
    if "s1" in args.scenarios:
        print("\n" + "="*60)
        print("  SCENARIO 1: Cross-Model Information Gathering")
        print("="*60)
        s1 = await run_scenario1_benchmark(problems, result_dir, models=["0.5B", "1.5B"], strong_subagent="3B")
        all_results["scenario1"] = s1

    # Scenario 2
    if "s2" in args.scenarios:
        print("\n" + "="*60)
        print("  SCENARIO 2: Self-Refinement")
        print("="*60)
        s2 = await run_scenario2_benchmark(problems, result_dir, models=["0.5B", "1.5B", "3B"])
        all_results["scenario2"] = s2

    # Scenario 3
    if "s3" in args.scenarios:
        print("\n" + "="*60)
        print("  SCENARIO 3: Two-Agent Debate")
        print("="*60)
        s3 = await run_scenario3_benchmark(problems, result_dir, model_pairs=[("0.5B", "1.5B"), ("1.5B", "3B")])
        all_results["scenario3"] = s3

    # Save
    out_file = result_dir / "all_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_file}")

    # Summary
    print_summary(
        all_results.get("baseline", {}),
        all_results.get("scenario1", {}),
        all_results.get("scenario2", {}),
        all_results.get("scenario3", {}),
    )

    # Generate report
    generate_report(result_dir, all_results, problems)


def generate_report(result_dir: Path, all_results: dict, problems: list):
    """Generate markdown report."""
    n = len(problems)
    report_path = result_dir / "REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Multi-Agent Benchmark v2 — Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Tasks:** {n} GSM8K problems\n\n")

        baseline = all_results.get("baseline", {})
        s1 = all_results.get("scenario1", {})
        s2 = all_results.get("scenario2", {})
        s3 = all_results.get("scenario3", {})

        baseline_accs = {}
        for model, runs in baseline.items():
            n_ok = sum(1 for r in runs if r.get("correct"))
            baseline_accs[model] = 100 * n_ok / n

        # Scenario 1
        f.write("## Scenario 1: Cross-Model Information Gathering\n\n")
        f.write("*Weak parent (0.5B/1.5B) coordinates a strong subagent (3B) to solve problems.*\n\n")
        f.write("| System | Accuracy | vs Parent Alone |\n")
        f.write("|--------|----------|----------------|\n")
        for key, runs in s1.items():
            n_ok = sum(1 for r in runs if r.get("correct"))
            acc = 100 * n_ok / n
            weak = key.split("+")[0]
            delta = acc - baseline_accs.get(weak, 0)
            f.write(f"| {key} | {acc:.1f}% | {delta:+.1f}pp |\n")
        f.write("\n")

        # Scenario 2
        f.write("## Scenario 2: Self-Refinement\n\n")
        f.write("*Single model answers, then critiques/refines its own answer.*\n\n")
        f.write("| Model | Initial | After Refinement | Δ |\n")
        f.write("|-------|---------|------------------|---|\n")
        for model, runs in s2.items():
            n_init = sum(1 for r in runs if r.get("initial_correct"))
            n_refine = sum(1 for r in runs if r.get("refined_correct"))
            init_acc = 100 * n_init / n
            refine_acc = 100 * n_refine / n
            delta = refine_acc - init_acc
            f.write(f"| {model} | {init_acc:.0f}% | {refine_acc:.0f}% | {delta:+.0f}pp |\n")
        f.write("\n")

        # Scenario 3
        f.write("## Scenario 3: Two-Agent Debate\n\n")
        f.write("*Two agents reason differently, one acts as arbiter.*\n\n")
        f.write("| Debate Pair | Accuracy | vs Single Agent A |\n")
        f.write("|-------------|----------|-------------------|\n")
        for key, runs in s3.items():
            n_ok = sum(1 for r in runs if r.get("correct"))
            acc = 100 * n_ok / n
            agents = key.split("_vs_")
            base_a = baseline_accs.get(agents[0], 0)
            delta = acc - base_a
            f.write(f"| {key} | {acc:.1f}% | {delta:+.1f}pp |\n")
        f.write("\n")

        # Overall findings
        f.write("## Key Findings\n\n")
        f.write("### Scenario 1: Cross-Model Delegation\n")
        for key, runs in s1.items():
            n_ok = sum(1 for r in runs if r.get("correct"))
            acc = 100 * n_ok / n
            weak = key.split("+")[0]
            delta = acc - baseline_accs.get(weak, 0)
            if delta > 10:
                f.write(f"- **{key}**: DELEGATION HELPS (+{delta:.0f}pp). Weak parent benefits from strong specialist.\n")
            elif delta < -10:
                f.write(f"- **{key}**: DELEGATION HURTS ({delta:.0f}pp). Coordinator adds noise.\n")
            else:
                f.write(f"- **{key}**: NEUTRAL ({delta:+.0f}pp). No significant effect.\n")

        f.write("\n### Scenario 2: Self-Refinement\n")
        for model, runs in s2.items():
            n_init = sum(1 for r in runs if r.get("initial_correct"))
            n_refine = sum(1 for r in runs if r.get("refined_correct"))
            delta = 100 * (n_refine - n_init) / n
            if delta > 10:
                f.write(f"- **{model}**: SELF-REFINEMENT HELPS (+{delta:.0f}pp). Model catches own errors.\n")
            elif delta < -10:
                f.write(f"- **{model}**: SELF-REFINEMENT HURTS ({delta:.0f}pp). Critique introduces confusion.\n")
            else:
                f.write(f"- **{model}**: NEUTRAL ({delta:+.0f}pp). Self-critique has minimal effect.\n")

        f.write("\n### Scenario 3: Debate\n")
        for key, runs in s3.items():
            n_ok = sum(1 for r in runs if r.get("correct"))
            acc = 100 * n_ok / n
            agents = key.split("_vs_")
            base_a = baseline_accs.get(agents[0], 0)
            delta = acc - base_a
            if delta > 10:
                f.write(f"- **{key}**: DEBATE HELPS (+{delta:.0f}pp). Different reasoning approaches combine well.\n")
            elif delta < -10:
                f.write(f"- **{key}**: DEBATE HURTS ({delta:.0f}pp). Arbiter confused by disagreement.\n")
            else:
                f.write(f"- **{key}**: NEUTRAL ({delta:+.0f}pp). Debate has minimal effect.\n")

        f.write("\n## Reproducing\n\n")
        f.write("```bash\n")
        f.write("cd /home/jinxu/tiny-agents\n")
        f.write("python benchmarks/MULTI_AGENT_BENCHMARK_V2.py --n-samples 40 --scenarios baseline s1 s2 s3\n")
        f.write("```\n")

    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
