"""
Multi-Hop QA Goal Alignment Benchmark — V3
==========================================

Purpose: Isolate GOAL MISALIGNMENT from the "extra reasoning step" confound.

Design — THREE modes:
  A (No-Delegate):   Single 3B answers end-to-end, NO subagent context.
  B (Real Delegate): 3B parent spawns 3B subagent → gets real output → answers.
  C (Oracle Delegate): Single 3B gets ORACLE (ground-truth) subagent output → answers.

If B < C: goal misalignment exists (parent can't use subagent output properly)
If B = C: all loss comes from extra step, not goal misalignment
If C < A: even oracle subagent output hurts (extra step is harmful)
"""

import os
import re
import sys
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_BASE = "/home/jinxu/.cache/tiny-agents/models/Qwen"
MODEL_PATH = {
    "0.5B": f"{MODEL_BASE}/Qwen2.5-0.5B-Instruct",
    "1.5B": f"{MODEL_BASE}/Qwen2.5-1.5B-Instruct",
    "3B":   f"{MODEL_BASE}/Qwen2.5-3B-Instruct",
}

PARENT_GPU   = {"3B": 2}
SUBAGENT_GPU = {"3B": 2}
MEM_CONFIG = {"3B": 0.68}

# ── Dataset ───────────────────────────────────────────────────────────────────
#
# Each case has:
#   question:         the full question (parent's actual task)
#   answer:          ground-truth final answer
#   subagent_q:      what parent delegates to subagent
#   subagent_a:      ground-truth for subagent question
#   oracle_output:    the ORACLE subagent answer (ground-truth, Mode C uses this)
#   misalignment_type: what TYPE of misalignment this case tests
#   why:             one-line mechanism explanation
#
# Cases are designed so that:
#   - Mode A (no subagent): single model answers directly
#   - Mode C (oracle): single model uses oracle subagent output, same prompt as parent
#   - Mode B (real): parent uses real subagent output (noisy/imperfect)
#
# The comparison C vs B isolates goal misalignment.
# The comparison A vs C isolates the "extra step" effect.

MISALIGNMENT_CASES = [
    {
        "id": "gm01",
        "question": "Python was created by Guido van Rossum and first released in 1991. Java was created by James Gosling and first released in 1995. Which was released first?",
        "answer": "Python",
        "subagent_q": "In what year was Java first released? Answer with just the year number.",
        "subagent_a": "1995",
        "oracle_output": "1995",
        "misalignment_type": "irrelevant_delegation",
        "why": "Parent could answer from its own context (Python=1991), but delegates. Subagent tells 1995. No real misalignment here.",
    },
    {
        "id": "gm02",
        "question": "Mount Everest is 8848 meters tall. K2 is 8611 meters tall. Which is taller? Answer with just the name.",
        "answer": "Mount Everest",
        "subagent_q": "What is the height of K2 in meters? Give the height as a plain number only.",
        "subagent_a": "8611",
        "oracle_output": "8611",
        "misalignment_type": "format_robustness",
        "why": "Parent can read 8848 from question. Subagent gives 8611. Parent compares 8848 vs 8611 correctly.",
    },
    {
        "id": "gm03",
        "question": "The speed of light is 299,792 km/s. The speed of sound is 343 m/s. Which is faster?",
        "answer": "speed of light",
        "subagent_q": "What is the speed of sound in km/s? Answer with just the number.",
        "subagent_a": "0.343",
        "oracle_output": "0.343",
        "misalignment_type": "unit_conversion",
        "why": "Subagent converts 343 m/s to 0.343 km/s. Parent compares 299792 vs 0.343. Parent might misread 0.343 as 343.",
    },
    {
        "id": "gm04",
        "question": "Which letter appears more times in 'Strawberry': 'r' or 'b'?",
        "answer": "r",
        "subagent_q": "How many times does the letter 'r' appear in 'Strawberry'? Answer with just the number.",
        "subagent_a": "3",
        "oracle_output": "3",
        "misalignment_type": "partial_only",
        "why": "Subagent gives r=3. Parent must still count b (2). But parent might forget and answer based only on subagent.",
    },
    {
        "id": "gm05",
        "question": "Which is longer: the word 'Wednesday' (9 letters) or the word 'Thursday' (8 letters)?",
        "answer": "Wednesday",
        "subagent_q": "How many letters does 'Wednesday' have? Answer with just the number.",
        "subagent_a": "9",
        "oracle_output": "9",
        "misalignment_type": "limited_delegation",
        "why": "Parent asks Wednesday's length only. Subagent says 9. Parent doesn't know Thursday=8 without asking. Parent might say Wednesday based on partial info.",
    },
    {
        "id": "gm06",
        "question": "Is 77 a prime number? Answer yes or no.",
        "answer": "no",
        "subagent_q": "Is 77 divisible by any number other than 1 and itself? Answer yes or no.",
        "subagent_a": "yes",
        "oracle_output": "yes",
        "misalignment_type": "context_collapse",
        "why": "Subagent says 'yes' (77=7×11). Parent needs to answer 'no'. The yes/no framing collapses the reasoning.",
    },
    {
        "id": "gm07",
        "question": "The Great Wall was first built in the 7th century BC. The Colosseum was built in 72 AD. Which is older?",
        "answer": "Great Wall of China",
        "subagent_q": "In what century was the Great Wall first built? Answer with the century like '7th century BC'.",
        "subagent_a": "7th century BC",
        "oracle_output": "7th century BC",
        "misalignment_type": "cross_era_comparison",
        "why": "Subagent gives '7th century BC'. Parent must compare with 72 AD. Cross-era comparison is non-trivial and error-prone.",
    },
    {
        "id": "gm08",
        "question": "Albert Einstein was born in 1879. Marie Curie was born in 1867. Who was born earlier?",
        "answer": "Marie Curie",
        "subagent_q": "In what year was Marie Curie born? Answer with just the year.",
        "subagent_a": "1867",
        "oracle_output": "1867",
        "misalignment_type": "parent_redundant",
        "why": "Parent already knows Einstein=1879 from question. Subagent says 1867. Parent CAN compare. But with oracle (1867) vs real (maybe wrong), Mode B differs from Mode C.",
    },
    {
        "id": "gm09",
        "question": "Which is greater: the number 9 or the number 7?",
        "answer": "9",
        "subagent_q": "Is 9 greater than 7? Answer yes or no.",
        "subagent_a": "yes",
        "oracle_output": "yes",
        "misalignment_type": "redundant_delegation",
        "why": "Parent already knows 9>7. Subagent confirms yes. With oracle=YES, Mode C might answer 'yes' instead of '9'. With real subagent might give wrong yes/no.",
    },
    {
        "id": "gm10",
        "question": "The Amazon River is about 6400 km long. The Nile River is about 6650 km long. Which is longer?",
        "answer": "Nile River",
        "subagent_q": "What is the length of the Amazon River in km? Answer with just the number.",
        "subagent_a": "6400",
        "oracle_output": "6400",
        "misalignment_type": "partial_knowledge",
        "why": "Subagent gives Amazon=6400. Parent has Nile=6650 in question. Oracle mode compares correctly. Real subagent might give wrong number.",
    },
    {
        "id": "gm11",
        "question": "Python was created by Guido van Rossum in 1991. Java was created by James Gosling in 1995. Which programming language was created first?",
        "answer": "Python",
        "subagent_q": "In what year was Java created? Answer with just the year.",
        "subagent_a": "1995",
        "oracle_output": "1995",
        "misalignment_type": "direct_vs_delegate",
        "why": "Both years in question. Parent could answer directly. Delegation adds noise. Oracle=subagent gives 1995. Real subagent might say wrong year.",
    },
    {
        "id": "gm12",
        "question": "Leonardo da Vinci was born in 1452. Michelangelo was born in 1475. Who was born first?",
        "answer": "Leonardo da Vinci",
        "subagent_q": "In what year was Michelangelo born? Answer with just the year.",
        "subagent_a": "1475",
        "oracle_output": "1475",
        "misalignment_type": "info_in_question",
        "why": "Parent knows 1452 from question. Subagent says 1475. With oracle, parent can compare. With real subagent, might say wrong year.",
    },
]

# ── Experiment Runner ─────────────────────────────────────────────────────────

class ExperimentRunner:
    def __init__(self):
        from tiny_agents.models.vllm_backend import VLLMBackend
        self.backend = VLLMBackend(default_gpu=1)
        self._loaded = set()

    def load(self, model_key: str, gpu_id: int):
        key = f"{model_key}_gpu{gpu_id}"
        if key in self._loaded:
            return
        print(f"  [Load {model_key} on GPU {gpu_id}]")
        self.backend.load_model(
            model_key=key,
            model_name=MODEL_PATH[model_key],
            gpu=gpu_id,
            max_model_len=2048,
            gpu_memory_utilization=MEM_CONFIG[model_key],
        )
        self._loaded.add(key)

    def generate(self, model_key: str, gpu_id: int, messages,
                 temperature=0.0, max_tokens=128) -> str:
        if model_key not in self._loaded:
            self.load(model_key, gpu_id)
        return self.backend.generate(
            model_key=f"{model_key}_gpu{gpu_id}",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

# ── Mode A: No-Delegate baseline ─────────────────────────────────────────────
# Single 3B answers directly, NO subagent context.

def run_mode_a(runner: ExperimentRunner, case: dict) -> dict:
    prompt = f"""Answer the following question. Output only the answer, nothing else.

Question: {case['question']}

Answer: """

    runner.load("3B", PARENT_GPU["3B"])
    output = runner.generate("3B", PARENT_GPU["3B"],
                           [{"role": "user", "content": prompt}],
                           temperature=0.0, max_tokens=64)

    return {
        "mode": "A_no_delegate",
        "model": "3B",
        "case_id": case["id"],
        "question": case["question"],
        "expected_answer": case["answer"],
        "model_output": output,
        "misalignment_type": case["misalignment_type"],
    }

# ── Mode B: Real Delegate ─────────────────────────────────────────────────────
# 3B parent spawns 3B subagent → gets real subagent output → produces final answer

def run_mode_b(runner: ExperimentRunner, case: dict) -> dict:
    # Step 1: Real subagent
    runner.load("3B", SUBAGENT_GPU["3B"])
    subagent_output = runner.generate(
        "3B", SUBAGENT_GPU["3B"],
        [{"role": "user", "content": case["subagent_q"]}],
        temperature=0.0, max_tokens=128
    )

    # Step 2: Parent uses real subagent output
    runner.load("3B", PARENT_GPU["3B"])
    parent_prompt = f"""A sub-agent was asked this question:

Sub-agent question: {case['subagent_q']}

The sub-agent answered: {subagent_output}

Now answer the ORIGINAL question using the sub-agent's response:

Original question: {case['question']}

Output only the answer to the original question:"""

    parent_output = runner.generate(
        "3B", PARENT_GPU["3B"],
        [{"role": "user", "content": parent_prompt}],
        temperature=0.0, max_tokens=64
    )

    return {
        "mode": "B_real_delegate",
        "parent_model": "3B",
        "subagent_model": "3B",
        "case_id": case["id"],
        "question": case["question"],
        "expected_answer": case["answer"],
        "subagent_question": case["subagent_q"],
        "subagent_output": subagent_output,
        "subagent_expected": case["subagent_a"],
        "parent_final_answer": parent_output,
        "misalignment_type": case["misalignment_type"],
    }

# ── Mode C: Oracle Delegate ────────────────────────────────────────────────────
# Same 3B model, but instead of real subagent output, it gets the GROUND-TRUTH oracle.
# This removes subagent noise but keeps the "extra reasoning step" context.

def run_mode_c(runner: ExperimentRunner, case: dict) -> dict:
    runner.load("3B", PARENT_GPU["3B"])

    parent_prompt = f"""A sub-agent was asked this question:

Sub-agent question: {case['subagent_q']}

The sub-agent answered: {case['oracle_output']}

Now answer the ORIGINAL question using the sub-agent's response:

Original question: {case['question']}

Output only the answer to the original question:"""

    parent_output = runner.generate(
        "3B", PARENT_GPU["3B"],
        [{"role": "user", "content": parent_prompt}],
        temperature=0.0, max_tokens=64
    )

    return {
        "mode": "C_oracle_delegate",
        "model": "3B",
        "case_id": case["id"],
        "question": case["question"],
        "expected_answer": case["answer"],
        "subagent_question": case["subagent_q"],
        "oracle_subagent_output": case["oracle_output"],
        "parent_final_answer": parent_output,
        "misalignment_type": case["misalignment_type"],
    }

# ── Evaluation ────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^(答案|Answer|result|回答|yes|no)[:：\s]*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^[\-*\d\.)]+\s*", "", text.strip())
    return text.strip().lower()

def check_match(output: str, expected: str, threshold: float = 0.6) -> bool:
    o = extract_answer(output)
    e = expected.lower()
    if not o or not e:
        return False
    if e in o or o in e:
        return True
    ow = set(o.split())
    ew = set(e.split())
    if not ow or not ew:
        return False
    jaccard = len(ow & ew) / len(ow | ew)
    return jaccard >= threshold

# ── Main ──────────────────────────────────────────────────────────────────────

async def run_experiment(output_dir: str, n_samples: int = None):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cases = MISALIGNMENT_CASES[:n_samples] if n_samples else MISALIGNMENT_CASES

    print("=" * 65)
    print("Goal Alignment Benchmark V3 — 3 Modes (isolate misalignment)")
    print(f"Cases: {len(cases)}")
    print("=" * 65)

    runner = ExperimentRunner()

    # Mode A: No-delegate baseline
    print(f"\n{'─'*65}")
    print("  Mode A: Single 3B (no subagent — true baseline)")
    print(f"{'─'*65}")
    mode_a_results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        r = run_mode_a(runner, case)
        mode_a_results.append(r)
        print("OK")
        time.sleep(0.05)
    mode_a_path = os.path.join(output_dir, f"modeA_no_delegate_{ts}.json")
    with open(mode_a_path, "w", encoding="utf-8") as f:
        json.dump(mode_a_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {mode_a_path}")

    # Mode B: Real delegate
    print(f"\n{'─'*65}")
    print("  Mode B: 3B parent + 3B real subagent")
    print(f"{'─'*65}")
    mode_b_results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        r = run_mode_b(runner, case)
        mode_b_results.append(r)
        print("OK")
        time.sleep(0.05)
    mode_b_path = os.path.join(output_dir, f"modeB_real_delegate_{ts}.json")
    with open(mode_b_path, "w", encoding="utf-8") as f:
        json.dump(mode_b_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {mode_b_path}")

    # Mode C: Oracle delegate
    print(f"\n{'─'*65}")
    print("  Mode C: Single 3B with ORACLE subagent output (no noise)")
    print(f"{'─'*65}")
    mode_c_results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        r = run_mode_c(runner, case)
        mode_c_results.append(r)
        print("OK")
        time.sleep(0.05)
    mode_c_path = os.path.join(output_dir, f"modeC_oracle_delegate_{ts}.json")
    with open(mode_c_path, "w", encoding="utf-8") as f:
        json.dump(mode_c_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {mode_c_path}")

    # ── Evaluation ──────────────────────────────────────────────────────────────
    for r in mode_a_results:
        r["is_correct"] = check_match(r["model_output"], r["expected_answer"])

    for r in mode_b_results:
        r["final_is_correct"] = check_match(r["parent_final_answer"], r["expected_answer"])
        r["subagent_is_correct"] = check_match(r["subagent_output"], r["subagent_expected"])

    for r in mode_c_results:
        r["final_is_correct"] = check_match(r["parent_final_answer"], r["expected_answer"])

    acc_a = sum(1 for r in mode_a_results if r["is_correct"]) / len(mode_a_results) * 100
    acc_b = sum(1 for r in mode_b_results if r["final_is_correct"]) / len(mode_b_results) * 100
    acc_c = sum(1 for r in mode_c_results if r["final_is_correct"]) / len(mode_c_results) * 100

    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  Mode A (single 3B, no subagent):         {acc_a:.0f}%  ({int(acc_a*len(mode_a_results)/100)}/{len(mode_a_results)})")
    print(f"  Mode B (3B + real subagent):             {acc_b:.0f}%  ({int(acc_b*len(mode_b_results)/100)}/{len(mode_b_results)})")
    print(f"  Mode C (3B + oracle subagent):           {acc_c:.0f}%  ({int(acc_c*len(mode_c_results)/100)}/{len(mode_c_results)})")
    print(f"{'='*65}")
    print(f"  A − B  (extra step + misalignment loss):  {acc_a - acc_b:+.0f} pp")
    print(f"  A − C  (extra step only):                 {acc_a - acc_c:+.0f} pp")
    print(f"  C − B  (goal misalignment loss):          {acc_c - acc_b:+.0f} pp")
    print(f"{'='*65}")

    # ── Detailed analysis ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CASE-BY-CASE ANALYSIS")
    print(f"{'='*65}")
    for a, b, c in zip(mode_a_results, mode_b_results, mode_c_results):
        a_ok = "✓" if a["is_correct"] else "✗"
        b_ok = "✓" if b["final_is_correct"] else "✗"
        c_ok = "✓" if c["final_is_correct"] else "✗"
        s_ok = "✓" if b["subagent_is_correct"] else "✗"
        # Flag patterns
        pattern = ""
        if not a["is_correct"] and not c["final_is_correct"] and not b["final_is_correct"]:
            pattern = "A✗C✗B✗ hard_case"
        elif c["final_is_correct"] and not b["final_is_correct"]:
            pattern = "C✓B✗ → MISALIGNMENT"
        elif not c["final_is_correct"] and b["final_is_correct"]:
            pattern = "C✗B✓ → subagent_noise_helped"
        elif not a["is_correct"] and c["final_is_correct"]:
            pattern = "A✗C✓ → step_helped"
        print(f"\n  [{a['case_id']}] type={a['misalignment_type']}")
        print(f"    A(no sub)={a_ok}  sub={s_ok}  B(real)={b_ok}  C(oracle)={c_ok}")
        if "MISALIGNMENT" in pattern or "step" in pattern.lower():
            print(f"    Pattern: {pattern}")
            print(f"    Q: {a['question'][:70]}...")
            print(f"    A out:     {a['model_output'][:50]}")
            print(f"    Sub out:   {b['subagent_output'][:40]}")
            print(f"    Oracle:    {c['oracle_subagent_output'][:40]}")
            print(f"    B parent:  {b['parent_final_answer'][:50]}")
            print(f"    C parent:  {c['parent_final_answer'][:50]}")
            print(f"    Expected:  {a['expected_answer']}")

    # Goal misalignment summary
    print(f"\n{'='*65}")
    print("  GOAL MISALIGNMENT SUMMARY (C−B > 0 means misalignment exists)")
    print(f"{'='*65}")
    misalignment_cases = [(b, c) for b, c in zip(mode_b_results, mode_c_results)
                           if c["final_is_correct"] and not b["final_is_correct"]]
    print(f"  Cases where oracle succeeds but real fails: {len(misalignment_cases)}/{len(cases)}")
    for b, c in misalignment_cases:
        print(f"    [{b['case_id']}] type={b['misalignment_type']}")
        print(f"      Oracle={c['oracle_subagent_output'][:40]}")
        print(f"      Real subagent={b['subagent_output'][:40]}")
        print(f"      C answered={c['parent_final_answer'][:40]}  B answered={b['parent_final_answer'][:40]}")

    return mode_a_results, mode_b_results, mode_c_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/home/jinxu/tiny-agents/benchmark_results/multi_hop_qa_alignment")
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(args.output_dir, args.n_samples))
