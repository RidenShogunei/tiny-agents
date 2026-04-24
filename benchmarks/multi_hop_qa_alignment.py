"""
Multi-Hop QA Goal Alignment Benchmark — Revised
================================================

Purpose: Detect goal misalignment between parent and subagent.

Hypothesis: Subagent correctly answers the delegated sub-question,
but the answer format/detail/framing is misaligned with what the
parent actually needs to complete the main task.

Design (revised):
  - Mode A (Single): 3B model answers the full question end-to-end
  - Mode B (Multi):  3B parent spawns 3B subagent for sub-task.
                    Both same model → no capability gap.
                    Misalignment is purely due to goal/interface mismatch.

Goal misalignment types tested:
  1. Precision mismatch: parent needs a number, subagent gives a range
  2. Framing mismatch: parent needs count, subagent gives names
  3. Context mismatch: subagent's answer is correct in wrong context
  4. Detail level: parent needs specific, subagent gives general
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
# Each case is designed to produce GOAL MISALIGNMENT, not能力 gap.
#
# Fields:
#   question:        the full question (what parent is ultimately asked)
#   answer:         ground-truth final answer
#   subagent_q:     what parent delegates to subagent (the sub-question)
#   subagent_a:     ground-truth for subagent question
#   misalignment:    what TYPE of goal misalignment occurs
#   expected_behavior: what subagent will likely do (misaligned)
#   why_misaligned: one-sentence explanation

MISALIGNMENT_CASES = [
    # ── Type 1: Subagent answers WRONG question, but parent uses it ───────────────
    # Goal misalignment: parent's delegated question is slightly off
    {
        "id": "m01_wrong_question",
        "question": "Python was created by Guido van Rossum and released in 1991. Java was created by James Gosling and released in 1995. Which language was released first?",
        "answer": "Python",
        "subagent_q": "Who created Java? Answer with just the name.",
        "subagent_a": "James Gosling",
        "misalignment": "irrelevant_delegation",
        "why": "Parent asks 'who created Java' but actually needs the release year to compare. Subagent answers correctly but provides useless info.",
    },
    # ── Type 2: Subagent correct but format unusable ─────────────────────────────
    {
        "id": "m02_unusable_format",
        "question": "Mount Everest is 8848 m tall. K2 is 8611 m tall. Which is taller? Answer with just the name.",
        "answer": "Mount Everest",
        "subagent_q": "What is the height of K2? Give the answer in a sentence like 'K2 is approximately X meters tall.'",
        "subagent_a": "K2 is approximately 8611 meters tall",
        "misalignment": "format_mismatch",
        "why": "Subagent gives a sentence but parent needs a plain number to compare. Parent can't extract 8611 from the sentence reliably.",
    },
    # ── Type 3: Parent relies on subagent but subagent gives wrong magnitude ─────
    {
        "id": "m03_wrong_magnitude",
        "question": "The speed of light is 299,792 km/s. The speed of sound is 343 m/s. Which is faster?",
        "answer": "speed of light",
        "subagent_q": "What is the speed of sound in km/s? Give just the number.",
        "subagent_a": "0.343",
        "misalignment": "unit_confusion",
        "why": "Subagent converts correctly (343 m/s = 0.343 km/s) but parent compares 299,792 vs 343 without unit context, gets wrong answer.",
    },
    # ── Type 4: Subagent gives name when parent needs count ──────────────────────
    {
        "id": "m04_count_vs_name",
        "question": "Which color appears more times in the word 'Strawberry': 'r' or 'b'?",
        "answer": "r (appears 3 times)",
        "subagent_q": "How many times does the letter 'r' appear in 'Strawberry'? Answer with just the number.",
        "subagent_a": "3",
        "misalignment": "parent_recovery_needed",
        "why": "Subagent gives 3 for 'r', but parent still needs to count 'b' (appears 2). Parent has to do extra work or forgets to count both.",
    },
    {
        "id": "m05_count_vs_name",
        "question": "Which has more letters: 'Wednesday' or 'Thursday'?",
        "answer": "Wednesday",
        "subagent_q": "How many letters does 'Wednesday' have? Answer with just the number.",
        "subagent_a": "9",
        "misalignment": "parent_limited_to_subagent",
        "why": "Parent only asks about Wednesday's count, never asks about Thursday. Parent can't compare without Thursday's count.",
    },
    # ── Type 5: Subagent correct but parent misinterprets ────────────────────────
    {
        "id": "m06_interpretation",
        "question": "Is 77 a prime number?",
        "answer": "no (77 = 7 × 11)",
        "subagent_q": "Is 77 divisible by any number besides 1 and itself? Answer yes or no.",
        "subagent_a": "yes",
        "misalignment": "context_collapse",
        "why": "Subagent correctly says 'yes' (77=7×11) but doesn't say what divides it. Parent needs the factorization to explain 'no'.",
    },
    # ── Type 6: Parent delegates wrong granularity ────────────────────────────────
    {
        "id": "m07_granularity",
        "question": "The Great Wall started building in 7th century BC and the Colosseum was built in 72 AD. Which is older?",
        "answer": "Great Wall of China",
        "subagent_q": "In what century was the Great Wall of China first built? Answer with the century like '7th century BC'.",
        "subagent_a": "7th century BC",
        "misalignment": "comparison_difficulty",
        "why": "Parent knows Colosseum=72AD. Subagent gives '7th century BC' but parent must compare centuries to AD dates — non-trivial.",
    },
    # ── Type 7: Subagent gives approximate, parent needs exact ─────────────────
    {
        "id": "m08_approximate",
        "question": "Who was born earlier: someone born in 1452 or someone born in 1475?",
        "answer": "1452",
        "subagent_q": "In what year was Michelangelo born? Answer with just the year number.",
        "subagent_a": "1475",
        "misalignment": "parent_context_lost",
        "why": "Parent has 1452 in the question itself but asks about Michelangelo's year. Subagent correctly says 1475. Parent CAN compare but subagent was unnecessary.",
    },
    # ── Type 8: Parent asks subagent to verify something parent already knows ──
    {
        "id": "m09_redundant",
        "question": "Which is bigger: the number 9 or the number 7?",
        "answer": "9",
        "subagent_q": "Is 9 greater than 7? Answer yes or no.",
        "subagent_a": "yes",
        "misalignment": "redundant_delegation",
        "why": "Parent already knows the answer by looking at the numbers. Delegating to subagent adds unnecessary steps.",
    },
    # ── Type 10: Subagent correct but parent ignores it ─────────────────────────
    {
        "id": "m10_ignored",
        "question": "Albert Einstein was born in 1879. Marie Curie was born in 1867. Who was born earlier?",
        "answer": "Marie Curie",
        "subagent_q": "In what year was Marie Curie born? Answer with just the year.",
        "subagent_a": "1867",
        "misalignment": "parent_overrides_subagent",
        "why": "Subagent correctly gives 1867. Parent already knows Einstein=1879, so parent CAN answer correctly. But parent might hallucinate and say Einstein instead.",
    },
]

# ── Experiment Runner ─────────────────────────────────────────────────────────

def load_llm(model_key: str, gpu_id: int):
    """Lazy-load LLM via VLLMBackend."""
    cache_key = f"{model_key}::{gpu_id}"
    if not hasattr(load_llm, "_backend"):
        from tiny_agents.models.vllm_backend import VLLMBackend
        load_llm._backend = VLLMBackend(default_gpu=gpu_id)
        load_llm._loaded = set()
    if cache_key not in load_llm._loaded:
        print(f"  [Load {model_key} on GPU {gpu_id}]")
        load_llm._backend.load_model(
            model_key=cache_key,
            model_name=MODEL_PATH[model_key],
            gpu=gpu_id,
            max_model_len=2048,
            gpu_memory_utilization=MEM_CONFIG[model_key],
        )
        load_llm._loaded.add(cache_key)
    return load_llm._backend

def generate(llm, messages, temperature=0.0, max_tokens=128):
    """Generate with the LLM."""
    return llm.generate(model_key=list(load_llm._loaded)[-1], messages=messages,
                       temperature=temperature, max_tokens=max_tokens)

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

    def generate(self, model_key: str, gpu_id: int, messages, temperature=0.0, max_tokens=128):
        if model_key not in self._loaded:
            self.load(model_key, gpu_id)
        return self.backend.generate(
            model_key=f"{model_key}_gpu{gpu_id}", messages=messages,
            temperature=temperature, max_tokens=max_tokens)

# ── Mode A: Single 3B Agent ───────────────────────────────────────────────────

def run_single_agent(runner: ExperimentRunner, case: dict) -> dict:
    prompt = f"""Answer the following question. Output only the answer, nothing else.

Question: {case['question']}

Answer: """

    runner.load("3B", PARENT_GPU["3B"])
    output = runner.generate("3B", PARENT_GPU["3B"],
                            [{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=64)

    return {
        "mode": "single_agent",
        "model": "3B",
        "case_id": case["id"],
        "question": case["question"],
        "expected_answer": case["answer"],
        "model_output": output,
        "is_correct": None,
        "misalignment_type": case["misalignment"],
        "why_misaligned": case["why"],
    }

# ── Mode B: 3B Parent + 3B Subagent ──────────────────────────────────────────

def run_multi_agent_same_model(runner: ExperimentRunner, case: dict) -> dict:
    """
    Both parent and subagent use 3B model.
    This isolates goal misalignment from capability differences.
    """
    # Step 1: Subagent answers the delegated sub-question
    runner.load("3B", SUBAGENT_GPU["3B"])
    subagent_output = runner.generate(
        "3B", SUBAGENT_GPU["3B"],
        [{"role": "user", "content": case["subagent_q"]}],
        temperature=0.0, max_tokens=128
    )

    # Step 2: Parent receives subagent's answer and produces final answer
    runner.load("3B", PARENT_GPU["3B"])
    parent_prompt = f"""You are a helpful assistant. A sub-agent was asked the following question:

Sub-agent question: {case['subagent_q']}

The sub-agent answered: {subagent_output}

Now answer the ORIGINAL question based on the sub-agent's response:

Original question: {case['question']}

Output only the answer to the original question:"""

    parent_output = runner.generate(
        "3B", PARENT_GPU["3B"],
        [{"role": "user", "content": parent_prompt}],
        temperature=0.0, max_tokens=64
    )

    return {
        "mode": "multi_agent_same_model",
        "parent_model": "3B",
        "subagent_model": "3B",
        "case_id": case["id"],
        "question": case["question"],
        "expected_answer": case["answer"],
        "subagent_question": case["subagent_q"],
        "subagent_output": subagent_output,
        "subagent_expected": case["subagent_a"],
        "parent_final_answer": parent_output,
        "misalignment_type": case["misalignment"],
        "why_misaligned": case["why"],
        "final_is_correct": None,
        "subagent_correct": None,  # Did subagent answer its own question correctly?
    }

# ── Mode C: Single 1.5B Agent (for comparison) ────────────────────────────────

def run_single_1_5b(runner: ExperimentRunner, case: dict) -> dict:
    """1.5B agent alone — to check if the multi-agent setup adds anything beyond using smaller model."""
    prompt = f"""Answer the following question. Output only the answer, nothing else.

Question: {case['question']}

Answer: """

    runner.load("1.5B", 1)
    output = runner.generate("1.5B", 1,
                            [{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=64)

    return {
        "mode": "single_1.5B_agent",
        "model": "1.5B",
        "case_id": case["id"],
        "question": case["question"],
        "expected_answer": case["answer"],
        "model_output": output,
        "is_correct": None,
    }

# ── Evaluation ────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Strip markdown, explanations, and noise."""
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^(答案|Answer|result|回答|结果是?)[:：\s]*", "", text.strip())
    text = re.sub(r"^[\-*\d\.)]+\s*", "", text.strip())
    return text.strip().lower()

def check_match(output: str, expected: str, threshold: float = 0.6) -> bool:
    out = extract_answer(output)
    exp = expected.lower()
    if not out or not exp:
        return False
    if exp in out or out in exp:
        return True
    out_words = set(out.split())
    exp_words = set(exp.split())
    if not exp_words or not out_words:
        return False
    jaccard = len(out_words & exp_words) / len(out_words | exp_words)
    return jaccard >= threshold

# ── Main ───────────────────────────────────────────────────────────────────────

async def run_experiment(output_dir: str, n_samples: int = None):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cases = MISALIGNMENT_CASES[:n_samples] if n_samples else MISALIGNMENT_CASES

    print("=" * 65)
    print("Multi-Agent Goal Alignment Benchmark (Revised)")
    print(f"Cases: {len(cases)}")
    print(f"Model config: Both parent and subagent use 3B (same model)")
    print("=" * 65)

    runner = ExperimentRunner()

    # ── Mode A: Single 3B Agent ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Mode A: Single 3B Agent (end-to-end)")
    print(f"{'─'*65}")
    single_results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        result = run_single_agent(runner, case)
        single_results.append(result)
        print(f"OK")
        time.sleep(0.1)

    # Save Mode A immediately
    mode_a_path = os.path.join(output_dir, f"modeA_single_3B_{ts}.json")
    with open(mode_a_path, "w", encoding="utf-8") as f:
        json.dump(single_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved Mode A to {mode_a_path}")

    # Mode B: Multi-agent (3B parent + 3B subagent)
    print(f"\n{'─'*65}")
    print("  Mode B: 3B Parent + 3B Subagent (same model)")
    print(f"{'─'*65}")
    multi_results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        result = run_multi_agent_same_model(runner, case)
        multi_results.append(result)
        print(f"OK")
        time.sleep(0.1)

    # Save Mode B immediately
    mode_b_path = os.path.join(output_dir, f"modeB_multi_3B+3B_{ts}.json")
    with open(mode_b_path, "w", encoding="utf-8") as f:
        json.dump(multi_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved Mode B to {mode_b_path}")

    # Mode C: Single 1.5B agent (baseline comparison)
    print(f"\n{'─'*65}")
    print("  Mode C: Single 1.5B Agent (weaker model baseline)")
    print(f"{'─'*65}")
    single_1_5b_results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        result = run_single_1_5b(runner, case)
        single_1_5b_results.append(result)
        print(f"OK")
        time.sleep(0.1)

    # Save Mode C
    mode_c_path = os.path.join(output_dir, f"modeC_single_1.5B_{ts}.json")
    with open(mode_c_path, "w", encoding="utf-8") as f:
        json.dump(single_1_5b_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved Mode C to {mode_c_path}")

    # ── Evaluation ──────────────────────────────────────────────────────────────
    for r in single_results:
        r["is_correct"] = check_match(r["model_output"], r["expected_answer"])

    for r in multi_results:
        r["final_is_correct"] = check_match(r["parent_final_answer"], r["expected_answer"])
        r["subagent_correct"] = check_match(r["subagent_output"], r["subagent_expected"])

    for r in single_1_5b_results:
        r["is_correct"] = check_match(r["model_output"], r["expected_answer"])

    # Print summary
    acc_single = sum(1 for r in single_results if r["is_correct"]) / len(single_results) * 100
    acc_multi  = sum(1 for r in multi_results  if r["final_is_correct"]) / len(multi_results) * 100
    acc_1_5b   = sum(1 for r in single_1_5b_results if r["is_correct"]) / len(single_1_5b_results) * 100

    print(f"\n{'='*65}")
    print(f"  Mode A (single 3B):      {acc_single:.0f}%  ({int(acc_single*len(single_results)/100)}/{len(single_results)})")
    print(f"  Mode B (3B+3B multi):    {acc_multi:.0f}%  ({int(acc_multi*len(multi_results)/100)}/{len(multi_results)})")
    print(f"  Mode C (single 1.5B):    {acc_1_5b:.0f}%  ({int(acc_1_5b*len(single_1_5b_results)/100)}/{len(single_1_5b_results)})")
    print(f"{'='*65}")

    # Analyze goal misalignment
    print(f"\n{'='*65}")
    print("  GOAL MISALIGNMENT ANALYSIS")
    print(f"{'='*65}")

    # Cases where subagent was correct but final was wrong
    misaligned = [r for r in multi_results
                  if r["subagent_correct"] and not r["final_is_correct"]]
    print(f"  Subagent CORRECT but final WRONG: {len(misaligned)}/{len(multi_results)}")
    for r in misaligned:
        print(f"    [{r['case_id']}] misalignment={r['misalignment_type']}")
        print(f"      subagent_output: {r['subagent_output'][:60]}")
        print(f"      parent_output:  {r['parent_final_answer'][:60]}")

    # Cases where subagent was wrong but final was correct
    recovered = [r for r in multi_results
                 if not r["subagent_correct"] and r["final_is_correct"]]
    print(f"\n  Subagent WRONG but final CORRECT (parent recovered): {len(recovered)}/{len(multi_results)}")

    # Subagent accuracy
    sub_ok = sum(1 for r in multi_results if r["subagent_correct"])
    print(f"\n  Subagent task accuracy: {sub_ok}/{len(multi_results)} ({sub_ok/len(multi_results)*100:.0f}%)")

    # Save
    with open(os.path.join(output_dir, f"modeA_single_3B_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(single_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, f"modeB_multi_3B+3B_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(multi_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, f"modeC_single_1.5B_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(single_1_5b_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_dir}")
    return single_results, multi_results, single_1_5b_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/home/jinxu/tiny-agents/benchmark_results/multi_hop_qa_alignment")
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(args.output_dir, args.n_samples))
