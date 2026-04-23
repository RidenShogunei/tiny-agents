"""S3 vs S4 stopping reason comparison analysis.

For each question, compare:
  - S3: why did it stop (rule_id sequence, last 3 steps)
  - S4: why did it stop (rule_id sequence, last 3 steps)
  - S3 vs S4: who stopped earlier, was it "smart early stop" or "premature stop"

Goal: determine whether S4's early stopping is "intelligent" or "overly aggressive".
"""

import csv
import json
import os
import sys
from collections import defaultdict

RULE_NAMES = {
    1: "R1_BUDGET_EXHAUSTED",
    2: "R2_CALL_VERIFY",
    3: "R3_CONSECUTIVE_LOW_GAIN",
    4: "R4_ENTROPY_HIGH",
    5: "R5_CONCENTRATION_LOW",
    6: "R6_UNCERTAINTY_HIGH",
    7: "R7_DEFAULT_CONTINUE",
    8: "R8_DISAGREEMENT_HIGH",
}


def load_results(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["final_correct"] = bool(int(row["final_correct"]))
            row["budget_used"] = int(row["budget_used"])
            row["total_steps"] = int(row["total_steps"])
            row["num_continue"] = int(row["num_continue"])
            row["num_verify"] = int(row["num_verify"])
            row["num_stop"] = int(row["num_stop"])
            row["avg_msg_gain"] = float(row["avg_msg_gain"])
            row["triggered_rule_sequence"] = json.loads(row["triggered_rule_sequence"])
            row["answer_disagreement_trace"] = json.loads(row["answer_disagreement_trace"])
            rows.append(row)
    return rows


def last_n_steps(rule_seq: list, n: int = 3):
    """Return last n rule IDs with their names."""
    tail = rule_seq[-n:] if len(rule_seq) >= n else rule_seq
    return [(rid, RULE_NAMES.get(rid, f"R{rid}_UNKNOWN")) for rid in tail]


def analyze_pair(s3_row: dict, s4_row: dict) -> dict:
    """Compare stopping behavior for one question."""
    s3_steps = s3_row["total_steps"]
    s4_steps = s4_row["total_steps"]
    s3_stop_reason = s3_row["triggered_rule_sequence"]
    s4_stop_reason = s4_row["triggered_rule_sequence"]
    s3_last3 = last_n_steps(s3_stop_reason, 3)
    s4_last3 = last_n_steps(s4_stop_reason, 3)
    s3_disagree_trace = s3_row["answer_disagreement_trace"]
    s4_disagree_trace = s4_row["answer_disagreement_trace"]

    # Classification
    if s4_steps < s3_steps * 0.7:
        stop_type = "S4_MUCH_EARLIER"
    elif s4_steps < s3_steps:
        stop_type = "S4_SLIGHTLY_EARLIER"
    elif s4_steps == s3_steps:
        stop_type = "SAME_STEPS"
    else:
        stop_type = "S4_LATER"

    # Was S4's stop "smart"?
    # Smart = S4 stopped with higher msg_gain or higher disagreement
    # Premature = S4 stopped with lower msg_gain and same/low disagreement
    s4_avg_gain = s4_row["avg_msg_gain"]
    s3_avg_gain = s3_row["avg_msg_gain"]
    s4_final_disagree = s4_disagree_trace[-1] if s4_disagree_trace else 0.0
    s3_final_disagree = s3_disagree_trace[-1] if s3_disagree_trace else 0.0

    if stop_type in ("S4_MUCH_EARLIER", "S4_SLIGHTLY_EARLIER"):
        if s4_avg_gain >= s3_avg_gain and s4_final_disagree <= s3_final_disagree:
            quality = "SMART_STOP"
        elif s4_avg_gain < s3_avg_gain * 0.5:
            quality = "PREMATURE_STOP"
        else:
            quality = "AMBIGUOUS"
    else:
        quality = "N/A"

    return {
        "question_id": s3_row["question_id"],
        "s3_steps": s3_steps,
        "s4_steps": s4_steps,
        "stop_type": stop_type,
        "stop_quality": quality,
        "s3_last3_rules": s3_last3,
        "s4_last3_rules": s4_last3,
        "s3_avg_gain": s3_avg_gain,
        "s4_avg_gain": s4_avg_gain,
        "s3_final_disagree": s3_final_disagree,
        "s4_final_disagree": s4_final_disagree,
        "s3_num_verify": s3_row["num_verify"],
        "s4_num_verify": s4_row["num_verify"],
        "s3_correct": s3_row["final_correct"],
        "s4_correct": s4_row["final_correct"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    s3_path = os.path.join(args.output_dir, [f for f in os.listdir(args.output_dir) if "S3_" in f and f.endswith(".csv")][0])
    s4_path = os.path.join(args.output_dir, [f for f in os.listdir(args.output_dir) if "S4_" in f and f.endswith(".csv")][0])

    s3_rows = load_results(s3_path)
    s4_rows = load_results(s4_path)

    # Index by question_id
    s3_by_q = {r["question_id"]: r for r in s3_rows}
    s4_by_q = {r["question_id"]: r for r in s4_rows}

    qids = sorted(s3_by_q.keys())

    results = []
    for qid in qids:
        r = analyze_pair(s3_by_q[qid], s4_by_q[qid])
        results.append(r)

    # Print per-question table
    print(f"\n{'QID':<12} {'S3':>4} {'S4':>4} {'TYPE':<20} {'QUALITY':<16} {'S3_LAST3':<40} {'S4_LAST3':<40}")
    print("-" * 160)
    for r in results:
        s3_last3 = str(r["s3_last3_rules"])
        s4_last3 = str(r["s4_last3_rules"])
        print(f"{r['question_id']:<12} {r['s3_steps']:>4} {r['s4_steps']:>4} {r['stop_type']:<20} {r['stop_quality']:<16} {s3_last3:<40} {s4_last3:<40}")

    # Aggregate stats
    stop_types = defaultdict(int)
    qualities = defaultdict(int)
    for r in results:
        stop_types[r["stop_type"]] += 1
        qualities[r["stop_quality"]] += 1

    print(f"\n{'='*60}")
    print("STOP_TYPE DISTRIBUTION")
    print("="*60)
    for k, v in sorted(stop_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}/{len(results)}")

    print(f"\n{'='*60}")
    print("STOP_QUALITY DISTRIBUTION")
    print("="*60)
    for k, v in sorted(qualities.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}/{len(results)}")

    print(f"\n{'='*60}")
    print("S3 vs S4 STEP COMPARISON")
    print("="*60)
    s3_total = sum(r["s3_steps"] for r in results)
    s4_total = sum(r["s4_steps"] for r in results)
    s3_avg = s3_total / len(results)
    s4_avg = s4_total / len(results)
    print(f"  S3 avg steps: {s3_avg:.2f}")
    print(f"  S4 avg steps: {s4_avg:.2f}")
    print(f"  S4 savings: {s3_avg - s4_avg:.2f} steps ({100*(s3_avg-s4_avg)/s3_avg:.1f}%)")

    # S3 vs S4 accuracy
    s3_correct = sum(1 for r in results if r["s3_correct"])
    s4_correct = sum(1 for r in results if r["s4_correct"])
    print(f"\n  S3 accuracy: {s3_correct}/{len(results)} = {100*s3_correct/len(results):.1f}%")
    print(f"  S4 accuracy: {s4_correct}/{len(results)} = {100*s4_correct/len(results):.1f}%")

    # VERIFY usage
    s3_verify = sum(r["s3_num_verify"] for r in results)
    s4_verify = sum(r["s4_num_verify"] for r in results)
    print(f"\n  S3 total VERIFY calls: {s3_verify}")
    print(f"  S4 total VERIFY calls: {s4_verify}")

    # Output JSON
    out_path = os.path.join(args.output_dir, "s3_vs_s4_comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "per_question": results,
            "stop_type_dist": dict(stop_types),
            "quality_dist": dict(qualities),
            "s3_avg_steps": s3_avg,
            "s4_avg_steps": s4_avg,
            "s3_accuracy": s3_correct / len(results),
            "s4_accuracy": s4_correct / len(results),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
