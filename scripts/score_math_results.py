"""Score MATH benchmark results from JSONL.

Extracts \\boxed{} answers and compares with ground truth using
symbolic normalization (strip latex wrappers, spaces, case).
"""

import json
import re
from pathlib import Path


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        # Fallback: last token
        parts = text.strip().split()
        return parts[-1] if parts else ""

    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1

    # end now points one past the matching }
    return text[start:end - 1].strip()


def normalize_answer(ans: str) -> str:
    """Normalize answer string for comparison."""
    ans = ans.strip()
    # Remove surrounding $...$
    ans = re.sub(r"^\$+|\$+$", "", ans)
    # Remove \\text wrappers
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    # Remove spaces
    ans = ans.replace(" ", "")
    # Normalize common latex commands
    ans = ans.replace("\\dfrac", "\\frac")
    return ans.lower()


def grade(pred: str, truth: str) -> bool:
    """Grade a single prediction against ground truth."""
    pred_norm = normalize_answer(pred)
    truth_norm = normalize_answer(truth)
    return pred_norm == truth_norm


def main():
    results_path = Path("benchmark_results/math_quick_results.jsonl")
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    correct = 0
    total = 0

    print("=" * 70)
    print("MATH Benchmark Scoring")
    print("=" * 70)

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            pred_raw = extract_boxed(item["predicted"])
            truth = item["ground_truth"]
            is_correct = grade(pred_raw, truth)
            correct += int(is_correct)

            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"\n[{total}] {item['type']} | {item['level']}")
            print(f"    Ground Truth: {truth}")
            print(f"    Predicted:    {pred_raw}")
            print(f"    Status:       {status}")

    print("\n" + "=" * 70)
    print(f"Total: {correct}/{total} = {correct/total*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
