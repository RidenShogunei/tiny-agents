"""MATH benchmark runner for Tiny Agents."""

import json
from typing import Any, Dict, List
from datasets import load_dataset


class MATHRunner:
    """Evaluates mathematical reasoning on MATH dataset."""

    def __init__(self, agent=None):
        self.agent = agent
        self.dataset = load_dataset("hendrycks/competition_math", split="test")
        self.results: List[Dict[str, Any]] = []

    def run(self, limit: int = 50) -> Dict[str, Any]:
        """Run evaluation on a subset of MATH."""
        subset = self.dataset.select(range(min(limit, len(self.dataset))))

        for item in subset:
            problem = item["problem"]
            answer = item["solution"]
            level = item.get("level", "unknown")
            _type = item.get("type", "unknown")

            # TODO: invoke reasoning agent(s)
            predicted = "TODO"

            self.results.append({
                "problem": problem,
                "ground_truth": answer,
                "predicted": predicted,
                "level": level,
                "type": _type,
            })

        return {
            "benchmark": "math",
            "total": len(self.results),
            "completed": len(self.results),
            "results": self.results,
        }

    def save(self, path: str) -> None:
        """Save results to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
