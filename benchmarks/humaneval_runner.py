"""HumanEval benchmark runner for Tiny Agents."""

import json
from typing import Any, Dict, List
from datasets import load_dataset


class HumanEvalRunner:
    """Evaluates code generation on HumanEval."""

    def __init__(self, agent=None):
        self.agent = agent
        self.dataset = load_dataset("openai_humaneval", split="test")
        self.results: List[Dict[str, Any]] = []

    def run(self, limit: int = 10) -> Dict[str, Any]:
        """Run evaluation on a subset of HumanEval."""
        subset = self.dataset.select(range(min(limit, len(self.dataset))))

        for item in subset:
            task_id = item["task_id"]
            prompt = item["prompt"]
            entry_point = item["entry_point"]

            # TODO: invoke coder agent
            generated_code = "# TODO: agent-generated code"

            self.results.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated": generated_code,
                "entry_point": entry_point,
            })

        return {
            "benchmark": "humaneval",
            "total": len(self.results),
            "completed": len(self.results),
            "results": self.results,
        }

    def save(self, path: str) -> None:
        """Save results to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
