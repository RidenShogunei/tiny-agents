"""MathVista benchmark runner for Tiny Agents."""

import json
from typing import Any, Dict, List
from datasets import load_dataset


class MathVistaRunner:
    """Evaluates multimodal reasoning on MathVista."""

    def __init__(self, vl_agent=None, llm_agent=None):
        self.vl_agent = vl_agent
        self.llm_agent = llm_agent
        self.dataset = load_dataset("AI4Math/MathVista", split="testmini")
        self.results: List[Dict[str, Any]] = []

    def run(self, limit: int = 50) -> Dict[str, Any]:
        """Run evaluation on MathVista testmini subset."""
        subset = self.dataset.select(range(min(limit, len(self.dataset))))

        for item in subset:
            question = item["question"]
            answer = item.get("answer", "")
            image = item.get("image", None)
            choices = item.get("choices", [])

            # Pipeline: VLM perception -> LLM reasoning
            # TODO: invoke agents
            predicted = "TODO"

            self.results.append({
                "question": question,
                "ground_truth": answer,
                "predicted": predicted,
                "has_image": image is not None,
                "choices": choices,
            })

        return {
            "benchmark": "mathvista",
            "total": len(self.results),
            "completed": len(self.results),
            "results": self.results,
        }

    def save(self, path: str) -> None:
        """Save results to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
