"""ClusterTool — groups papers by research theme/topic for organized survey writing.

Given a list of paper summaries, clusters them into thematic groups using
keyword-based scoring. Writers can then pull from relevant clusters for their sections.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tiny_agents.tools.base import BaseTool, ToolResult


# Common vision/LLM research theme keywords
THEME_KEYWORDS = {
    "lora_finetuning": [
        "lora", "low-rank", "adapter", "fine-tuning", "fine-tune",
        "parameter-efficient", "PEFT", "adapters", "adaptation",
    ],
    "vision_transformer": [
        "vit", "vision transformer", "transformer", "self-attention",
        "patch", "multi-head", "positional encoding", "ViT",
    ],
    "multimodal": [
        "multimodal", "vision-language", "image-text", "cross-modal",
        "VLM", "visual question", "BLIP", "LLaVA", "CLIP", "alignment",
    ],
    "efficient_ml": [
        "quantization", "pruning", "distillation", "compression",
        "efficient", "speed", "memory", "latency", "throughput",
        "knowledge distillation", "model compression",
    ],
    "rl_alignment": [
        "RLHF", "DPO", "RL", "reward", "alignment", "preference",
        "reinforcement learning", "PPO", "instruction tuning",
    ],
    "reasoning_planning": [
        "reasoning", "chain-of-thought", "CoT", "planning",
        "problem solving", "logical", "step-by-step", "self-consistency",
    ],
    "architecture": [
        "attention", "feed-forward", "layer norm", "embedding",
        "decoder", "encoder", "architecture", "backbone", "neck", "head",
    ],
    "training_strategy": [
        "training", "optimizer", "learning rate", "warmup", "batch size",
        "epochs", "convergence", "regularization", "augmentation",
    ],
    "downstream_tasks": [
        "detection", "segmentation", "classification", "recognition",
        "object detection", "semantic", "instance", "parsing", "captioning",
    ],
    "dataset_benchmark": [
        "benchmark", "dataset", "evaluation", "metrics", "accuracy",
        "COCO", "ImageNet", "VQA", "OK-VQA", "ScienceQA",
    ],
}


class ClusterTool(BaseTool):
    """Cluster papers into research themes.

    Args:
        papers: List of paper summaries (each with title, abstract/summary, method, category).
        num_clusters: Target number of clusters (default 5, max 10).
        theme_keywords: Optional dict of theme_name -> keywords for custom clustering.
            If not provided, uses built-in THEME_KEYWORDS.
    """

    def __init__(
        self,
        num_clusters: int = 5,
        theme_keywords: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            name="cluster",
            description="Group papers by research theme/topic using keyword scoring. "
                        "Returns clusters with theme name, description, and paper indices.",
        )
        self.num_clusters = max(1, min(num_clusters, 10))
        self.theme_keywords = theme_keywords or THEME_KEYWORDS

    def _execute(
        self,
        papers: List[Dict[str, Any]] = None,
        summaries: List[Dict[str, Any]] = None,
        num_clusters: int = None,
    ) -> ToolResult:
        """Cluster papers by theme.

        Args:
            papers: List of paper dicts (title, summary/abstract, method, category).
            summaries: Alias for papers (for compatibility).
            num_clusters: Override number of clusters.
        """
        source = papers or summaries or []
        if not source:
            return ToolResult(
                tool_name=self.name,
                args={"papers": papers, "summaries": summaries},
                success=False,
                error="No papers or summaries provided",
            )

        n_clusters = num_clusters or self.num_clusters
        paper_list = source if isinstance(source, list) else []

        # Score each paper against each theme
        theme_scores: Dict[str, Dict[str, Any]] = {}
        for theme, keywords in self.theme_keywords.items():
            theme_papers = []
            for idx, paper in enumerate(paper_list):
                score = self._score_paper(paper, keywords)
                if score > 0:
                    theme_papers.append({"idx": idx, "score": score, "paper": paper})
            if theme_papers:
                theme_papers.sort(key=lambda x: x["score"], reverse=True)
                theme_scores[theme] = {
                    "papers": theme_papers,
                    "count": len(theme_papers),
                    "top_score": theme_papers[0]["score"] if theme_papers else 0,
                }

        # Sort themes by total score and take top n
        sorted_themes = sorted(
            theme_scores.items(),
            key=lambda x: sum(p["score"] for p in x[1]["papers"]),
            reverse=True,
        )
        top_themes = sorted_themes[:n_clusters]

        # Build cluster results
        clusters = []
        for theme, data in top_themes:
            theme_desc = self._theme_description(theme)
            clusters.append({
                "theme": theme,
                "description": theme_desc,
                "paper_indices": [p["idx"] for p in data["papers"]],
                "papers": [
                    {
                        "title": p["paper"].get("title", "Unknown"),
                        "method": p["paper"].get("method", "Unknown"),
                        "score": p["score"],
                    }
                    for p in data["papers"][:5]  # top 5 per theme
                ],
                "count": data["count"],
            })

        # Papers not yet assigned
        assigned = set()
        for c in clusters:
            assigned.update(c["paper_indices"])

        unassigned = [
            {"idx": i, "title": p.get("title", "Unknown")}
            for i, p in enumerate(paper_list)
            if i not in assigned
        ]

        return ToolResult(
            tool_name=self.name,
            args={"papers_count": len(paper_list), "num_clusters": n_clusters},
            success=True,
            output={
                "clusters": clusters,
                "unassigned": unassigned,
                "total_papers": len(paper_list),
                "num_clusters": len(clusters),
            },
        )

    def _score_paper(self, paper: Dict[str, Any], keywords: List[str]) -> float:
        """Score how relevant a paper is to a theme based on keywords."""
        text = " ".join(
            str(v)
            for v in [
                paper.get("title", ""),
                paper.get("summary", ""),
                paper.get("abstract", ""),
                paper.get("method", ""),
                paper.get("category", ""),
                paper.get("contribution", ""),
            ]
        ).lower()

        score = 0.0
        for kw in keywords:
            pattern = re.compile(re.escape(kw.lower()))
            matches = pattern.findall(text)
            score += len(matches)

        return score

    def _theme_description(self, theme: str) -> str:
        """Human-readable description of a theme."""
        descriptions = {
            "lora_finetuning": "Low-rank adaptation and parameter-efficient fine-tuning methods",
            "vision_transformer": "Vision Transformer architectures and self-attention mechanisms",
            "multimodal": "Multimodal vision-language models and cross-modal alignment",
            "efficient_ml": "Model compression, quantization, and efficient inference",
            "rl_alignment": "RLHF, DPO, and reinforcement learning alignment techniques",
            "reasoning_planning": "Chain-of-thought reasoning and planning capabilities",
            "architecture": "Model architecture components and design choices",
            "training_strategy": "Training strategies, optimizers, and convergence",
            "downstream_tasks": "Application to detection, segmentation, and recognition tasks",
            "dataset_benchmark": "Datasets, benchmarks, and evaluation metrics",
        }
        return descriptions.get(theme, theme.replace("_", " ").title())

    def to_definition(self):
        """Return JSON Schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "List of paper summaries (title, summary/abstract, method, category)",
                        "items": {"type": "object"},
                    },
                    "summaries": {
                        "type": "array",
                        "description": "Alias for papers — use this or 'papers'",
                        "items": {"type": "object"},
                    },
                    "num_clusters": {
                        "type": "integer",
                        "description": f"Number of theme clusters to create (1-{self.num_clusters})",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
            },
        }
