"""PaperKnowledgeGraph — cross-reference and relationship tracking for survey writing.

Builds a graph of papers and their relationships as papers are read.
Writers can query the graph to:
1. Get related papers (by method, citation, or theme)
2. Check what's already covered in other sections (to avoid duplication)
3. Find the most impactful/consequential papers

Usage:
    graph = PaperKnowledgeGraph()
    graph.add_paper(paper_data)           # for each paper from ReaderAgent
    graph.add_relationship(p1, p2, "extends")
    graph.get_related(paper_id, max_hops=2)
    graph.get_section_coverage()           # which papers are already covered
    graph.get_cross_references(section_ids)  # overlaps between sections
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class PaperNode:
    """A paper in the knowledge graph."""
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    method: str = ""
    contribution: str = ""
    limitation: str = ""
    category: str = ""
    relevance_score: float = 0.0
    paper_url: str = ""
    abstract: str = ""
    # Extracted via NLP-like heuristics
    key_phrases: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    # Relationships
    relationships: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph of papers and their relationships."""
    nodes: Dict[str, PaperNode] = field(default_factory=dict)
    section_assignments: Dict[str, str] = field(default_factory=dict)  # paper_id -> section_id
    section_topics: Dict[str, List[str]] = field(default_factory=dict)  # section_id -> [keywords]

    def add_paper(self, paper_data: Dict[str, Any], section_id: str = "") -> str:
        """Add a paper to the graph. Returns the paper_id."""
        paper_id = paper_data.get("paper_id", paper_data.get("entry_id", ""))
        if not paper_id:
            # Generate from URL or title hash
            title = paper_data.get("title", "")
            paper_id = f"paper_{abs(hash(title)) % 100000}"

        self.nodes[paper_id] = PaperNode(
            paper_id=paper_id,
            title=paper_data.get("title", ""),
            authors=paper_data.get("authors", []),
            year=paper_data.get("year"),
            venue=paper_data.get("venue", ""),
            method=paper_data.get("method", ""),
            contribution=paper_data.get("contribution", ""),
            limitation=paper_data.get("limitation", ""),
            category=paper_data.get("category", ""),
            relevance_score=paper_data.get("relevance_score", 0.0),
            paper_url=paper_data.get("paper_url", paper_data.get("entry_id", "")),
            abstract=paper_data.get("abstract", paper_data.get("summary", "")),
        )

        if section_id:
            self.section_assignments[paper_id] = section_id

        # Extract key phrases from title + method + abstract
        node = self.nodes[paper_id]
        node.key_phrases = self._extract_key_phrases(node)
        node.datasets = self._extract_datasets(node)
        node.tasks = self._extract_tasks(node)

        return paper_id

    def assign_section(self, paper_id: str, section_id: str) -> None:
        """Assign a paper to a section (writer will cover it)."""
        self.section_assignments[paper_id] = section_id

    def set_section_keywords(self, section_id: str, keywords: List[str]) -> None:
        """Set the keywords/topics of a section."""
        self.section_topics[section_id] = keywords

    def get_paper(self, paper_id: str) -> Optional[PaperNode]:
        return self.nodes.get(paper_id)

    def get_all_papers(self) -> List[PaperNode]:
        return list(self.nodes.values())

    def get_papers_by_section(self, section_id: str) -> List[PaperNode]:
        """Get all papers assigned to a given section."""
        return [
            self.nodes[pid]
            for pid, sid in self.section_assignments.items()
            if sid == section_id and pid in self.nodes
        ]

    def get_papers_by_category(self, category: str) -> List[PaperNode]:
        """Get all papers in a given category."""
        return [n for n in self.nodes.values() if n.category == category]

    def get_uncovered_papers(self, written_section_ids: List[str]) -> List[PaperNode]:
        """Get papers not yet assigned to any written section."""
        covered = set()
        for pid, sid in self.section_assignments.items():
            if sid in written_section_ids:
                covered.add(pid)
        return [n for pid, n in self.nodes.items() if pid not in covered]

    def get_cross_references(self, section_a: str, section_b: str) -> Dict[str, Any]:
        """Find overlapping topics/papers between two sections to detect duplication."""
        papers_a = set(pid for pid, sid in self.section_assignments.items() if sid == section_a)
        papers_b = set(pid for pid, sid in self.section_assignments.items() if sid == section_b)
        shared_papers = papers_a & papers_b

        # Find shared key phrases
        phr_a = set()
        phr_b = set()
        for pid in papers_a:
            if pid in self.nodes:
                phr_a.update(self.nodes[pid].key_phrases)
        for pid in papers_b:
            if pid in self.nodes:
                phr_b.update(self.nodes[pid].key_phrases)
        shared_phrases = phr_a & phr_b

        return {
            "shared_papers": list(shared_papers),
            "shared_phrases": list(shared_phrases),
            "overlap_score": len(shared_papers) / max(len(papers_a), 1),
        }

    def detect_section_overlap(self, new_section_keywords: List[str],
                                written_sections: Dict[str, str]) -> List[Dict]:
        """Check if new section keywords overlap significantly with already-written sections.
        
        Returns list of (section_id, overlap_score, shared_keywords).
        """
        new_kw = set(k.lower() for k in new_section_keywords)
        overlaps = []

        for sec_id, sec_content in written_sections.items():
            if not sec_content:
                continue
            sec_lower = sec_content.lower()
            shared = [kw for kw in new_section_keywords if kw.lower() in sec_lower]
            if shared:
                overlaps.append({
                    "section_id": sec_id,
                    "overlap_score": len(shared) / max(len(new_kw), 1),
                    "shared_keywords": shared,
                })

        return sorted(overlaps, key=lambda x: x["overlap_score"], reverse=True)

    def get_related_papers(
        self,
        paper_id: str,
        max_results: int = 5,
        same_category_only: bool = False,
    ) -> List[PaperNode]:
        """Find papers related to the given paper by method/category/phrase similarity."""
        if paper_id not in self.nodes:
            return []

        source = self.nodes[paper_id]
        candidates = [
            (pid, node)
            for pid, node in self.nodes.items()
            if pid != paper_id
            and (not same_category_only or node.category == source.category)
        ]

        scored = []
        for pid, node in candidates:
            score = 0.0
            # Method similarity
            if node.method and source.method:
                shared_method_words = set(node.method.lower().split()) & set(source.method.lower().split())
                score += len(shared_method_words) * 2.0
            # Category match
            if node.category == source.category and node.category:
                score += 3.0
            # Shared key phrases
            shared_phrases = set(node.key_phrases) & set(source.key_phrases)
            score += len(shared_phrases) * 1.5
            # Task overlap
            shared_tasks = set(node.tasks) & set(source.tasks)
            score += len(shared_tasks) * 2.0
            # Dataset overlap
            shared_datasets = set(node.datasets) & set(source.datasets)
            score += len(shared_datasets) * 1.0
            # Contribution phrase overlap
            if node.contribution and source.contribution:
                shared_contrib = set(node.contribution.lower().split()) & set(source.contribution.lower().split())
                score += len(shared_contrib) * 0.5

            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:max_results]]

    def get_methodology_coverage(self) -> Dict[str, List[str]]:
        """Group papers by their methodology for writing thematic sections."""
        method_groups: Dict[str, List[str]] = {}
        for node in self.nodes.values():
            if node.method:
                # Normalize method name (lowercase, strip)
                method_key = node.method.lower().strip()
                if method_key not in method_groups:
                    method_groups[method_key] = []
                method_groups[method_key].append(node.paper_id)
        return method_groups

    def get_paper_stats(self) -> Dict[str, Any]:
        """Return statistics about the knowledge graph."""
        categories = {}
        methods = {}
        years = {}
        for node in self.nodes.values():
            if node.category:
                categories[node.category] = categories.get(node.category, 0) + 1
            if node.method:
                methods[node.method] = methods.get(node.method, 0) + 1
            if node.year:
                years[node.year] = years.get(node.year, 0) + 1

        return {
            "total_papers": len(self.nodes),
            "categories": categories,
            "methods": methods,
            "years": years,
            "papers_with_relationships": sum(1 for n in self.nodes.values() if n.relationships),
        }

    # ── Extraction helpers ──────────────────────────────────────────────

    def _extract_key_phrases(self, node: PaperNode) -> List[str]:
        """Extract meaningful phrases from a paper's text."""
        text = f"{node.title} {node.method} {node.contribution} {node.abstract}".lower()
        # Common ML/AI terms to look for
        key_terms = [
            "lora", "adapter", "fine-tuning", "vision transformer", "vit",
            "distillation", "quantization", "pruning", "sparse",
            "reinforcement learning", "rlhf", "dpo", "group relative policy",
            "mixture of experts", "moe", "speculative decoding",
            "chain-of-thought", "cot", "reasoning",
            "multimodal", "vision-language", "vlm",
            "object detection", "segmentation", "classification",
            "knowledge distillation", "model compression",
            "prefix tuning", "prompt tuning", "p-tuning",
            " continual learning", "domain adaptation",
            "zero-shot", "few-shot", "in-context learning",
        ]
        found = [t for t in key_terms if t in text]
        # Also extract 2-3 word noun phrases from title
        title_words = node.title.split()
        for i in range(len(title_words) - 1):
            phrase = f"{title_words[i]} {title_words[i+1]}".lower()
            if len(phrase) > 6 and phrase not in found:
                found.append(phrase)
        return found

    def _extract_datasets(self, node: PaperNode) -> List[str]:
        """Extract dataset names from paper text."""
        datasets = [
            "imagenet", "cifar-10", "cifar-100", "coco", "vqa", "vqav2",
            "glue", "squad", "mmlu", "humaneval", "mbpp",
            "alpacaeval", "mt-bench", "flask",
        ]
        text = f"{node.title} {node.abstract} {node.contribution}".lower()
        return [d for d in datasets if d in text]

    def _extract_tasks(self, node: PaperNode) -> List[str]:
        """Extract task types from paper text."""
        tasks = [
            "text generation", "code generation", "translation",
            "summarization", "question answering", "qa",
            "image classification", "object detection", "semantic segmentation",
            "visual question answering", "vqa",
            "dialogue", "chat",
        ]
        text = f"{node.title} {node.abstract} {node.contribution}".lower()
        return [t for t in tasks if t in text]
