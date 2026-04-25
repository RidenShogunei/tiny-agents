"""ReaderAgent — reads paper metadata and extracts structured summaries.

Given a list of papers (with abstracts), extracts key information from each:
- method/approach
- main contribution
- limitations
- category/theme

Works in batches to avoid context overflow. Each batch = up to 10 papers.
Supports true parallel processing of batches using asyncio for speed.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


READER_PROMPT = """You are a research paper analyst. Given a list of academic papers (with titles and abstracts), extract structured information from each.

For each paper, extract:
- title: the exact paper title
- method: the core method/approach used (e.g. "low-rank adaptation", "attention mechanism")
- contribution: the main contribution in 1-2 sentences
- limitation: key limitations or weaknesses (1 sentence, or "Not clearly stated")
- category: the research theme (e.g. "fine-tuning", "model compression", "vision-language", "efficient training")
- relevance_score: 1-5 based on how relevant this paper is to the research topic

Input format: a list of papers with titles and abstracts.
Output format (JSON array only):

[
  {{
    "title": "Paper Title",
    "method": "core method or approach",
    "contribution": "main contribution in 1-2 sentences",
    "limitation": "key limitation or weakness",
    "category": "research theme",
    "relevance_score": 4,
    "authors": ["Author1", "Author2"],
    "year": 2024,
    "paper_url": "URL if available"
  }},
  ...
]

Rules:
- Output ONLY valid JSON array, no markdown fences
- Extract exactly one entry per paper
- relevance_score: 5=essential/core paper, 4=highly relevant, 3=somewhat relevant, 2=tangentially relevant, 1=not relevant
- Use "Not clearly stated" for missing fields
"""


# Batch size for reading papers (to avoid context overflow)
BATCH_SIZE = 8


class ReaderAgent(BaseAgent):
    """Extracts structured summaries from academic papers (stateless, batched)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__(
            name="reader",
            model_name=model_name,
            role_prompt=READER_PROMPT,
            **kwargs,
        )
        self.batch_size = batch_size

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Read and summarize a batch of papers (max batch_size at a time)."""
        papers = input_data.get("papers", [])
        topic = input_data.get("topic", "the research topic")

        if not papers:
            return AgentOutput(
                thought="No papers provided to read",
                action="respond",
                payload={"summaries": [], "topic": topic},
                finished=True,
            )

        # Process in batches — all at once (parallel)
        all_summaries: List[Dict] = []
        batch_list: List[List] = []
        for i in range(0, len(papers), self.batch_size):
            batch_list.append(papers[i:i + self.batch_size])

        async def process_batch(batch: List, batch_num: int) -> List[Dict]:
            batch_input = self._build_batch_input(batch, topic)
            messages = context.get_messages(self.name)
            messages.append({"role": "user", "content": batch_input})

            if self.backend is not None:
                temp = context.config.get("temperature", 0.2)
                response = self.backend.generate(
                    model_key=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=2048,
                )
            else:
                response = "[]"

            summaries = self._parse_summaries(response, batch)
            context.add_message(self.name, "user", f"[Batch {batch_num}] Read {len(batch)} papers")
            context.add_message(self.name, "assistant", f"Extracted {len(summaries)} summaries")
            return summaries

        # Run all batches in parallel
        results = await asyncio.gather(
            *[process_batch(batch, i + 1) for i, batch in enumerate(batch_list)],
            return_exceptions=True,
        )

        # Collect successful results
        for result in results:
            if isinstance(result, Exception):
                continue
            all_summaries.extend(result)

        # Sort by relevance score descending
        all_summaries.sort(key=lambda s: s.get("relevance_score", 0), reverse=True)

        return AgentOutput(
            thought=f"Read {len(papers)} papers → {len(all_summaries)} summaries, top relevance: {all_summaries[0].get('relevance_score', 0) if all_summaries else 0}",
            action="respond",
            payload={
                "summaries": all_summaries,
                "total_papers": len(papers),
                "topic": topic,
            },
            finished=True,
        )

    def _build_batch_input(self, papers: List[Dict], topic: str) -> str:
        """Build a readable text prompt from a batch of papers."""
        lines = [f"Research topic: {topic}\n"]
        lines.append(f"Number of papers to analyze: {len(papers)}\n")
        lines.append("=" * 60)

        for idx, paper in enumerate(papers):
            lines.append(f"\n[PAPER {idx + 1}]")
            lines.append(f"Title: {paper.get('title', 'Unknown')}")

            authors = paper.get('authors', [])
            if isinstance(authors, list) and authors:
                lines.append(f"Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}")
            elif isinstance(authors, str):
                lines.append(f"Authors: {authors}")

            if paper.get('published'):
                lines.append(f"Published: {str(paper.get('published', ''))[:10]}")
            if paper.get('entry_id'):
                lines.append(f"URL: {paper.get('entry_id')}")

            abstract = paper.get('summary', paper.get('abstract', ''))
            if abstract:
                # Truncate very long abstracts
                if len(abstract) > 1000:
                    abstract = abstract[:1000] + "..."
                lines.append(f"Abstract: {abstract}")
            else:
                snippet = paper.get('snippet', '')
                if snippet:
                    lines.append(f"Snippet: {snippet[:500]}")

            lines.append("")

        lines.append("=" * 60)
        lines.append(f"\nNow extract structured information for each of the {len(papers)} papers above.")
        lines.append("Output a JSON array with one entry per paper:\n")

        return "\n".join(lines)

    def _parse_summaries(self, response: str, original_papers: List[Dict]) -> List[Dict[str, Any]]:
        """Parse JSON summaries from LLM response, with fallback."""
        try:
            json_str = self._extract_json(response)
            summaries = json.loads(json_str)
            if isinstance(summaries, list):
                return summaries
        except Exception:
            pass

        # Fallback: return minimal summaries for each paper
        return [
            {
                "title": p.get("title", "Unknown"),
                "method": "Unknown",
                "contribution": p.get("summary", p.get("abstract", ""))[:200],
                "limitation": "Not clearly stated",
                "category": "general",
                "relevance_score": 3,
                "authors": p.get("authors", [])[:3] if isinstance(p.get("authors"), list) else [],
                "year": None,
                "paper_url": p.get("entry_id", p.get("url", "")),
            }
            for p in original_papers
        ]

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "")
        start = re.search(r"[\[{]", text)
        if start:
            text = text[start.start():]
        depth = 0
        for i, c in enumerate(text):
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return text[:i + 1]
        return text
