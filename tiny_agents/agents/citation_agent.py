"""CitationAgent — deduplicates, formats, and manages references.

Takes references collected from multiple section drafts, deduplicates them,
formats them according to a chosen citation style, and generates the final
reference list and appendix.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


CITATION_PROMPT = """You are a citation manager. Given a list of academic papers, produce a clean, deduplicated reference list in a standard academic format.

Tasks:
1. Remove exact duplicates (same title)
2. Format each reference as: Authors. (Year). Title. Venue/Conference/Journal.
3. Sort by first author's last name alphabetically
4. Assign sequential numbers [1], [2], [3]...
5. Also generate a BibTeX entry for each paper

Input: list of papers with title, authors, year, venue
Output format (plain text, NOT JSON):

REFERENCES
==========
[1] Author1, A., Author2, B. (2024). Paper Title. Conference/Journal Name.
[2] Author, C. (2023). Another Paper. Journal Name.

BIBTEX
======
@article{key2024,
  author = {Author, A. and Author2, B.},
  title = {Paper Title},
  year = {2024},
  venue = {Conference/Journal Name},
}
"""


class CitationAgent(BaseAgent):
    """Manages citations: deduplication, formatting, BibTeX generation (stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        style: str = "numbered",  # numbered | apa | gbt7714
        **kwargs,
    ):
        super().__init__(
            name="citation",
            model_name=model_name,
            role_prompt=CITATION_PROMPT,
            **kwargs,
        )
        self.style = style

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Format and deduplicate references."""
        papers = input_data.get("papers", [])  # List[PaperSummary]
        full_paper = input_data.get("full_paper", "")  # the complete paper text
        style = input_data.get("style", self.style)

        if not papers:
            return AgentOutput(
                thought="No papers to format as citations",
                action="respond",
                payload={"references": "", "bibtex": "", "count": 0},
                finished=True,
            )

        # Deduplicate by title
        seen_titles = set()
        unique_papers = []
        for p in papers:
            title = p.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(p)

        # Build paper list text for prompt
        papers_text = self._format_paper_list(unique_papers)

        prompt = f"""Format the following papers into a clean reference list.

Citation style: {style} (numbered = [1], [2], ... format)
Papers:\n{papers_text}\n\n{CITATION_PROMPT}\n"""

        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.1)  # low temp for formatting
            response = self.backend.generate(
                model_key=self.name,
                messages=messages,
                temperature=temp,
                max_tokens=2048,
            )
        else:
            response = self._format_fallback(unique_papers, style)

        # Parse out references and bibtex sections
        references, bibtex = self._parse_output(response)

        context.add_message(self.name, "user", f"Formatted {len(unique_papers)} unique references")
        context.add_message(self.name, "assistant", f"{style} format, {len(unique_papers)} references")

        return AgentOutput(
            thought=f"Formatted {len(unique_papers)} references ({style} style)",
            action="respond",
            payload={
                "references": references,
                "bibtex": bibtex,
                "count": len(unique_papers),
                "deduplicated_from": len(papers),
                "style": style,
            },
            finished=True,
        )

    def _format_paper_list(self, papers: List[Dict]) -> str:
        """Format papers for the prompt."""
        lines = []
        for i, p in enumerate(papers):
            title = p.get("title", "Unknown")
            authors = p.get("authors", [])
            if isinstance(authors, list):
                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += " et al."
            else:
                author_str = str(authors)
            year = p.get("year", p.get("published", ""))
            if isinstance(year, str) and len(year) >= 4:
                year = year[:4]
            venue = p.get("journal_ref", p.get("primary_category", ""))
            url = p.get("paper_url", p.get("entry_id", ""))
            lines.append(f"[{i + 1}] {author_str}. ({year}). {title}. {venue}. {url}")
        return "\n".join(lines)

    def _parse_output(self, text: str) -> tuple:
        """Split output into references and bibtex sections."""
        # Look for BIBTEX separator
        if "BIBTEX" in text.upper() or "BIBTEX" in text:
            parts = re.split(r'BIBTEX\s*\n+\s*={3,}\s*\n', text, maxsplit=1, flags=re.IGNORECASE)
        elif "REFERENCES" in text.upper():
            parts = re.split(r'REFERENCES\s*\n+={3,}\s*\n', text, maxsplit=1)
        else:
            parts = [text, ""]

        references = parts[0].strip()
        bibtex = parts[1].strip() if len(parts) > 1 else ""
        return references, bibtex

    def _format_fallback(self, papers: List[Dict], style: str) -> str:
        """Fallback formatting when no LLM is available."""
        lines = ["REFERENCES\n=========="]
        for i, p in enumerate(papers, 1):
            authors = p.get("authors", [])
            if isinstance(authors, list):
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
            else:
                author_str = str(authors)
            year = str(p.get("year", ""))[:4]
            title = p.get("title", "Unknown")
            venue = p.get("journal_ref", p.get("primary_category", ""))
            lines.append(f"[{i}] {author_str}. ({year}). {title}. {venue}.")
        return "\n".join(lines)
