"""WriterAgent — drafts one section of a survey/review paper.

Given a section specification (title, keywords) and a set of relevant paper
summaries, writes a well-structured section with citations to the provided papers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


WRITER_PROMPT = """You are an academic paper writer. Your task is to write one section of a survey/review paper.

STYLE RULES (strictly follow these):
1. Write in academic prose — formal, impersonal, precise
2. Use citations like [1], [2], [3] inline — NOT parenthetical author-year style
3. Group papers by THEME, not by individual paper
4. For each theme, discuss 2-4 papers together, explaining WHY they are grouped
5. Do NOT list papers one-by-one (no "The paper [1] did X, then [2] did Y...")
6. Write 3-6 paragraphs per section, each 3-5 sentences
7. Acknowledge limitations and contradictions honestly
8. Do NOT overstate contributions — be objective
9. Include a "References" subsection at the end listing only the papers actually cited in this section

CITATION FORMAT:
- Inline: [1], [2], [3] (consecutive citations as [1-3])
- Reference list at bottom: numbered list with title, authors, year, venue
- Papers with no clear citation in your text should NOT appear in the reference list

OUTPUT FORMAT:
## Section Title

Opening paragraph:motivate why this topic matters, give an overview...

### Theme A (2-4 sentences)
Paragraph discussing 2-4 related papers and what they collectively show...

### Theme B
...

### References
[1] Title. Authors. Year. Venue.
[2] ...
"""


class WriterAgent(BaseAgent):
    """Drafts one section of a survey paper (stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        **kwargs,
    ):
        super().__init__(
            name="writer",
            model_name=model_name,
            role_prompt=WRITER_PROMPT,
            **kwargs,
        )

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Draft one section given section spec and paper summaries."""
        section = input_data.get("section", {})  # {id, title, keywords}
        section_id = section.get("id", "?")
        section_title = section.get("title", "Unknown Section")
        keywords = section.get("keywords", [])

        summaries = input_data.get("summaries", [])  # List[PaperSummary]
        topic = input_data.get("topic", "the research topic")
        overview = input_data.get("overview", "")  # overall survey title

        # Select top relevant papers for this section
        # Filter by keywords overlap, or just take top-N by relevance
        if keywords:
            scored = []
            for s in summaries:
                # Score by keyword overlap
                paper_text = f"{s.get('title', '')} {s.get('method', '')} {s.get('category', '')}".lower()
                score = sum(1 for kw in keywords if kw.lower() in paper_text)
                # Add relevance score
                score += s.get("relevance_score", 0)
                scored.append((score, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            relevant = [s for score, s in scored[:12]]  # max 12 papers per section
        else:
            relevant = summaries[:10]

        if not relevant:
            content = f"## {section_title}\n\n*No relevant papers found for this section.*\n"
            return AgentOutput(
                thought=f"Section {section_id} ({section_title}): no relevant papers",
                action="respond",
                payload={"section_id": section_id, "content": content, "papers_cited": []},
                finished=True,
            )

        # Build prompt
        prompt = self._build_prompt(section_id, section_title, keywords, relevant, topic)
        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.3)
            max_tok = context.config.get("max_tokens", 1536)
            content = self.backend.generate(
                model_key=self.name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        else:
            content = f"## {section_title}\n\n[Generated content placeholder]\n"

        # Extract cited paper indices for the citation agent
        cited_titles = self._extract_citations(content)
        cited_papers = [s for s in relevant if s.get("title", "") in cited_titles]

        context.add_message(self.name, "user", prompt[:200] + "...")
        context.add_message(self.name, "assistant", content[:200] + "...")

        return AgentOutput(
            thought=f"Drafted section {section_id} ({section_title}): {len(relevant)} papers provided, {len(cited_papers)} cited",
            action="respond",
            payload={
                "section_id": section_id,
                "section_title": section_title,
                "content": content,
                "papers_cited": cited_papers,
                "papers_provided": relevant,
            },
            finished=True,
        )

    def _build_prompt(
        self,
        section_id: str,
        section_title: str,
        keywords: List[str],
        summaries: List[Dict],
        topic: str,
    ) -> str:
        """Build the writing prompt."""
        lines = []
        lines.append(f"Survey topic: {topic}")
        lines.append(f"Section to write: {section_id} — {section_title}")
        lines.append(f"Section keywords: {', '.join(keywords)}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("RELEVANT PAPERS FOR THIS SECTION:")
        lines.append("=" * 60)

        for idx, s in enumerate(summaries):
            lines.append(f"\n[Paper {idx + 1}]")
            lines.append(f"Title: {s.get('title', 'Unknown')}")
            authors = s.get("authors", [])
            if authors:
                lines.append(f"Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}")
            lines.append(f"Method: {s.get('method', 'Unknown')}")
            lines.append(f"Contribution: {s.get('contribution', 'Unknown')}")
            lines.append(f"Limitation: {s.get('limitation', 'Not clearly stated')}")
            lines.append(f"Category: {s.get('category', 'general')}")
            if s.get("year"):
                lines.append(f"Year: {s.get('year')}")
            lines.append(f"Relevance: {s.get('relevance_score', 0)}/5")
            lines.append(f"URL: {s.get('paper_url', s.get('entry_id', 'N/A'))}")

        lines.append("")
        lines.append("=" * 60)
        lines.append(f"\nNow write section {section_id} ({section_title}) following the style rules.")
        lines.append("IMPORTANT: Cite papers as [1], [2], [3] inline. Group by theme.\n")

        return "\n".join(lines)

    def _extract_citations(self, content: str) -> List[str]:
        """Extract paper titles from cited references in the content."""
        # Look for reference list at bottom of content
        import re
        titles = []
        # Match patterns like: [1] Title. Authors. Year.
        refs = re.findall(r'\[\d+\]\s+(.+?)(?:\n|$)', content)
        for ref in refs:
            # Clean up: take everything before the first period or authors pattern
            title = ref.strip()
            title = re.sub(r'\s+\d+\s*$', '', title)  # trailing year/number
            if title and len(title) > 5:
                titles.append(title)
        return titles
