"""SynthesizerAgent — merges section drafts into a cohesive survey paper.

Given all drafted sections, generates:
- Introduction
- Conclusion / Future Directions
- Abstract (optional)

And handles section ordering and transitions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


SYNTHESIZER_PROMPT = """You are a research paper synthesizer. Your job is to write the framing sections of a survey paper and merge all sections into a coherent document.

TASKS:
1. Write the INTRODUCTION section: motivate the field, explain why it matters, list the main contributions of this survey
2. Write the CONCLUSION section: summarize key findings, open challenges, future research directions
3. Write the ABSTRACT (200-300 words): standard academic abstract — context, gap, contribution, findings
4. Merge all drafted sections with proper ordering and transitions

INPUT:
- Survey title
- All drafted sections (with content and citations)
- List of all cited papers across all sections

OUTPUT FORMAT (plain text, NOT JSON):

---
ABSTRACT:
[200-300 word abstract]

---

## 1. Introduction

[introduction content]

---

## [Other drafted sections in order...]

---

## 8. Conclusion

[conclusion content]
---

REFERENCES:
[numbered reference list]
"""


class SynthesizerAgent(BaseAgent):
    """Merges section drafts and generates introduction/conclusion/abstract (stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        **kwargs,
    ):
        super().__init__(
            name="synthesizer",
            model_name=model_name,
            role_prompt=SYNTHESIZER_PROMPT,
            **kwargs,
        )

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Synthesize all sections into a complete survey paper."""
        sections = input_data.get("sections", [])  # [{id, title, content, papers_cited}]
        topic = input_data.get("topic", "the research topic")
        title = input_data.get("title", f"Survey on {topic}")

        # Collect all cited papers
        all_cited: Dict[str, Dict] = {}
        for sec in sections:
            for paper in sec.get("papers_cited", []):
                key = paper.get("title", "")
                if key and key not in all_cited:
                    all_cited[key] = paper

        # Build sections summary for prompt
        sections_text = []
        for sec in sorted(sections, key=lambda s: s.get("id", "")):
            sid = sec.get("id", "?")
            stitle = sec.get("title", "Unknown")
            content = sec.get("content", "")
            sections_text.append(f"\n{'=' * 60}\nSection {sid}: {stitle}\n{'=' * 60}\n{content}")

        prompt = f"""Survey title: {title}
Topic: {topic}
Number of sections: {len(sections)}

{' '.join(sections_text)}

---

Now write the framing sections (Abstract, Introduction, Conclusion) and merge everything into a complete survey paper.
Follow the output format exactly.
"""
        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.3)
            max_tok = context.config.get("max_tokens", 2048)
            content = self.backend.generate(
                model_key=self.name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        else:
            content = "[Synthesized content placeholder]"

        context.add_message(self.name, "user", prompt[:300] + "...")
        context.add_message(self.name, "assistant", "Synthesized full paper with introduction, conclusion, and references")

        return AgentOutput(
            thought=f"Synthesized {len(sections)} sections + intro/conclusion into full paper (~{len(content)} chars)",
            action="respond",
            payload={
                "title": title,
                "topic": topic,
                "full_paper": content,
                "sections_drafted": len(sections),
                "total_cited_papers": len(all_cited),
            },
            finished=True,
        )
