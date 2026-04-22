"""PlanningAgent — analyzes research topic and generates section outline.

Given a research topic, decides the structure of the survey/review paper
and outputs a structured outline with section titles and keywords for each.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


PLANNING_PROMPT = """You are a research paper planner. Given a research topic, generate a structured outline for a comprehensive survey/review paper.

Your task:
1. Analyze the research topic and identify 6-10 main thematic sections
2. For each section, provide: section_id (e.g. "§1"), title, 3-5 keywords
3. The outline should follow standard survey paper structure:
   - §1 Introduction (overview of the field, motivation, contributions)
   - §2 Background & Foundations (core concepts, terminology)
   - §3-{N} Thematic sections (specific topics within the field)
   - §N+1 Challenges & Limitations
   - §N+2 Future Directions & Conclusion

Output format (JSON only, no extra text):
{{
  "title": "Survey Title: [Topic]",
  "abstract_hint": "2-3 sentences describing what this survey covers",
  "sections": [
    {{"id": "§1", "title": "Introduction", "keywords": ["field overview", "motivation", "contributions"]}},
    {{"id": "§2", "title": "Background", "keywords": ["foundational concepts"]}},
    ...
  ]
}}

Rules:
- Output ONLY valid JSON, no markdown fences, no explanations
- section titles should be descriptive (not just "Introduction")
- keywords help writers understand what content belongs in each section
"""


class PlanningAgent(BaseAgent):
    """Generates structured outline for a research survey (stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        **kwargs,
    ):
        super().__init__(
            name="planner",
            model_name=model_name,
            role_prompt=PLANNING_PROMPT,
            **kwargs,
        )

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Generate section outline for the given research topic."""
        topic = input_data.get("topic", "")
        desired_sections = input_data.get("num_sections", 8)  # target number of sections

        prompt = f"Research topic: {topic}\n\nGenerate a survey paper outline targeting approximately {desired_sections} sections.\n\n{PLANNING_PROMPT}\n\nOutput JSON:"

        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.3)
            response = self.backend.generate(
                model_key=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=2048,
            )
        else:
            response = "{}"

        # Parse JSON
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            outline = json.loads(json_str)
        except Exception:
            outline = self._fallback_outline(topic)

        context.add_message(self.name, "user", prompt)
        context.add_message(self.name, "assistant", json.dumps(outline, ensure_ascii=False))

        return AgentOutput(
            thought=f"Generated outline with {len(outline.get('sections', []))} sections",
            action="respond",
            payload={
                "outline": outline,
                "topic": topic,
            },
            finished=True,
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response (remove markdown fences, trailing text)."""
        text = text.strip()
        # Remove markdown code fences
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "")
        # Find JSON object or array
        start = re.search(r"[\[{]", text)
        if start:
            text = text[start.start():]
        # Find matching closing bracket
        depth = 0
        for i, c in enumerate(text):
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return text[:i + 1]
        return text

    def _fallback_outline(self, topic: str) -> Dict[str, Any]:
        """Fallback outline when JSON parsing fails."""
        return {
            "title": f"Survey on {topic}",
            "abstract_hint": f"A comprehensive survey covering {topic}.",
            "sections": [
                {"id": "§1", "title": "Introduction", "keywords": ["overview", "motivation"]},
                {"id": "§2", "title": "Background", "keywords": ["foundations", "concepts"]},
                {"id": "§3", "title": "Methods", "keywords": ["techniques", "approaches"]},
                {"id": "§4", "title": "Applications", "keywords": ["use cases", "domains"]},
                {"id": "§5", "title": "Challenges", "keywords": ["limitations", "open problems"]},
                {"id": "§6", "title": "Future Directions", "keywords": ["research gaps", "opportunities"]},
            ],
        }
