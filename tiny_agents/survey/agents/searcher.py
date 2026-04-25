"""SearchAgent — discovers relevant papers using arXiv and web search.

Takes a research topic and keywords, calls arXiv and web search tools
to find relevant papers, then returns a deduplicated, sorted list of papers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.survey.tools.arxiv_tool import ArxivTool
from tiny_agents.tools.web_search_tool import WebSearchTool


SEARCH_PROMPT = """You are a research paper search specialist. Given a research topic and section keywords, generate effective search queries to find the most relevant academic papers.

Your task:
1. Generate 2-3 arXiv search queries from the topic and keywords
2. Each query should be specific enough to return relevant papers (use arXiv operators like AND, OR, ti:, abs:, all:)
3. Set num_results appropriately (more for broad topics, fewer for niche ones)

Output format (JSON only):
{{
  "queries": [
    {{"query": "ti:transformer AND abs:vision", "source": "arxiv", "num_results": 20}},
    {{"query": "LoRA fine-tuning vision model", "source": "arxiv", "num_results": 15}}
  ]
}}

Rules:
- Output ONLY valid JSON, no markdown fences
- arXiv queries: use ti: (title), abs: (abstract), all: (everywhere)
- Prioritize specificity over breadth
"""


class SearchAgent(BaseAgent):
    """Discovers relevant academic papers via arXiv and web search (stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_papers: int = 50,
        **kwargs,
    ):
        super().__init__(
            name="searcher",
            model_name=model_name,
            role_prompt=SEARCH_PROMPT,
            **kwargs,
        )
        self.max_papers = max_papers
        self.arxiv_tool = ArxivTool(max_results=30)
        self.websearch_tool = WebSearchTool(num_results=5)

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Search for papers matching the topic and section keywords."""
        topic = input_data.get("topic", "")
        section_keywords = input_data.get("keywords", [])  # list of keyword lists, one per section
        all_keywords = []
        for kws in section_keywords:
            if isinstance(kws, list):
                all_keywords.extend(kws)
            else:
                all_keywords.append(str(kws))

        keywords_str = ", ".join(all_keywords[:15])  # limit keywords to avoid overly long prompts

        # Generate search queries
        query_prompt = f"Topic: {topic}\nKeywords: {keywords_str}\n\n{SEARCH_PROMPT}\n\nOutput JSON:"
        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": query_prompt})

        queries = []
        if self.backend is not None:
            temp = context.config.get("temperature", 0.1)
            response = self.backend.generate(
                model_key=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=512,
            )
            try:
                json_str = self._extract_json(response)
                queries = json.loads(json_str).get("queries", [])
            except Exception:
                pass

        # Fallback queries if parsing failed
        if not queries:
            queries = [
                {"query": topic, "source": "arxiv", "num_results": 30},
            ]

        # Execute searches
        all_papers: Dict[str, Dict] = {}
        search_log = []

        for qspec in queries[:5]:  # max 5 query rounds
            query = qspec.get("query", "")
            source = qspec.get("source", "arxiv")
            num = min(qspec.get("num_results", 20), 50)

            if source == "arxiv":
                result = self.arxiv_tool.execute({"query": query, "max_results": num})
            else:
                result = self.websearch_tool.execute({"query": query, "num_results": num})

            search_log.append({"query": query, "source": source, "success": result.success})

            if result.success:
                if source == "arxiv" and "papers" in result.output:
                    for paper in result.output["papers"]:
                        key = paper.get("entry_id", paper.get("title", ""))
                        if key and key not in all_papers:
                            all_papers[key] = paper
                elif "results" in result.output:
                    for item in result.output["results"]:
                        key = item.get("url", item.get("title", ""))
                        if key and key not in all_papers:
                            all_papers[key] = item

            # Respect arXiv rate limit
            if source == "arxiv":
                import time
                time.sleep(3.1)

        # Convert to sorted list (arXiv papers first, then web results)
        papers_list = list(all_papers.values())
        arxiv_papers = [p for p in papers_list if "arxiv" in p.get("entry_id", "") or "arxiv.org" in p.get("pdf_url", "")]
        web_papers = [p for p in papers_list if p not in arxiv_papers]

        # Sort each group by date (newest first if available)
        arxiv_papers.sort(key=lambda p: p.get("published", ""), reverse=True)
        web_papers.sort(key=lambda p: p.get("snippet", "")[:50])

        sorted_papers = arxiv_papers + web_papers
        sorted_papers = sorted_papers[:self.max_papers]

        context.add_message(self.name, "user", query_prompt)
        context.add_message(self.name, "assistant", f"Found {len(sorted_papers)} papers via {len(search_log)} search queries")

        return AgentOutput(
            thought=f"Found {len(sorted_papers)} papers ({len(arxiv_papers)} from arXiv, {len(web_papers)} from web)",
            action="respond",
            payload={
                "papers": sorted_papers,
                "arxiv_count": len(arxiv_papers),
                "web_count": len(web_papers),
                "topic": topic,
                "search_log": search_log,
            },
            finished=True,
        )

    def _extract_json(self, text: str) -> str:
        import re
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
