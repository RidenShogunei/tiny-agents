"""arXivTool — search and retrieve papers from arXiv.org.

Uses the `arxiv` library for searching and downloading paper metadata.
No API key required. Rate-limited by arXiv (1 request/3 seconds recommended).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from tiny_agents.tools.base import BaseTool, ToolResult


class ArxivTool(BaseTool):
    """Search arXiv for academic papers.

    Downloads paper metadata (title, authors, abstract, etc.) but NOT PDFs.
    For full-text analysis, use WebBrowseTool to access arXiv HTML pages.

    Args:
        query: Search query string (supports arXiv advanced search operators).
        max_results: Maximum number of results to return (default 20, max 100).
        sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'.
        sort_order: 'descending' or 'ascending'.
    """

    def __init__(
        self,
        max_results: int = 20,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        delay: float = 3.0,  # arXiv asks for 1 req/3 sec
    ):
        super().__init__(name="arxiv", description="Search arXiv for academic papers by topic or keyword.")
        self.max_results = max(1, min(max_results, 100))
        self.sort_by = sort_by  # relevance | lastUpdatedDate | submittedDate
        self.sort_order = sort_order  # descending | ascending
        self.delay = delay
        self._last_call: float = 0

    def _execute(
        self,
        query: str,
        max_results: int = None,
    ) -> ToolResult:
        """Search arXiv and return structured paper metadata."""
        import arxiv

        max_results = max_results or self.max_results
        max_results = max(1, min(max_results, 100))

        # Rate limiting
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.time()

        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=getattr(arxiv.SortCriterion, self.sort_by.capitalize())
                    if hasattr(arxiv.SortCriterion, self.sort_by.capitalize())
                    else arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending if self.sort_order == "descending" else arxiv.SortOrder.Ascending,
            )
            papers = []
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [str(a) for a in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat() if result.published else None,
                    "updated": result.updated.isoformat() if result.updated else None,
                    "entry_id": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "comment": result.comment,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": list(result.categories),
                    "links": [{"title": l.title, "href": l.href, "rel": l.rel} for l in result.links],
                })
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "max_results": max_results},
                success=True,
                output={"papers": papers, "count": len(papers), "query": query},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "max_results": max_results},
                success=False,
                error=f"arXiv search failed: {str(e)}",
            )

    def to_definition(self):
        """Return JSON Schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Supports arXiv operators: AND, OR, ANDNOT, all:field:value. E.g. 'ti:transformer AND (cat:cs.LG OR cat:cs.AI)'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-100, default 20)",
                        "default": self.max_results,
                    },
                },
                "required": ["query"],
            },
        }
