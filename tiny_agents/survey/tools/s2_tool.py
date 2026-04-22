"""Semantic Scholar API tool — fetch real paper metadata, citations, and references."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from tiny_agents.tools.base import BaseTool, ToolResult


S2_BASE = "https://api.semanticscholar.org/graph/v1"
# Free tier: 100 req/5min without key, 1000/5min with key
S2_KEY = os.environ.get("S2_API_KEY", "")


class S2Tool(BaseTool):
    """Fetch real paper metadata from Semantic Scholar API.

    Methods:
        search: Search papers by query, return metadata + citation counts
        get_paper: Get detailed info for specific papers by DOI or URL
        get_references: Get references for a specific paper
        get_citations: Get papers that cite a specific paper

    Note: Without an API key, rate limits apply. Set S2_API_KEY env var for higher limits.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        super().__init__(name="semantic_scholar", description="Semantic Scholar API for academic paper metadata, citations, and references.")
        self.api_key = api_key or S2_KEY
        self.timeout = timeout
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"x-api-key": self.api_key})

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _rate_limit(self) -> None:
        """Simple rate limiting: 100 req/min without key."""
        now = time.time()
        if not hasattr(self, "_last_request"):
            self._last_request = now
            return
        elapsed = now - self._last_request
        min_interval = 0.6 if self.api_key else 1.0  # ~100/min or ~60/min
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        self._rate_limit()
        url = f"{S2_BASE}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            elif resp.status_code == 429:
                # Rate limited — wait and retry once
                time.sleep(5)
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
            return None
        except Exception:
            return None

    def search(
        self,
        query: str,
        top_k: int = 10,
        year: Optional[int] = None,
        return_fields: Optional[List[str]] = None,
    ) -> ToolResult:
        """Search papers by query and return metadata.

        Args:
            query: Search query (paper title, topic, author)
            top_k: Number of results to return (max 100)
            year: Filter by year (optional)
            return_fields: Which fields to include (default: title,authors,year,abstract,citationCount,externalIds)

        Returns:
            ToolResult with list of paper metadata dicts
        """
        if return_fields is None:
            return_fields = [
                "title",
                "authors",
                "year",
                "abstract",
                "citationCount",
                "externalIds",
                "venue",
                "publicationDate",
            ]

        params = {
            "query": query,
            "limit": min(top_k, 100),
            "fields": ",".join(return_fields),
        }
        if year:
            params["year"] = str(year)

        data = self._get("paper/search", params)
        if data is None:
            return ToolResult(tool_name=self.name, args={"query": query}, success=True, output={"papers": [], "query": query}, error="API request failed")

        papers = data.get("data", [])
        return ToolResult(
            tool_name=self.name,
            args={"query": query, "top_k": top_k},
            success=True,
            output={
                "papers": papers,
                "total": data.get("total", len(papers)),
                "query": query,
            },
        )

    def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None,
    ) -> ToolResult:
        """Get detailed metadata for a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID, DOI (10.xxxx/...), or arXiv ID
            fields: Which fields to return (default: all major fields)

        Returns:
            ToolResult with paper metadata dict
        """
        if fields is None:
            fields = [
                "title",
                "authors",
                "year",
                "abstract",
                "citationCount",
                "references",
                "citations",
                "externalIds",
                "venue",
                "publicationDate",
                "journal",
                "conferenceVenue",
            ]

        params = {"fields": ",".join(fields)}
        data = self._get(f"paper/{paper_id}", params)

        if data is None:
            return ToolResult(tool_name=self.name, args={"paper_id": paper_id}, success=True, output=None, error=f"Paper not found: {paper_id}")

        return ToolResult(tool_name=self.name, args={}, success=True, output=data)

    def get_references(self, paper_id: str, top_k: int = 20) -> ToolResult:
        """Get referenced papers for a given paper.

        Args:
            paper_id: Semantic Scholar paper ID or DOI
            top_k: Number of references to return

        Returns:
            ToolResult with list of reference metadata dicts
        """
        params = {
            "fields": "title,authors,year,abstract,citationCount,externalIds",
            "limit": top_k,
        }
        data = self._get(f"paper/{paper_id}/references", params)
        if data is None:
            return ToolResult(tool_name=self.name, args={"paper_id": paper_id}, success=True, output={"references": [], "paper_id": paper_id})
        refs = [r.get("citedPaper", {}) for r in data.get("data", []) if r.get("citedPaper")]
        return ToolResult(tool_name=self.name, args={"paper_id": paper_id}, success=True, output={"references": refs, "paper_id": paper_id})

    def get_citations(self, paper_id: str, top_k: int = 20) -> ToolResult:
        """Get citing papers for a given paper.

        Args:
            paper_id: Semantic Scholar paper ID or DOI
            top_k: Number of citations to return

        Returns:
            ToolResult with list of citation metadata dicts
        """
        params = {
            "fields": "title,authors,year,abstract,citationCount,externalIds",
            "limit": top_k,
        }
        data = self._get(f"paper/{paper_id}/citations", params)
        if data is None:
            return ToolResult(tool_name=self.name, args={"paper_id": paper_id}, success=True, output={"citations": [], "paper_id": paper_id})
        cites = [r.get("citingPaper", {}) for r in data.get("data", []) if r.get("citingPaper")]
        return ToolResult(tool_name=self.name, args={"paper_id": paper_id}, success=True, output={"citations": cites, "paper_id": paper_id})

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute a Semantic Scholar operation.

        Args:
            input_data: {
                "operation": "search" | "get_paper" | "get_references" | "get_citations",
                "query": str,           # for search
                "paper_id": str,         # for get_paper / get_references / get_citations
                "top_k": int,           # number of results (default 10)
                "year": int,            # for search (optional)
                "fields": List[str],   # which fields to return (optional)
            }
        """
        operation = input_data.get("operation", "search")
        top_k = min(input_data.get("top_k", 10), 100)

        if operation == "search":
            return self.search(
                query=input_data.get("query", ""),
                top_k=top_k,
                year=input_data.get("year"),
                return_fields=input_data.get("fields"),
            )
        elif operation == "get_paper":
            result = self.get_paper(
                paper_id=input_data.get("paper_id", ""),
                fields=input_data.get("fields"),
            )
            return result
        elif operation == "get_references":
            return self.get_references(
                paper_id=input_data.get("paper_id", ""),
                top_k=top_k,
            )
        elif operation == "get_citations":
            return self.get_citations(
                paper_id=input_data.get("paper_id", ""),
                top_k=top_k,
            )
        else:
            return ToolResult(tool_name=self.name, args={"operation": operation}, success=False, output=None, error=f"Unknown operation: {operation}")

    def _execute(self, **kwargs) -> ToolResult:
        """Required by BaseTool abstract method — delegates to execute()."""
        return self.execute(kwargs)
