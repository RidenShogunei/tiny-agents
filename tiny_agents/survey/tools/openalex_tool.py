"""OpenAlex API tool — fetch real paper metadata, citations, and references.

OpenAlex is a free/open catalog of scholarly papers:
  - No API key required
  - ~10 req/sec rate limit (10k/month free tier)
  - Real citation counts, references, abstracts, venue info
  - Good CS/AI coverage
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from tiny_agents.tools.base import BaseTool, ToolResult


OA_BASE = "https://api.openalex.org"


class OpenAlexTool(BaseTool):
    """Fetch real paper metadata from OpenAlex API.

    Methods:
        search: Search papers by query, return metadata + citation counts
        get_paper: Get detailed info for a specific paper by DOI or OpenAlex ID
        get_references: Get referenced papers for a specific paper
        get_citations: Get papers that cite a specific paper

    Note: Rate limit ~10 req/sec. Set OPENALEX_EMAIL env var for polite pool.
    """

    def __init__(self, email: Optional[str] = None, timeout: int = 15):
        super().__init__(
            name="openalex",
            description="Search academic papers via OpenAlex API — free, no key required. Returns real citation counts, abstracts, and references.",
        )
        import os as _os
        self.email = email or _os.environ.get("OPENALEX_EMAIL", "tiny-agents@example.com")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": f"tiny-agents/1.0 (mailto:{self.email})"})
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < 0.1:  # 10 req/sec max
            time.sleep(0.1 - elapsed)
        self._last_request = time.time()

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        self._rate_limit()
        url = f"{OA_BASE}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            elif resp.status_code == 429:
                time.sleep(2)
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
            return None
        except Exception:
            return None

    def _format_author_list(self, authorships: List[Dict]) -> List[str]:
        """Extract author names from OpenAlex authorship list."""
        result = []
        for a in authorships[:5]:
            au = a.get("author", {})
            name = au.get("display_name", "")
            if name:
                result.append(name)
        return result

    def _paper_from_result(self, r: Dict) -> Dict:
        """Convert OpenAlex work result to our standard paper dict."""
        # Extract abstract (inverted index -> plain text)
        abstract = ""
        inv_idx = r.get("abstract_inverted_index")
        if inv_idx:
            words = []
            for word, positions in sorted(inv_idx.items(), key=lambda x: x[1][0]):
                words.append(word)
            abstract = " ".join(words)

        # Year
        year = r.get("publication_year")

        # Authors
        authorships = r.get("authorships", [])
        authors = self._format_author_list(authorships)

        # DOI
        doi = r.get("doi", "")

        # Concepts (research themes)
        concepts = [c.get("display_name", "") for c in r.get("concepts", [])[:5] if c.get("display_name")]

        # OpenAlex ID (strip https://openalex.org/)
        oa_id = r.get("id", "")
        if oa_id and oa_id.startswith("https://openalex.org/"):
            oa_id = oa_id[len("https://openalex.org/"):]

        return {
            "title": r.get("title", "Unknown"),
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "openalex_id": oa_id,
            "citation_count": r.get("cited_by_count", 0),
            "venue": r.get("display_name", ""),  # journal or conference
            "paper_url": r.get("doi", ""),
            "openalex_url": r.get("id", ""),
            "concepts": concepts,
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        year: Optional[int] = None,
        concepts: Optional[List[str]] = None,
    ) -> ToolResult:
        """Search papers by query and return metadata.

        Args:
            query: Search query (paper title, topic, author)
            top_k: Number of results to return (max 100)
            year: Filter by year (optional)
            concepts: Filter by research concepts (optional, e.g. ["computer vision"])

        Returns:
            ToolResult with list of paper metadata dicts
        """
        params: Dict[str, Any] = {
            "search": query,
            "per-page": min(top_k, 100),
            "select": ",".join([
                "id", "title", "authorships", "publication_year",
                "doi", "cited_by_count", "display_name",
                "abstract_inverted_index", "concepts",
            ]),
        }
        if year:
            params["filter"] = f"publication_year:{year}"
        if concepts:
            concept_filter = "|".join(concepts)
            params["filter"] = (params.get("filter", "") + f",concepts:{concept_filter}").lstrip(",")

        data = self._get("works", params)
        if data is None:
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "top_k": top_k},
                success=True,
                output={"papers": [], "query": query},
                error="API request failed",
            )

        papers = [self._paper_from_result(r) for r in data.get("results", [])]
        return ToolResult(
            tool_name=self.name,
            args={"query": query, "top_k": top_k},
            success=True,
            output={
                "papers": papers,
                "total": data.get("meta", {}).get("count", len(papers)),
                "query": query,
            },
        )

    def get_paper(self, paper_id: str) -> ToolResult:
        """Get detailed metadata for a specific paper.

        Args:
            paper_id: OpenAlex ID (e.g. 'W2897658767') or DOI URL

        Returns:
            ToolResult with paper metadata dict
        """
        # If it's a DOI URL, use the works/doi: endpoint
        if paper_id.startswith("10."):
            data = self._get("works/doi:" + paper_id, {})
        else:
            # Strip prefix if present
            oid = paper_id
            if oid.startswith("https://openalex.org/"):
                oid = oid[len("https://openalex.org/"):]
            data = self._get(f"works/{oid}", {})

        if data is None:
            return ToolResult(
                tool_name=self.name,
                args={"paper_id": paper_id},
                success=True,
                output=None,
                error=f"Paper not found: {paper_id}",
            )

        paper = self._paper_from_result(data)
        return ToolResult(
            tool_name=self.name,
            args={"paper_id": paper_id},
            success=True,
            output=paper,
        )

    def get_references(self, paper_id: str) -> ToolResult:
        """Get referenced papers for a given paper.

        Args:
            paper_id: OpenAlex ID or DOI URL

        Returns:
            ToolResult with list of reference metadata dicts
        """
        if paper_id.startswith("10."):
            data = self._get("works/doi:" + paper_id, {"filter": "referenced_works"})
        else:
            oid = paper_id
            if oid.startswith("https://openalex.org/"):
                oid = oid[len("https://openalex.org/"):]
            data = self._get(f"works/{oid}", {})

        if data is None:
            return ToolResult(
                tool_name=self.name,
                args={"paper_id": paper_id},
                success=True,
                output={"references": [], "paper_id": paper_id},
            )

        refs = data.get("referenced_works", [])
        # Resolve referenced work IDs to basic metadata
        ref_papers = []
        for ref_id in refs[:20]:
            if ref_id.startswith("https://openalex.org/"):
                ref_id = ref_id[len("https://openalex.org/"):]
            ref_data = self._get(f"works/{ref_id}", {
                "select": "id,title,authorships,publication_year,cited_by_count,doi"
            })
            if ref_data:
                ref_papers.append(self._paper_from_result(ref_data))

        return ToolResult(
            tool_name=self.name,
            args={"paper_id": paper_id},
            success=True,
            output={"references": ref_papers, "paper_id": paper_id},
        )

    def get_citations(self, paper_id: str, top_k: int = 20) -> ToolResult:
        """Get citing papers for a given paper.

        Args:
            paper_id: OpenAlex ID or DOI URL

        Returns:
            ToolResult with list of citing paper metadata dicts
        """
        if paper_id.startswith("10."):
            filter_str = f"referenced_works:doi:{paper_id}"
        else:
            oid = paper_id
            if oid.startswith("https://openalex.org/"):
                oid = oid[len("https://openalex.org/"):]
            filter_str = f"referenced_works:{oid}"

        data = self._get("works", {
            "filter": filter_str,
            "per-page": top_k,
            "select": "id,title,authorships,publication_year,cited_by_count,doi",
        })

        if data is None:
            return ToolResult(
                tool_name=self.name,
                args={"paper_id": paper_id},
                success=True,
                output={"citations": [], "paper_id": paper_id},
            )

        citations = [self._paper_from_result(r) for r in data.get("results", [])]
        return ToolResult(
            tool_name=self.name,
            args={"paper_id": paper_id},
            success=True,
            output={"citations": citations, "paper_id": paper_id},
        )

    def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute an OpenAlex operation.

        Args:
            input_data: {
                "operation": "search" | "get_paper" | "get_references" | "get_citations",
                "query": str,           # for search
                "paper_id": str,        # for get_paper / get_references / get_citations
                "top_k": int,           # number of results (default 10)
                "year": int,            # filter by year (optional)
                "concepts": List[str],  # filter by concepts (optional)
            }
        """
        operation = input_data.get("operation", "search")
        top_k = min(input_data.get("top_k", 10), 100)

        if operation == "search":
            return self.search(
                query=input_data.get("query", ""),
                top_k=top_k,
                year=input_data.get("year"),
                concepts=input_data.get("concepts"),
            )
        elif operation == "get_paper":
            return self.get_paper(paper_id=input_data.get("paper_id", ""))
        elif operation == "get_references":
            return self.get_references(paper_id=input_data.get("paper_id", ""))
        elif operation == "get_citations":
            return self.get_citations(
                paper_id=input_data.get("paper_id", ""),
                top_k=top_k,
            )
        else:
            return ToolResult(
                tool_name=self.name,
                args={"operation": operation},
                success=False,
                output=None,
                error=f"Unknown operation: {operation}",
            )

    def _execute(self, **kwargs) -> ToolResult:
        """Required by BaseTool abstract method — delegates to execute()."""
        return self.execute(kwargs)
