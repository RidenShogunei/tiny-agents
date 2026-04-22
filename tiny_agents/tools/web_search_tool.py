"""WebSearchTool — search the web for information.

NOTE: Web search through most free services (Google, DuckDuckGo) is blocked
by the current network environment. This tool uses:
1. Bing search (accessible via proxy) when available
2. Falls back to returning instructions for manual search

For academic paper discovery, prefer ArxivTool which works reliably.

Set SERPER_API_KEY for full Serper API support (free tier: 2500/mo).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from tiny_agents.tools.base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Search the web for relevant information.

    Supports two backends:
    - Serper API (preferred, set SERPER_API_KEY env var)
    - Bing HTML (fallback when no API key)

    Args:
        query: Search query string.
        num_results: Number of results to return (default 10).
        api_key: Serper API key. If None, reads from SERPER_API_KEY env var.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_results: int = 10,
        timeout: int = 10,
    ):
        super().__init__(name="web_search", description="Search the web for information.")
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.num_results = num_results
        self.timeout = timeout
        self._use_serper = bool(self.api_key)

    def _execute(
        self,
        query: str,
        num_results: int = None,
    ) -> ToolResult:
        num_results = num_results or self.num_results

        if self._use_serper:
            results = self._search_serper(query, num_results)
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "num_results": num_results},
                success=True,
                output={"results": results, "engine": "serper"},
            )

        # Try Bing as fallback
        bing_results = self._search_bing(query, num_results)
        if bing_results:
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "num_results": num_results},
                success=True,
                output={"results": bing_results, "engine": "bing"},
            )

        # Bing failed too — graceful fallback
        return ToolResult(
            tool_name=self.name,
            args={"query": query, "num_results": num_results},
            success=True,
            output={
                "results": [],
                "engine": "none",
                "message": (
                    "Web search is unavailable in this environment. "
                    "For academic papers, use the arxiv tool instead."
                ),
                "fallback_suggestion": f"Manual search query: {query}",
            },
        )

    def _search_serper(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Serper API."""
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "numResults": num_results}
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            return [{"error": f"Serper API error: {resp.status_code} {resp.text[:200]}"}]
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "serper",
            })
        return results

    def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Fallback: scrape Bing HTML via proxy."""
        proxies = {
            "http": "socks5://127.0.0.1:39211",
            "https": "socks5://127.0.0.1:39211",
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&count={num_results}"
        resp = requests.get(url, headers=headers, proxies=proxies, timeout=self.timeout)
        if resp.status_code != 200:
            return []
        results = []
        # Parse Bing HTML for results
        # Format: <li class="b_algo"> ... <h2><a href="URL">TITLE</a></h2> ... <p>SNIPPET</p>
        html = resp.text
        # Split into result blocks
        algo_blocks = re.split(r'<li class="b_algo"', html)
        for block in algo_blocks[1:num_results + 1]:
            # Extract URL and title from <a> inside <h2>
            title_match = re.search(r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', block, re.DOTALL)
            snippet_match = re.search(r'<p class="b_paractl"[^>]*>(.*?)</p>', block, re.DOTALL)
            if title_match:
                url = title_match.group(1)
                title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "bing",
                })
        return results
