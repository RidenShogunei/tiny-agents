"""WebSearchTool — search the web for information.

Network environment: Google/DuckDuckGo blocked. Available free sources:
1. cn.bing.com — Bing Chinese, accessible via proxy
2. 360 so.com — Chinese search engine, accessible via proxy
3. Tavily API — free tier (1000/mo), set TAVILY_API_KEY env var

For academic papers, use ArxivTool which works reliably.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from tiny_agents.tools.base import BaseTool, ToolResult

_PROXY = {"http": "socks5://127.0.0.1:39211", "https": "socks5://127.0.0.1:39211"}
_BING_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class WebSearchTool(BaseTool):
    """Search the web for relevant information.

    Backends (tried in order):
    1. Tavily API — set TAVILY_API_KEY env var (most reliable, 1000/mo free)
    2. cn.bing.com — Bing Chinese, no key required
    3. 360 so.com — Chinese search fallback, no key required

    Args:
        query: Search query string.
        num_results: Number of results to return (default 10).
    """

    def __init__(
        self,
        num_results: int = 10,
        timeout: int = 10,
    ):
        super().__init__(name="web_search", description="Search the web for information.")
        self.num_results = num_results
        self.timeout = timeout
        self.tavily_key = os.environ.get("TAVILY_API_KEY")

    def _execute(
        self,
        query: str,
        num_results: int = None,
    ) -> ToolResult:
        num_results = num_results or self.num_results

        # 1. Try Tavily API
        if self.tavily_key:
            results = self._search_tavily(query, num_results)
            if results:
                return ToolResult(
                    tool_name=self.name,
                    args={"query": query, "num_results": num_results},
                    success=True,
                    output={"results": results, "engine": "tavily"},
                )

        # 2. Try cn.bing.com
        results = self._search_bing(query, num_results)
        if results:
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "num_results": num_results},
                success=True,
                output={"results": results, "engine": "bing"},
            )

        # 3. Try 360 search
        results = self._search_360(query, num_results)
        if results:
            return ToolResult(
                tool_name=self.name,
                args={"query": query, "num_results": num_results},
                success=True,
                output={"results": results, "engine": "360"},
            )

        # All failed
        return ToolResult(
            tool_name=self.name,
            args={"query": query, "num_results": num_results},
            success=True,
            output={
                "results": [],
                "engine": "none",
                "message": (
                    "Web search unavailable. For academic papers use arxiv tool instead."
                ),
                "fallback_suggestion": f"Manual search: {query}",
            },
        )

    def _search_tavily(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search via Tavily API (free tier: 1000 queries/month)."""
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": num_results},
                headers={"Content-Type": "application/json", "api-key": self.tavily_key},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", r.get("description", "")),
                    "source": "tavily",
                }
                for r in data.get("results", [])[:num_results]
            ]
        except Exception:
            return []

    def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Scrape cn.bing.com via socks5 proxy."""
        try:
            resp = requests.get(
                f"https://cn.bing.com/search?q={requests.utils.quote(query)}&count={num_results}",
                headers={"User-Agent": _BING_UA},
                proxies=_PROXY,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            return self._parse_bing_html(resp.text, num_results)
        except Exception:
            return []

    def _parse_bing_html(self, html: str, num_results: int) -> List[Dict[str, Any]]:
        """Parse Bing HTML results."""
        results = []
        # Each result is in a <li class="b_algo"> block
        blocks = re.split(r'<li class="b_algo"', html)
        for block in blocks[1:num_results + 1]:
            title_m = re.search(
                r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', block, re.DOTALL
            )
            snippet_m = re.search(r'<p class="b_paractl"[^>]*>(.*?)</p>', block, re.DOTALL)
            if title_m:
                url = title_m.group(1)
                title = re.sub(r'<[^>]+>', "", title_m.group(2)).strip()
                snippet = re.sub(r'<[^>]+>', "", snippet_m.group(1)).strip() if snippet_m else ""
                results.append({"title": title, "url": url, "snippet": snippet, "source": "bing"})
        return results

    def _search_360(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Scrape 360 so.com via socks5 proxy (Chinese search engine)."""
        try:
            resp = requests.get(
                f"https://www.so.com/s?q={requests.utils.quote(query)}&pn=1&rn={num_results}",
                headers={"User-Agent": _BING_UA},
                proxies=_PROXY,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            return self._parse_360_html(resp.text, num_results)
        except Exception:
            return []

    def _parse_360_html(self, html: str, num_results: int) -> List[Dict[str, Any]]:
        """Parse 360 so.com HTML results."""
        results = []
        # 360 uses <h3 class="res-title"> and <p class="res-desc">
        titles = re.findall(r'<h3[^>]*class="res-title[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', html, re.DOTALL)
        descs = re.findall(r'<p class="res-desc[^"]*"[^>]*>(.*?)</p>', html, re.DOTALL)
        for i, (url, title) in enumerate(titles[:num_results]):
            title = re.sub(r'<[^>]+>', "", title).strip()
            snippet = re.sub(r'<[^>]+>', "", descs[i]).strip() if i < len(descs) else ""
            results.append({"title": title, "url": url, "snippet": snippet, "source": "360"})
        return results
