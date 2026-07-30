"""Kimi Search backend.

Calls Kimi's agent-gw search endpoint (the same one used by the Kimi-Search
OpenClaw plugin). Reachable from CN servers, unlike brave/startpage/yahoo.

Config (env vars):
    KIMI_SEARCH_API_KEY  Kimi apiKey (Bearer token). Required.
    KIMI_SEARCH_BASE_URL Search endpoint. Default: https://agent-gw.kimi.com/coding/v1/search
    KIMI_SEARCH_TIMEOUT  Per-request timeout seconds. Default: 30
"""

import os
from typing import Any

import httpx

from .base import WebSearchBackend
from .registry import register_backend

_DEFAULT_BASE_URL = "https://agent-gw.kimi.com/coding/v1/search"
_DEFAULT_TIMEOUT = 30
_DEFAULT_USER_AGENT = "Kimi Claw Plugin"


@register_backend
class KimiBackend(WebSearchBackend):
    """Kimi Search API backend (agent-gw.kimi.com)."""

    name = "kimi"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("KIMI_SEARCH_API_KEY", "")
        self.base_url = (
            os.environ.get("KIMI_SEARCH_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        )
        try:
            parsed = int(os.environ.get("KIMI_SEARCH_TIMEOUT", "") or "0")
            self.timeout = parsed if parsed >= 1 else _DEFAULT_TIMEOUT
        except ValueError:
            self.timeout = _DEFAULT_TIMEOUT

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, count: int, **kwargs: Any) -> str:
        if not self.api_key:
            return "Error: KIMI_SEARCH_API_KEY not configured"

        n = min(max(count, 1), 20)
        tool_call_id = kwargs.get("tool_call_id") or "ov-vikingbot"

        headers = {
            "User-Agent": _DEFAULT_USER_AGENT,
            "Authorization": f"Bearer {self.api_key}",
            "X-Msh-Tool-Call-Id": str(tool_call_id),
            "Content-Type": "application/json",
        }
        payload = {
            "text_query": query,
            "limit": n,
            "enable_page_crawling": False,
            "timeout_seconds": self.timeout,
        }

        try:
            # Kimi gateway occasionally returns 408 on slow queries; retry once.
            for attempt in range(2):
                async with httpx.AsyncClient(timeout=self.timeout + 15) as client:
                    resp = await client.post(self.base_url, headers=headers, json=payload)
                if resp.status_code == 200:
                    break
                if resp.status_code != 408 or attempt:
                    return f"Error: Kimi search failed (status {resp.status_code})"
            data = resp.json()
        except Exception as e:
            return f"Error: {e}"

        results = data.get("search_results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return f"No results for: {query}"

        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results[:n], 1):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            content = item.get("content", "") or ""
            date = item.get("date", "")
            lines.append(f"{i}. {title}\n   {url}")
            if date:
                lines.append(f"   Date: {date}")
            if snippet:
                snippet_text = snippet[:500]
                suffix = "..." if len(snippet) > 500 else ""
                lines.append(f"   {snippet_text}{suffix}")
            if content:
                content_text = content[:1000]
                suffix = "..." if len(content) > 1000 else ""
                lines.append(f"   {content_text}{suffix}")
        return "\n".join(lines)
