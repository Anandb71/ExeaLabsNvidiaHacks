"""Tavily Search API client for web research (appliance manuals, clearances, safety guidelines).

Supports standard TAVILY_API_KEY from environment or local .env.
Falls back to TavilyNotConfigured when no API key is available.
"""

from __future__ import annotations

import json
import urllib.request

from arrival import config as settings
from arrival.config import load_dotenv  # noqa: F401  (kept importable from here)

DEFAULT_BASE_URL = "https://api.tavily.com/search"


class TavilyNotConfigured(RuntimeError):
    pass


def config() -> dict:
    return {
        "api_key": settings.get("TAVILY_API_KEY"),
        "base_url": settings.get("TAVILY_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
    }


def validate_reply(data) -> dict:
    """Validate the reply envelope and return safe text fields, retaining usable results and extra API metadata.

    Valid replies are unchanged. Malformed result rows are skipped; null/non-string text is never passed to
    research consumers or stored in the canonical cache. An entirely unusable malformed result list is an error.
    """
    if not isinstance(data, dict):
        raise ValueError("Tavily reply must be a JSON object")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Tavily reply results must be a list")
    answer = data.get("answer", "")
    if not isinstance(answer, str):
        if answer is not None and not results:
            raise ValueError("Tavily reply answer must be text or null")
        answer = ""
    safe = []
    for result in results:
        if not isinstance(result, dict):
            continue
        row = dict(result)
        for field in ("title", "url", "content"):
            if not isinstance(row.get(field), str):
                row[field] = ""
        if any(isinstance(row.get(field), str) and row[field].strip() for field in ("title", "url", "content")):
            safe.append(row)
    if results and not safe and not answer.strip():
        raise ValueError("Tavily reply has no usable result text or answer")
    out = {**data, "results": safe}
    if "answer" in data:
        out["answer"] = answer
    return out


def search(
    query: str, search_depth: str = "basic", include_answer: bool = True, max_results: int = 5, timeout: float = 30.0
) -> dict:
    """Execute a web search using the Tavily Search API.

    Returns the parsed JSON dictionary from Tavily containing 'query', 'answer',
    and list of 'results'. Malformed envelopes raise ValueError; unusable result rows are filtered and null or
    non-string text fields normalized before consumers can inspect them.
    """
    cfg = config()
    if not cfg["api_key"]:
        raise TavilyNotConfigured("TAVILY_API_KEY is not set in environment or .env")

    payload = {
        "api_key": cfg["api_key"],
        "query": query,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "max_results": max_results,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        cfg["base_url"],
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        try:
            data = json.load(resp)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Tavily reply is not valid JSON") from error
    return validate_reply(data)
