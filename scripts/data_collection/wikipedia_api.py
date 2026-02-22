"""
Wikipedia / MediaWiki API client for player disambiguation and bio data.

Uses the public MediaWiki API (e.g. en.wikipedia.org/w/api.php) – no API key required.
Use for: search by name, get page extract/summary (DOB, nationality, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "IPM-Player-Identity/1.0 (Python; Player Identity Layer)"


def search(
    query: str,
    limit: int = 5,
    api_url: str = DEFAULT_API_URL,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Search Wikipedia by query (e.g. player name).

    Returns:
        List of dicts with keys: title, pageid, snippet (optional).
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": min(limit, 50),
        "srnamespace": 0,
    }
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
    try:
        r = sess.get(api_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("query", {}).get("search", [])
    except Exception as e:
        LOGGER.warning("Wikipedia search failed for %s: %s", query, e)
        return []


def get_page_extract(
    page_title: str,
    intro_only: bool = True,
    api_url: str = DEFAULT_API_URL,
    session: Optional[requests.Session] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get page extract (summary) for a Wikipedia page title.

    Returns:
        Dict with keys: title, extract, url (and optionally pageid). None on failure.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "exintro": "1" if intro_only else "0",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
    }
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
    try:
        r = sess.get(api_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page_id = next(iter(pages))
        page = pages[page_id]
        if page_id == "-1":
            return None
        extract = page.get("extract", "").strip()
        title = page.get("title", page_title)
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
        return {"title": title, "extract": extract, "url": url, "pageid": page_id}
    except Exception as e:
        LOGGER.warning("Wikipedia get_page_extract failed for %s: %s", page_title, e)
        return None


def fetch_player_wikipedia(page_title_or_id: str) -> Dict[str, Any]:
    """
    Fetch Wikipedia data for a player (page title or ID from external_id_mappings).

    Used by the player identity fetcher when source=wikipedia and external_id is set.
    external_id is typically the page title (e.g. "Harry_Kane") or the full title with spaces.

    Returns:
        Dict with title, extract, url (and pageid). Empty dict on failure.
    """
    title = page_title_or_id.replace("_", " ") if "_" in page_title_or_id else page_title_or_id
    result = get_page_extract(title)
    return result if result else {}
