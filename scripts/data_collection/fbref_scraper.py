"""
High-level FBRef web-scraper utilities.

This module focuses on fetching raw HTML tables from FBRef that later get
transformed into structured datasets. It follows the same pattern as the
TransferMarkt scraper for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "fbref_config.json"


def _load_config() -> Dict[str, Any]:
    """Load FBRef configuration from JSON file."""
    if not CONFIG_PATH.exists():
        LOGGER.warning("Missing FBRef config file at %s, using defaults", CONFIG_PATH)
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


CONFIG_DATA = _load_config()


@dataclass
class FBRefConfig:
    """Runtime configuration for the FBRef scraper."""

    base_url: str = CONFIG_DATA.get("base_url", "https://fbref.com")
    rate_limit_seconds: float = CONFIG_DATA.get("rate_limit_seconds", 3.0)  # Increased to 3 seconds
    max_retries: int = CONFIG_DATA.get("max_retries", 5)  # Increased retries
    timeout_sec: int = CONFIG_DATA.get("timeout_sec", 30)
    verify_ssl: bool = True
    user_agent: str = CONFIG_DATA.get(
        "user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )
    # Additional rate limiting settings
    initial_retry_delay: float = 5.0  # Initial delay for retries (seconds)
    max_retry_delay: float = 60.0  # Maximum delay between retries (seconds)
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier


class FBRefScraper:
    """Primary interface for fetching FBRef data."""

    def __init__(
        self,
        config: Optional[FBRefConfig] = None,
        session: Optional[Any] = None,
    ) -> None:
        self.config = config or FBRefConfig()
        # Use cloudscraper if available, otherwise fall back to requests
        if session is None:
            if HAS_CLOUDSCRAPER:
                self.session = cloudscraper.create_scraper()
                LOGGER.info("Using cloudscraper to bypass Cloudflare protection")
            else:
                self.session = requests.Session()
                LOGGER.warning(
                    "cloudscraper not installed. Install it with 'pip install cloudscraper' "
                    "for better success rate with FBRef (bypasses Cloudflare protection)"
                )
        else:
            self.session = session
        self._last_request_ts: float = 0.0
        self._setup_session()
        # Visit homepage first to establish session and get cookies
        self._initialize_session()

    def close(self) -> None:
        """Close the HTTP session and clean up resources."""
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass

    def _setup_session(self) -> None:
        """Configure HTTP session headers."""
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
        )

    def _initialize_session(self) -> None:
        """Visit homepage to establish session and get cookies."""
        try:
            # Visit homepage first to get initial cookies
            # Use a longer delay for initial request to let cloudscraper solve challenges
            homepage_url = f"{self.config.base_url}/en/"
            LOGGER.debug("Initializing session by visiting homepage...")
            # Add extra delay before first request
            time.sleep(2.0)
            self._request(homepage_url)
            # Add delay after initial request to ensure session is established
            time.sleep(1.0)
            LOGGER.debug("Session initialized successfully")
        except Exception as e:
            LOGGER.warning(f"Failed to initialize session: {e}. Continuing anyway...")

    def _respect_rate_limit(self) -> None:
        """Enforce rate limiting between requests with random jitter."""
        import random
        delta = time.monotonic() - self._last_request_ts
        # Add random jitter (0-1 second) to avoid predictable patterns
        jitter = random.uniform(0, 1.0)
        remaining = self.config.rate_limit_seconds - delta + jitter
        if remaining > 0:
            LOGGER.debug(f"Rate limiting: sleeping {remaining:.2f} seconds")
            time.sleep(remaining)

    def _request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """Make HTTP request with rate limiting and retry logic."""
        self._respect_rate_limit()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_sec,
                    verify=self.config.verify_ssl,
                )
                # Handle rate limiting and forbidden errors
                if response.status_code in [429, 403]:
                    self._handle_retry(attempt, status=response.status_code)
                    continue
                response.raise_for_status()
                self._last_request_ts = time.monotonic()
                return response
            except requests.RequestException as exc:
                self._handle_retry(attempt, error=exc)

    def _handle_retry(
        self,
        attempt: int,
        status: Optional[int] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Handle retry logic with exponential backoff for failed requests."""
        if attempt >= self.config.max_retries:
            raise RuntimeError(
                f"Failed to fetch FBRef page after {attempt} attempts; "
                f"status={status}, error={error}"
            ) from error
        
        # Exponential backoff: initial_delay * (multiplier ^ (attempt - 1))
        # Cap at max_retry_delay
        wait = min(
            self.config.initial_retry_delay * (self.config.backoff_multiplier ** (attempt - 1)),
            self.config.max_retry_delay
        )
        
        # Add random jitter to avoid thundering herd
        import random
        jitter = random.uniform(0, wait * 0.2)  # Up to 20% jitter
        wait += jitter
        
        # Special handling for 403 errors (rate limiting)
        if status == 403:
            wait = max(wait, 10.0)  # Minimum 10 seconds for 403 errors
            LOGGER.warning(
                "FBRef rate limited (403). Waiting %.1fs before retry %d/%d...",
                wait,
                attempt,
                self.config.max_retries,
            )
        else:
            LOGGER.warning(
                "FBRef request failed (status=%s, error=%s). Retrying in %.1fs (attempt %d/%d)...",
                status,
                error,
                wait,
                attempt,
                self.config.max_retries,
            )
        
        time.sleep(wait)

    def _fetch_soup(self, url: str) -> BeautifulSoup:
        """Fetch URL and return BeautifulSoup object."""
        response = self._request(url)
        return BeautifulSoup(response.content, "html.parser")

    def search_player(self, name: str, club: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search FBRef for players by name.

        Args:
            name: Player name to search for
            club: Optional club name to narrow search

        Returns:
            List of dicts with: {'fbref_id': str, 'name': str, 'url': str, 'club': str}
        """
        # FBRef search URL format
        search_url = f"{self.config.base_url}/en/search/search.fcgi"
        params = {"search": name}
        if club:
            params["club"] = club

        LOGGER.info(f"Searching FBRef for: {name}")
        soup = self._fetch_soup(search_url)

        results = []
        # FBRef search results are typically in a table or list
        # Look for player links in search results
        player_links = soup.select("div.search-result a[href*='/en/players/']")
        
        for link in player_links:
            href = link.get("href", "")
            # Extract player ID from URL: /en/players/{id}/{slug}
            match = re.search(r"/en/players/([a-f0-9]+)/", href)
            if not match:
                continue
            
            fbref_id = match.group(1)
            player_name = link.get_text(strip=True)
            full_url = f"{self.config.base_url}{href}" if href.startswith("/") else href
            
            # Try to extract club from nearby text
            club_name = None
            parent = link.parent
            if parent:
                club_elem = parent.find_next("span", class_="search-result-club")
                if club_elem:
                    club_name = club_elem.get_text(strip=True)
            
            results.append({
                "fbref_id": fbref_id,
                "name": player_name,
                "url": full_url,
                "club": club_name,
            })
        
        LOGGER.info(f"Found {len(results)} search results for {name}")
        return results

    def fetch_player_profile(self, fbref_id: str) -> Dict[str, Any]:
        """
        Fetch player profile page.

        Args:
            fbref_id: FBRef player ID (e.g., 'dc7f8a28')

        Returns:
            Dict with player profile data: name, DOB, nationality, height, position, etc.
        """
        url = f"{self.config.base_url}/en/players/{fbref_id}/"
        LOGGER.info(f"Fetching FBRef profile for player {fbref_id}")
        soup = self._fetch_soup(url)

        data = {}

        # Extract player name from header
        header = soup.select_one("h1")
        if header:
            data["name"] = header.get_text(strip=True)

        # Extract profile info from info box
        info_box = soup.select_one("div.info")
        if info_box:
            # Look for key-value pairs
            info_items = info_box.select("p")
            for item in info_items:
                text = item.get_text(strip=True)
                if ":" in text:
                    key, value = text.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    # Map common keys
                    if "born" in key or "date of birth" in key:
                        data["date_of_birth"] = value
                    elif "position" in key:
                        data["position"] = value
                    elif "height" in key:
                        data["height"] = value
                    elif "nationality" in key or "citizenship" in key:
                        data["nationality"] = value

        return data

    def fetch_player_match_logs(
        self,
        fbref_id: str,
        season: Optional[int] = None,
        competition: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch all match statistics for a player.

        Args:
            fbref_id: FBRef player ID
            season: Optional season year (e.g., 2024). If None, fetches all available seasons.
            competition: Optional competition slug (e.g., 'Premier-League'). If None, fetches all competitions.

        Returns:
            DataFrame with match-by-match statistics
        """
        if season:
            # Fetch specific season
            return self._fetch_season_match_logs(fbref_id, season, competition)
        else:
            # Fetch all available seasons
            all_matches = []
            # FBRef typically shows last 5-6 seasons on the main page
            # We'll need to iterate through season links
            url = f"{self.config.base_url}/en/players/{fbref_id}/matchlogs/"
            soup = self._fetch_soup(url)
            
            # Find all season links
            season_links = soup.select("div.filter a[href*='/matchlogs/']")
            seasons_to_fetch = []
            
            for link in season_links:
                href = link.get("href", "")
                # Extract season from URL: /en/players/{id}/matchlogs/{season}/{competition}/
                match = re.search(r"/matchlogs/(\d{4})/", href)
                if match:
                    season_year = int(match.group(1))
                    seasons_to_fetch.append(season_year)
            
            # If no season links found, try to extract from current page
            if not seasons_to_fetch:
                # Look for current season in page content
                current_year = pd.Timestamp.now().year
                # Try last 10 seasons
                for year in range(current_year, current_year - 10, -1):
                    seasons_to_fetch.append(year)
            
            # Fetch each season
            for season_year in seasons_to_fetch:
                try:
                    season_matches = self._fetch_season_match_logs(fbref_id, season_year, competition)
                    if not season_matches.empty:
                        all_matches.append(season_matches)
                except Exception as e:
                    LOGGER.warning(f"Failed to fetch season {season_year} for player {fbref_id}: {e}")
                    continue
            
            if not all_matches:
                return pd.DataFrame()
            
            return pd.concat(all_matches, ignore_index=True)

    def _fetch_season_match_logs(
        self,
        fbref_id: str,
        season: int,
        competition: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch match logs for a specific season and competition.

        Args:
            fbref_id: FBRef player ID
            season: Season year (e.g., 2024 for 2024-25 season)
            competition: Optional competition slug (e.g., 'Premier-League')

        Returns:
            DataFrame with match statistics
        """
        # Build URL
        if competition:
            url = f"{self.config.base_url}/en/players/{fbref_id}/matchlogs/{season}/{competition}/"
        else:
            # Fetch all competitions for this season
            url = f"{self.config.base_url}/en/players/{fbref_id}/matchlogs/{season}/"
        
        LOGGER.info(f"Fetching match logs for player {fbref_id}, season {season}, competition {competition or 'all'}")
        soup = self._fetch_soup(url)

        # FBRef match logs are in tables with class "stats_table"
        match_tables = soup.select("table.stats_table")
        
        if not match_tables:
            LOGGER.warning(f"No match tables found for player {fbref_id}, season {season}")
            return pd.DataFrame()

        all_matches = []
        
        for table in match_tables:
            try:
                # Use pandas to parse the table
                from io import StringIO
                df = pd.read_html(StringIO(str(table)), flavor="bs4")[0]
                
                # Clean up the DataFrame
                # Remove summary rows (they contain "Squad:" or "Team:")
                mask = pd.Series([True] * len(df))
                for col in df.columns:
                    if df[col].dtype == 'object':
                        mask = mask & ~df[col].astype(str).str.contains('Squad:', na=False, case=False)
                        mask = mask & ~df[col].astype(str).str.contains('Team:', na=False, case=False)
                
                df = df[mask].copy()
                
                if not df.empty:
                    # Add metadata columns
                    df['fbref_player_id'] = fbref_id
                    df['season'] = f"{season}-{season+1}"
                    if competition:
                        df['competition'] = competition
                    
                    all_matches.append(df)
            except Exception as e:
                LOGGER.warning(f"Failed to parse match table: {e}")
                continue
        
        if not all_matches:
            return pd.DataFrame()
        
        # Combine all tables
        combined = pd.concat(all_matches, ignore_index=True)
        
        # Standardize column names (will be done in transformer, but basic cleanup here)
        # Remove multi-level column headers if present
        if isinstance(combined.columns, pd.MultiIndex):
            combined.columns = ['_'.join(col).strip() for col in combined.columns.values]
        
        return combined


__all__ = [
    "FBRefConfig",
    "FBRefScraper",
]

