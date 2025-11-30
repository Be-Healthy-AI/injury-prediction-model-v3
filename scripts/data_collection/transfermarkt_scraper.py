"""
High-level Transfermarkt web-scraper utilities.

This module focuses on fetching raw HTML tables/blocks that later get
transformed into the canonical datasets defined in
`documentation/transfermarkt_schema.md`. It purposely keeps the parsing
lightweight so downstream transformers can standardize values per schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    """Runtime configuration for the Transfermarkt scraper."""

    base_url: str = "https://www.transfermarkt.com"
    api_base_url: str = "https://tmapi-alpha.transfermarkt.technology"
    language_header: str = "en-US,en;q=0.9"
    timeout_sec: int = 30
    rate_limit_seconds: float = 1.0
    max_retries: int = 3
    verify_ssl: bool = True
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )


class TransfermarktScraper:
    """Primary interface for fetching Transfermarkt tables."""

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or ScraperConfig()
        self.session = session or requests.Session()
        self._last_request_ts: float = 0.0
        self._transfer_history_cache: Dict[int, Dict[str, any]] = {}
        self._club_cache: Dict[str, Dict[str, any]] = {}
        self._setup_session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_current_squad(
        self,
        club_slug: str,
        club_id: int,
        path: str = "startseite",
    ) -> pd.DataFrame:
        """
        Fetch the current squad table for a club.

        Args:
            club_slug: URL slug shown in Transfermarkt (e.g., 'sl-benfica').
            club_id: Numeric club identifier (e.g., 294).
            path: Optional path override (default 'startseite').
        """
        url = f"{self.config.base_url}/{club_slug}/{path}/verein/{club_id}"
        soup = self._fetch_soup(url)
        return self._extract_table(soup, "table.items")

    def fetch_player_transfer_history_api(self, player_id: int) -> Dict[str, Any]:
        """
        Retrieve structured transfer history from the Transfermarkt TMAPI.

        The response contains the list of completed (`terminated`) transfers as well as
        upcoming transfers. Results are cached per player to avoid duplicate requests.
        """
        if player_id in self._transfer_history_cache:
            return self._transfer_history_cache[player_id]

        url = f"{self.config.api_base_url}/transfer/history/player/{player_id}"
        payload = self._get_json(url, params={"locale": "en"})
        history = payload.get("data", {}).get("history", {}) if isinstance(payload, dict) else {}
        self._transfer_history_cache[player_id] = history
        return history

    def fetch_club_profile(self, club_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return metadata for a given club id using TMAPI."""
        if not club_id:
            return None
        club_key = str(club_id)
        if club_key in self._club_cache:
            return self._club_cache[club_key]
        url = f"{self.config.api_base_url}/club/{club_key}"
        payload = self._get_json(url, params={"locale": "en"})
        data = payload.get("data") if isinstance(payload, dict) else None
        if data:
            self._club_cache[club_key] = data
        return data

    def get_club_name(self, club_id: Optional[str]) -> Optional[str]:
        profile = self.fetch_club_profile(club_id)
        if not profile:
            return None
        return profile.get("name") or profile.get("baseDetails", {}).get("shortName")

    def get_latest_completed_transfer(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Return the most recent completed transfer entry for the given player."""
        history = self.fetch_player_transfer_history_api(player_id)
        transfers = history.get("terminated") if isinstance(history, dict) else None
        if not transfers:
            return None

        def _sort_key(entry: Dict[str, Any]) -> str:
            return entry.get("details", {}).get("date") or ""

        # API already returns transfers in descending order, but sort to be safe.
        for transfer in sorted(transfers, key=_sort_key, reverse=True):
            details = transfer.get("details", {})
            if details.get("isPending"):
                continue
            return transfer
        return None

    def get_squad_players(
        self,
        club_slug: str,
        club_id: int,
        path: str = "kader",
        season_id: Optional[int] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """
        Parse player name, slug, and Transfermarkt ID from a squad page.
        
        Args:
            club_slug: URL slug shown in Transfermarkt (e.g., 'sl-benfica').
            club_id: Numeric club identifier (e.g., 294).
            path: Path component (default 'kader' for squad page).
            season_id: Optional season ID (e.g., 2025 for 2025/26 season).
        """
        if season_id:
            url = f"{self.config.base_url}/{club_slug}/{path}/verein/{club_id}/saison_id/{season_id}"
        else:
            url = f"{self.config.base_url}/{club_slug}/{path}/verein/{club_id}"
        soup = self._fetch_soup(url)
        players: List[Dict[str, Optional[str]]] = []
        seen_ids: set = set()
        
        # Find all player profile links (filter out market value links)
        for anchor in soup.select("table.items a[href*='/spieler/']"):
            href = anchor.get("href", "")
            # Only process profile links, not market value or other links
            if "/profil/spieler/" not in href:
                continue
                
            match = re.search(r"/spieler/(\d+)", href)
            if not match:
                continue
                
            player_id = int(match.group(1))
            # Skip duplicates
            if player_id in seen_ids:
                continue
            seen_ids.add(player_id)
            
            # Extract slug from href (format: /slug/profil/spieler/id)
            parts = href.split("/")
            if len(parts) >= 2:
                slug = parts[1]
            else:
                continue
                
            player_name = anchor.get_text(" ", strip=True)
            if not player_name:
                continue
                
            players.append(
                {
                    "player_id": player_id,
                    "player_slug": slug,
                    "player_name": player_name,
                }
            )
        return players

    def fetch_player_profile(
        self,
        player_slug: Optional[str],
        player_id: int,
    ) -> Dict[str, str]:
        """Return a dictionary with the key-value pairs from the player profile."""
        slug = player_slug or "spieler"
        url = f"{self.config.base_url}/{slug}/profil/spieler/{player_id}"
        soup = self._fetch_soup(url)
        data = {}
        
        # Extract data from info-table: labels have --regular class, values have --bold class
        info_table = soup.select_one("div.info-table")
        if info_table:
            # Get all spans with info-table__content class
            spans = info_table.select("span.info-table__content")
            
            # Pair them: regular (label) followed by bold (value)
            i = 0
            while i < len(spans):
                span = spans[i]
                classes = span.get("class", [])
                
                # Check if this is a label (regular class)
                if "info-table__content--regular" in classes:
                    label_text = span.get_text(strip=True).rstrip(":")
                    # Next span should be the value (bold class)
                    if i + 1 < len(spans):
                        value_span = spans[i + 1]
                        value_classes = value_span.get("class", [])
                        if "info-table__content--bold" in value_classes:
                            value_text = value_span.get_text(" ", strip=True)
                            # Normalize key and map to expected transformer keys
                            key = label_text.lower().replace(" ", "_").replace("/", "_").replace(":", "").strip()
                            
                            # Map common variations to expected keys
                            key_mapping = {
                                "date_of_birth_age": "date_of_birth",
                                "date_of_birth/age": "date_of_birth",
                                "nasc._idade": "date_of_birth",
                                "citizenship": "nationality",
                                "nacionalidade": "nationality",  # Portuguese
                                "nationalities": "nationality",  # Will be handled by transformer
                                "joined": "joined_on",
                                "na_equipa_desde": "joined_on",  # Portuguese
                                "current_club": "signed_from",
                                "clube_atual": "signed_from",  # Portuguese
                            }
                            
                            # Apply mapping
                            mapped_key = key_mapping.get(key, key)
                            
                            # For date_of_birth, extract just the date part (before the age in parentheses)
                            if mapped_key == "date_of_birth" and "(" in value_text:
                                value_text = value_text.split("(")[0].strip()
                            
                            # For nationality/citizenship, extract all countries from flag images
                            if mapped_key == "nationality":
                                # Check for flag images (multiple nationalities are shown as multiple flag images)
                                flag_images = value_span.select('img.flaggenrahmen')
                                if flag_images:
                                    # Extract country names from flag image alt or title attributes
                                    nationalities = []
                                    for img in flag_images:
                                        country_name = img.get('alt') or img.get('title')
                                        if country_name and country_name not in nationalities:
                                            nationalities.append(country_name)
                                    if nationalities:
                                        # Store as list for transformer to handle
                                        data["nationalities"] = nationalities
                                        # Also keep single value for backward compatibility
                                        data[mapped_key] = nationalities[0]
                                    else:
                                        data[mapped_key] = value_text
                                else:
                                    # No flag images, just use text (might be single nationality)
                                    data[mapped_key] = value_text
                            else:
                                data[mapped_key] = value_text
                            i += 2  # Skip both label and value
                            continue
                i += 1
        
        # Extract position - look in info-table first (from the "Position:" label we already extracted)
        # If not found there, try other locations
        if "position" not in data or data.get("position") == "First Tier":
            # Try to find position in data-header
            header_items = soup.select("div.data-header__items span")
            for span in header_items:
                text = span.get_text(strip=True)
                # Look for position text that's not "League level"
                if text and "League level" not in text and ":" not in text and len(text) > 3:
                    # Check if it looks like a position (common positions)
                    common_positions = ["Goalkeeper", "Defender", "Midfielder", "Forward", 
                                       "Centre-Back", "Right-Back", "Left-Back", "Defensive Midfield",
                                       "Central Midfield", "Right Midfield", "Left Midfield",
                                       "Attacking Midfield", "Right Winger", "Left Winger",
                                       "Centre-Forward", "Second Striker"]
                    if any(pos.lower() in text.lower() for pos in common_positions):
                        data["position"] = text
                        break
            
            # Also try table.items td.pos
            if "position" not in data or data.get("position") == "First Tier":
                pos_table = soup.select_one("table.items td.pos")
                if pos_table:
                    pos_text = pos_table.get_text(strip=True)
                    if pos_text and pos_text != "First Tier":
                        data["position"] = pos_text
        
        # Extract joined date and signed from (current club info)
        # Look for "Joined:" in the info table
        
        # Fall back to header values when available.
        header_name = soup.select_one("div.data-header__headline-wrapper h1")
        if header_name:
            data.setdefault("name", header_name.get_text(" ", strip=True))
        
        return data

    def fetch_player_career(
        self,
        player_slug: Optional[str],
        player_id: int,
        include_loans: bool = True,
    ) -> pd.DataFrame:
        """Retrieve the transfer history table for a player using the API."""
        try:
            history_data = self.fetch_player_transfer_history_api(player_id)
            if not history_data:
                return pd.DataFrame()
            
            # The API method returns the history dict directly (data.history)
            transfers = history_data.get("terminated", [])
            if not transfers:
                return pd.DataFrame()
            
            rows = []
            for transfer in transfers:
                # Skip loans if not included
                if not include_loans:
                    transfer_type = transfer.get("typeDetails", {}).get("type", "")
                    if "LOAN" in transfer_type.upper():
                        continue
                
                details = transfer.get("details", {})
                season_info = details.get("season", {})
                season_display = season_info.get("display", "") or season_info.get("nonCyclicalName", "")
                
                # Get club names
                source_id = transfer.get("transferSource", {}).get("clubId")
                dest_id = transfer.get("transferDestination", {}).get("clubId")
                source_name = self.get_club_name(source_id) if source_id else None
                dest_name = self.get_club_name(dest_id) if dest_id else None
                
                # Market value
                mv_info = details.get("marketValue", {})
                mv_value = mv_info.get("value") if mv_info else None
                
                # Transfer fee - use the integer value directly, or special strings
                fee_info = details.get("fee", {})
                fee_value = fee_info.get("value") if fee_info else None
                # Use the integer value if available, otherwise use "-"
                # The API may also have special values like "Zero cost" or "End of loan" in the compact field
                fee_compact = fee_info.get("compact", {}) if fee_info else {}
                if fee_value is not None:
                    fee_display = fee_value  # Use integer value directly
                elif fee_compact:
                    # Check for special strings like "Zero cost" or "End of loan"
                    content = fee_compact.get("content", "")
                    if content and content.lower() in ("zero cost", "end of loan", "free transfer", "free", "?"):
                        fee_display = content
                    else:
                        fee_display = "-"
                else:
                    fee_display = "-"
                
                # Date
                date_str = details.get("date", "")
                if date_str:
                    # Parse ISO date and convert to datetime
                    from datetime import datetime
                    try:
                        date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        date_str = date_obj.strftime("%Y-%m-%d")
                    except:
                        pass
                
                rows.append({
                    "Season": season_display,
                    "Date": date_str,
                    "From": source_name or "-",
                    "To": dest_name or "-",
                    "VM": mv_value,
                    "Value": fee_display,
                })
            
            if not rows:
                return pd.DataFrame()
            
            return pd.DataFrame(rows)
        except Exception as e:
            LOGGER.warning(f"Failed to fetch career data for player {player_id}: {e}")
            return pd.DataFrame()

    def fetch_player_injuries(
        self,
        player_slug: Optional[str],
        player_id: int,
    ) -> pd.DataFrame:
        """Fetch the injury history table from all pages."""
        slug = player_slug or "spieler"
        base_url = f"{self.config.base_url}/{slug}/verletzungen/spieler/{player_id}"
        
        # Fetch first page to check for pagination
        soup = self._fetch_soup(base_url)
        all_dfs = []
        all_clubs = []
        
        # Check for pagination
        pager = soup.select_one('div.pager')
        max_page = 1
        if pager:
            page_links = pager.select('a.tm-pagination__link')
            for link in page_links:
                href = link.get('href', '')
                # Extract page number from href like "/spieler/verletzungen/spieler/452607/page/2"
                if '/page/' in href:
                    try:
                        page_num = int(href.split('/page/')[-1])
                        max_page = max(max_page, page_num)
                    except ValueError:
                        pass
        
        # Fetch all pages
        for page_num in range(1, max_page + 1):
            if page_num == 1:
                url = base_url
                page_soup = soup
            else:
                url = f"{base_url}/page/{page_num}"
                page_soup = self._fetch_soup(url)
            
            page_df = self._extract_table(page_soup, "table.items")
            if page_df.empty:
                continue
            
            # Extract club information from this page
            table = page_soup.select_one("table.items")
            page_clubs = []
            if table:
                rows = table.select("tbody tr")
                for row in rows:
                    # Look for club links in the row
                    club_links = row.select('a[href*="/verein/"]')
                    club_names = []
                    for link in club_links:
                        href = link.get("href", "")
                        link_text = link.get_text(strip=True)
                        if link_text and "/verein/" in href:
                            club_names.append(link_text)
                    page_clubs.append(" / ".join(club_names) if club_names else None)
            
            all_dfs.append(page_df)
            all_clubs.extend(page_clubs)
        
        # Combine all pages
        if not all_dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Add clubs column if we found any
        if all_clubs and any(c for c in all_clubs if c):
            combined_df["Club"] = all_clubs[:len(combined_df)]
        
        return combined_df

    def fetch_player_match_log(
        self,
        player_slug: Optional[str],
        player_id: int,
        season: Optional[int] = None,
        competition_id: int = 0,
        club_id: int = 0,
    ) -> pd.DataFrame:
        """
        Fetch the match log table for a player.

        Args:
            season: Season year (e.g., 2025) or None for all seasons.
            competition_id: Transfermarkt competition id (0 = all).
            club_id: Transfermarkt club id (0 = all clubs).
        """
        slug = player_slug or "spieler"
        season_segment = f"/saison/{season}" if season else ""
        url = (
            f"{self.config.base_url}/{slug}/leistungsdatendetails/spieler/{player_id}"
            f"{season_segment}/verein/{club_id}/liga/{competition_id}/wettbewerb//pos/0"
            "/trainer_id/0/plus/1"
        )
        soup = self._fetch_soup(url)
        
        # Extract all tables (one per competition) and combine them
        # Look for tables with match data (they have headers like "Matchday", "Date", etc.)
        all_tables = soup.select('table')
        match_tables = []
        
        for table in all_tables:
            headers = table.select('thead th')
            if headers:
                header_texts = [h.get_text(strip=True) for h in headers]
                # Check if this table has match data columns
                if any(keyword in ' '.join(header_texts).lower() for keyword in ['matchday', 'date', 'venue', 'opponent', 'result']):
                    # Extract competition name from the div with class 'content-box-headline' before the table
                    competition_name = None
                    # Look for the closest heading before this table
                    parent = table.parent
                    while parent and parent.name != 'body':
                        # Find the content-box-headline div that appears before this table
                        prev_headline = parent.find_previous('div', class_='content-box-headline')
                        if prev_headline:
                            competition_name = prev_headline.get_text(strip=True)
                            # Clean up the competition name (remove extra whitespace)
                            if competition_name:
                                competition_name = ' '.join(competition_name.split())
                            break
                        parent = parent.parent
                    
                    table_df = self._to_frame(table)
                    if not table_df.empty:
                        # Filter out summary rows (they contain "Squad:" or "Starting eleven:")
                        # Check all columns for summary text
                        mask = pd.Series([True] * len(table_df))
                        for col in table_df.columns:
                            if table_df[col].dtype == 'object':
                                mask = mask & ~table_df[col].astype(str).str.contains('Squad:', na=False)
                                mask = mask & ~table_df[col].astype(str).str.contains('Starting eleven:', na=False)
                        
                        table_df = table_df[mask].copy()
                        
                        if not table_df.empty:
                            # Extract home and away teams from the .1 columns
                            # The table has "Home team" (empty) and "Home team.1" (actual team)
                            if "Home team.1" in table_df.columns:
                                table_df["Home team"] = table_df["Home team.1"]
                            if "Away team.1" in table_df.columns:
                                table_df["Away team"] = table_df["Away team.1"]
                            
                            # Clean team names (remove rankings like "(23.)" or "(1.)")
                            for col in ["Home team", "Away team"]:
                                if col in table_df.columns:
                                    table_df[col] = table_df[col].astype(str).str.replace(r'\s*\(\d+\.\)\s*', '', regex=True)
                                    table_df[col] = table_df[col].replace('nan', pd.NA)
                            
                            # Add season column if we have it
                            if season:
                                table_df["Season"] = f"{season-1}/{str(season)[-2:]}"
                            # Add competition column
                            if competition_name:
                                table_df["Competition"] = competition_name
                            match_tables.append(table_df)
        
        if not match_tables:
            return pd.DataFrame()
        
        # Combine all competition tables
        combined = pd.concat(match_tables, ignore_index=True)
        return combined

    def fetch_competition_directory(self, url: str) -> pd.DataFrame:
        """Generic helper for endpoints that expose simple lookup tables."""
        soup = self._fetch_soup(url)
        return self._extract_table(soup, "table.items")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_session(self) -> None:
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept-Language": self.config.language_header,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Connection": "keep-alive",
            }
        )

    def _fetch_soup(self, url: str) -> BeautifulSoup:
        html = self._get(url)
        return BeautifulSoup(html, "lxml")

    def _get(self, url: str) -> str:
        response = self._request(url)
        return response.text

    def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self._request(
            url,
            params=params,
            headers={"Accept": "application/json"},
        )
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Invalid JSON payload from {url}") from exc

    def _request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
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
                if response.status_code == 429:
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
        if attempt >= self.config.max_retries:
            raise RuntimeError(
                f"Failed to fetch Transfermarkt page after {attempt} attempts; "
                f"status={status}, error={error}"
            ) from error
        wait = self.config.rate_limit_seconds * attempt
        LOGGER.warning(
            "Transfermarkt request failed (status=%s, error=%s). Retrying in %.1fs...",
            status,
            error,
            wait,
        )
        time.sleep(wait)

    def _respect_rate_limit(self) -> None:
        delta = time.monotonic() - self._last_request_ts
        remaining = self.config.rate_limit_seconds - delta
        if remaining > 0:
            time.sleep(remaining)

    def _extract_table(self, soup: BeautifulSoup, selector: str) -> pd.DataFrame:
        table = soup.select_one(selector)
        if table is None:
            LOGGER.warning("Could not find table with selector '%s'", selector)
            return pd.DataFrame()
        return self._to_frame(table)

    @staticmethod
    def _to_frame(table_tag) -> pd.DataFrame:
        dfs = pd.read_html(str(table_tag), flavor="bs4")
        return dfs[0] if dfs else pd.DataFrame()


def fetch_multiple_match_logs(
    scraper: TransfermarktScraper,
    player_slug: Optional[str],
    player_id: int,
    seasons: Iterable[int],
    competition_id: int = 0,
    club_id: int = 0,
) -> pd.DataFrame:
    """
    Convenience wrapper that concatenates match logs across seasons.
    """
    frames = []
    for season in seasons:
        frame = scraper.fetch_player_match_log(
            player_slug=player_slug,
            player_id=player_id,
            season=season,
            competition_id=competition_id,
            club_id=club_id,
        )
        if frame.empty:
            LOGGER.info(
                "No match data for player_id=%s season=%s competition=%s club=%s",
                player_id,
                season,
                competition_id,
                club_id,
            )
        else:
            frame["season_id"] = season
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "ScraperConfig",
    "TransfermarktScraper",
    "fetch_multiple_match_logs",
]

