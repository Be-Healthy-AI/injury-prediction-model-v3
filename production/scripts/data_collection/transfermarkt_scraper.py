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
import traceback
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
    
    def close(self) -> None:
        """Close the HTTP session and clean up resources."""
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass
            # Clear caches to free memory
            self._transfer_history_cache.clear()
            self._club_cache.clear()

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

        print(f"      [API] Fetching transfer history for player {player_id}...", end=" ", flush=True)
        url = f"{self.config.api_base_url}/transfer/history/player/{player_id}"
        payload = self._get_json(url, params={"locale": "en"})
        history = payload.get("data", {}).get("history", {}) if isinstance(payload, dict) else {}
        transfers_count = len(history.get("terminated", []))
        print(f"[OK] ({transfers_count} transfers)", flush=True)
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
        """Get club name by ID, with error handling to prevent hangs."""
        try:
            # Check cache first to avoid unnecessary API calls
            club_key = str(club_id) if club_id else None
            if club_key and club_key in self._club_cache:
                return self._club_cache[club_key].get("name") or self._club_cache[club_key].get("baseDetails", {}).get("shortName")
            
            # Log API call if not in cache
            if club_id:
                print(f"      [API] Fetching club name for ID {club_id}...", end=" ", flush=True)
            
            profile = self.fetch_club_profile(club_id)
            if not profile:
                if club_id:
                    print("(not found)", flush=True)
                return None
            
            club_name = profile.get("name") or profile.get("baseDetails", {}).get("shortName")
            if club_id and club_name:
                print(f"[OK] {club_name}", flush=True)
            return club_name
        except Exception as e:
            if club_id:
                print(f"✗ Error: {e}", flush=True)
            LOGGER.warning(f"Failed to get club name for club_id {club_id}: {e}")
            return None

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

    def fetch_league_clubs(
        self,
        competition_slug: str,
        competition_id: Any,
        season_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all clubs that participated in a league/competition for a given season.
        
        Args:
            competition_slug: URL slug (e.g., 'premier-league')
            competition_id: Transfermarkt competition ID (can be int like 1, or str like 'GB1' for Premier League)
            season_id: Season year (e.g., 2025 for 2025/26 season)
        
        Returns:
            List of dicts with: {'club_id': int, 'club_slug': str, 'club_name': str}
        
        Example URL for Premier League 2024/25:
        https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id=2024
        """
        # competition_id can be numeric (1) or string code (GB1) - convert to string for URL
        competition_id_str = str(competition_id)
        url = (
            f"{self.config.base_url}/{competition_slug}/startseite/wettbewerb/{competition_id_str}"
            f"/plus/?saison_id={season_id}"
        )
        
        LOGGER.info(f"Fetching league clubs from: {url}")
        print(f"    Making HTTP request to Transfermarkt...", end=" ", flush=True)
        soup = self._fetch_soup(url)
        print("[OK]", flush=True)
        clubs: List[Dict[str, Any]] = []
        seen_ids: set = set()
        
        # Try to find the standings table - it could be in different formats
        # Look for table.items which is the standard Transfermarkt table format
        standings_table = soup.select_one("table.items")
        
        if standings_table:
            # Extract club links from the table
            # Group by club_id first to find the best name for each club
            club_candidates: Dict[int, Dict[str, Any]] = {}
            
            for anchor in standings_table.select("a[href*='/verein/']"):
                href = anchor.get("href", "")
                # Extract club ID from href (format: /slug/startseite/verein/{club_id} or /slug/kader/verein/{club_id})
                match = re.search(r"/verein/(\d+)", href)
                if not match:
                    continue
                
                club_id = int(match.group(1))
                
                # Extract slug from href
                parts = href.split("/")
                if len(parts) >= 2:
                    club_slug = parts[1]
                else:
                    continue
                
                # Get club name from anchor text
                club_name = anchor.get_text(" ", strip=True)
                
                # Skip if name looks like a number, value, or is too short
                if club_name:
                    # Check if it's a valid club name (not a number, not a currency value, not too short)
                    is_number = club_name.replace(",", "").replace(".", "").isdigit()
                    is_currency = club_name.startswith("€") or club_name.startswith("$") or club_name.startswith("£")
                    is_valid_name = (
                        not is_number
                        and not is_currency
                        and len(club_name) > 2
                        and not club_name.isdigit()
                    )
                    
                    if is_valid_name:
                        # Store candidate - prefer longer names (more likely to be full club name)
                        if club_id not in club_candidates or len(club_name) > len(
                            club_candidates[club_id].get("club_name", "")
                        ):
                            club_candidates[club_id] = {
                                "club_id": club_id,
                                "club_slug": club_slug,
                                "club_name": club_name,
                            }
            
            # Add all valid clubs
            for club_id, club_info in club_candidates.items():
                clubs.append(club_info)
        
        # If no clubs found in standings table, try alternative structure
        # Some leagues might have a participants list instead
        if not clubs:
            # Look for club links in other sections
            for anchor in soup.select("a[href*='/verein/']"):
                href = anchor.get("href", "")
                match = re.search(r"/verein/(\d+)", href)
                if not match:
                    continue
                
                club_id = int(match.group(1))
                if club_id in seen_ids:
                    continue
                seen_ids.add(club_id)
                
                parts = href.split("/")
                if len(parts) >= 2:
                    club_slug = parts[1]
                else:
                    continue
                
                club_name = anchor.get_text(" ", strip=True)
                if club_name and len(club_name) > 1:  # Filter out single characters/noise
                    clubs.append({
                        "club_id": club_id,
                        "club_slug": club_slug,
                        "club_name": club_name,
                    })
        
        if not clubs:
            LOGGER.warning(
                "No clubs found for competition_slug=%s competition_id=%s season_id=%s",
                competition_slug,
                competition_id,
                season_id,
            )
        else:
            LOGGER.info(f"Found {len(clubs)} clubs for season {season_id}")
        
        return clubs

    def fetch_player_profile(
        self,
        player_slug: Optional[str],
        player_id: int,
    ) -> Dict[str, str]:
        """Return a dictionary with the key-value pairs from the player profile."""
        print(f"      [API] Fetching profile page for player {player_id}...", end=" ", flush=True)
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
                                "current_club": "current_club",
                                "clube_atual": "current_club",  # Portuguese
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
        
        # Extract player name from header - try multiple selectors
        # Always try to extract name, even if it's already in data (to ensure it's set)
        header_name = soup.select_one("div.data-header__headline-wrapper h1")
        if not header_name:
            # Try alternative selectors
            header_name = soup.select_one("h1.data-header__headline")
        if not header_name:
            header_name = soup.select_one("div.data-header h1")
        if not header_name:
            # Try title tag as last resort
            title_tag = soup.select_one("title")
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                # Title format is usually "Player Name | Transfermarkt"
                if "|" in title_text:
                    name_text = title_text.split("|")[0].strip()
                    # Remove suffixes like "- Player profile 25/26" or "- Player profile"
                    name_text = re.sub(r'\s*-\s*Player profile(\s+\d+/\d+)?\s*$', '', name_text, flags=re.IGNORECASE)
                    name_text = name_text.strip()
                    if name_text:
                        data["name"] = name_text
        else:
            name_text = header_name.get_text(" ", strip=True)
            if name_text:
                # Remove suffixes like "- Player profile 25/26" or "- Player profile"
                # Pattern to match "- Player profile" followed by optional season (e.g., "25/26")
                name_text = re.sub(r'\s*-\s*Player profile(\s+\d+/\d+)?\s*$', '', name_text, flags=re.IGNORECASE)
                name_text = name_text.strip()
                if name_text:
                    data["name"] = name_text
        
        print("[OK]", flush=True)
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
            total_transfers = len(transfers)
            print(f"      [INFO] Processing {total_transfers} transfers (2 API calls per transfer)...", flush=True)
            
            for idx, transfer in enumerate(transfers, 1):
                # Skip loans if not included
                if not include_loans:
                    transfer_type = transfer.get("typeDetails", {}).get("type", "")
                    if "LOAN" in transfer_type.upper():
                        continue
                
                details = transfer.get("details", {})
                season_info = details.get("season", {})
                season_display = season_info.get("display", "") or season_info.get("nonCyclicalName", "")
                
                # Get club names (this is the slow part - 2 API calls per transfer with rate limiting)
                source_id = transfer.get("transferSource", {}).get("clubId")
                dest_id = transfer.get("transferDestination", {}).get("clubId")
                
                # Show progress for each transfer
                if total_transfers > 3:  # Only show progress if there are multiple transfers
                    print(f"      [Transfer {idx}/{total_transfers}] Fetching club names...", flush=True)
                
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
        except KeyboardInterrupt:
            # Re-raise keyboard interrupts
            raise
        except Exception as e:
            LOGGER.warning(f"Failed to fetch career data for player {player_id}: {e}")
            LOGGER.debug(f"Career fetch error traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def fetch_player_injuries(
        self,
        player_slug: Optional[str],
        player_id: int,
    ) -> pd.DataFrame:
        """Fetch the injury history table from all pages."""
        print(f"      [API] Fetching injuries page for player {player_id}...", end=" ", flush=True)
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
            # First check if pd.read_html already extracted a club column (check common names)
            club_col = None
            for col_name in ["Club", "Verein", "Team", "Clubs"]:
                if col_name in page_df.columns:
                    club_col = col_name
                    break
            
            if club_col:
                # Already extracted, rename to "Club" for consistency
                if club_col != "Club":
                    page_df["Club"] = page_df[club_col]
            else:
                # Manually extract club information from HTML
                # The club column in injury tables is typically in a specific position
                # We need to identify which column contains clubs by looking at the table structure
                table = page_soup.select_one("table.items")
                page_clubs = []
                if table:
                    # First, try to identify the club column from the header
                    headers = table.select("thead th")
                    club_col_idx = None
                    for idx, header in enumerate(headers):
                        header_text = header.get_text(strip=True).lower()
                        if any(term in header_text for term in ["club", "verein", "team"]):
                            club_col_idx = idx
                            break
                    
                    rows = table.select("tbody tr")
                    for row in rows:
                        club_names = []
                        
                        # If we found the club column index, extract from that column
                        if club_col_idx is not None:
                            cells = row.select("td")
                            if club_col_idx < len(cells):
                                club_cell = cells[club_col_idx]
                                # Look for club links in this cell
                                club_links = club_cell.select('a[href*="/verein/"]')
                                for link in club_links:
                                    link_text = link.get_text(strip=True)
                                    if link_text:
                                        club_names.append(link_text)
                                # If no links, try the cell text itself (might be plain text club name)
                                if not club_names:
                                    cell_text = club_cell.get_text(strip=True)
                                    # Only use if it doesn't look like an injury type or other data
                                    if cell_text and not any(skip in cell_text.lower() for skip in 
                                                             ['injury', 'muscle', 'knee', 'ankle', 'shoulder', 'back', 
                                                              'hamstring', 'groin', 'calf', 'thigh', 'foot', 'wrist',
                                                              'fracture', 'strain', 'tear', 'rupture', 'sprain',
                                                              'dislocation', 'bruise', 'contusion', 'laceration']):
                                        club_names.append(cell_text)
                        else:
                            # No header found, search the entire row for club links
                            club_links = row.select('a[href*="/verein/"]')
                            for link in club_links:
                                href = link.get("href", "")
                                link_text = link.get_text(strip=True)
                                if link_text and "/verein/" in href:
                                    club_name = link_text.strip()
                                    if club_name and club_name not in club_names:
                                        club_names.append(club_name)
                        
                        page_clubs.append(" / ".join(club_names) if club_names else None)
                    
                    # Ensure we have the same number of clubs as rows
                    if len(page_clubs) == len(page_df):
                        page_df["Club"] = page_clubs
                    elif len(page_clubs) > 0:
                        # Try to align: pad with None if needed
                        while len(page_clubs) < len(page_df):
                            page_clubs.append(None)
                        page_df["Club"] = page_clubs[:len(page_df)]
                    else:
                        # No clubs found, create empty column
                        page_df["Club"] = None
            
            all_dfs.append(page_df)
        
        # Combine all pages
        if not all_dfs:
            print("(no injuries)", flush=True)
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Ensure Club column exists (even if empty)
        if "Club" not in combined_df.columns:
            combined_df["Club"] = None
        
        # Deduplicate injuries based on key fields
        # Use Injury/Injury type, from/From, until/Until columns
        injury_col = None
        from_col = None
        until_col = None
        
        for col in combined_df.columns:
            col_lower = col.lower()
            if col_lower in ['injury', 'injury type']:
                injury_col = col
            elif col_lower in ['from', 'fromdate']:
                from_col = col
            elif col_lower in ['until', 'untildate']:
                until_col = col
        
        if injury_col and from_col:
            dedup_cols = [injury_col, from_col]
            if until_col:
                dedup_cols.append(until_col)
            
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep='first')
            after_dedup = len(combined_df)
            
            if before_dedup > after_dedup:
                print(f" (removed {before_dedup - after_dedup} duplicates)", end="", flush=True)
        
        print(f"[OK] ({len(combined_df)} injuries)", flush=True)
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
        season_str = f"season {season}" if season else "all seasons"
        print(f"      [API] Fetching match log for player {player_id} ({season_str})...", end=" ", flush=True)
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
                            LOGGER.info(f"  Processing table with {len(table_df)} rows for competition: {competition_name}")
                            
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
                            
                            # Map position column (Pos. -> Position) - transformer also handles this
                            if "Pos." in table_df.columns and "Position" not in table_df.columns:
                                table_df["Position"] = table_df["Pos."]
                            
                            # ============================================================
                            # APPLY DIRECT COLUMN MAPPING (Unnamed: 8 -> Goals, etc.)
                            # ============================================================
                            mapping = {
                                "Unnamed: 8": "Goals",
                                "Unnamed: 9": "Assists",
                                "Unnamed: 10": "Own goals",
                                "Unnamed: 11": "Yellow cards",
                                "Unnamed: 12": "Second yellow cards",
                                "Unnamed: 13": "Red cards",
                                "Unnamed: 14": "Sub on",
                                "Unnamed: 15": "Sub off",
                                "Unnamed: 16": "Minutes played",
                                "Unnamed: 17": "TM-Whoscored grade",
                            }
                            
                            # Apply mappings
                            for source_col, target_col in mapping.items():
                                if source_col in table_df.columns:
                                    # Clean the data: remove invalid texts
                                    series = table_df[source_col].copy()
                                    series_str = series.astype(str)
                                    
                                    # Filter out invalid texts
                                    invalid_texts = [
                                        'on the bench',
                                        'not in squad',
                                        'suspended',
                                        'injured',
                                        'Squad:',
                                        'Starting eleven:',
                                    ]
                                    
                                    for invalid in invalid_texts:
                                        mask = series_str.str.contains(invalid, case=False, na=False)
                                        series.loc[mask] = pd.NA
                                    
                                    # For time-based columns, handle time formats like "88'", "45 + 1'", and "90+3'"
                                    # This includes substitutions, minutes played, and card times
                                    if target_col in ["Sub on", "Sub off", "Minutes played", "Yellow cards", 
                                                      "Second yellow cards", "Red cards"]:
                                        def extract_time(val):
                                            if pd.isna(val) or str(val) == 'nan':
                                                return pd.NA
                                            val_str = str(val).strip()
                                            
                                            # Handle "90+3'" or "45 + 1'" format (injury time)
                                            if "+" in val_str:
                                                import re
                                                parts = re.split(r'\s*\+\s*', val_str)
                                                if len(parts) == 2:
                                                    base_min = re.findall(r'\d+', parts[0])
                                                    extra_min = re.findall(r'\d+', parts[1])
                                                    if base_min and extra_min:
                                                        total = int(base_min[0]) + int(extra_min[0])
                                                        # For substitution times, return as string with quote
                                                        if target_col in ["Sub on", "Sub off"]:
                                                            return f"{total}'"
                                                        # For minutes played and cards, return as integer
                                                        else:
                                                            return total
                                            
                                            # Remove quotes and extract number
                                            val_str = val_str.replace("'", "").replace('"', '').strip()
                                            # If it contains comma, take first part (for multiple values)
                                            if ',' in val_str:
                                                val_str = val_str.split(',')[0].strip()
                                            # Handle multiple times separated by space (e.g., "45' 90'" for two yellow cards)
                                            # For cards, we want the first occurrence (minute of first card)
                                            if ' ' in val_str and target_col in ["Yellow cards", "Second yellow cards", "Red cards"]:
                                                # Extract first time value
                                                import re
                                                first_match = re.search(r'\d+', val_str)
                                                if first_match:
                                                    val_str = first_match.group(0)
                                            
                                            try:
                                                num = float(val_str)
                                                # For substitution times, return as string with quote
                                                if target_col in ["Sub on", "Sub off"]:
                                                    return f"{int(num)}'"
                                                # For minutes played and cards, return as integer
                                                else:
                                                    return int(num)
                                            except (ValueError, TypeError):
                                                return pd.NA
                                            return pd.NA
                                        
                                        series = series.apply(extract_time)
                                    
                                    # Only map if target column doesn't already exist
                                    if target_col not in table_df.columns:
                                        table_df[target_col] = series
                                        LOGGER.info(f"  Mapped {source_col} -> {target_col}")
                                    else:
                                        LOGGER.info(f"  Warning: {target_col} already exists, skipping {source_col}")
                                else:
                                    LOGGER.info(f"  Warning: {source_col} not found in DataFrame")
                            
                            # Check for existing columns with alternative names and rename them
                            # Handle own goals column name variations
                            own_goals_col_found = None
                            for col_name in ["Own goals", "Own goal", "OG", "Golos próprios", "Golo próprio", "Eigentore", "Eigentor"]:
                                if col_name in table_df.columns:
                                    own_goals_col_found = col_name
                                    break
                            
                            if own_goals_col_found and own_goals_col_found != "Own goals":
                                LOGGER.info(f"  Found own goals column as '{own_goals_col_found}', renaming to 'Own goals'")
                                table_df["Own goals"] = table_df[own_goals_col_found]
                                
                            
                            # Add season column if we have it
                            if season:
                                table_df["Season"] = f"{season-1}/{str(season)[-2:]}"
                            # Add competition column
                            if competition_name:
                                table_df["Competition"] = competition_name
                            match_tables.append(table_df)
        
        if not match_tables:
            print("(no matches)", flush=True)
            return pd.DataFrame()
        
        # Combine all competition tables
        combined = pd.concat(match_tables, ignore_index=True)
        print(f"[OK] ({len(combined)} matches)", flush=True)
        
        # After combining, ensure Position column exists (map Pos. -> Position if needed)
        # The transformer handles both, but we ensure Position exists for consistency
        if "Pos." in combined.columns and "Position" not in combined.columns:
            combined["Position"] = combined["Pos."]
        
        # Remove duplicate columns if both Pos. and Position exist
        if "Pos." in combined.columns and "Position" in combined.columns:
            combined = combined.drop(columns=["Pos."])
        
        # Direct column mapping is applied per table, no final pass remapping needed
        # The mapping is already correct and should not be overwritten
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

