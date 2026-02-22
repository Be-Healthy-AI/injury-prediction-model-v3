"""
Multi-source fetcher: given internal_id, fetch data from TransferMarkt, FBRef, Wikipedia,
and (placeholders) SofaScore, Understat, WhoScored.

Uses players_raw_data external_id_mappings to resolve internal_id -> external IDs per source.
Does not overwrite admin-verified mappings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from scripts.data_collection.fbref_scraper import FBRefConfig, FBRefScraper
    HAS_FBREF = True
except ImportError:
    HAS_FBREF = False

try:
    from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper
    HAS_TM = True
except ImportError:
    HAS_TM = False

try:
    from scripts.data_collection.wikipedia_api import fetch_player_wikipedia
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False
    fetch_player_wikipedia = None

from scripts.player_identity.store import PlayerIdentityStore, get_external_id

LOGGER = logging.getLogger(__name__)

# All supported sources (design: documentation/player_identity_design.md)
SOURCES = ["transfermarkt", "fbref", "sofascore", "understat", "wikipedia", "whoscored"]


def _fetch_sofascore_placeholder(external_id: str) -> Dict[str, Any]:
    """Placeholder until SofaScore scraper is implemented."""
    LOGGER.debug("SofaScore adapter not implemented; external_id=%s", external_id)
    return {}


def _fetch_understat_placeholder(external_id: str) -> Dict[str, Any]:
    """Placeholder until Understat scraper is implemented."""
    LOGGER.debug("Understat adapter not implemented; external_id=%s", external_id)
    return {}


def _fetch_whoscored_placeholder(external_id: str) -> Dict[str, Any]:
    """Placeholder until WhoScored scraper is implemented."""
    LOGGER.debug("WhoScored adapter not implemented; external_id=%s", external_id)
    return {}


def fetch_all(
    internal_id: str,
    store: Optional[PlayerIdentityStore] = None,
    tm_scraper: Optional[Any] = None,
    fbref_scraper: Optional[Any] = None,
    include_tm_career: bool = True,
    include_tm_injuries: bool = True,
    include_tm_matches: bool = False,
    include_fbref_match_logs: bool = False,
    season: Optional[int] = None,
    include_fbref: bool = True,
    include_wikipedia: bool = True,
    include_sofascore: bool = True,
    include_understat: bool = True,
    include_whoscored: bool = True,
) -> Dict[str, Any]:
    """
    Fetch data from all mapped sources for the given internal_id.

    Uses external_id_mappings to get external IDs per source. Missing mappings are skipped.
    Returns dict with keys: transfermarkt, fbref, wikipedia, sofascore, understat, whoscored.
    """
    store = store or PlayerIdentityStore()
    tm_id_str = get_external_id(internal_id, "transfermarkt", path=store.mappings_path)
    fbref_id_str = get_external_id(internal_id, "fbref", path=store.mappings_path)
    wikipedia_id = get_external_id(internal_id, "wikipedia", path=store.mappings_path)
    sofascore_id = get_external_id(internal_id, "sofascore", path=store.mappings_path)
    understat_id = get_external_id(internal_id, "understat", path=store.mappings_path)
    whoscored_id = get_external_id(internal_id, "whoscored", path=store.mappings_path)

    out: Dict[str, Any] = {
        "transfermarkt": {},
        "fbref": {},
        "wikipedia": {},
        "sofascore": {},
        "understat": {},
        "whoscored": {},
    }

    # TransferMarkt
    if HAS_TM and tm_id_str:
        tm_id = int(tm_id_str)
        scraper = tm_scraper or TransfermarktScraper(ScraperConfig())
        try:
            # Profile (slug optional; scraper uses "spieler" if missing)
            profile = scraper.fetch_player_profile(player_slug="spieler", player_id=tm_id)
            if profile:
                out["transfermarkt"]["profile"] = profile
            # Transfer history (API)
            history = scraper.fetch_player_transfer_history_api(tm_id)
            if history:
                out["transfermarkt"]["transfer_history"] = history
            # Career table (optional, can be slow)
            if include_tm_career:
                career_df = scraper.fetch_player_career(player_slug="spieler", player_id=tm_id)
                if career_df is not None and not career_df.empty:
                    out["transfermarkt"]["career"] = career_df.to_dict(orient="records")
            # Injuries (optional)
            if include_tm_injuries:
                injuries_df = scraper.fetch_player_injuries(player_slug="spieler", player_id=tm_id)
                if injuries_df is not None and not injuries_df.empty:
                    out["transfermarkt"]["injuries"] = injuries_df.to_dict(orient="records")
            if include_tm_matches:
                matches_df = scraper.fetch_player_match_log(
                    player_slug="spieler", player_id=tm_id,
                    season=season, competition_id=0, club_id=0,
                )
                if matches_df is not None and not matches_df.empty:
                    out["transfermarkt"]["matches"] = matches_df.to_dict(orient="records")
        finally:
            if not tm_scraper and hasattr(scraper, "close"):
                scraper.close()

    # FBRef (continue on rate limit / errors so other sources still run)
    if include_fbref and HAS_FBREF and fbref_id_str:
        try:
            scraper = fbref_scraper or FBRefScraper(FBRefConfig())
            try:
                profile = scraper.fetch_player_profile(fbref_id_str)
                if profile:
                    out["fbref"]["profile"] = profile
                if include_fbref_match_logs:
                    logs = scraper.fetch_player_match_logs(
                        fbref_id_str, season=season, competition=None
                    )
                    if logs is not None and not (hasattr(logs, "empty") and logs.empty):
                        out["fbref"]["match_logs"] = logs.to_dict(orient="records") if hasattr(logs, "to_dict") else logs
            finally:
                if not fbref_scraper and hasattr(scraper, "close"):
                    scraper.close()
        except Exception as e:
            LOGGER.warning("FBRef fetch failed for %s: %s", fbref_id_str, e)
            out["fbref"] = {}

    # Wikipedia (MediaWiki API; continue on errors)
    if include_wikipedia and wikipedia_id and HAS_WIKIPEDIA and fetch_player_wikipedia:
        try:
            out["wikipedia"] = fetch_player_wikipedia(wikipedia_id)
        except Exception as e:
            LOGGER.warning("Wikipedia fetch failed for %s: %s", wikipedia_id, e)
            out["wikipedia"] = {}

    # SofaScore (placeholder until scraper implemented)
    if include_sofascore and sofascore_id:
        out["sofascore"] = _fetch_sofascore_placeholder(sofascore_id)

    # Understat (placeholder until scraper implemented)
    if include_understat and understat_id:
        out["understat"] = _fetch_understat_placeholder(understat_id)

    # WhoScored (placeholder until scraper implemented)
    if include_whoscored and whoscored_id:
        out["whoscored"] = _fetch_whoscored_placeholder(whoscored_id)

    return out
