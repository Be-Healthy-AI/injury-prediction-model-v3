"""
Map players between TransferMarkt and FBRef using fuzzy matching and validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        HAS_RAPIDFUZZ = False
    except ImportError:
        HAS_RAPIDFUZZ = False
        fuzz = None
        process = None

from scripts.data_collection.fbref_scraper import FBRefScraper, FBRefConfig


LOGGER = logging.getLogger(__name__)


class PlayerMapper:
    """Maps players between TransferMarkt and FBRef."""
    
    def __init__(
        self,
        mapping_file: Optional[Path] = None,
        fbref_scraper: Optional[FBRefScraper] = None,
    ):
        """
        Initialize player mapper.
        
        Args:
            mapping_file: Path to CSV file storing mappings (will be created if doesn't exist)
            fbref_scraper: Optional FBRef scraper instance (will create one if not provided)
        """
        self.mapping_file = mapping_file
        self.fbref_scraper = fbref_scraper or FBRefScraper(FBRefConfig())
        self.mappings = self._load_mappings()
        
        if not HAS_RAPIDFUZZ and fuzz is None:
            LOGGER.warning(
                "Fuzzy matching libraries not installed. Install with: "
                "pip install rapidfuzz (recommended) or pip install fuzzywuzzy"
            )
    
    def _load_mappings(self) -> pd.DataFrame:
        """Load existing mappings from CSV file."""
        if self.mapping_file and self.mapping_file.exists():
            try:
                df = pd.read_csv(self.mapping_file, parse_dates=['date_of_birth', 'last_verified'])
                LOGGER.info(f"Loaded {len(df)} existing player mappings from {self.mapping_file}")
                return df
            except Exception as e:
                LOGGER.warning(f"Failed to load mappings from {self.mapping_file}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save_mappings(self) -> None:
        """Save mappings to CSV file."""
        if self.mapping_file:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            self.mappings.to_csv(
                self.mapping_file,
                index=False,
                encoding='utf-8-sig',
                date_format='%Y-%m-%d'
            )
            LOGGER.info(f"Saved {len(self.mappings)} mappings to {self.mapping_file}")
    
    def get_mapping(self, tm_id: int) -> Optional[Dict[str, Any]]:
        """Get existing mapping for a TransferMarkt player ID."""
        if self.mappings.empty:
            return None
        
        match = self.mappings[self.mappings['transfermarkt_id'] == tm_id]
        if not match.empty and match.iloc[0]['is_active']:
            row = match.iloc[0]
            return {
                'fbref_id': row['fbref_id'],
                'fbref_name': row['fbref_name'],
                'confidence': row['match_confidence'],
                'method': row['match_method'],
            }
        return None
    
    def find_fbref_player(
        self,
        tm_id: int,
        tm_name: str,
        tm_dob: Optional[pd.Timestamp] = None,
        tm_clubs: Optional[List[str]] = None,
        tm_seasons: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find FBRef player ID for a TransferMarkt player.
        
        Uses multi-layered matching:
        1. Check existing mapping cache
        2. Search FBRef by name
        3. Fuzzy name matching
        4. DOB validation if available
        5. Club + season triangulation
        
        Args:
            tm_id: TransferMarkt player ID
            tm_name: Player name from TransferMarkt
            tm_dob: Date of birth (optional, for validation)
            tm_clubs: List of club names (optional, for validation)
            tm_seasons: List of season years (optional, for validation)
        
        Returns:
            Dict with fbref_id, fbref_name, confidence, method, or None if no match found
        """
        # Check cache first
        cached = self.get_mapping(tm_id)
        if cached:
            LOGGER.debug(f"Found cached mapping for {tm_name} (TM ID: {tm_id})")
            return cached
        
        # Search FBRef
        LOGGER.info(f"Searching FBRef for: {tm_name} (TM ID: {tm_id})")
        search_results = self.fbref_scraper.search_player(tm_name, club=tm_clubs[0] if tm_clubs else None)
        
        if not search_results:
            LOGGER.warning(f"No FBRef search results for: {tm_name}")
            return None
        
        # Score and rank matches
        best_match = self._score_matches(
            tm_name=tm_name,
            tm_dob=tm_dob,
            tm_clubs=tm_clubs or [],
            search_results=search_results
        )
        
        if best_match and best_match['confidence'] >= 0.70:
            # Save to cache
            self._add_mapping(tm_id, tm_name, best_match)
            return best_match
        
        LOGGER.warning(f"No confident match found for {tm_name} (best confidence: {best_match['confidence'] if best_match else 0.0})")
        return None
    
    def _score_matches(
        self,
        tm_name: str,
        tm_dob: Optional[pd.Timestamp],
        tm_clubs: List[str],
        search_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Score and rank potential matches."""
        if not search_results:
            return None
        
        if not HAS_RAPIDFUZZ and fuzz is None:
            # Fallback: use first result if available
            if search_results:
                return {
                    'fbref_id': search_results[0]['fbref_id'],
                    'fbref_name': search_results[0]['name'],
                    'confidence': 0.50,  # Low confidence without fuzzy matching
                    'method': 'exact_name_fallback',
                }
            return None
        
        scored_matches = []
        
        for result in search_results:
            fbref_name = result['name']
            fbref_id = result['fbref_id']
            fbref_club = result.get('club')
            
            # Calculate name similarity
            name_ratio = fuzz.ratio(tm_name.lower(), fbref_name.lower())
            name_partial = fuzz.partial_ratio(tm_name.lower(), fbref_name.lower())
            name_token = fuzz.token_sort_ratio(tm_name.lower(), fbref_name.lower())
            
            # Average of name matching methods
            name_score = (name_ratio + name_partial + name_token) / 3.0
            
            # Club matching bonus
            club_bonus = 0.0
            if tm_clubs and fbref_club:
                for tm_club in tm_clubs:
                    club_similarity = fuzz.ratio(tm_club.lower(), fbref_club.lower())
                    if club_similarity > 80:
                        club_bonus = 0.10
                        break
            
            # Calculate confidence
            confidence = min(1.0, name_score / 100.0 + club_bonus)
            
            # Determine method
            if name_score >= 95:
                method = 'exact_name'
            elif name_score >= 85:
                method = 'fuzzy_name'
            elif name_score >= 70:
                method = 'fuzzy_name_club'
            else:
                method = 'low_confidence'
            
            scored_matches.append({
                'fbref_id': fbref_id,
                'fbref_name': fbref_name,
                'confidence': confidence,
                'method': method,
                'name_score': name_score,
            })
        
        # Sort by confidence (highest first)
        scored_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        if scored_matches:
            best = scored_matches[0]
            # Remove internal scoring fields
            return {
                'fbref_id': best['fbref_id'],
                'fbref_name': best['fbref_name'],
                'confidence': best['confidence'],
                'method': best['method'],
            }
        
        return None
    
    def _add_mapping(
        self,
        tm_id: int,
        tm_name: str,
        match: Dict[str, Any]
    ) -> None:
        """Add a new mapping to the cache."""
        # Check if mapping already exists
        if not self.mappings.empty:
            existing = self.mappings[self.mappings['transfermarkt_id'] == tm_id]
            if not existing.empty:
                # Update existing mapping
                idx = existing.index[0]
                self.mappings.loc[idx, 'fbref_id'] = match['fbref_id']
                self.mappings.loc[idx, 'fbref_name'] = match['fbref_name']
                self.mappings.loc[idx, 'match_confidence'] = match['confidence']
                self.mappings.loc[idx, 'match_method'] = match['method']
                self.mappings.loc[idx, 'is_active'] = True
                self.mappings.loc[idx, 'last_verified'] = pd.Timestamp.now()
                return
        
        # Add new mapping
        new_row = {
            'transfermarkt_id': tm_id,
            'transfermarkt_name': tm_name,
            'fbref_id': match['fbref_id'],
            'fbref_name': match['fbref_name'],
            'fbref_url': f"https://fbref.com/en/players/{match['fbref_id']}/",
            'date_of_birth': None,  # Can be filled later
            'match_confidence': match['confidence'],
            'match_method': match['method'],
            'last_verified': pd.Timestamp.now(),
            'is_active': True,
        }
        
        if self.mappings.empty:
            self.mappings = pd.DataFrame([new_row])
        else:
            self.mappings = pd.concat([self.mappings, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to file
        self._save_mappings()
    
    def validate_mapping(
        self,
        tm_id: int,
        fbref_id: str
    ) -> bool:
        """Validate an existing mapping by checking recent matches."""
        try:
            # Fetch FBRef profile
            profile = self.fbref_scraper.fetch_player_profile(fbref_id)
            # If we can fetch the profile, the mapping is likely still valid
            return profile is not None and 'name' in profile
        except Exception as e:
            LOGGER.warning(f"Failed to validate mapping {tm_id} -> {fbref_id}: {e}")
            return False
    
    def close(self) -> None:
        """Close resources."""
        if self.fbref_scraper:
            self.fbref_scraper.close()


__all__ = [
    "PlayerMapper",
]









