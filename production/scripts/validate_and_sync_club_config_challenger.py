#!/usr/bin/env python3
"""
Validate and sync club config.json for V4 challenger pipeline (writes ONLY to challenger/).

This script is used by deploy_all_clubs_v4.py. It does NOT touch V3 paths.
- Reads/writes: production/deployments/{country}/challenger/{club}/config.json
- Cleans: challenger/{club}/daily_features/, daily_features_enhanced/, predictions/ (v4 files only)
- Sync result: challenger/.sync_result_challenger_{data_date}.json

Usage:
    python validate_and_sync_club_config_challenger.py --country England --club "Arsenal FC" --data-date 20260204
    python validate_and_sync_club_config_challenger.py --country England --club "Arsenal FC" --data-date 20260204 --auto-fix --sync-transfermarkt
"""

import json
import sys
import io
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Add root directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.data_collection.transfermarkt_scraper import ScraperConfig, TransfermarktScraper

try:
    from production.scripts.transfer_utils import get_pl_club_names
except ImportError:
    from transfer_utils import get_pl_club_names


def get_challenger_deployments_dir(country: str) -> Path:
    """Challenger root for V4: production/deployments/{country}/challenger. Never touch V3 paths."""
    return PRODUCTION_ROOT / "deployments" / country / "challenger"

# Club name mapping (our names -> Transfermarkt names)
CLUB_NAME_MAPPING = {
    "Arsenal FC": "Arsenal FC",
    "Manchester City": "Manchester City",
    "Chelsea FC": "Chelsea FC",
    "Liverpool FC": "Liverpool FC",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Manchester United": "Manchester United",
    "Aston Villa": "Aston Villa",
    "Newcastle United": "Newcastle United",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Crystal Palace": "Crystal Palace",
    "Fulham FC": "Fulham FC",
    "AFC Bournemouth": "AFC Bournemouth",
    "Everton FC": "Everton FC",
    "Brentford FC": "Brentford FC",
    "Nottingham Forest": "Nottingham Forest",
    "Leeds United": "Leeds United",
    "Sunderland AFC": "Sunderland AFC",
    "Burnley FC": "Burnley FC",
}


def get_latest_raw_data_folder(country: str = "england") -> Optional[Path]:
    """Get the latest raw data folder."""
    base_dir = PRODUCTION_ROOT / "raw_data" / country.lower().replace(" ", "_")
    if not base_dir.exists():
        return None
    
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    if not date_folders:
        return None
    
    return max(date_folders, key=lambda x: x.name)


def load_config(club_name: str, country: Optional[str] = None) -> Tuple[Optional[Dict], Path]:
    """Load config.json for a club from challenger path only."""
    country = country or "England"
    base = get_challenger_deployments_dir(country)
    config_path = base / club_name / "config.json"
    
    if not config_path.exists():
        return None, config_path
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config, config_path
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return None, config_path


def save_config(config: Dict, config_path: Path) -> bool:
    """Save config.json for a club."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"ERROR saving config: {e}")
        return False


def get_players_from_raw_data(data_dir: Path) -> Set[int]:
    """Get all player IDs that have raw data (profile, career, match data)."""
    player_ids = set()
    
    # Check profile file
    profile_path = data_dir / "players_profile.csv"
    if profile_path.exists():
        try:
            profile_df = pd.read_csv(profile_path, sep=';', encoding='utf-8-sig', low_memory=False)
            player_id_col = 'player_id' if 'player_id' in profile_df.columns else 'id'
            if player_id_col in profile_df.columns:
                player_ids.update(profile_df[player_id_col].dropna().astype(int).unique())
        except Exception as e:
            print(f"  Warning: Could not read profile file: {e}")
    
    return player_ids


def check_player_has_raw_data(player_id: int, data_dir: Path, club_name: str) -> Dict[str, any]:
    """Check if a player has required raw data files and assess completeness."""
    checks = {
        'profile': False,
        'career': False,
        'match_data_current': False,
        'match_data_previous': False,
        'injury_data': False,
        'completeness_level': None,
        'missing_critical': [],
        'missing_optional': [],
    }
    
    # Check profile
    profile_path = data_dir / "players_profile.csv"
    if profile_path.exists():
        try:
            profile_df = pd.read_csv(profile_path, sep=';', encoding='utf-8-sig', low_memory=False)
            player_id_col = 'player_id' if 'player_id' in profile_df.columns else 'id'
            if player_id_col in profile_df.columns:
                checks['profile'] = int(player_id) in profile_df[player_id_col].values
        except Exception:
            pass
    
    # Check career
    career_path = data_dir / "players_career.csv"
    if career_path.exists():
        try:
            career_df = pd.read_csv(career_path, sep=';', encoding='utf-8-sig', low_memory=False)
            player_id_col = 'player_id' if 'player_id' in career_df.columns else 'id'
            if player_id_col in career_df.columns:
                checks['career'] = int(player_id) in career_df[player_id_col].values
        except Exception:
            pass
    
    # Check current season match data
    match_data_dir = data_dir / "match_data"
    if match_data_dir.exists():
        match_files = list(match_data_dir.glob(f"match_{player_id}_*.csv"))
        checks['match_data_current'] = len(match_files) > 0
    
    # Check previous seasons match data
    previous_seasons_dir = PRODUCTION_ROOT / "raw_data" / data_dir.parent.name / "previous_seasons"
    if previous_seasons_dir.exists():
        match_files = list(previous_seasons_dir.glob(f"match_{player_id}_*.csv"))
        checks['match_data_previous'] = len(match_files) > 0
    
    # Check injury data
    injury_path = data_dir / "injuries_data.csv"
    if injury_path.exists():
        try:
            injury_df = pd.read_csv(injury_path, sep=';', encoding='utf-8-sig', low_memory=False)
            player_id_col = 'player_id' if 'player_id' in injury_df.columns else 'id'
            if player_id_col in injury_df.columns:
                checks['injury_data'] = int(player_id) in injury_df[player_id_col].values
        except Exception:
            pass
    
    # Assess completeness
    has_critical = checks['profile'] and checks['career']
    has_any_data = checks['profile'] or checks['career'] or checks['match_data_current'] or checks['match_data_previous']
    
    if not has_any_data:
        checks['completeness_level'] = 'orphaned'
    elif not has_critical:
        checks['completeness_level'] = 'incomplete'
        if not checks['profile']:
            checks['missing_critical'].append('profile')
        if not checks['career']:
            checks['missing_critical'].append('career')
    else:
        checks['completeness_level'] = 'complete'
        # Track optional missing data
        if not checks['match_data_current'] and not checks['match_data_previous']:
            checks['missing_optional'].append('match_data')
        if not checks['injury_data']:
            checks['missing_optional'].append('injury_data')
    
    return checks


def get_player_current_club(player_id: int, data_dir: Path) -> Optional[str]:
    """Get player's current club from profile data."""
    profile_path = data_dir / "players_profile.csv"
    if not profile_path.exists():
        return None
    
    try:
        profile_df = pd.read_csv(profile_path, sep=';', encoding='utf-8-sig', low_memory=False)
        player_id_col = 'player_id' if 'player_id' in profile_df.columns else 'id'
        current_club_col = 'current_club'
        
        if player_id_col in profile_df.columns and current_club_col in profile_df.columns:
            player_row = profile_df[profile_df[player_id_col] == player_id]
            if not player_row.empty:
                current_club = player_row.iloc[0][current_club_col]
                return str(current_club) if pd.notna(current_club) else None
    except Exception:
        pass
    
    return None


def fetch_transfermarkt_players(club_name: str, season_year: int = 2025) -> Dict[int, Dict]:
    """Fetch current players from Transfermarkt for a club."""
    scraper = TransfermarktScraper(ScraperConfig())
    
    # Get club name for Transfermarkt
    tm_club_name = CLUB_NAME_MAPPING.get(club_name, club_name)
    
    try:
        # Fetch clubs for the season
        clubs = scraper.fetch_league_clubs("premier-league", "GB1", season_year)
        
        # Find our club
        target_club = None
        for club in clubs:
            if club['club_name'] == tm_club_name:
                target_club = club
                break
        
        if not target_club:
            print(f"  Warning: Club {tm_club_name} not found on Transfermarkt")
            scraper.close()
            return {}
        
        # Fetch players
        players = scraper.get_squad_players(
            target_club['club_slug'],
            target_club['club_id'],
            "kader",
            season_year
        )
        
        scraper.close()
        
        # Convert to dict with player_id as key
        players_dict = {}
        for player in players:
            player_id = player['player_id']
            players_dict[player_id] = {
                'player_id': player_id,
                'player_name': player.get('player_name', f'Player {player_id}'),
                'player_slug': player.get('player_slug'),
                'position': player.get('position', ''),
            }
        
        return players_dict
        
    except Exception as e:
        print(f"  Warning: Could not fetch from Transfermarkt: {e}")
        return {}


def validate_and_sync_club_config(
    club_name: str,
    country: str,
    data_date: str,
    auto_fix: bool = False,
    sync_transfermarkt: bool = False
) -> Dict:
    """
    Validate and optionally sync a club's config.json.
    
    Returns:
        {
            'players_to_add': List[Dict],  # Have raw data, missing from config
            'players_to_remove': List[Dict],  # In config, no raw data or transferred
            'orphaned_players': List[Dict],  # In config, no raw data
            'config_updated': bool,
            'config_path': Path
        }
    """
    print("=" * 80)
    print(f"VALIDATING CLUB CONFIG: {club_name}")
    print("=" * 80)
    print(f"Data date: {data_date}")
    print(f"Auto-fix: {auto_fix}")
    print(f"Sync Transfermarkt: {sync_transfermarkt}")
    print()
    
    # Load config
    config, config_path = load_config(club_name, country)
    if not config:
        print(f"ERROR: Could not load config from {config_path}")
        return {
            'players_to_add': [],
            'players_to_remove': [],
            'orphaned_players': [],
            'incomplete_players': [],
            'config_updated': False,
            'config_path': config_path
        }
    
    current_player_ids = set(config.get('player_ids', []))
    print(f"Current players in config: {len(current_player_ids)}")
    
    # Get raw data folder
    data_dir = get_latest_raw_data_folder(country.lower())
    if not data_dir:
        print(f"ERROR: No raw data folder found for {country}")
        return {
            'players_to_add': [],
            'players_to_remove': [],
            'orphaned_players': [],
            'incomplete_players': [],
            'config_updated': False,
            'config_path': config_path
        }
    
    print(f"Using raw data from: {data_dir.name}")
    print()
    
    # Get all players with raw data
    all_raw_data_players = get_players_from_raw_data(data_dir)
    print(f"Players with raw data: {len(all_raw_data_players)}")
    
    # Optionally fetch from Transfermarkt
    tm_players = {}
    if sync_transfermarkt:
        print("Fetching current squad from Transfermarkt...")
        tm_players = fetch_transfermarkt_players(club_name, season_year=2025)
        print(f"Players on Transfermarkt: {len(tm_players)}")
        print()
    
    # Analyze players in config
    print("Analyzing players in config...")
    players_to_remove = []
    orphaned_players = []
    
    for player_id in sorted(current_player_ids):
        # Check if player has raw data
        data_checks = check_player_has_raw_data(player_id, data_dir, club_name)
        has_profile = data_checks['profile']
        has_career = data_checks['career']
        has_match_data = data_checks['match_data_current'] or data_checks['match_data_previous']
        completeness = data_checks['completeness_level']
        
        # Get current club from profile
        current_club = get_player_current_club(player_id, data_dir)
        club_name_normalized = club_name.lower().strip()
        current_club_normalized = str(current_club).lower().strip() if current_club else ''
        is_in_club = club_name_normalized in current_club_normalized or current_club_normalized in club_name_normalized
        
        # Check Transfermarkt if enabled
        on_transfermarkt = player_id in tm_players if sync_transfermarkt else None
        
        # Check for orphaned or incomplete data first
        if completeness == 'orphaned':
            # Orphaned: no raw data at all
            orphaned_players.append({
                'player_id': player_id,
                'reason': 'No raw data (profile, career, or match data)',
                'completeness': completeness,
                'has_profile': has_profile,
                'has_career': has_career,
                'has_match_data': has_match_data,
                'missing_critical': data_checks['missing_critical'],
                'missing_optional': data_checks['missing_optional'],
            })
        elif completeness == 'incomplete':
            # Incomplete: missing critical data (profile or career)
            orphaned_players.append({
                'player_id': player_id,
                'reason': f"Missing critical data: {', '.join(data_checks['missing_critical'])}",
                'completeness': completeness,
                'has_profile': has_profile,
                'has_career': has_career,
                'has_match_data': has_match_data,
                'missing_critical': data_checks['missing_critical'],
                'missing_optional': data_checks['missing_optional'],
            })
        elif not is_in_club and current_club:
            # Transferred out
            players_to_remove.append({
                'player_id': player_id,
                'reason': f'Transferred to {current_club}',
                'current_club': current_club,
                'on_transfermarkt': on_transfermarkt,
            })
        elif sync_transfermarkt and on_transfermarkt is False:
            # Not on Transfermarkt squad
            players_to_remove.append({
                'player_id': player_id,
                'reason': 'Not on Transfermarkt squad',
                'current_club': current_club,
                'on_transfermarkt': False,
            })
    
    # Classify players_to_remove: PL→PL if current_club is another PL club
    pl_clubs = get_pl_club_names(country)
    for p in players_to_remove:
        current = p.get('current_club')
        if current:
            current_str = str(current).strip().lower()
            for c in pl_clubs:
                c_lower = c.lower().strip()
                if c_lower in current_str or current_str in c_lower:
                    p['transfer_type'] = 'pl_to_pl'
                    p['to_club'] = c
                    break
            if 'transfer_type' not in p:
                p['transfer_type'] = 'pl_to_non_pl'
        else:
            p['transfer_type'] = 'pl_to_non_pl'
    
    # Find players with raw data but missing from config
    print("Finding players with raw data but missing from config...")
    players_to_add = []
    
    # Get players in raw data but not in config
    missing_from_config = all_raw_data_players - current_player_ids
    
    for player_id in sorted(missing_from_config):
        # Check if player is currently in this club
        current_club = get_player_current_club(player_id, data_dir)
        club_name_normalized = club_name.lower().strip()
        current_club_normalized = str(current_club).lower().strip() if current_club else ''
        is_in_club = club_name_normalized in current_club_normalized or current_club_normalized in club_name_normalized
        
        if is_in_club:
            data_checks = check_player_has_raw_data(player_id, data_dir, club_name)
            players_to_add.append({
                'player_id': player_id,
                'current_club': current_club,
                'has_profile': data_checks['profile'],
                'has_career': data_checks['career'],
                'has_match_data': data_checks['match_data_current'] or data_checks['match_data_previous'],
                'on_transfermarkt': player_id in tm_players if sync_transfermarkt else None,
            })
    
    # Separate orphaned vs incomplete players
    orphaned = [p for p in orphaned_players if p.get('completeness') == 'orphaned']
    incomplete = [p for p in orphaned_players if p.get('completeness') == 'incomplete']
    
    # Print summary
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Players to add: {len(players_to_add)}")
    print(f"Players to remove: {len(players_to_remove)}")
    print(f"Orphaned players (no data): {len(orphaned)}")
    print(f"Incomplete players (missing critical data): {len(incomplete)}")
    print()
    
    if players_to_add:
        print("PLAYERS TO ADD:")
        for p in players_to_add:
            print(f"  - {p['player_id']}: {p.get('current_club', 'Unknown')}")
    print()
    
    if players_to_remove:
        print("PLAYERS TO REMOVE:")
        for p in players_to_remove:
            print(f"  - {p['player_id']}: {p['reason']}")
    print()
    
    if orphaned:
        print("ORPHANED PLAYERS (no data at all):")
        for p in orphaned:
            print(f"  - {p['player_id']}: {p['reason']}")
    print()
    
    if incomplete:
        print("INCOMPLETE PLAYERS (missing critical data):")
        for p in incomplete:
            missing = ', '.join(p.get('missing_critical', []))
            has_match = p.get('has_match_data', False)
            print(f"  - {p['player_id']}: Missing {missing} (has match data: {has_match})")
    print()
    
    # Auto-fix if requested
    config_updated = False
    if auto_fix and (players_to_add or players_to_remove or orphaned_players):
        print("=" * 80)
        print("AUTO-FIXING CONFIG...")
        print("=" * 80)
        
        new_player_ids = set(current_player_ids)
        club_path = config_path.parent  # challenger/{country}/{club}
        daily_features_dir = club_path / "daily_features"
        daily_features_enhanced_dir = club_path / "daily_features_enhanced"
        players_dir = club_path / "predictions" / "players"
        
        # Remove transferred/orphaned players and clean up their files (non-PL→PL only)
        for p in players_to_remove + orphaned_players:
            pid = p['player_id']
            if pid in new_player_ids:
                new_player_ids.remove(pid)
                print(f"  [-] Removing player {pid}: {p['reason']}")
            
            # Delete daily_features (Layer 1) for non-PL→PL
            is_pl_to_pl = p.get('transfer_type') == 'pl_to_pl'
            if not is_pl_to_pl and daily_features_dir.exists():
                df_file = daily_features_dir / f"player_{pid}_daily_features.csv"
                if df_file.exists():
                    try:
                        df_file.unlink()
                        print(f"  [CLEANUP] Removed daily_features: {df_file.name}")
                    except OSError as e:
                        print(f"  [WARNING] Could not remove {df_file}: {e}")
            
            # Delete daily_features_enhanced (Layer 2) for non-PL→PL
            if not is_pl_to_pl and daily_features_enhanced_dir.exists():
                df_enh_file = daily_features_enhanced_dir / f"player_{pid}_daily_features.csv"
                if df_enh_file.exists():
                    try:
                        df_enh_file.unlink()
                        print(f"  [CLEANUP] Removed daily_features_enhanced: {df_enh_file.name}")
                    except OSError as e:
                        print(f"  [WARNING] Could not remove {df_enh_file}: {e}")
            
            # Remove per-player V4 prediction files for removed players
            if not is_pl_to_pl and players_dir.exists():
                for pred_file in players_dir.glob(f"player_{pid}_predictions_v4_*.csv"):
                    try:
                        pred_file.unlink()
                        print(f"  [CLEANUP] Removed prediction: {pred_file.name}")
                    except OSError as e:
                        print(f"  [WARNING] Could not remove {pred_file}: {e}")
        
        # Add missing players
        for p in players_to_add:
            new_player_ids.add(p['player_id'])
            print(f"  [+] Adding player {p['player_id']}")
        
        # Update config (convert to regular Python ints for JSON serialization)
        config['player_ids'] = sorted([int(pid) for pid in new_player_ids])
        
        if save_config(config, config_path):
            config_updated = True
            print(f"\n[OK] Config updated: {len(new_player_ids)} players (was {len(current_player_ids)})")
        else:
            print(f"\n[ERROR] Failed to save config")
    elif auto_fix:
        print("No changes needed - config is already in sync")
    
    # Persist sync result for orchestrator (deploy_all_clubs_v3) to read PL→PL moves
    result_data = {
        'players_to_add': players_to_add,
        'players_to_remove': players_to_remove,
        'orphaned_players': orphaned,
        'incomplete_players': incomplete,
        'config_updated': config_updated,
        'config_path': str(config_path),
        'club_name': club_name,
        'country': country,
        'data_date': data_date,
    }
    sync_result_path = get_challenger_deployments_dir(country) / f".sync_result_challenger_{data_date}.json"
    try:
        existing = {}
        if sync_result_path.exists():
            with open(sync_result_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        if 'clubs' not in existing:
            existing['clubs'] = {}
        # Merge this club's result (overwrite key)
        existing['clubs'][club_name] = {
            'players_to_remove': players_to_remove,
            'players_to_add': [{'player_id': p['player_id'], 'current_club': p.get('current_club')} for p in players_to_add],
            'pl_to_pl': [{'player_id': p['player_id'], 'to_club': p['to_club']} for p in players_to_remove if p.get('transfer_type') == 'pl_to_pl'],
        }
        existing['data_date'] = data_date
        existing['country'] = country
        with open(sync_result_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [WARNING] Could not write sync result to {sync_result_path}: {e}")
    
    return {
        'players_to_add': players_to_add,
        'players_to_remove': players_to_remove,
        'orphaned_players': orphaned,
        'incomplete_players': incomplete,
        'config_updated': config_updated,
        'config_path': config_path
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate and sync club config (challenger/V4 only; does not touch V3 paths)'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., "England")'
    )
    parser.add_argument(
        '--club',
        type=str,
        required=True,
        help='Club name (e.g., "AFC Bournemouth")'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Data date (YYYYMMDD format, default: latest available)'
    )
    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help='Automatically update config.json (default: report only)'
    )
    parser.add_argument(
        '--sync-transfermarkt',
        action='store_true',
        help='Sync with Transfermarkt to detect transfers (slower but more accurate)'
    )
    
    args = parser.parse_args()
    
    data_date = args.data_date or get_latest_raw_data_folder(args.country.lower())
    if isinstance(data_date, Path):
        data_date = data_date.name
    
    if not data_date:
        print("ERROR: No data date available and none specified")
        return 1
    
    result = validate_and_sync_club_config(
        args.club,
        args.country,
        data_date,
        auto_fix=args.auto_fix,
        sync_transfermarkt=args.sync_transfermarkt
    )
    
    if result['config_updated']:
        return 0
    elif result['players_to_add'] or result['players_to_remove'] or result['orphaned_players']:
        print("\n[NOTE] Use --auto-fix to automatically update the config")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
