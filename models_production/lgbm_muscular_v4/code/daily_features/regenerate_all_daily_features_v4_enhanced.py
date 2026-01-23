#!/usr/bin/env python3
"""
Regenerate daily features files for V4 Enhanced (excluding goalkeepers).

Supports two modes:
  a) Test mode: Process a subset of X players (default: 10)
  b) Full mode: Process all non-goalkeeper players

Features:
  - Group-based processing with checkpoint/resume capability
  - Enhanced monitoring and progress tracking

Usage:
  # Test mode (10 players)
  python regenerate_all_daily_features_v4_enhanced.py --test
  
  # Test mode (custom number)
  python regenerate_all_daily_features_v4_enhanced.py --test --num-players 50
  
  # Test mode (specific player IDs)
  python regenerate_all_daily_features_v4_enhanced.py --test --player-ids 123456 789012 345678
  
  # Full regeneration (all non-goalkeeper players)
  python regenerate_all_daily_features_v4_enhanced.py --full
  
  # Full regeneration with groups (auto-resume)
  python regenerate_all_daily_features_v4_enhanced.py --full --group-size 100
  
  # Resume from checkpoint
  python regenerate_all_daily_features_v4_enhanced.py --full --resume
  
  # Start fresh (ignore checkpoint)
  python regenerate_all_daily_features_v4_enhanced.py --full --clear-checkpoint
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import argparse
import pandas as pd
from pathlib import Path
import random
import json
from datetime import datetime
import time

# Add script directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import V4 Enhanced script
from create_daily_features_v4_enhanced import (
    generate_daily_features_for_player,
    DATA_DIR,
    REFERENCE_DATE,
    OUTPUT_DIR
)

# Checkpoint file
DEFAULT_CHECKPOINT_FILE = "checkpoint_v4_enhanced.json"

def is_goalkeeper(position):
    """Check if position indicates a goalkeeper"""
    if pd.isna(position):
        return False
    position_str = str(position).strip().lower()
    goalkeeper_indicators = ['goalkeeper', 'gk', 'keeper']
    return any(indicator in position_str for indicator in goalkeeper_indicators)

def get_non_goalkeeper_players(players_path: Path, seed: int = 42) -> tuple:
    """Load and filter out goalkeepers from players_profile.csv"""
    if not players_path.exists():
        raise FileNotFoundError(f"Players profile not found: {players_path}")
    
    players_df = pd.read_csv(players_path, sep=';', encoding='utf-8')
    total_players = len(players_df)
    
    # Filter out goalkeepers
    if 'position' in players_df.columns:
        non_gk_mask = ~players_df['position'].apply(is_goalkeeper)
        players_df = players_df[non_gk_mask].copy()
        gk_count = total_players - len(players_df)
        print(f"📊 Total players in profile: {total_players}")
        print(f"🚫 Goalkeepers excluded: {gk_count}")
        print(f"✅ Non-goalkeeper players available: {len(players_df)}")
    else:
        print("⚠️  Warning: 'position' column not found, including all players")
        gk_count = 0
    
    return players_df, gk_count

def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint if exists"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            return None
    return None

def save_checkpoint(checkpoint_path: Path, data: dict):
    """Save checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)

def detect_existing_files(output_dir: str) -> set:
    """Scan output directory for existing daily features files and return set of player IDs"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()
    
    existing_ids = set()
    for csv_file in output_path.glob("player_*_daily_features.csv"):
        try:
            # Extract player_id from filename: player_123456_daily_features.csv
            player_id = int(csv_file.stem.split('_')[1])
            existing_ids.add(player_id)
        except (ValueError, IndexError):
            # Skip files with unexpected naming
            continue
    
    return existing_ids

def regenerate_players(
    player_ids: list,
    data_dir: str,
    reference_date,
    output_dir: str,
    verbose: bool = False,
    group_size: int = None,
    checkpoint_path: Path = None,
    processed_ids: set = None
) -> dict:
    """Regenerate daily features for a list of player IDs"""
    
    results = {
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    processed_ids = processed_ids or set()
    total_players = len(player_ids)
    start_time = time.time()
    
    # Group-based processing
    if group_size and group_size > 0:
        total_groups = (total_players + group_size - 1) // group_size
        print(f"📦 Processing in {total_groups} groups of {group_size} players")
        
        for group_num in range(total_groups):
            group_start_idx = group_num * group_size
            group_end_idx = min(group_start_idx + group_size, total_players)
            group_player_ids = player_ids[group_start_idx:group_end_idx]
            
            # Skip already processed players
            group_player_ids = [pid for pid in group_player_ids if pid not in processed_ids]
            
            if not group_player_ids:
                print(f"⏭️  Group {group_num + 1}/{total_groups}: All players already processed, skipping")
                continue
            
            print(f"\n{'=' * 80}")
            print(f"📦 GROUP {group_num + 1}/{total_groups}")
            print(f"{'=' * 80}")
            group_start_time = time.time()
            
            for i, player_id in enumerate(group_player_ids, 1):
                global_idx = group_start_idx + i
                if verbose:
                    print(f"\n[{global_idx}/{total_players}] Processing player {player_id}...")
                else:
                    if i % 10 == 0 or i == len(group_player_ids):
                        print(f"Progress: {global_idx}/{total_players} ({global_idx/total_players*100:.1f}%)")
                
                try:
                    df = generate_daily_features_for_player(
                        player_id=player_id,
                        data_dir=data_dir,
                        reference_date=reference_date,
                        output_dir=output_dir
                    )
                    if not df.empty:
                        # Save the file
                        output_path = Path(output_dir) / f"player_{player_id}_daily_features.csv"
                        os.makedirs(output_dir, exist_ok=True)
                        # Always reset index first
                        if isinstance(df.index, pd.DatetimeIndex) and 'date' not in df.columns:
                            df = df.reset_index()
                            index_col = None
                            for col in df.columns:
                                if col == '' or (isinstance(col, float) and pd.isna(col)) or col == df.index.name:
                                    index_col = col
                                    break
                            if index_col:
                                df = df.rename(columns={index_col: 'date'})
                            elif len(df.columns) > 0 and df.columns[0] not in ['player_id', 'date']:
                                df = df.rename(columns={df.columns[0]: 'date'})
                        else:
                            df = df.reset_index(drop=True)
                            cols_to_drop = [col for col in df.columns 
                                            if (col == '' or (isinstance(col, float) and pd.isna(col)))]
                            if cols_to_drop:
                                df = df.drop(columns=cols_to_drop)
                        df.to_csv(output_path, index=False, encoding='utf-8-sig')
                        results['successful'] += 1
                        processed_ids.add(player_id)
                        if verbose:
                            print(f"✅ Player {player_id}: {len(df)} days generated, saved to {output_path.name}")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Player {player_id}: No data generated")
                        if verbose:
                            print(f"⚠️  Player {player_id}: No data generated")
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Player {player_id}: {str(e)}"
                    results['errors'].append(error_msg)
                    if verbose:
                        print(f"❌ {error_msg}")
                        import traceback
                        traceback.print_exc()
            
            # Save checkpoint after each group
            if checkpoint_path:
                checkpoint_data = {
                    'last_update': datetime.now().isoformat(),
                    'processed_player_ids': list(processed_ids),
                    'current_group': group_num + 1,
                    'total_groups': total_groups,
                    'statistics': {
                        'successful': results['successful'],
                        'failed': results['failed'],
                        'skipped': results['skipped']
                    },
                    'errors': results['errors'][-10:]  # Keep last 10 errors
                }
                save_checkpoint(checkpoint_path, checkpoint_data)
            
            group_time = time.time() - group_start_time
            print(f"\n✅ GROUP {group_num + 1} COMPLETED")
            print(f"   Processed: {len(group_player_ids)} players")
            print(f"   Successful: {sum(1 for pid in group_player_ids if pid in processed_ids)} | Failed: {results['failed']}")
            print(f"   Group time: {group_time/60:.1f} minutes")
            if checkpoint_path:
                print(f"   Checkpoint saved")
    else:
        # Non-grouped processing (original behavior)
        for i, player_id in enumerate(player_ids, 1):
            if verbose:
                print(f"\n[{i}/{len(player_ids)}] Processing player {player_id}...")
            else:
                if i % 10 == 0 or i == len(player_ids):
                    print(f"Progress: {i}/{len(player_ids)} ({i/len(player_ids)*100:.1f}%)")
            
            try:
                df = generate_daily_features_for_player(
                    player_id=player_id,
                    data_dir=data_dir,
                    reference_date=reference_date,
                    output_dir=output_dir
                )
                if not df.empty:
                    # FIXED: Save the file with date as a column (not index)
                    output_path = Path(output_dir) / f"player_{player_id}_daily_features.csv"
                    os.makedirs(output_dir, exist_ok=True)
                    # Always reset index first
                    if isinstance(df.index, pd.DatetimeIndex) and 'date' not in df.columns:
                        # Create date column from index
                        df = df.reset_index()
                        # Rename the index column (might be unnamed)
                        index_col = None
                        for col in df.columns:
                            if col == '' or (isinstance(col, float) and pd.isna(col)) or col == df.index.name:
                                index_col = col
                                break
                        if index_col:
                            df = df.rename(columns={index_col: 'date'})
                        elif len(df.columns) > 0 and df.columns[0] not in ['player_id', 'date']:
                            df = df.rename(columns={df.columns[0]: 'date'})
                    else:
                        # Date column exists or no DatetimeIndex, just drop the index
                        df = df.reset_index(drop=True)
                        # Remove any unnamed columns (empty string or NaN column names) that might contain dates
                        cols_to_drop = [col for col in df.columns 
                                        if (col == '' or (isinstance(col, float) and pd.isna(col)))]
                        if cols_to_drop:
                            df = df.drop(columns=cols_to_drop)
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    results['successful'] += 1
                    if verbose:
                        print(f"✅ Player {player_id}: {len(df)} days generated, saved to {output_path.name}")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Player {player_id}: No data generated")
                    if verbose:
                        print(f"⚠️  Player {player_id}: No data generated")
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Player {player_id}: {str(e)}"
                results['errors'].append(error_msg)
                if verbose:
                    print(f"❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Regenerate daily features files for V4 (excluding goalkeepers)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (10 players, default)
  python regenerate_all_daily_features_v4.py --test
  
  # Test mode (50 players)
  python regenerate_all_daily_features_v4.py --test --num-players 50
  
  # Test mode (specific players)
  python regenerate_all_daily_features_v4.py --test --player-ids 123456 789012 345678
  
  # Full regeneration (all non-goalkeeper players)
  python regenerate_all_daily_features_v4.py --full
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--test',
        action='store_true',
        help='Test mode: Process a subset of players'
    )
    mode_group.add_argument(
        '--full',
        action='store_true',
        help='Full mode: Process all non-goalkeeper players'
    )
    
    # Test mode options
    parser.add_argument(
        '--num-players',
        type=int,
        default=10,
        help='Number of players to process in test mode (default: 10)'
    )
    parser.add_argument(
        '--player-ids',
        type=int,
        nargs='+',
        help='Specific player IDs to process (overrides --num-players)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for player selection in test mode (default: 42)'
    )
    
    # General options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output (show details for each player)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help=f'Data directory (default: from create_daily_features_v4.py: {DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: from create_daily_features_v4_enhanced.py: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--group-size',
        type=int,
        default=100,
        help='Number of players per group for checkpoint/resume (default: 100, set to 0 to disable grouping)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Auto-resume from checkpoint if it exists'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Start fresh (ignore checkpoint)'
    )
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default=None,
        help=f'Custom checkpoint file path (default: {DEFAULT_CHECKPOINT_FILE})'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = args.data_dir if args.data_dir else str(Path(DATA_DIR).resolve())
    output_dir = args.output_dir if args.output_dir else str(Path(OUTPUT_DIR).resolve())
    players_path = Path(data_dir) / "players_profile.csv"
    
    # Setup checkpoint
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else Path(SCRIPT_DIR) / DEFAULT_CHECKPOINT_FILE
    checkpoint = None
    processed_ids = set()
    
    # Detect existing files in output directory
    existing_file_ids = detect_existing_files(output_dir)
    if existing_file_ids:
        print(f"📁 Detected {len(existing_file_ids)} existing daily features files in output directory")
        processed_ids.update(existing_file_ids)
    
    if args.resume or (not args.clear_checkpoint and checkpoint_path.exists()):
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            checkpoint_ids = set(checkpoint.get('processed_player_ids', []))
            processed_ids.update(checkpoint_ids)
            print(f"📊 RESUMING FROM CHECKPOINT")
            print(f"   ✅ Already processed (checkpoint): {len(checkpoint_ids)} players")
            if existing_file_ids:
                # Show overlap and differences
                overlap = checkpoint_ids & existing_file_ids
                only_in_files = existing_file_ids - checkpoint_ids
                only_in_checkpoint = checkpoint_ids - existing_file_ids
                if overlap:
                    print(f"   ✅ Overlap (in both): {len(overlap)} players")
                if only_in_files:
                    print(f"   📁 Found in files but not checkpoint: {len(only_in_files)} players")
                if only_in_checkpoint:
                    print(f"   ⚠️  In checkpoint but file missing: {len(only_in_checkpoint)} players")
            print(f"   📍 Last update: {checkpoint.get('last_update', 'unknown')}")
            if 'current_group' in checkpoint:
                print(f"   📦 Last group: {checkpoint['current_group']}/{checkpoint.get('total_groups', '?')}")
    
    print(f"   ✅ Total to skip: {len(processed_ids)} players (from files + checkpoint)")
    
    print("=" * 80)
    print("V4 ENHANCED DAILY FEATURES REGENERATION")
    print("=" * 80)
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Group size: {args.group_size if args.group_size > 0 else 'Disabled'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Resume mode: {'Enabled' if (args.resume or checkpoint) and not args.clear_checkpoint else 'Disabled'}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load and filter players
    try:
        players_df, gk_count = get_non_goalkeeper_players(players_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Select players based on mode
    if args.test:
        if args.player_ids:
            # Use specified player IDs
            selected_ids = [pid for pid in args.player_ids if pid in players_df['id'].values]
            missing_ids = [pid for pid in args.player_ids if pid not in players_df['id'].values]
            
            if missing_ids:
                print(f"⚠️  Warning: {len(missing_ids)} specified player IDs not found: {missing_ids}")
            
            if not selected_ids:
                print("❌ No valid player IDs found!")
                return 1
            
            player_ids = selected_ids
            print(f"🎯 Test mode: Processing {len(player_ids)} specified player(s)")
        else:
            # Random sample
            num_players = min(args.num_players, len(players_df))
            if num_players < args.num_players:
                print(f"⚠️  Warning: Only {len(players_df)} players available, processing all")
            
            random.seed(args.seed)
            player_ids = random.sample(players_df['id'].tolist(), num_players)
            print(f"🎯 Test mode: Processing {len(player_ids)} randomly selected player(s) (seed: {args.seed})")
    else:
        # Full mode: all non-goalkeeper players
        player_ids = players_df['id'].unique().tolist()
        print(f"🚀 Full mode: Processing all {len(player_ids)} non-goalkeeper players")
        if processed_ids:
            remaining = [pid for pid in player_ids if pid not in processed_ids]
            print(f"   🎯 Remaining: {len(remaining)} players ({len(processed_ids)} already processed)")
            player_ids = remaining
    
    print()
    
    # Process players
    results = regenerate_players(
        player_ids=player_ids,
        data_dir=data_dir,
        reference_date=REFERENCE_DATE,
        output_dir=output_dir,
        verbose=args.verbose,
        group_size=args.group_size if args.group_size > 0 else None,
        checkpoint_path=checkpoint_path if args.group_size > 0 else None,
        processed_ids=processed_ids
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("REGENERATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📊 Total processed: {results['successful'] + results['failed']}")
    print(f"📁 Output directory: {output_dir}")
    
    if results['errors']:
        print(f"\n⚠️  Errors encountered ({len(results['errors'])}):")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(results['errors']) > 10:
            print(f"   ... and {len(results['errors']) - 10} more errors")
    
    return 0 if results['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
