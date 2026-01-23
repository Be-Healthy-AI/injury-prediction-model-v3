#!/usr/bin/env python3
"""
Compare V2 and V3 predictions side-by-side.

This script loads predictions from both V2 and V3 models for the same club and date,
and provides comprehensive comparison metrics, rankings, and differences.
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent


def load_predictions(club_path: Path, date_str: str, model_version: str) -> Optional[pd.DataFrame]:
    """Load predictions for a given model version."""
    predictions_file = club_path / "predictions" / f"predictions_lgbm_v{model_version}_{date_str}.csv"
    
    if not predictions_file.exists():
        print(f"[WARN] Predictions file not found: {predictions_file}")
        return None
    
    df = pd.read_csv(predictions_file, encoding='utf-8-sig', low_memory=False)
    
    # Ensure reference_date is datetime
    if 'reference_date' in df.columns:
        df['reference_date'] = pd.to_datetime(df['reference_date'], errors='coerce')
    
    return df


def compare_statistics(df_v2: pd.DataFrame, df_v3: pd.DataFrame) -> dict:
    """Compare statistical distributions of predictions."""
    stats = {}
    
    for version, df in [('v2', df_v2), ('v3', df_v3)]:
        stats[version] = {
            'count': len(df),
            'mean': df['injury_probability'].mean(),
            'median': df['injury_probability'].median(),
            'std': df['injury_probability'].std(),
            'min': df['injury_probability'].min(),
            'max': df['injury_probability'].max(),
            'q25': df['injury_probability'].quantile(0.25),
            'q75': df['injury_probability'].quantile(0.75),
            'q90': df['injury_probability'].quantile(0.90),
            'q95': df['injury_probability'].quantile(0.95),
            'q99': df['injury_probability'].quantile(0.99),
        }
    
    # Calculate differences
    stats['difference'] = {
        'mean_diff': stats['v3']['mean'] - stats['v2']['mean'],
        'mean_pct_diff': ((stats['v3']['mean'] - stats['v2']['mean']) / stats['v2']['mean'] * 100) if stats['v2']['mean'] > 0 else 0,
        'max_diff': stats['v3']['max'] - stats['v2']['max'],
    }
    
    return stats


def merge_predictions(df_v2: pd.DataFrame, df_v3: pd.DataFrame) -> pd.DataFrame:
    """Merge V2 and V3 predictions on player_id and reference_date."""
    # Merge on player_id and reference_date
    merged = pd.merge(
        df_v2[['player_id', 'reference_date', 'injury_probability', 'risk_level']].copy(),
        df_v3[['player_id', 'reference_date', 'injury_probability', 'risk_level']].copy(),
        on=['player_id', 'reference_date'],
        how='inner',
        suffixes=('_v2', '_v3')
    )
    
    # Add player_name if available
    if 'player_name' in df_v2.columns:
        merged = merged.merge(
            df_v2[['player_id', 'reference_date', 'player_name']],
            on=['player_id', 'reference_date'],
            how='left'
        )
    
    # Calculate differences
    merged['prob_diff'] = merged['injury_probability_v3'] - merged['injury_probability_v2']
    merged['prob_diff_abs'] = merged['prob_diff'].abs()
    merged['prob_diff_pct'] = (merged['prob_diff'] / merged['injury_probability_v2'] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Risk level comparison
    merged['risk_level_match'] = merged['risk_level_v2'] == merged['risk_level_v3']
    
    return merged


def compare_rankings(merged: pd.DataFrame, top_n: int = 20) -> dict:
    """Compare rankings between V2 and V3."""
    # Get top N for each model
    top_v2 = merged.nlargest(top_n, 'injury_probability_v2')
    top_v3 = merged.nlargest(top_n, 'injury_probability_v3')
    
    # Create sets of (player_id, reference_date) tuples
    top_v2_set = set(zip(top_v2['player_id'], top_v2['reference_date']))
    top_v3_set = set(zip(top_v3['player_id'], top_v3['reference_date']))
    
    # Calculate overlap
    overlap = top_v2_set & top_v3_set
    only_v2 = top_v2_set - top_v3_set
    only_v3 = top_v3_set - top_v2_set
    
    return {
        'top_n': top_n,
        'overlap_count': len(overlap),
        'overlap_pct': len(overlap) / top_n * 100,
        'only_v2_count': len(only_v2),
        'only_v3_count': len(only_v3),
    }


def print_comparison_report(
    stats: dict,
    merged: pd.DataFrame,
    rankings: dict,
    club: str,
    date_str: str,
    output_file: Optional[Path] = None
):
    """Print comprehensive comparison report."""
    
    report_lines = []
    
    def add_line(text: str = ""):
        report_lines.append(text)
        print(text)
    
    add_line("=" * 80)
    add_line(f"V2 vs V3 PREDICTIONS COMPARISON")
    add_line("=" * 80)
    add_line(f"Club: {club}")
    add_line(f"Date: {date_str}")
    add_line(f"Total matched predictions: {len(merged):,}")
    add_line()
    
    # Statistical comparison
    add_line("=" * 80)
    add_line("STATISTICAL COMPARISON")
    add_line("=" * 80)
    add_line(f"{'Metric':<20} {'V2':<15} {'V3':<15} {'Difference':<15}")
    add_line("-" * 80)
    add_line(f"{'Count':<20} {stats['v2']['count']:<15,} {stats['v3']['count']:<15,} {merged.shape[0]:<15,}")
    add_line(f"{'Mean':<20} {stats['v2']['mean']:<15.4f} {stats['v3']['mean']:<15.4f} {stats['difference']['mean_diff']:+.4f} ({stats['difference']['mean_pct_diff']:+.2f}%)")
    add_line(f"{'Median':<20} {stats['v2']['median']:<15.4f} {stats['v3']['median']:<15.4f}")
    add_line(f"{'Std Dev':<20} {stats['v2']['std']:<15.4f} {stats['v3']['std']:<15.4f}")
    add_line(f"{'Min':<20} {stats['v2']['min']:<15.4f} {stats['v3']['min']:<15.4f}")
    add_line(f"{'Max':<20} {stats['v2']['max']:<15.4f} {stats['v3']['max']:<15.4f} {stats['difference']['max_diff']:+.4f}")
    add_line(f"{'25th Percentile':<20} {stats['v2']['q25']:<15.4f} {stats['v3']['q25']:<15.4f}")
    add_line(f"{'75th Percentile':<20} {stats['v2']['q75']:<15.4f} {stats['v3']['q75']:<15.4f}")
    add_line(f"{'90th Percentile':<20} {stats['v2']['q90']:<15.4f} {stats['v3']['q90']:<15.4f}")
    add_line(f"{'95th Percentile':<20} {stats['v2']['q95']:<15.4f} {stats['v3']['q95']:<15.4f}")
    add_line(f"{'99th Percentile':<20} {stats['v2']['q99']:<15.4f} {stats['v3']['q99']:<15.4f}")
    add_line()
    
    # Correlation
    correlation = merged['injury_probability_v2'].corr(merged['injury_probability_v3'])
    add_line("=" * 80)
    add_line("CORRELATION")
    add_line("=" * 80)
    add_line(f"Pearson correlation: {correlation:.4f}")
    add_line()
    
    # Difference analysis
    add_line("=" * 80)
    add_line("DIFFERENCE ANALYSIS")
    add_line("=" * 80)
    add_line(f"Mean absolute difference: {merged['prob_diff_abs'].mean():.4f}")
    add_line(f"Median absolute difference: {merged['prob_diff_abs'].median():.4f}")
    add_line(f"Max absolute difference: {merged['prob_diff_abs'].max():.4f}")
    add_line()
    add_line(f"V3 > V2 (higher risk): {len(merged[merged['prob_diff'] > 0]):,} ({len(merged[merged['prob_diff'] > 0])/len(merged)*100:.1f}%)")
    add_line(f"V3 < V2 (lower risk): {len(merged[merged['prob_diff'] < 0]):,} ({len(merged[merged['prob_diff'] < 0])/len(merged)*100:.1f}%)")
    add_line(f"V3 = V2 (same): {len(merged[merged['prob_diff'] == 0]):,} ({len(merged[merged['prob_diff'] == 0])/len(merged)*100:.1f}%)")
    add_line()
    
    # Risk level agreement
    risk_agreement = merged['risk_level_match'].sum() / len(merged) * 100
    add_line(f"Risk level agreement: {risk_agreement:.1f}% ({merged['risk_level_match'].sum():,}/{len(merged):,})")
    add_line()
    
    # Ranking comparison
    add_line("=" * 80)
    add_line(f"TOP {rankings['top_n']} RANKINGS COMPARISON")
    add_line("=" * 80)
    add_line(f"Overlap: {rankings['overlap_count']}/{rankings['top_n']} ({rankings['overlap_pct']:.1f}%)")
    add_line(f"Only in V2 top {rankings['top_n']}: {rankings['only_v2_count']}")
    add_line(f"Only in V3 top {rankings['top_n']}: {rankings['only_v3_count']}")
    add_line()
    
    # Top differences
    add_line("=" * 80)
    add_line("LARGEST DIFFERENCES (V3 - V2)")
    add_line("=" * 80)
    top_diff = merged.nlargest(10, 'prob_diff_abs')
    for idx, row in top_diff.iterrows():
        name = row.get('player_name', f"Player {row['player_id']}") if 'player_name' in row else f"Player {row['player_id']}"
        add_line(f"{name:<30} {row['reference_date'].date()} | V2: {row['injury_probability_v2']:.4f} | V3: {row['injury_probability_v3']:.4f} | Diff: {row['prob_diff']:+.4f}")
    add_line()
    
    # Top risks comparison
    add_line("=" * 80)
    add_line("TOP 10 HIGHEST RISK - V2")
    add_line("=" * 80)
    top_v2 = merged.nlargest(10, 'injury_probability_v2')
    for idx, row in top_v2.iterrows():
        name = row.get('player_name', f"Player {row['player_id']}") if 'player_name' in row else f"Player {row['player_id']}"
        add_line(f"{name:<30} {row['reference_date'].date()} | V2: {row['injury_probability_v2']:.4f} | V3: {row['injury_probability_v3']:.4f}")
    add_line()
    
    add_line("=" * 80)
    add_line("TOP 10 HIGHEST RISK - V3")
    add_line("=" * 80)
    top_v3 = merged.nlargest(10, 'injury_probability_v3')
    for idx, row in top_v3.iterrows():
        name = row.get('player_name', f"Player {row['player_id']}") if 'player_name' in row else f"Player {row['player_id']}"
        add_line(f"{name:<30} {row['reference_date'].date()} | V2: {row['injury_probability_v2']:.4f} | V3: {row['injury_probability_v3']:.4f}")
    add_line()
    
    add_line("=" * 80)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n[SAVE] Comparison report saved to: {output_file}")
    
    return report_lines


def main():
    parser = argparse.ArgumentParser(
        description='Compare V2 and V3 predictions side-by-side'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='England',
        help='Country name (default: England)'
    )
    parser.add_argument(
        '--club',
        type=str,
        required=True,
        help='Club name (required)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date in YYYY-MM-DD format (default: latest available)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for comparison report (optional)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top risks to compare (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Get club path
    club_path = PRODUCTION_ROOT / "deployments" / args.country / args.club
    
    if not club_path.exists():
        print(f"[ERROR] Club path not found: {club_path}")
        return 1
    
    # Determine date
    if args.date:
        date_str = args.date.replace('-', '')
    else:
        # Find latest predictions file
        predictions_dir = club_path / "predictions"
        v2_files = list(predictions_dir.glob("predictions_lgbm_v2_*.csv"))
        if not v2_files:
            print(f"[ERROR] No V2 predictions found in {predictions_dir}")
            return 1
        # Extract date from filename
        latest_v2 = max(v2_files, key=lambda x: x.stem.split('_')[-1])
        date_str = latest_v2.stem.split('_')[-1]
        print(f"[INFO] Using latest available date: {date_str}")
    
    # Load predictions
    print(f"\n[LOAD] Loading V2 predictions...")
    df_v2 = load_predictions(club_path, date_str, '2')
    if df_v2 is None:
        print(f"[ERROR] Could not load V2 predictions")
        return 1
    
    print(f"[LOAD] Loading V3 predictions...")
    df_v3 = load_predictions(club_path, date_str, '3')
    if df_v3 is None:
        print(f"[ERROR] Could not load V3 predictions")
        return 1
    
    print(f"[LOAD] V2: {len(df_v2):,} predictions")
    print(f"[LOAD] V3: {len(df_v3):,} predictions")
    
    # Compare statistics
    print(f"\n[ANALYZE] Comparing statistics...")
    stats = compare_statistics(df_v2, df_v3)
    
    # Merge predictions
    print(f"[ANALYZE] Merging predictions...")
    merged = merge_predictions(df_v2, df_v3)
    print(f"[ANALYZE] Matched {len(merged):,} predictions")
    
    # Compare rankings
    print(f"[ANALYZE] Comparing rankings...")
    rankings = compare_rankings(merged, args.top_n)
    
    # Generate report
    output_path = Path(args.output) if args.output else None
    print_comparison_report(stats, merged, rankings, args.club, date_str, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())

