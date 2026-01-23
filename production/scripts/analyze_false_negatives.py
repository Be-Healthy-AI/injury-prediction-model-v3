#!/usr/bin/env python3
"""
Analyze false negative cases (low predicted probability but injury occurred).

This script identifies cases where the model predicted low probability (< 0.3)
but the player got injured within 5 days, and conducts a thorough analysis
to identify patterns and potential improvements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Calculate paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_ROOT = SCRIPT_DIR.parent
ROOT_DIR = PRODUCTION_ROOT.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def get_all_clubs(country: str = "England") -> List[str]:
    """Get all club folders in the deployments directory."""
    deployments_dir = PRODUCTION_ROOT / "deployments" / country
    if not deployments_dir.exists():
        return []
    
    clubs = []
    for item in deployments_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            clubs.append(item.name)
    
    return sorted(clubs)


def load_all_predictions(
    clubs: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """Load all V3 predictions from all clubs within date range."""
    all_predictions = []
    
    for club in clubs:
        predictions_dir = PRODUCTION_ROOT / "deployments" / "England" / club / "predictions"
        pattern = "predictions_lgbm_v3_*.csv"
        prediction_files = list(predictions_dir.glob(pattern))
        
        if not prediction_files:
            continue
        
        for pred_file in prediction_files:
            try:
                df = pd.read_csv(pred_file, parse_dates=['reference_date'], low_memory=False)
                df_filtered = df[
                    (df['reference_date'] >= start_date) & 
                    (df['reference_date'] <= end_date)
                ].copy()
                
                if not df_filtered.empty:
                    df_filtered['club'] = club
                    all_predictions.append(df_filtered)
            except Exception as e:
                print(f"  [WARN] Could not load {pred_file}: {e}")
    
    if not all_predictions:
        raise ValueError("No predictions found for the specified date range")
    
    combined = pd.concat(all_predictions, ignore_index=True)
    return combined


def load_all_injuries(data_date: str = None) -> Dict[int, set]:
    """Load all muscular injury dates for all players."""
    if data_date is None:
        raw_data_dir = PRODUCTION_ROOT / "raw_data" / "england"
        date_folders = [d for d in raw_data_dir.iterdir() 
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
        if not date_folders:
            raise FileNotFoundError("No raw data folders found")
        latest_folder = max(date_folders, key=lambda x: x.name)
        injuries_file = latest_folder / "injuries_data.csv"
    else:
        injuries_file = PRODUCTION_ROOT / "raw_data" / "england" / data_date / "injuries_data.csv"
    
    if not injuries_file.exists():
        raise FileNotFoundError(f"Injuries file not found: {injuries_file}")
    
    injuries_df = pd.read_csv(injuries_file, sep=';', encoding='utf-8-sig', low_memory=False)
    
    # Parse dates
    if 'fromDate' in injuries_df.columns:
        injuries_df['fromDate_parsed'] = pd.to_datetime(
            injuries_df['fromDate'], 
            format='%d/%m/%Y', 
            errors='coerce'
        )
        valid_count = injuries_df['fromDate_parsed'].notna().sum()
        
        if valid_count < len(injuries_df) * 0.9:
            injuries_df['fromDate_parsed2'] = pd.to_datetime(
                injuries_df['fromDate'], 
                errors='coerce'
            )
            valid_count2 = injuries_df['fromDate_parsed2'].notna().sum()
            if valid_count2 > valid_count:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed2']
            else:
                injuries_df['fromDate'] = injuries_df['fromDate_parsed']
        else:
            injuries_df['fromDate'] = injuries_df['fromDate_parsed']
    
    # Derive injury_class if it doesn't exist
    if 'injury_class' not in injuries_df.columns:
        from production.scripts.update_timelines import derive_injury_class
        injuries_df['injury_class'] = injuries_df.apply(
            lambda row: derive_injury_class(
                row.get('injury_type', ''),
                row.get('no_physio_injury', None)
            ),
            axis=1
        )
    
    # Filter to only muscular injuries
    injuries_df['injury_class_lower'] = injuries_df['injury_class'].astype(str).str.lower()
    muscular_injuries = injuries_df[injuries_df['injury_class_lower'] == 'muscular'].copy()
    
    # Create mapping: player_id -> set of injury dates
    injury_dates = defaultdict(set)
    for _, row in muscular_injuries.iterrows():
        player_id = row.get('player_id')
        from_date = row.get('fromDate')
        
        if pd.notna(player_id) and pd.notna(from_date):
            injury_dates[int(player_id)].add(pd.Timestamp(from_date).normalize())
    
    return dict(injury_dates)


def check_injury_in_window(
    player_id: int,
    prediction_date: pd.Timestamp,
    injury_dates: Dict[int, set],
    window_days: int = 5
) -> Tuple[bool, pd.Timestamp]:
    """Check if player was injured within the next window_days after prediction_date."""
    if player_id not in injury_dates:
        return False, None
    
    window_end = prediction_date + timedelta(days=window_days)
    
    for injury_date in injury_dates[player_id]:
        if prediction_date < injury_date <= window_end:
            return True, injury_date
    
    return False, None


def classify_predictions(
    predictions_df: pd.DataFrame,
    injury_dates: Dict[int, set],
    window_days: int = 5,
    threshold: float = 0.3
) -> pd.DataFrame:
    """Classify predictions into: False Negatives, True Positives, True Negatives, False Positives."""
    
    results = []
    for _, row in predictions_df.iterrows():
        player_id = row['player_id']
        pred_date = row['reference_date']
        prob = row['injury_probability']
        
        injured, injury_date = check_injury_in_window(
            player_id, pred_date, injury_dates, window_days
        )
        
        # Classify
        if prob < threshold:
            if injured:
                category = "False Negative"
            else:
                category = "True Negative"
        else:
            if injured:
                category = "True Positive"
            else:
                category = "False Positive"
        
        result = row.to_dict()
        result['category'] = category
        result['injured'] = injured
        result['injury_date'] = injury_date if injured else None
        result['days_to_injury'] = (injury_date - pred_date).days if injured else None
        
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_top_features(df: pd.DataFrame, category: str, top_n: int = 20) -> pd.DataFrame:
    """Analyze top features for a specific category."""
    category_df = df[df['category'] == category].copy()
    
    if len(category_df) == 0:
        return pd.DataFrame()
    
    # Collect all top features
    feature_shap_pairs = []
    for idx, row in category_df.iterrows():
        for i in range(1, 11):
            feat_name = row.get(f'top_feature_{i}_name', '')
            feat_shap = row.get(f'top_feature_{i}_shap', 0.0)
            feat_value = row.get(f'top_feature_{i}_value', None)
            if feat_name:
                feature_shap_pairs.append({
                    'feature': feat_name,
                    'shap_value': abs(feat_shap),  # Use absolute value
                    'shap_raw': feat_shap,
                    'feature_value': feat_value
                })
    
    if not feature_shap_pairs:
        return pd.DataFrame()
    
    features_df = pd.DataFrame(feature_shap_pairs)
    
    # Aggregate by feature
    feature_stats = features_df.groupby('feature').agg({
        'shap_value': ['mean', 'std', 'count'],
        'shap_raw': 'mean',
        'feature_value': lambda x: np.mean([v for v in x if pd.notna(v) and v is not None]) if any(pd.notna(v) and v is not None for v in x) else None
    }).reset_index()
    
    feature_stats.columns = ['feature', 'mean_abs_shap', 'std_abs_shap', 'count', 'mean_shap', 'mean_value']
    feature_stats = feature_stats.sort_values('mean_abs_shap', ascending=False)
    
    return feature_stats.head(top_n)


def generate_analysis_report(
    predictions_df: pd.DataFrame,
    output_path: Path
):
    """Generate comprehensive analysis report."""
    
    print("\n" + "=" * 80)
    print("FALSE NEGATIVE ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary statistics
    category_counts = predictions_df['category'].value_counts()
    print(f"\n[SUMMARY] Prediction Categories:")
    for cat, count in category_counts.items():
        pct = (count / len(predictions_df)) * 100
        print(f"  {cat}: {count:,} ({pct:.2f}%)")
    
    # Focus on False Negatives
    fn_df = predictions_df[predictions_df['category'] == 'False Negative'].copy()
    tp_df = predictions_df[predictions_df['category'] == 'True Positive'].copy()
    tn_df = predictions_df[predictions_df['category'] == 'True Negative'].copy()
    
    print(f"\n[FALSE NEGATIVES] Total: {len(fn_df)}")
    if len(fn_df) > 0:
        print(f"  Average predicted probability: {fn_df['injury_probability'].mean():.4f}")
        print(f"  Median predicted probability: {fn_df['injury_probability'].median():.4f}")
        print(f"  Min predicted probability: {fn_df['injury_probability'].min():.4f}")
        print(f"  Max predicted probability: {fn_df['injury_probability'].max():.4f}")
        print(f"  Average days to injury: {fn_df['days_to_injury'].mean():.2f}")
        print(f"  Median days to injury: {fn_df['days_to_injury'].median():.2f}")
        print(f"  Days to injury distribution:")
        days_dist = fn_df['days_to_injury'].value_counts().sort_index()
        for days, count in days_dist.items():
            print(f"    Day {days}: {count} cases")
    
    # Compare to True Positives
    if len(tp_df) > 0:
        print(f"\n[TRUE POSITIVES] Total: {len(tp_df)}")
        print(f"  Average predicted probability: {tp_df['injury_probability'].mean():.4f}")
        print(f"  Median predicted probability: {tp_df['injury_probability'].median():.4f}")
    
    # Top features analysis
    print(f"\n[TOP FEATURES - FALSE NEGATIVES]")
    fn_features = analyze_top_features(predictions_df, "False Negative", top_n=20)
    if not fn_features.empty:
        print(fn_features.to_string(index=False))
    else:
        print("  No features found")
    
    print(f"\n[TOP FEATURES - TRUE POSITIVES]")
    tp_features = analyze_top_features(predictions_df, "True Positive", top_n=20)
    if not tp_features.empty:
        print(tp_features.to_string(index=False))
    else:
        print("  No features found")
    
    # Feature comparison
    if not fn_features.empty and not tp_features.empty:
        print(f"\n[FEATURE COMPARISON]")
        fn_top10 = set(fn_features.head(10)['feature'].values)
        tp_top10 = set(tp_features.head(10)['feature'].values)
        
        common = fn_top10 & tp_top10
        fn_only = fn_top10 - tp_top10
        tp_only = tp_top10 - fn_top10
        
        print(f"  Common top features: {len(common)}")
        if common:
            print(f"    {', '.join(sorted(common))}")
        print(f"  FN-only top features: {len(fn_only)}")
        if fn_only:
            print(f"    {', '.join(sorted(fn_only))}")
        print(f"  TP-only top features: {len(tp_only)}")
        if tp_only:
            print(f"    {', '.join(sorted(tp_only))}")
    
    # Player/club distribution
    print(f"\n[FALSE NEGATIVES - Player Distribution]")
    if len(fn_df) > 0:
        fn_players = fn_df['player_id'].value_counts()
        print(f"  Unique players: {fn_players.nunique()}")
        print(f"  Players with multiple FN cases: {(fn_players > 1).sum()}")
        if (fn_players > 1).sum() > 0:
            print(f"  Top players with FN cases:")
            for player_id, count in fn_players.head(10).items():
                player_name = fn_df[fn_df['player_id'] == player_id]['player_name'].iloc[0] if 'player_name' in fn_df.columns else 'Unknown'
                print(f"    Player {player_id} ({player_name}): {count} cases")
    
    print(f"\n[FALSE NEGATIVES - Club Distribution]")
    if len(fn_df) > 0:
        fn_clubs = fn_df['club'].value_counts()
        print(fn_clubs.to_string())
    
    # Date distribution
    print(f"\n[FALSE NEGATIVES - Date Distribution]")
    if len(fn_df) > 0:
        fn_df['pred_month'] = pd.to_datetime(fn_df['reference_date']).dt.to_period('M')
        fn_by_month = fn_df['pred_month'].value_counts().sort_index()
        print(fn_by_month.to_string())
    
    # Save detailed false negatives
    fn_output = output_path.parent / f"{output_path.stem}_false_negatives.csv"
    fn_df.to_csv(fn_output, index=False, encoding='utf-8-sig')
    print(f"\n[SAVED] Detailed false negatives to: {fn_output}")
    
    # Save feature analysis
    if not fn_features.empty:
        features_output = output_path.parent / f"{output_path.stem}_fn_features.csv"
        fn_features.to_csv(features_output, index=False, encoding='utf-8-sig')
        print(f"[SAVED] FN feature analysis to: {features_output}")
    
    if not tp_features.empty:
        features_output = output_path.parent / f"{output_path.stem}_tp_features.csv"
        tp_features.to_csv(features_output, index=False, encoding='utf-8-sig')
        print(f"[SAVED] TP feature analysis to: {features_output}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze false negative cases (low probability but injury occurred)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-12-06',
        help='Start date for observations (YYYY-MM-DD, default: 2025-12-06)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for observations (YYYY-MM-DD, default: today - 6 days)'
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=5,
        help='Observation window in days (default: 5)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Probability threshold for low predictions (default: 0.3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: false_negative_analysis_YYYYMMDD.txt)'
    )
    parser.add_argument(
        '--data-date',
        type=str,
        default=None,
        help='Raw data date to use for injuries (YYYYMMDD, default: latest)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = pd.to_datetime(args.start_date)
    
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    else:
        end_date = pd.Timestamp.now().normalize() - timedelta(days=6)
    
    print("=" * 80)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 80)
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    print(f"Observation window: {args.window_days} days")
    print(f"Low probability threshold: {args.threshold}")
    print()
    
    # Get all clubs
    clubs = get_all_clubs("England")
    print(f"[INFO] Found {len(clubs)} Premier League clubs")
    
    # Load predictions
    print(f"\n[INFO] Loading V3 predictions from all clubs...")
    predictions_df = load_all_predictions(clubs, start_date, end_date)
    print(f"       Loaded {len(predictions_df):,} predictions")
    
    # Load injuries
    injury_dates = load_all_injuries(data_date=args.data_date)
    print(f"       Loaded {sum(len(dates) for dates in injury_dates.values()):,} muscular injury dates")
    
    # Classify predictions
    print(f"\n[INFO] Classifying predictions...")
    classified_df = classify_predictions(
        predictions_df,
        injury_dates,
        window_days=args.window_days,
        threshold=args.threshold
    )
    
    # Generate analysis
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PRODUCTION_ROOT / "deployments" / "England" / f"false_negative_analysis_{end_date.strftime('%Y%m%d')}.txt"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Redirect output to file while also printing
    import io
    import contextlib
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(TeeOutput(sys.stdout, f)):
            generate_analysis_report(classified_df, output_path)
    
    print(f"\n[INFO] Analysis report saved to: {output_path}")


if __name__ == "__main__":
    main()

