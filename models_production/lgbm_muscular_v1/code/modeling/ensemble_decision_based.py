#!/usr/bin/env python3
"""
Decision-Based Ensemble Evaluation Script

Combines:
- LightGBM (10% target ratio, corr=0.8, high Gini ~0.62)
- Random Forest (50% target ratio, corr=0.5, high Recall ~0.75)

Evaluation is purely decision-based (team × week, Top-K):
- Precision@K
- Recall@K (injury capture rate)
- False Alerts per True Injury
"""

import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

# Import preprocessing functions from training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_models_seasonal_combined import prepare_data  # type: ignore


# ========== CONFIGURATION ==========
LGBM_MODEL_PATH = 'models/lgbm_model_seasonal_10pc_v4_muscular_corr08.joblib'
LGBM_COLUMNS_PATH = 'models/lgbm_model_seasonal_10pc_v4_muscular_corr08_columns.json'

RF_MODEL_PATH = 'models/rf_model_seasonal_50pc_v4_muscular_corr05.joblib'
RF_COLUMNS_PATH = 'models/rf_model_seasonal_50pc_v4_muscular_corr05_columns.json'

TEST_DATA_PATH = 'timelines_35day_season_2025_2026_v4_muscular.csv'

K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20]
OUTPUT_DIR = 'ensemble_decision_results'

# Known single-model Ginis (from baseline experiments) for weighting
LGBM_GINI = 0.6198
RF_GINI = 0.5771
# ===================================


def load_model_and_columns(model_path, columns_path, name):
    """Load a model and its feature column list."""
    print(f"\n📦 Loading {name}...")
    model = joblib.load(model_path)
    with open(columns_path, 'r', encoding='utf-8') as f:
        columns = json.load(f)
    print(f"   ✅ Loaded {name}: {type(model).__name__}, {len(columns)} features")
    return model, columns


def extract_team_week_info(df):
    """Extract team and week information from test dataset."""
    print("\n📊 Extracting team and week information...")

    # Ensure datetime
    if df['reference_date'].dtype == 'object':
        df['reference_date'] = pd.to_datetime(df['reference_date'])

    df['team'] = df['current_club'].fillna('Unknown')
    df['week'] = df['reference_date'].dt.isocalendar().week
    df['year'] = df['reference_date'].dt.isocalendar().year
    df['team_week'] = (
        df['team'].astype(str)
        + '_'
        + df['year'].astype(str)
        + '_W'
        + df['week'].astype(str)
    )

    print(f"✅ Extracted team-week information:")
    print(f"   Unique teams: {df['team'].nunique()}")
    print(f"   Unique weeks: {df['week'].nunique()}")
    print(f"   Unique team-weeks: {df['team_week'].nunique()}")
    print(f"   Date range: {df['reference_date'].min()} to {df['reference_date'].max()}")

    return df


def align_to_model_columns(X_base, model_columns, name):
    """Align a preprocessed feature matrix to a model's expected columns."""
    print(f"\n🔧 Aligning features for {name}...")

    X = X_base.copy()

    # Add missing features as zeros
    missing_features = [c for c in model_columns if c not in X.columns]
    if missing_features:
        print(f"   ⚠️  Missing {len(missing_features)} features (setting to 0)")
        missing_df = pd.DataFrame(0, index=X.index, columns=missing_features)
        X = pd.concat([X, missing_df], axis=1)

    # Drop any extra features not used by the model
    extra_features = [c for c in X.columns if c not in model_columns]
    if extra_features:
        print(f"   ⚠️  Dropping {len(extra_features)} extra features not used by {name}")

    X_aligned = X[model_columns].copy()

    print(f"   ✅ Aligned to {X_aligned.shape[1]} features, {X_aligned.shape[0]:,} samples")
    return X_aligned


def generate_predictions(model, X_aligned, name):
    """Generate probability predictions from a model."""
    print(f"\n🔮 Generating predictions for {name}...")
    probs = model.predict_proba(X_aligned)[:, 1]
    print(f"   ✅ {name}: {len(probs):,} predictions")
    print(f"   Risk score range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"   Mean risk score: {probs.mean():.4f}")
    return probs


# ---------- Decision-based metrics (copied/adapted from winner evaluation) ----------

def calculate_precision_at_k(results_df, k):
    precision_values = []
    team_weeks_evaluated = 0

    for _, group in results_df.groupby('team_week'):
        top_k = group.head(k)
        tp_k = top_k['actual_injury'].sum()
        precision_k = tp_k / len(top_k) if len(top_k) > 0 else 0.0
        precision_values.append(precision_k)
        team_weeks_evaluated += 1

    return {
        'mean': np.mean(precision_values) if precision_values else 0.0,
        'median': np.median(precision_values) if precision_values else 0.0,
        'p25': np.percentile(precision_values, 25) if precision_values else 0.0,
        'p75': np.percentile(precision_values, 75) if precision_values else 0.0,
        'std': np.std(precision_values) if precision_values else 0.0,
        'values': precision_values,
        'team_weeks_evaluated': team_weeks_evaluated,
    }


def calculate_recall_at_k(results_df, k):
    recall_values = []
    overall_captured = 0
    overall_total_injuries = 0

    for _, group in results_df.groupby('team_week'):
        total_injuries = group['actual_injury'].sum()
        if total_injuries > 0:
            top_k = group.head(k)
            captured_injuries = top_k['actual_injury'].sum()
            recall_k = captured_injuries / total_injuries
            recall_values.append(recall_k)
            overall_captured += captured_injuries
            overall_total_injuries += total_injuries

    overall_recall = (
        overall_captured / overall_total_injuries if overall_total_injuries > 0 else 0.0
    )

    return {
        'overall': overall_recall,
        'mean': np.mean(recall_values) if recall_values else 0.0,
        'median': np.median(recall_values) if recall_values else 0.0,
        'p25': np.percentile(recall_values, 25) if recall_values else 0.0,
        'p75': np.percentile(recall_values, 75) if recall_values else 0.0,
        'std': np.std(recall_values) if recall_values else 0.0,
        'values': recall_values,
        'total_captured': overall_captured,
        'total_injuries': overall_total_injuries,
    }


def calculate_false_alerts_per_injury(results_df, k):
    false_alerts_values = []
    total_alerts = 0
    total_tp = 0

    for _, group in results_df.groupby('team_week'):
        top_k = group.head(k)
        tp_k = top_k['actual_injury'].sum()
        fp_k = len(top_k) - tp_k

        if tp_k > 0:
            false_alerts_per_injury = fp_k / tp_k
            false_alerts_values.append(false_alerts_per_injury)

        total_alerts += len(top_k)
        total_tp += tp_k

    overall_false_alerts = (
        (total_alerts - total_tp) / total_tp if total_tp > 0 else float('inf')
    )

    return {
        'overall': overall_false_alerts,
        'mean': np.mean(false_alerts_values) if false_alerts_values else float('inf'),
        'median': np.median(false_alerts_values) if false_alerts_values else float('inf'),
        'p25': np.percentile(false_alerts_values, 25) if false_alerts_values else float('inf'),
        'p75': np.percentile(false_alerts_values, 75) if false_alerts_values else float('inf'),
        'std': np.std(false_alerts_values) if false_alerts_values else 0.0,
        'values': false_alerts_values,
        'total_alerts': total_alerts,
        'total_tp': total_tp,
    }


def evaluate_all_k_values(results_df, k_values, label):
    """Evaluate all K values and return summary dataframe with a model/ensemble label."""
    print(f"\n📊 Evaluating metrics for {label} at K values: {k_values}...")
    rows = []
    for k in tqdm(k_values, desc=f"   {label} - K loop"):
        precision_metrics = calculate_precision_at_k(results_df, k)
        recall_metrics = calculate_recall_at_k(results_df, k)
        false_alerts_metrics = calculate_false_alerts_per_injury(results_df, k)

        rows.append(
            {
                'Model': label,
                'K': k,
                'Precision@K_mean': precision_metrics['mean'],
                'Precision@K_median': precision_metrics['median'],
                'Precision@K_p25': precision_metrics['p25'],
                'Precision@K_p75': precision_metrics['p75'],
                'Recall@K_overall': recall_metrics['overall'],
                'Recall@K_mean': recall_metrics['mean'],
                'False_Alerts_per_Injury_overall': false_alerts_metrics['overall'],
                'False_Alerts_per_Injury_mean': false_alerts_metrics['mean'],
                'Team_Weeks_Evaluated': precision_metrics['team_weeks_evaluated'],
                'Total_Injuries': recall_metrics['total_injuries'],
                'Total_Captured': recall_metrics['total_captured'],
            }
        )
    return pd.DataFrame(rows)


# ---------- Ensemble combinations ----------

def ensemble_equal(p1, p2):
    return 0.5 * p1 + 0.5 * p2


def ensemble_gini_weighted(p1, p2):
    total = LGBM_GINI + RF_GINI
    w1 = LGBM_GINI / total
    w2 = RF_GINI / total
    return w1 * p1 + w2 * p2, w1, w2


def ensemble_rank_average(p1, p2):
    r1 = rankdata(p1, method='average')
    r2 = rankdata(p2, method='average')
    avg_rank = (r1 + r2) / 2.0
    n = len(p1)
    # Convert ranks back to [0,1] scores (higher rank -> higher score)
    return (n - avg_rank + 1) / n


def ensemble_geometric_mean(p1, p2):
    eps = 1e-10
    p1_safe = np.clip(p1, eps, 1 - eps)
    p2_safe = np.clip(p2, eps, 1 - eps)
    return np.sqrt(p1_safe * p2_safe)


def build_results_df(df_test_with_team, probs, y_true, label):
    """Build results dataframe for decision-based evaluation."""
    df = pd.DataFrame(
        {
            'player_id': df_test_with_team['player_id'].values,
            'player_name': df_test_with_team['player_name'].values,
            'reference_date': df_test_with_team['reference_date'].values,
            'team': df_test_with_team['team'].values,
            'week': df_test_with_team['week'].values,
            'year': df_test_with_team['year'].values,
            'team_week': df_test_with_team['team_week'].values,
            'risk_score': probs,
            'actual_injury': y_true,
        }
    )
    print(
        f"\n✅ Built results dataframe for {label}: "
        f"{len(df):,} rows, injuries={df['actual_injury'].sum():,} "
        f"({df['actual_injury'].mean():.2%})"
    )
    return df


def main():
    print("=" * 80)
    print("DECISION-BASED ENSEMBLE EVALUATION")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test data
    print(f"\n📂 Loading test dataset: {TEST_DATA_PATH}...")
    df_test = pd.read_csv(TEST_DATA_PATH, encoding='utf-8-sig', low_memory=False)
    print(f"✅ Loaded test dataset: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.2%}")

    # Extract team-week context
    df_test_with_team = extract_team_week_info(df_test.copy())

    # Preprocess features once (shared base)
    print("\n📊 Preprocessing test data (shared base features)...")
    X_base, y_test = prepare_data(df_test, cache_file=None, use_cache=False)
    print(f"   ✅ Base features: {X_base.shape[1]} columns, {X_base.shape[0]:,} samples")

    # Load models
    lgbm_model, lgbm_cols = load_model_and_columns(
        LGBM_MODEL_PATH, LGBM_COLUMNS_PATH, "LightGBM (10%, 0.8 corr)"
    )
    rf_model, rf_cols = load_model_and_columns(
        RF_MODEL_PATH, RF_COLUMNS_PATH, "Random Forest (50%, 0.5 corr)"
    )

    # Align test features for each model
    X_lgbm = align_to_model_columns(X_base, lgbm_cols, "LightGBM")
    X_rf = align_to_model_columns(X_base, rf_cols, "Random Forest")

    # Generate predictions
    p_lgbm = generate_predictions(lgbm_model, X_lgbm, "LightGBM")
    p_rf = generate_predictions(rf_model, X_rf, "Random Forest")

    # Ensembles
    p_equal = ensemble_equal(p_lgbm, p_rf)
    p_gini, w_lgbm, w_rf = ensemble_gini_weighted(p_lgbm, p_rf)
    p_rank = ensemble_rank_average(p_lgbm, p_rf)
    p_geo = ensemble_geometric_mean(p_lgbm, p_rf)

    print(
        f"\n📊 Ensemble weights (Gini-weighted): "
        f"LGBM={w_lgbm:.3f}, RF={w_rf:.3f}"
    )

    # Build result dataframes
    results = {}
    results['LGBM'] = build_results_df(df_test_with_team, p_lgbm, y_test, "LGBM")
    results['RF'] = build_results_df(df_test_with_team, p_rf, y_test, "RF")
    results['Ensemble_Equal'] = build_results_df(
        df_test_with_team, p_equal, y_test, "Ensemble_Equal"
    )
    results['Ensemble_Gini'] = build_results_df(
        df_test_with_team, p_gini, y_test, "Ensemble_Gini"
    )
    results['Ensemble_Rank'] = build_results_df(
        df_test_with_team, p_rank, y_test, "Ensemble_Rank"
    )
    results['Ensemble_Geo'] = build_results_df(
        df_test_with_team, p_geo, y_test, "Ensemble_Geo"
    )

    # Evaluate all models/ensembles
    all_summaries = []
    for label, df_res in results.items():
        summary_df = evaluate_all_k_values(df_res, K_VALUES, label)
        all_summaries.append(summary_df)

    summary_all = pd.concat(all_summaries, ignore_index=True)

    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'ensemble_decision_metrics_summary.csv')
    summary_all.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved combined summary to {summary_file}")

    # Also create a human-friendly pivoted view for quick inspection
    display_rows = []
    for model_name in summary_all['Model'].unique():
        sub = summary_all[summary_all['Model'] == model_name].copy()
        for _, row in sub.iterrows():
            display_rows.append(
                {
                    'Model': model_name,
                    'K': int(row['K']),
                    'Precision@K': f"{row['Precision@K_mean']:.1%}",
                    'Recall@K': f"{row['Recall@K_overall']:.1%}",
                    'False Alerts/Injury': (
                        f"{row['False_Alerts_per_Injury_overall']:.1f}"
                        if row['False_Alerts_per_Injury_overall'] != float('inf')
                        else "∞"
                    ),
                }
            )
    display_df = pd.DataFrame(display_rows)
    display_file = os.path.join(OUTPUT_DIR, 'ensemble_decision_metrics_formatted.csv')
    display_df.to_csv(display_file, index=False, encoding='utf-8-sig')
    print(f"✅ Saved formatted comparison table to {display_file}")

    # Print compact tables per model
    print("\n" + "=" * 80)
    print("DECISION-BASED ENSEMBLE COMPARISON (Precision@K, Recall@K, False Alerts/Injury)")
    print("=" * 80)
    for model_name in ['LGBM', 'RF', 'Ensemble_Equal', 'Ensemble_Gini', 'Ensemble_Rank', 'Ensemble_Geo']:
        sub = display_df[display_df['Model'] == model_name].copy()
        if sub.empty:
            continue
        print(f"\n--- {model_name} ---")
        print(sub[['K', 'Precision@K', 'Recall@K', 'False Alerts/Injury']].to_string(index=False))

    print("\n✅ Evaluation complete.")
    print(f"📁 Results directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()


