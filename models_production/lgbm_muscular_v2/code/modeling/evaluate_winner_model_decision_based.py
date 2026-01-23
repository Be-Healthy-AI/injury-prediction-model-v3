#!/usr/bin/env python3
"""
Decision-Based Evaluation Script for WINNER Model
Reframes injury prediction as a decision-support system using ranking and Top-K selection.

WINNER Model: LightGBM, 10% target ratio, baseline hyperparameters, 0.8 correlation threshold
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Import preprocessing functions from training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_models_seasonal_combined import (
    prepare_data, align_features, sanitize_feature_name, clean_categorical_value
)

# ========== CONFIGURATION ==========
# Model v2: LGBM trained on ALL 10% seasons (including 2025-2026)
WINNER_MODEL_PATH = 'models/lgbm_model_seasonal_10pc_v4_muscular_corr08_v2_allseasons.joblib'
WINNER_COLUMNS_PATH = 'models/lgbm_model_seasonal_10pc_v4_muscular_corr08_v2_allseasons_columns.json'

# Test set: NATURAL 2025-2026 timelines (no target-ratio sampling),
# using the canonical file from the v1 model bundle.
TEST_DATA_PATH = (
    'models_production/lgbm_muscular_v1/data/timelines/test/'
    'timelines_35day_season_2025_2026_v4_muscular.csv'
)

K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20]
PREDICTION_HORIZON_DAYS = 35
CALIBRATE_PROBABILITIES = False  # Optional, for interpretability only
OUTPUT_DIR = 'evaluation_results'
# ===================================


def load_winner_model(model_path, columns_path):
    """Load WINNER model and feature columns"""
    print(f"\n📦 Loading WINNER model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"📦 Loading feature columns from {columns_path}...")
    # Use UTF-8 with errors=replace to be robust to any non-ASCII characters
    with open(columns_path, 'r', encoding='utf-8', errors='replace') as f:
        model_columns = json.load(f)
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Feature columns: {len(model_columns)} features")
    
    return model, model_columns


def extract_team_week_info(df):
    """Extract team and week information from test dataset"""
    print("\n📊 Extracting team and week information...")
    
    # Convert reference_date to datetime if needed
    if df['reference_date'].dtype == 'object':
        df['reference_date'] = pd.to_datetime(df['reference_date'])
    
    # Extract team (current_club)
    df['team'] = df['current_club'].fillna('Unknown')
    
    # Extract week number (ISO week of year)
    df['week'] = df['reference_date'].dt.isocalendar().week
    df['year'] = df['reference_date'].dt.isocalendar().year
    
    # Create team-week identifier
    df['team_week'] = df['team'].astype(str) + '_' + df['year'].astype(str) + '_W' + df['week'].astype(str)
    
    # Check for missing data
    missing_team = df['team'].isna().sum() + (df['team'] == 'Unknown').sum()
    if missing_team > 0:
        print(f"   ⚠️  Found {missing_team} records with missing/unknown team")
    
    print(f"✅ Extracted team-week information:")
    print(f"   Unique teams: {df['team'].nunique()}")
    print(f"   Unique weeks: {df['week'].nunique()}")
    print(f"   Unique team-weeks: {df['team_week'].nunique()}")
    print(f"   Date range: {df['reference_date'].min()} to {df['reference_date'].max()}")
    
    return df


def generate_risk_scores(model, model_columns, df_test, df_test_original):
    """Generate risk scores for all test samples"""
    print("\n🔮 Generating risk scores...")
    
    # Prepare test data using same preprocessing as training
    print("   Preparing test data (preprocessing)...")
    X_test, y_test = prepare_data(df_test_original, cache_file=None, use_cache=False)
    
    # Align features with model
    print("   Aligning features with model...")
    # Get features that exist in both
    available_features = [col for col in model_columns if col in X_test.columns]
    missing_features = [col for col in model_columns if col not in X_test.columns]
    
    if missing_features:
        print(f"   ⚠️  Missing {len(missing_features)} features from model (will be set to 0)")
        # Add missing features with zeros (using concat for better performance)
        missing_df = pd.DataFrame(0, index=X_test.index, columns=missing_features)
        X_test = pd.concat([X_test, missing_df], axis=1)
    
    # Also check for extra features in test data that aren't in model
    extra_features = [col for col in X_test.columns if col not in model_columns]
    if extra_features:
        print(f"   ⚠️  Found {len(extra_features)} extra features in test data (will be dropped)")
    
    # Select only model features in correct order
    X_test_aligned = X_test[model_columns].copy()
    
    print(f"   ✅ Aligned features: {X_test_aligned.shape[1]} features")
    print(f"   ✅ Test samples: {X_test_aligned.shape[0]:,}")
    
    # Generate predictions
    print("   Generating risk scores...")
    risk_scores = model.predict_proba(X_test_aligned)[:, 1]

    # Classic test metrics on natural 2025-2026
    test_classic_metrics = compute_classic_metrics_from_scores(
        y_true=y_test.values,
        y_proba=risk_scores,
        threshold=0.5,
        dataset_name="Natural 2025-2026 (Test)",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classic_metrics_path = os.path.join(
        OUTPUT_DIR, "classic_metrics_v2_natural_2025_2026.json"
    )
    with open(classic_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_classic_metrics, f, indent=2)
    print(f"✅ Saved classic test metrics to {classic_metrics_path}")

    # Create results dataframe
    results_df = pd.DataFrame({
        'player_id': df_test_original['player_id'].values,
        'player_name': df_test_original['player_name'].values,
        'reference_date': df_test_original['reference_date'].values,
        'team': df_test['team'].values,
        'week': df_test['week'].values,
        'year': df_test['year'].values,
        'team_week': df_test['team_week'].values,
        'risk_score': risk_scores,
        'actual_injury': y_test.values
    })
    
    print(f"✅ Generated risk scores for {len(results_df):,} samples")
    print(f"   Risk score range: [{risk_scores.min():.4f}, {risk_scores.max():.4f}]")
    print(f"   Mean risk score: {risk_scores.mean():.4f}")
    print(f"   Actual injuries: {results_df['actual_injury'].sum():,} ({results_df['actual_injury'].mean():.1%})")
    
    return results_df


def compute_classic_metrics_from_scores(y_true, y_proba, threshold=0.5, dataset_name="Test"):
    """Compute accuracy, precision, recall, F1, ROC AUC, Gini, confusion matrix."""
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = 0.0
    metrics["roc_auc"] = roc_auc
    metrics["gini"] = 2 * roc_auc - 1

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

    print(f"\n📊 Classic metrics – {dataset_name}:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"   Gini: {metrics['gini']:.4f}")
    cmv = metrics.get("confusion_matrix")
    if cmv is not None:
        print(f"   TP: {cmv['tp']}, FP: {cmv['fp']}, TN: {cmv['tn']}, FN: {cmv['fn']}")

    return metrics


def rank_players_by_team_week(results_df):
    """Rank players within each team-week by risk score"""
    print("\n📊 Ranking players by team-week...")
    
    # Sort by team_week and risk_score (descending)
    results_df = results_df.sort_values(['team_week', 'risk_score'], ascending=[True, False])
    
    # Add rank within each team-week
    results_df['rank'] = results_df.groupby('team_week').cumcount() + 1
    
    # Count players per team-week
    team_week_counts = results_df.groupby('team_week').size()
    print(f"✅ Ranked players in {len(team_week_counts)} team-weeks")
    print(f"   Average players per team-week: {team_week_counts.mean():.1f}")
    print(f"   Min players: {team_week_counts.min()}, Max players: {team_week_counts.max()}")
    
    return results_df


def calculate_precision_at_k(results_df, k):
    """Calculate Precision@K for each team-week"""
    precision_values = []
    team_weeks_evaluated = 0
    
    for team_week, group in results_df.groupby('team_week'):
        # Select top K players
        top_k = group.head(k)
        
        # Count true injuries among top K
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
        'team_weeks_evaluated': team_weeks_evaluated
    }


def calculate_recall_at_k(results_df, k):
    """Calculate Recall@K (injury capture rate) for each team-week"""
    recall_values = []
    overall_captured = 0
    overall_total_injuries = 0
    
    for team_week, group in results_df.groupby('team_week'):
        total_injuries = group['actual_injury'].sum()
        
        if total_injuries > 0:
            # Select top K players
            top_k = group.head(k)
            captured_injuries = top_k['actual_injury'].sum()
            recall_k = captured_injuries / total_injuries
            
            recall_values.append(recall_k)
            overall_captured += captured_injuries
            overall_total_injuries += total_injuries
    
    overall_recall = overall_captured / overall_total_injuries if overall_total_injuries > 0 else 0.0
    
    return {
        'overall': overall_recall,
        'mean': np.mean(recall_values) if recall_values else 0.0,
        'median': np.median(recall_values) if recall_values else 0.0,
        'p25': np.percentile(recall_values, 25) if recall_values else 0.0,
        'p75': np.percentile(recall_values, 75) if recall_values else 0.0,
        'std': np.std(recall_values) if recall_values else 0.0,
        'values': recall_values,
        'total_captured': overall_captured,
        'total_injuries': overall_total_injuries
    }


def calculate_false_alerts_per_injury(results_df, k):
    """Calculate False Alerts per True Injury"""
    false_alerts_values = []
    total_alerts = 0
    total_tp = 0
    
    for team_week, group in results_df.groupby('team_week'):
        # Select top K players
        top_k = group.head(k)
        tp_k = top_k['actual_injury'].sum()
        fp_k = len(top_k) - tp_k
        
        if tp_k > 0:
            false_alerts_per_injury = fp_k / tp_k
            false_alerts_values.append(false_alerts_per_injury)
        
        total_alerts += len(top_k)
        total_tp += tp_k
    
    overall_false_alerts = (total_alerts - total_tp) / total_tp if total_tp > 0 else float('inf')
    
    return {
        'overall': overall_false_alerts,
        'mean': np.mean(false_alerts_values) if false_alerts_values else float('inf'),
        'median': np.median(false_alerts_values) if false_alerts_values else float('inf'),
        'p25': np.percentile(false_alerts_values, 25) if false_alerts_values else float('inf'),
        'p75': np.percentile(false_alerts_values, 75) if false_alerts_values else float('inf'),
        'std': np.std(false_alerts_values) if false_alerts_values else 0.0,
        'values': false_alerts_values,
        'total_alerts': total_alerts,
        'total_tp': total_tp
    }


def evaluate_all_k_values(results_df, k_values):
    """Evaluate all K values and return summary"""
    print(f"\n📊 Evaluating metrics for K values: {k_values}...")
    
    results_summary = []
    
    for k in tqdm(k_values, desc="   Evaluating K values"):
        precision_metrics = calculate_precision_at_k(results_df, k)
        recall_metrics = calculate_recall_at_k(results_df, k)
        false_alerts_metrics = calculate_false_alerts_per_injury(results_df, k)
        
        results_summary.append({
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
            'Total_Captured': recall_metrics['total_captured']
        })
    
    return pd.DataFrame(results_summary)


def generate_report(summary_df, results_df, output_dir):
    """Generate comprehensive report with tables and visualizations"""
    print(f"\n📝 Generating report in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary table
    summary_file = os.path.join(output_dir, 'decision_based_metrics_summary.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✅ Saved summary table to {summary_file}")
    
    # Create formatted summary table for display
    display_df = summary_df.copy()
    display_df['Precision@K'] = display_df['Precision@K_mean'].apply(lambda x: f"{x:.1%}")
    display_df['Recall@K'] = display_df['Recall@K_overall'].apply(lambda x: f"{x:.1%}")
    display_df['False Alerts/Injury'] = display_df['False_Alerts_per_Injury_overall'].apply(
        lambda x: f"{x:.1f}" if x != float('inf') else "∞"
    )
    
    display_columns = ['K', 'Precision@K', 'Recall@K', 'False Alerts/Injury', 'Team_Weeks_Evaluated']
    display_table = display_df[display_columns]
    
    # Save formatted table
    formatted_file = os.path.join(output_dir, 'decision_based_metrics_formatted.csv')
    display_table.to_csv(formatted_file, index=False, encoding='utf-8-sig')
    print(f"✅ Saved formatted table to {formatted_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("DECISION-BASED EVALUATION RESULTS - WINNER MODEL")
    print("="*80)
    print("\nSummary Table:")
    print(display_table.to_string(index=False))
    
    # Generate visualizations
    generate_visualizations(summary_df, output_dir)
    
    # Generate detailed report
    generate_detailed_report(summary_df, results_df, output_dir)
    
    return display_table


def generate_visualizations(summary_df, output_dir):
    """Generate visualization plots"""
    print("\n📊 Generating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Precision@K and Recall@K vs K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(summary_df['K'], summary_df['Precision@K_mean'] * 100, 'o-', linewidth=2, markersize=8, label='Mean Precision@K')
    ax1.fill_between(summary_df['K'], 
                     summary_df['Precision@K_p25'] * 100,
                     summary_df['Precision@K_p75'] * 100,
                     alpha=0.3, label='IQR')
    ax1.set_xlabel('K (Alert Budget)', fontsize=12)
    ax1.set_ylabel('Precision@K (%)', fontsize=12)
    ax1.set_title('Precision@K vs Alert Budget', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(summary_df['K'], summary_df['Recall@K_overall'] * 100, 'o-', linewidth=2, markersize=8, 
             color='green', label='Overall Recall@K')
    ax2.set_xlabel('K (Alert Budget)', fontsize=12)
    ax2.set_ylabel('Recall@K (%)', fontsize=12)
    ax2.set_title('Injury Capture Rate vs Alert Budget', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_vs_k.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved precision/recall plot")
    plt.close()
    
    # 2. False Alerts per Injury vs K
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out infinite values for plotting
    plot_df = summary_df[summary_df['False_Alerts_per_Injury_overall'] != float('inf')]
    
    ax.plot(plot_df['K'], plot_df['False_Alerts_per_Injury_overall'], 'o-', 
            linewidth=2, markersize=8, color='red', label='False Alerts per Injury')
    ax.set_xlabel('K (Alert Budget)', fontsize=12)
    ax.set_ylabel('False Alerts per True Injury', fontsize=12)
    ax.set_title('False Alerts per Injury vs Alert Budget', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'false_alerts_vs_k.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved false alerts plot")
    plt.close()
    
    # 3. Combined metrics plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(summary_df['K'], summary_df['Precision@K_mean'] * 100, 'o-', 
             linewidth=2, markersize=8, label='Precision@K (%)', color='blue')
    ax1.set_xlabel('K (Alert Budget)', fontsize=12)
    ax1.set_ylabel('Precision@K (%)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(summary_df['K'], summary_df['Recall@K_overall'] * 100, 's-', 
             linewidth=2, markersize=8, label='Recall@K (%)', color='green')
    ax2.set_ylabel('Recall@K (%)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax1.set_title('Precision@K and Recall@K vs Alert Budget', fontsize=14, fontweight='bold')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics_vs_k.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined metrics plot")
    plt.close()


def generate_detailed_report(summary_df, results_df, output_dir):
    """Generate detailed markdown report"""
    report_file = os.path.join(output_dir, 'decision_based_evaluation_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Decision-Based Evaluation Report - WINNER Model\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Model Information\n\n")
        f.write(f"- **Model:** LightGBM (WINNER)\n")
        f.write(f"- **Configuration:** 10% target ratio, baseline hyperparameters, 0.8 correlation threshold\n")
        f.write(f"- **Test Dataset:** 2025-2026 season (natural target ratio)\n")
        f.write(f"- **Prediction Horizon:** {PREDICTION_HORIZON_DAYS} days\n\n")
        
        f.write("## Evaluation Approach\n\n")
        f.write("This evaluation reframes injury prediction as a decision-support system:\n")
        f.write("- **Decision Unit:** Team × Week\n")
        f.write("- **Selection Method:** Top-K players by risk score (ranking-based)\n")
        f.write("- **Primary Metric:** Precision@K (accuracy of alerts)\n")
        f.write("- **Secondary Metric:** Recall@K (injury capture rate)\n")
        f.write("- **Operational Metric:** False Alerts per True Injury\n\n")
        
        f.write("## Summary Results\n\n")
        f.write("| K | Precision@K | Recall@K | False Alerts/Injury | Team-Weeks |\n")
        f.write("|---|-------------|----------|---------------------|------------|\n")
        
        for _, row in summary_df.iterrows():
            precision = f"{row['Precision@K_mean']:.1%}"
            recall = f"{row['Recall@K_overall']:.1%}"
            false_alerts = f"{row['False_Alerts_per_Injury_overall']:.1f}" if row['False_Alerts_per_Injury_overall'] != float('inf') else "∞"
            f.write(f"| {int(row['K'])} | {precision} | {recall} | {false_alerts} | {int(row['Team_Weeks_Evaluated'])} |\n")
        
        f.write("\n## Key Insights\n\n")
        
        # Find best K for precision
        best_precision_k = summary_df.loc[summary_df['Precision@K_mean'].idxmax()]
        f.write(f"- **Best Precision@K:** {best_precision_k['Precision@K_mean']:.1%} at K={int(best_precision_k['K'])}\n")
        
        # Find K that captures most injuries
        best_recall_k = summary_df.loc[summary_df['Recall@K_overall'].idxmax()]
        f.write(f"- **Best Recall@K:** {best_recall_k['Recall@K_overall']:.1%} at K={int(best_recall_k['K'])}\n")
        
        # Find K with lowest false alerts
        valid_false_alerts = summary_df[summary_df['False_Alerts_per_Injury_overall'] != float('inf')]
        if len(valid_false_alerts) > 0:
            best_false_alerts_k = valid_false_alerts.loc[valid_false_alerts['False_Alerts_per_Injury_overall'].idxmin()]
            f.write(f"- **Lowest False Alerts/Injury:** {best_false_alerts_k['False_Alerts_per_Injury_overall']:.1f} at K={int(best_false_alerts_k['K'])}\n")
        
        f.write("\n## Dataset Statistics\n\n")
        f.write(f"- **Total Test Samples:** {len(results_df):,}\n")
        f.write(f"- **Total Injuries:** {results_df['actual_injury'].sum():,}\n")
        f.write(f"- **Injury Rate:** {results_df['actual_injury'].mean():.1%}\n")
        f.write(f"- **Unique Teams:** {results_df['team'].nunique()}\n")
        f.write(f"- **Unique Weeks:** {results_df['week'].nunique()}\n")
        f.write(f"- **Unique Team-Weeks:** {results_df['team_week'].nunique()}\n")
    
    print(f"✅ Saved detailed report to {report_file}")


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("DECISION-BASED EVALUATION - WINNER MODEL")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"   Model: {WINNER_MODEL_PATH}")
    print(f"   Test Data: {TEST_DATA_PATH}")
    print(f"   K Values: {K_VALUES}")
    print(f"   Prediction Horizon: {PREDICTION_HORIZON_DAYS} days")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load WINNER model
    model, model_columns = load_winner_model(WINNER_MODEL_PATH, WINNER_COLUMNS_PATH)
    
    # Load test data
    print(f"\n📂 Loading test dataset: {TEST_DATA_PATH}...")
    df_test_original = pd.read_csv(TEST_DATA_PATH, encoding='utf-8-sig', low_memory=False)
    print(f"✅ Loaded test dataset: {len(df_test_original):,} records")
    print(f"   Injury ratio: {df_test_original['target'].mean():.1%}")
    
    # Extract team and week information
    df_test = extract_team_week_info(df_test_original.copy())
    
    # Generate risk scores
    results_df = generate_risk_scores(model, model_columns, df_test, df_test_original)
    
    # Rank players by team-week
    results_df = rank_players_by_team_week(results_df)
    
    # Evaluate all K values
    summary_df = evaluate_all_k_values(results_df, K_VALUES)
    
    # Generate report
    display_table = generate_report(summary_df, results_df, OUTPUT_DIR)
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    elapsed_time = datetime.now() - start_time
    print(f"⏱️  Total time: {elapsed_time}")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == '__main__':
    main()

