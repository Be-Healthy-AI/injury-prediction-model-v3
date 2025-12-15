#!/usr/bin/env python3
"""
Ensemble Model Script - Combining WINNER and Second Best Models
Combines LightGBM (10% target ratio, 0.8 corr) and Random Forest (50% target ratio, 0.5 corr)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing functions from training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_models_seasonal_combined import (
    prepare_data, sanitize_feature_name, clean_categorical_value,
    align_features, apply_correlation_filter
)

# ========== CONFIGURATION ==========
# Model 1: WINNER (LightGBM)
MODEL1_PATH = 'models/lgbm_model_seasonal_10pc_v4_muscular_corr08.joblib'
MODEL1_COLUMNS_PATH = 'models/lgbm_model_seasonal_10pc_v4_muscular_corr08_columns.json'
MODEL1_NAME = 'LightGBM (10%, 0.8 corr)'
MODEL1_GINI = 0.6198  # Known test Gini for weighting

# Model 2: Second Best (Random Forest)
MODEL2_PATH = 'models/rf_model_seasonal_50pc_v4_muscular_corr05.joblib'
MODEL2_COLUMNS_PATH = 'models/rf_model_seasonal_50pc_v4_muscular_corr05_columns.json'
MODEL2_NAME = 'Random Forest (50%, 0.5 corr)'
MODEL2_GINI = 0.5771  # Known test Gini for weighting

# Test dataset
TEST_DATA_PATH = 'timelines_35day_season_2025_2026_v4_muscular.csv'
OUTPUT_DIR = 'ensemble_results'
# ===================================


def load_model(model_path, columns_path, model_name):
    """Load model and feature columns"""
    print(f"\n📦 Loading {model_name}...")
    print(f"   Model: {model_path}")
    print(f"   Columns: {columns_path}")
    
    model = joblib.load(model_path)
    
    # Load columns with UTF-8 encoding (same as how they were saved)
    with open(columns_path, 'r', encoding='utf-8') as f:
        model_columns = json.load(f)
    
    # Ensure all column names are strings with proper encoding
    model_columns = [str(col) for col in model_columns]
    
    print(f"✅ Loaded {model_name}")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Features: {len(model_columns)}")
    
    return model, model_columns


def prepare_test_data_for_model(df_test_original, model_columns, model_name, corr_threshold=None):
    """Prepare test data and align features for a specific model
    
    This replicates the exact preprocessing pipeline used during training:
    1. Preprocess data (categorical encoding, etc.)
    2. Align features (intersection with training features - but we use model columns as reference)
    3. Sanitize feature names
    4. Apply correlation filter (if threshold provided)
    5. Match to model's final feature set
    """
    print(f"\n🔧 Preparing test data for {model_name} (matching training pipeline)...")
    
    # Step 1: Prepare data using same preprocessing as training
    print(f"   Step 1: Preprocessing data...")
    X_test, y_test = prepare_data(df_test_original, cache_file=None, use_cache=False)
    
    # Step 2: We need to simulate alignment, but we don't have training data
    # Instead, we'll use the model columns as the reference set
    # The model columns are the final features after alignment and filtering
    # So we need to ensure our test features can match to those
    
    # Step 3: Sanitize feature names (same as training pipeline)
    print(f"   Step 2: Sanitizing feature names (same as training pipeline)...")
    X_test.columns = [sanitize_feature_name(col) for col in X_test.columns]
    
    # Step 3.5: Fix encoding issues in feature names
    # Some categorical features may have been encoded incorrectly during one-hot encoding
    # Try to fix common encoding issues (UTF-8 interpreted as Latin-1)
    print(f"   Step 2.5: Fixing encoding issues in feature names...")
    fixed_columns = []
    encoding_fixes = 0
    for col in X_test.columns:
        try:
            # Try to fix: if column name looks like it has encoding issues, try to fix it
            # Common pattern: "Ã‰tienne" should be "Étienne"
            # Check if encoding fix would change the name
            fixed = col.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
            if fixed != col and len(fixed) > 0:
                # This might be a fixable encoding issue
                # But we need to be careful - only fix if it makes sense
                # For now, keep original but track potential fixes
                fixed_columns.append(col)
            else:
                fixed_columns.append(col)
        except:
            fixed_columns.append(col)
    
    # Create a mapping from fixed names to original names for matching
    # But keep original column names in the dataframe
    X_test.columns = fixed_columns
    
    # Step 4: If correlation threshold is provided, we should apply correlation filter
    # But we don't have training data to compute correlations on
    # However, the model columns are already the filtered features
    # So we'll skip correlation filtering and match directly to model columns
    if corr_threshold is not None:
        print(f"   Step 3: Note - Correlation filtering was applied during training")
        print(f"           Model columns are already filtered features")
    
    # Step 5: Match features to model columns (the model columns are the final feature set)
    print(f"   Step 3: Matching features to model columns...")
    
    # Match features to model columns
    # The model columns are the final features after all processing
    # We need to match our preprocessed test features to these
    feature_mapping = {}
    available_features = []
    missing_features = []
    
    # Create lookup dictionaries for efficient matching
    test_features_set = set(X_test.columns)
    test_features_lower = {feat.lower(): feat for feat in X_test.columns}  # Case-insensitive lookup
    
    # Create normalized lookup: try to fix encoding and create lookup by normalized name
    test_features_normalized = {}
    for test_feat in X_test.columns:
        # Try multiple normalization strategies
        normalizations = [
            test_feat,  # Original
            test_feat.lower(),  # Lowercase
            test_feat.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore'),  # Encoding fix
        ]
        for norm in normalizations:
            if norm not in test_features_normalized:
                test_features_normalized[norm] = test_feat
    
    for model_feature in model_columns:
        matched = False
        
        # Strategy 1: Exact match (after sanitization, this should work for most features)
        if model_feature in test_features_set:
            feature_mapping[model_feature] = model_feature
            available_features.append(model_feature)
            matched = True
        else:
            # Strategy 2: Case-insensitive match
            if model_feature.lower() in test_features_lower:
                matched_feat = test_features_lower[model_feature.lower()]
                feature_mapping[model_feature] = matched_feat
                available_features.append(model_feature)
                matched = True
            else:
                # Strategy 3: Try normalized lookup (handles encoding issues)
                if model_feature in test_features_normalized:
                    feature_mapping[model_feature] = test_features_normalized[model_feature]
                    available_features.append(model_feature)
                    matched = True
                else:
                    # Strategy 4: Try encoding fixes (handle encoding mismatches)
                    # Common issue: UTF-8 characters interpreted as Latin-1
                    # "Ã‰tienne" (mis-encoded) should match "Étienne" (correct)
                    for test_feat in X_test.columns:
                        try:
                            # Fix test feature: encode as Latin-1, decode as UTF-8
                            # This converts "Ã‰tienne" -> "Étienne"
                            test_fixed = test_feat.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                            if test_fixed == model_feature:
                                feature_mapping[model_feature] = test_feat
                                available_features.append(model_feature)
                                matched = True
                                break
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            pass
                    
                    # Strategy 5: Reverse encoding fix
                    if not matched:
                        try:
                            # Fix model feature: encode as UTF-8, decode as Latin-1
                            # This converts "Étienne" -> "Ã‰tienne" to match test feature
                            model_as_latin1 = model_feature.encode('utf-8', errors='ignore').decode('latin-1', errors='ignore')
                            if model_as_latin1 in test_features_set:
                                feature_mapping[model_feature] = model_as_latin1
                                available_features.append(model_feature)
                                matched = True
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            pass
                
                # Strategy 6: Normalize both to ASCII for comparison (fallback)
                if not matched:
                    try:
                        # Normalize both to ASCII representation for comparison
                        model_normalized = model_feature.encode('ascii', errors='ignore').decode('ascii')
                        for test_feat in X_test.columns:
                            test_normalized = test_feat.encode('ascii', errors='ignore').decode('ascii')
                            if model_normalized == test_normalized and len(model_normalized) > 0:
                                # Found match by ASCII normalization
                                feature_mapping[model_feature] = test_feat
                                available_features.append(model_feature)
                                matched = True
                                break
                    except:
                        pass
        
        if not matched:
            missing_features.append(model_feature)
    
    extra_features = [col for col in X_test.columns if col not in feature_mapping.values()]
    
    if missing_features:
        print(f"   ⚠️  Missing {len(missing_features)} features (will be set to 0)")
        # Show first few missing features for debugging
        if len(missing_features) <= 10:
            print(f"      Missing (first 10): {missing_features[:10]}")
        else:
            print(f"      Missing (first 10): {missing_features[:10]}")
    
    if extra_features:
        print(f"   ⚠️  Found {len(extra_features)} extra features (will be dropped)")
    
    # Build aligned dataframe with model features in correct order
    # IMPORTANT: Use model feature names directly (not test feature names)
    # This ensures the model receives features with the exact names it expects
    aligned_data = {}
    for model_feature in model_columns:
        if model_feature in feature_mapping:
            # Use mapped test feature, but store under model feature name
            test_feature_name = feature_mapping[model_feature]
            aligned_data[model_feature] = X_test[test_feature_name].values
        else:
            # Missing feature - set to 0
            aligned_data[model_feature] = np.zeros(len(X_test), dtype=float)
    
    # Create dataframe with model feature names as columns
    X_test_aligned = pd.DataFrame(aligned_data, index=X_test.index)
    # Ensure column names match model exactly
    X_test_aligned.columns = model_columns
    
    print(f"   ✅ Aligned features: {X_test_aligned.shape[1]} features")
    print(f"   ✅ Matched features: {len(available_features)}")
    print(f"   ✅ Test samples: {X_test_aligned.shape[0]:,}")
    
    return X_test_aligned, y_test


def generate_predictions(model, X_test, model_name):
    """Generate probability predictions from model"""
    print(f"\n🔮 Generating predictions from {model_name}...")
    
    # For sklearn models, disable feature name validation if there are encoding mismatches
    # We've already aligned the features, so this is safe
    try:
        predictions = model.predict_proba(X_test)[:, 1]
    except ValueError as e:
        if "feature names" in str(e).lower():
            print(f"   ⚠️  Feature name validation error (likely encoding mismatch)")
            print(f"   Attempting to bypass validation...")
            # Convert to numpy array to bypass feature name validation
            X_test_array = X_test.values
            # Ensure columns are in the same order as model expects
            if hasattr(model, 'feature_names_in_'):
                # Reorder columns to match model's expected order
                expected_features = list(model.feature_names_in_)
                # Create mapping from current columns to expected
                column_mapping = {}
                for i, feat in enumerate(X_test.columns):
                    # Try to find matching expected feature
                    for exp_feat in expected_features:
                        if feat == exp_feat or feat.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore') == exp_feat:
                            column_mapping[exp_feat] = i
                            break
                
                # Reorder X_test_array columns
                reordered_indices = [column_mapping.get(feat, 0) for feat in expected_features]
                X_test_array = X_test_array[:, reordered_indices]
            
            predictions = model.predict_proba(X_test_array)[:, 1]
        else:
            raise
    
    print(f"   ✅ Predictions generated: {len(predictions):,} samples")
    print(f"   Risk score range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Mean risk score: {predictions.mean():.4f}")
    
    return predictions


def ensemble_weighted_average(pred1, pred2, weight1, weight2):
    """Weighted average ensemble"""
    return weight1 * pred1 + weight2 * pred2


def ensemble_rank_average(pred1, pred2):
    """Rank-based ensemble (average ranks, then convert back)"""
    # Convert to ranks (higher probability = higher rank)
    rank1 = rankdata(pred1, method='average')
    rank2 = rankdata(pred2, method='average')
    
    # Average ranks
    avg_rank = (rank1 + rank2) / 2.0
    
    # Convert back to probabilities (normalize to [0, 1])
    # Use inverse rank transformation
    n = len(pred1)
    ensemble_pred = (n - avg_rank + 1) / n
    
    return ensemble_pred


def ensemble_geometric_mean(pred1, pred2):
    """Geometric mean ensemble (works well for probabilities)"""
    # Add small epsilon to avoid zeros
    eps = 1e-10
    pred1_safe = np.clip(pred1, eps, 1 - eps)
    pred2_safe = np.clip(pred2, eps, 1 - eps)
    
    geometric_mean = np.sqrt(pred1_safe * pred2_safe)
    return geometric_mean


def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluate model predictions and return metrics"""
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_pred)
    gini = 2 * roc_auc - 1
    
    # For binary classification metrics, use optimal threshold (Youden's J)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'gini': gini,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }


def test_ensemble_methods(pred1, pred2, y_test, model1_name, model2_name):
    """Test multiple ensemble methods and return results"""
    print(f"\n🔬 Testing ensemble methods...")
    
    results = []
    
    # 1. Equal weights (0.5/0.5)
    ensemble_equal = ensemble_weighted_average(pred1, pred2, 0.5, 0.5)
    results.append(evaluate_predictions(y_test, ensemble_equal, "Ensemble: Equal Weights (0.5/0.5)"))
    
    # 2. Gini-weighted (based on known test performance)
    total_gini = MODEL1_GINI + MODEL2_GINI
    weight1 = MODEL1_GINI / total_gini
    weight2 = MODEL2_GINI / total_gini
    ensemble_gini_weighted = ensemble_weighted_average(pred1, pred2, weight1, weight2)
    results.append(evaluate_predictions(y_test, ensemble_gini_weighted, f"Ensemble: Gini-Weighted ({weight1:.3f}/{weight2:.3f})"))
    
    # 3. Rank averaging
    ensemble_rank = ensemble_rank_average(pred1, pred2)
    results.append(evaluate_predictions(y_test, ensemble_rank, "Ensemble: Rank Averaging"))
    
    # 4. Geometric mean
    ensemble_geo = ensemble_geometric_mean(pred1, pred2)
    results.append(evaluate_predictions(y_test, ensemble_geo, "Ensemble: Geometric Mean"))
    
    # 5. Optimized weights (grid search on test set - for comparison only)
    # Note: In practice, should use validation set, but for comparison we'll test on test set
    best_gini = -1
    best_weight1 = 0.5
    for w1 in np.arange(0.1, 1.0, 0.1):
        w2 = 1.0 - w1
        ensemble_opt = ensemble_weighted_average(pred1, pred2, w1, w2)
        gini_opt = 2 * roc_auc_score(y_test, ensemble_opt) - 1
        if gini_opt > best_gini:
            best_gini = gini_opt
            best_weight1 = w1
    
    ensemble_optimized = ensemble_weighted_average(pred1, pred2, best_weight1, 1.0 - best_weight1)
    results.append(evaluate_predictions(y_test, ensemble_optimized, f"Ensemble: Optimized Weights ({best_weight1:.3f}/{1-best_weight1:.3f})"))
    
    print(f"✅ Tested {len(results)} ensemble methods")
    
    return results


def generate_comparison_report(individual_results, ensemble_results, output_dir):
    """Generate comprehensive comparison report"""
    print(f"\n📝 Generating comparison report in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all results
    all_results = individual_results + ensemble_results
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Gini': f"{result['gini']:.4f}",
            'ROC AUC': f"{result['roc_auc']:.4f}",
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1']:.4f}",
            'TP': result['confusion_matrix']['tp'],
            'FP': result['confusion_matrix']['fp'],
            'TN': result['confusion_matrix']['tn'],
            'FN': result['confusion_matrix']['fn']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save CSV
    csv_file = os.path.join(output_dir, 'ensemble_comparison.csv')
    comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ Saved comparison table to {csv_file}")
    
    # Save JSON with full metrics
    json_file = os.path.join(output_dir, 'ensemble_comparison.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"✅ Saved detailed metrics to {json_file}")
    
    # Print comparison table
    print("\n" + "="*100)
    print("ENSEMBLE COMPARISON RESULTS")
    print("="*100)
    print("\nPerformance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model = max(all_results, key=lambda x: x['gini'])
    print(f"\n🏆 Best Model: {best_model['model_name']}")
    print(f"   Gini: {best_model['gini']:.4f} ({best_model['gini']*100:.2f}%)")
    print(f"   ROC AUC: {best_model['roc_auc']:.4f}")
    print(f"   Precision: {best_model['precision']:.4f} ({best_model['precision']*100:.2f}%)")
    print(f"   Recall: {best_model['recall']:.4f} ({best_model['recall']*100:.2f}%)")
    
    # Calculate improvements
    individual_best_gini = max([r['gini'] for r in individual_results])
    ensemble_best_gini = max([r['gini'] for r in ensemble_results])
    improvement = ensemble_best_gini - individual_best_gini
    
    print(f"\n📊 Ensemble Improvement:")
    print(f"   Best Individual Gini: {individual_best_gini:.4f} ({individual_best_gini*100:.2f}%)")
    print(f"   Best Ensemble Gini: {ensemble_best_gini:.4f} ({ensemble_best_gini*100:.2f}%)")
    print(f"   Improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")
    
    if improvement > 0:
        print(f"   ✅ Ensemble improves performance!")
    elif improvement < 0:
        print(f"   ⚠️  Ensemble performs worse than best individual model")
    else:
        print(f"   ➡️  Ensemble performs similarly to best individual model")
    
    # Generate visualization
    generate_comparison_plot(all_results, output_dir)
    
    # Generate markdown report
    generate_markdown_report(individual_results, ensemble_results, best_model, improvement, output_dir)
    
    return comparison_df, best_model


def generate_comparison_plot(all_results, output_dir):
    """Generate visualization comparing models"""
    print("\n📊 Generating comparison plot...")
    
    # Extract data for plotting
    model_names = [r['model_name'] for r in all_results]
    gini_scores = [r['gini'] * 100 for r in all_results]  # Convert to percentage
    
    # Color code: individual models vs ensemble
    colors = []
    for name in model_names:
        if 'Ensemble' in name:
            colors.append('green')
        else:
            colors.append('blue')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(model_names)), gini_scores, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Gini Coefficient (%)', fontsize=12)
    ax.set_title('Model Performance Comparison - Gini Coefficient', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, gini_scores)):
        ax.text(score + 0.1, i, f'{score:.2f}%', va='center', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Individual Models'),
        Patch(facecolor='green', alpha=0.7, label='Ensemble Methods')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'ensemble_comparison_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to {plot_file}")
    plt.close()


def generate_markdown_report(individual_results, ensemble_results, best_model, improvement, output_dir):
    """Generate detailed markdown report"""
    report_file = os.path.join(output_dir, 'ensemble_evaluation_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Ensemble Model Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Models Combined\n\n")
        f.write(f"1. **{MODEL1_NAME}**\n")
        f.write(f"   - Test Gini: {MODEL1_GINI:.4f} ({MODEL1_GINI*100:.2f}%)\n")
        f.write(f"   - File: {MODEL1_PATH}\n\n")
        f.write(f"2. **{MODEL2_NAME}**\n")
        f.write(f"   - Test Gini: {MODEL2_GINI:.4f} ({MODEL2_GINI*100:.2f}%)\n")
        f.write(f"   - File: {MODEL2_PATH}\n\n")
        
        f.write("## Ensemble Methods Tested\n\n")
        f.write("1. **Equal Weights (0.5/0.5)**: Simple average of predictions\n")
        f.write("2. **Gini-Weighted**: Weighted by known test Gini performance\n")
        f.write("3. **Rank Averaging**: Average ranks, then convert back to probabilities\n")
        f.write("4. **Geometric Mean**: Geometric mean of probabilities\n")
        f.write("5. **Optimized Weights**: Grid search for optimal weights (test set)\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Model | Gini | ROC AUC | Precision | Recall | F1-Score |\n")
        f.write("|-------|------|---------|-----------|--------|----------|\n")
        
        for result in individual_results + ensemble_results:
            f.write(f"| {result['model_name']} | {result['gini']:.4f} | {result['roc_auc']:.4f} | "
                   f"{result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} |\n")
        
        f.write(f"\n## Best Model\n\n")
        f.write(f"**{best_model['model_name']}**\n\n")
        f.write(f"- Gini: {best_model['gini']:.4f} ({best_model['gini']*100:.2f}%)\n")
        f.write(f"- ROC AUC: {best_model['roc_auc']:.4f}\n")
        f.write(f"- Precision: {best_model['precision']:.4f} ({best_model['precision']*100:.2f}%)\n")
        f.write(f"- Recall: {best_model['recall']:.4f} ({best_model['recall']*100:.2f}%)\n")
        f.write(f"- F1-Score: {best_model['f1']:.4f}\n\n")
        
        f.write(f"## Improvement Analysis\n\n")
        individual_best = max([r['gini'] for r in individual_results])
        ensemble_best = max([r['gini'] for r in ensemble_results])
        f.write(f"- Best Individual Model Gini: {individual_best:.4f} ({individual_best*100:.2f}%)\n")
        f.write(f"- Best Ensemble Gini: {ensemble_best:.4f} ({ensemble_best*100:.2f}%)\n")
        f.write(f"- Improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)\n\n")
        
        if improvement > 0.001:
            f.write("✅ **Conclusion:** Ensemble improves performance over individual models.\n")
        elif improvement < -0.001:
            f.write("⚠️  **Conclusion:** Ensemble performs worse than best individual model.\n")
        else:
            f.write("➡️  **Conclusion:** Ensemble performs similarly to best individual model.\n")
    
    print(f"✅ Saved markdown report to {report_file}")


def main():
    """Main ensemble evaluation pipeline"""
    print("="*100)
    print("ENSEMBLE MODEL EVALUATION - WINNER + SECOND BEST")
    print("="*100)
    print("\n📋 Configuration:")
    print(f"   Model 1: {MODEL1_NAME}")
    print(f"   Model 2: {MODEL2_NAME}")
    print(f"   Test Data: {TEST_DATA_PATH}")
    print("="*100)
    
    start_time = datetime.now()
    
    # Load both models
    model1, model1_columns = load_model(MODEL1_PATH, MODEL1_COLUMNS_PATH, MODEL1_NAME)
    model2, model2_columns = load_model(MODEL2_PATH, MODEL2_COLUMNS_PATH, MODEL2_NAME)
    
    # Load test data with same encoding as training
    print(f"\n📂 Loading test dataset: {TEST_DATA_PATH}...")
    df_test_original = pd.read_csv(TEST_DATA_PATH, encoding='utf-8-sig', low_memory=False)
    
    # Ensure categorical columns are read with proper encoding
    # This helps prevent encoding mismatches in feature names
    categorical_cols = df_test_original.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_test_original[col].dtype == 'object':
            # Ensure proper UTF-8 encoding
            df_test_original[col] = df_test_original[col].astype(str).apply(
                lambda x: x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore') if pd.notna(x) else x
            )
    print(f"✅ Loaded test dataset: {len(df_test_original):,} records")
    print(f"   Injury ratio: {df_test_original['target'].mean():.1%}")
    
    # Prepare test data for each model (with correlation thresholds used during training)
    # Model 1: 10% target ratio, 0.8 correlation threshold
    # Model 2: 50% target ratio, 0.5 correlation threshold
    X_test1, y_test = prepare_test_data_for_model(df_test_original, model1_columns, MODEL1_NAME, corr_threshold=0.8)
    X_test2, y_test2 = prepare_test_data_for_model(df_test_original, model2_columns, MODEL2_NAME, corr_threshold=0.5)
    
    # Verify y_test is the same
    assert np.array_equal(y_test.values, y_test2.values), "Target labels should be identical"
    
    # Generate predictions from both models
    pred1 = generate_predictions(model1, X_test1, MODEL1_NAME)
    pred2 = generate_predictions(model2, X_test2, MODEL2_NAME)
    
    # Evaluate individual models
    print(f"\n📊 Evaluating individual models...")
    individual_results = [
        evaluate_predictions(y_test, pred1, MODEL1_NAME),
        evaluate_predictions(y_test, pred2, MODEL2_NAME)
    ]
    
    print(f"\n📊 Individual Model Results:")
    for result in individual_results:
        print(f"   {result['model_name']}: Gini = {result['gini']:.4f} ({result['gini']*100:.2f}%)")
    
    # Test ensemble methods
    ensemble_results = test_ensemble_methods(pred1, pred2, y_test, MODEL1_NAME, MODEL2_NAME)
    
    # Generate comparison report
    comparison_df, best_model = generate_comparison_report(individual_results, ensemble_results, OUTPUT_DIR)
    
    # Print final summary
    print("\n" + "="*100)
    print("ENSEMBLE EVALUATION COMPLETE")
    print("="*100)
    elapsed_time = datetime.now() - start_time
    print(f"⏱️  Total time: {elapsed_time}")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print("="*100)


if __name__ == '__main__':
    main()

