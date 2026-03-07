#!/usr/bin/env python3
"""
Comprehensive Performance Analysis and Enhancement Recommendations
for V4 Dual Target LGBM Models
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models'
TRAIN_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'train'
TEST_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'data' / 'timelines' / 'test'
OUTPUT_DIR = SCRIPT_DIR / 'analysis'
TIMELINES_DIR = SCRIPT_DIR.parent / 'timelines'
sys.path.insert(0, str(TIMELINES_DIR.resolve()))

# Copy filter function directly to avoid import issues
def filter_timelines_for_model(timelines_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Filter timelines dataset for a specific model training.
    
    This function ensures that when training a model for a specific injury type,
    we exclude other injury types from the negative class.
    
    Args:
        timelines_df: DataFrame with target1 and target2 columns
        target_column: 'target1' (muscular) or 'target2' (skeletal)
    
    Returns:
        Filtered DataFrame ready for model training
    """
    if target_column not in ['target1', 'target2']:
        raise ValueError(f"target_column must be 'target1' or 'target2', got: {target_column}")
    
    # Create a copy to avoid modifying the original
    df = timelines_df.copy()
    
    if target_column == 'target1':
        # Model 1: Muscular injuries
        # Positives: target1=1 (muscular injuries)
        # Negatives: target1=0 AND target2=0 (non-injuries only, exclude skeletal)
        filtered = df[(df['target1'] == 1) | ((df['target1'] == 0) & (df['target2'] == 0))]
    else:
        # Model 2: Skeletal injuries
        # Positives: target2=1 (skeletal injuries)
        # Negatives: target1=0 AND target2=0 (non-injuries only, exclude muscular)
        filtered = df[(df['target2'] == 1) | ((df['target1'] == 0) & (df['target2'] == 0))]
    
    return filtered

# Copy necessary helper functions to avoid import issues
def clean_categorical_value(value):
    """Clean categorical values to remove special characters"""
    if pd.isna(value) or value is None:
        return 'Unknown'
    value_str = str(value).strip()
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    replacements = {
        ':': '_', "'": '_', ',': '_', '"': '_', ';': '_', '/': '_', '\\': '_',
        '{': '_', '}': '_', '[': '_', ']': '_', '(': '_', ')': '_', '|': '_',
        '&': '_', '?': '_', '!': '_', '*': '_', '+': '_', '=': '_', '@': '_',
        '#': '_', '$': '_', '%': '_', '^': '_', ' ': '_',
    }
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)
    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    value_str = value_str.strip('_')
    return value_str if value_str else 'Unknown'

def sanitize_feature_name(name):
    """Sanitize feature names to be JSON-safe"""
    name_str = str(name)
    replacements = {
        '"': '_quote_', '\\': '_backslash_', '/': '_slash_',
        '\b': '_bs_', '\f': '_ff_', '\n': '_nl_', '\r': '_cr_', '\t': '_tab_', ' ': '_',
        "'": '_apostrophe_', ':': '_colon_', ';': '_semicolon_', ',': '_comma_',
        '{': '_lbrace_', '}': '_rbrace_', '[': '_lbracket_', ']': '_rbracket_', '&': '_amp_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    return name_str.strip('_')

def load_test_dataset():
    """Load test dataset (2025/26 season)"""
    test_file = TEST_DIR / 'timelines_35day_season_2025_2026_v4_muscular_test.csv'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    print(f"\n📂 Loading test dataset: {test_file.name}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    if 'target1' not in df_test.columns or 'target2' not in df_test.columns:
        raise ValueError("Test dataset missing target1/target2 columns")
    return df_test

def prepare_data(df, cache_file=None, use_cache=False):
    """Prepare data with basic preprocessing"""
    # Drop non-feature columns
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'date', 'player_name', 'target1', 'target2', 'target', 'has_minimum_activity']
    ]
    X = df[feature_columns].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    if len(categorical_features) > 0:
        for feature in categorical_features:
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    if len(numeric_features) > 0:
        for feature in numeric_features:
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    return X_encoded

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model_and_data(model_num, target_col):
    """Load model, columns, and prepare test data"""
    model_path = MODEL_DIR / f'lgbm_muscular_v4_natural_model{model_num}.joblib'
    cols_path = MODEL_DIR / f'lgbm_muscular_v4_natural_model{model_num}_columns.json'
    
    # Load model
    model = joblib.load(model_path)
    with open(cols_path, 'r') as f:
        model_columns = json.load(f)
    
    # Load and filter test data
    df_test = load_test_dataset()
    df_test_filtered = filter_timelines_for_model(df_test, target_col)
    
    # Prepare features
    X_test = prepare_data(df_test_filtered, cache_file=None, use_cache=False)
    y_test = df_test_filtered[target_col].values
    
    # Align to model columns
    X_test = X_test.reindex(columns=model_columns, fill_value=0)
    
    return model, X_test, y_test, df_test_filtered

def analyze_feature_importance(model, feature_names, top_n=30):
    """Analyze feature importance"""
    importance = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_imp_df.head(top_n)

def analyze_prediction_distributions(y_true, y_proba, model_name):
    """Analyze prediction probability distributions"""
    results = {}
    
    # Separate by class
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    
    results['positive_mean'] = float(pos_proba.mean())
    results['positive_median'] = float(np.median(pos_proba))
    results['positive_std'] = float(pos_proba.std())
    results['positive_min'] = float(pos_proba.min())
    results['positive_max'] = float(pos_proba.max())
    
    results['negative_mean'] = float(neg_proba.mean())
    results['negative_median'] = float(np.median(neg_proba))
    results['negative_std'] = float(neg_proba.std())
    results['negative_min'] = float(neg_proba.min())
    results['negative_max'] = float(neg_proba.max())
    
    # Separation metric
    results['separation'] = float(results['positive_mean'] - results['negative_mean'])
    results['overlap_ratio'] = float(len(neg_proba[neg_proba > results['positive_median']]) / len(neg_proba)) if len(neg_proba) > 0 else 0.0
    
    return results, pos_proba, neg_proba

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal threshold using different metrics"""
    thresholds = np.arange(0.01, 0.99, 0.01)
    results = []
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_youden_idx = np.argmax(youden_j)
    optimal_youden_threshold = roc_thresholds[optimal_youden_idx]
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if y_pred.sum() == 0:
            continue
            
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Find closest Youden's J threshold
        closest_youden_idx = np.argmin(np.abs(roc_thresholds - threshold))
        youden_j_value = youden_j[closest_youden_idx] if closest_youden_idx < len(youden_j) else 0
        
        results.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'youden_j': float(youden_j_value)
        })
    
    df_results = pd.DataFrame(results)
    
    # Find best by different metrics
    if len(df_results) > 0:
        best_f1 = df_results.loc[df_results['f1'].idxmax()].to_dict()
        best_youden = df_results.loc[df_results['youden_j'].idxmax()].to_dict()
        
        # Find threshold that balances precision and recall
        df_results['pr_balance'] = (df_results['precision'] + df_results['recall']) / 2
        best_balance = df_results.loc[df_results['pr_balance'].idxmax()].to_dict()
        
        return {
            'best_f1': best_f1,
            'best_youden': best_youden,
            'best_balance': best_balance,
            'optimal_youden_threshold': float(optimal_youden_threshold),
            'all_results': df_results.to_dict('records')
        }
    else:
        return None

def analyze_overfitting(train_metrics, test_metrics):
    """Analyze overfitting patterns"""
    overfitting_analysis = {
        'roc_auc_gap': float(train_metrics['roc_auc'] - test_metrics['roc_auc']),
        'recall_gap': float(train_metrics['recall'] - test_metrics['recall']),
        'precision_gap': float(train_metrics['precision'] - test_metrics['precision']),
        'f1_gap': float(train_metrics['f1'] - test_metrics['f1']),
        'severity': 'severe' if (train_metrics['roc_auc'] - test_metrics['roc_auc']) > 0.2 else 'moderate'
    }
    
    return overfitting_analysis

def generate_recommendations(analysis_results):
    """Generate enhancement recommendations"""
    recommendations = []
    
    # Overfitting recommendations
    if analysis_results['overfitting']['roc_auc_gap'] > 0.2:
        recommendations.append({
            'category': 'Overfitting',
            'priority': 'HIGH',
            'issue': f"Severe overfitting: Training ROC-AUC {analysis_results['train_metrics']['roc_auc']:.4f} vs Test {analysis_results['test_metrics']['roc_auc']:.4f} (gap: {analysis_results['overfitting']['roc_auc_gap']:.4f})",
            'recommendations': [
                'Apply correlation filtering (threshold=0.8) to reduce redundant features',
                'Increase regularization: reg_alpha=0.5, reg_lambda=2.0',
                'Reduce model complexity: max_depth=7, min_child_samples=50',
                'Add early stopping with validation set',
                'Reduce learning rate: learning_rate=0.05',
                'Increase subsample and colsample_bytree: subsample=0.7, colsample_bytree=0.7'
            ]
        })
    
    # Low recall recommendations
    if analysis_results['test_metrics']['recall'] < 0.3:
        recommendations.append({
            'category': 'Low Recall',
            'priority': 'HIGH',
            'issue': f"Very low recall on test: {analysis_results['test_metrics']['recall']:.2%} - missing most injuries",
            'recommendations': [
                'Optimize prediction threshold (currently using 0.5, optimal may be lower)',
                'Use class_weight parameter more aggressively',
                'Consider focal loss or other imbalanced learning techniques',
                'Use SMOTE or other oversampling techniques for training',
                'Focus on features that improve recall (check feature importance)'
            ]
        })
    
    # Low precision recommendations
    if analysis_results['test_metrics']['precision'] < 0.1:
        recommendations.append({
            'category': 'Low Precision',
            'priority': 'MEDIUM',
            'issue': f"Low precision on test: {analysis_results['test_metrics']['precision']:.2%} - many false positives",
            'recommendations': [
                'Optimize prediction threshold (may need to be higher)',
                'Improve feature quality - remove noisy features',
                'Apply correlation filtering to reduce redundant features',
                'Use ensemble methods to reduce variance',
                'Focus on features with high precision impact'
            ]
        })
    
    # Threshold optimization
    if 'optimal_threshold' in analysis_results and analysis_results['optimal_threshold']:
        current_threshold = 0.5
        optimal_threshold = analysis_results['optimal_threshold']['best_f1']['threshold']
        if abs(current_threshold - optimal_threshold) > 0.1:
            recommendations.append({
                'category': 'Threshold Optimization',
                'priority': 'HIGH',
                'issue': f"Using suboptimal threshold: current=0.5, optimal≈{optimal_threshold:.3f}",
                'recommendations': [
                    f"Use threshold={optimal_threshold:.3f} for F1 optimization",
                    f"Use threshold={analysis_results['optimal_threshold']['best_balance']['threshold']:.3f} for precision-recall balance",
                    'Implement threshold optimization on validation set',
                    'Consider different thresholds for different use cases'
                ]
            })
    
    # Feature importance recommendations
    if 'feature_importance' in analysis_results:
        feature_imp = analysis_results['feature_importance']
        if feature_imp is not None and (isinstance(feature_imp, pd.DataFrame) and not feature_imp.empty) or (isinstance(feature_imp, list) and len(feature_imp) > 0):
            if isinstance(feature_imp, list):
                top_features = pd.DataFrame(feature_imp).head(10)
            else:
                top_features = feature_imp.head(10)
            top_feature_names = [f['feature'] for f in top_features.to_dict('records')[:5]]
            recommendations.append({
                'category': 'Feature Engineering',
                'priority': 'MEDIUM',
                'issue': f"Top features identified - focus on these for improvements",
                'recommendations': [
                    f"Top 5 features: {', '.join(top_feature_names)}",
                    'Consider feature interactions for top features',
                    'Remove features with zero importance',
                    'Apply correlation filtering to reduce redundancy'
                ]
            })
    
    return recommendations

def main():
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS - V4 DUAL TARGET MODELS")
    print("="*80)
    
    # Load metrics
    metrics_file = MODEL_DIR / 'lgbm_muscular_v4_natural_metrics.json'
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    analysis_results = {}
    
    # Analyze Model 1 (Muscular)
    print("\n" + "="*80)
    print("ANALYZING MODEL 1: MUSCULAR INJURIES")
    print("="*80)
    
    print("\nLoading model and test data...")
    model1, X_test1, y_test1, df_test1 = load_model_and_data(1, 'target1')
    
    # Feature importance
    print("\n1. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)
    feature_imp1 = analyze_feature_importance(model1, X_test1.columns, top_n=30)
    print(f"Top 10 features:")
    for idx, row in feature_imp1.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    analysis_results['model1'] = {'feature_importance': feature_imp1}
    
    # Prediction distributions
    print("\n2. PREDICTION DISTRIBUTION ANALYSIS")
    print("-" * 80)
    y_proba1 = model1.predict_proba(X_test1)[:, 1]
    dist_analysis1, pos_proba1, neg_proba1 = analyze_prediction_distributions(y_test1, y_proba1, "Model 1")
    print(f"Positive class probabilities:")
    print(f"   Mean: {dist_analysis1['positive_mean']:.4f}, Median: {dist_analysis1['positive_median']:.4f}")
    print(f"   Range: [{dist_analysis1['positive_min']:.4f}, {dist_analysis1['positive_max']:.4f}]")
    print(f"Negative class probabilities:")
    print(f"   Mean: {dist_analysis1['negative_mean']:.4f}, Median: {dist_analysis1['negative_median']:.4f}")
    print(f"   Range: [{dist_analysis1['negative_min']:.4f}, {dist_analysis1['negative_max']:.4f}]")
    print(f"Separation: {dist_analysis1['separation']:.4f}")
    print(f"Overlap ratio: {dist_analysis1['overlap_ratio']:.2%}")
    analysis_results['model1']['prediction_distribution'] = dist_analysis1
    
    # Threshold optimization
    print("\n3. THRESHOLD OPTIMIZATION")
    print("-" * 80)
    optimal_thresh1 = find_optimal_threshold(y_test1, y_proba1)
    if optimal_thresh1:
        print(f"Current threshold: 0.5")
        print(f"Optimal F1 threshold: {optimal_thresh1['best_f1']['threshold']:.3f}")
        print(f"   At this threshold: Precision={optimal_thresh1['best_f1']['precision']:.4f}, Recall={optimal_thresh1['best_f1']['recall']:.4f}, F1={optimal_thresh1['best_f1']['f1']:.4f}")
        print(f"Optimal balance threshold: {optimal_thresh1['best_balance']['threshold']:.3f}")
        print(f"   At this threshold: Precision={optimal_thresh1['best_balance']['precision']:.4f}, Recall={optimal_thresh1['best_balance']['recall']:.4f}")
        print(f"Optimal Youden's J threshold: {optimal_thresh1['optimal_youden_threshold']:.3f}")
        analysis_results['model1']['optimal_threshold'] = optimal_thresh1
    
    # Overfitting analysis
    print("\n4. OVERFITTING ANALYSIS")
    print("-" * 80)
    train_metrics1 = metrics['model1_muscular']['train']
    test_metrics1 = metrics['model1_muscular']['test']
    overfitting1 = analyze_overfitting(train_metrics1, test_metrics1)
    print(f"ROC-AUC gap: {overfitting1['roc_auc_gap']:.4f} ({overfitting1['severity']} overfitting)")
    print(f"Recall gap: {overfitting1['recall_gap']:.4f}")
    print(f"F1 gap: {overfitting1['f1_gap']:.4f}")
    analysis_results['model1']['train_metrics'] = train_metrics1
    analysis_results['model1']['test_metrics'] = test_metrics1
    analysis_results['model1']['overfitting'] = overfitting1
    
    # Analyze Model 2 (Skeletal)
    print("\n" + "="*80)
    print("ANALYZING MODEL 2: SKELETAL INJURIES")
    print("="*80)
    
    print("\nLoading model and test data...")
    model2, X_test2, y_test2, df_test2 = load_model_and_data(2, 'target2')
    
    # Feature importance
    print("\n1. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)
    feature_imp2 = analyze_feature_importance(model2, X_test2.columns, top_n=30)
    print(f"Top 10 features:")
    for idx, row in feature_imp2.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    analysis_results['model2'] = {'feature_importance': feature_imp2}
    
    # Prediction distributions
    print("\n2. PREDICTION DISTRIBUTION ANALYSIS")
    print("-" * 80)
    y_proba2 = model2.predict_proba(X_test2)[:, 1]
    dist_analysis2, pos_proba2, neg_proba2 = analyze_prediction_distributions(y_test2, y_proba2, "Model 2")
    print(f"Positive class probabilities:")
    print(f"   Mean: {dist_analysis2['positive_mean']:.4f}, Median: {dist_analysis2['positive_median']:.4f}")
    print(f"Negative class probabilities:")
    print(f"   Mean: {dist_analysis2['negative_mean']:.4f}, Median: {dist_analysis2['negative_median']:.4f}")
    print(f"Separation: {dist_analysis2['separation']:.4f}")
    print(f"Overlap ratio: {dist_analysis2['overlap_ratio']:.2%}")
    analysis_results['model2']['prediction_distribution'] = dist_analysis2
    
    # Threshold optimization
    print("\n3. THRESHOLD OPTIMIZATION")
    print("-" * 80)
    optimal_thresh2 = find_optimal_threshold(y_test2, y_proba2)
    if optimal_thresh2:
        print(f"Current threshold: 0.5")
        print(f"Optimal F1 threshold: {optimal_thresh2['best_f1']['threshold']:.3f}")
        print(f"   At this threshold: Precision={optimal_thresh2['best_f1']['precision']:.4f}, Recall={optimal_thresh2['best_f1']['recall']:.4f}, F1={optimal_thresh2['best_f1']['f1']:.4f}")
        analysis_results['model2']['optimal_threshold'] = optimal_thresh2
    
    # Overfitting analysis
    print("\n4. OVERFITTING ANALYSIS")
    print("-" * 80)
    train_metrics2 = metrics['model2_skeletal']['train']
    test_metrics2 = metrics['model2_skeletal']['test']
    overfitting2 = analyze_overfitting(train_metrics2, test_metrics2)
    print(f"ROC-AUC gap: {overfitting2['roc_auc_gap']:.4f} ({overfitting2['severity']} overfitting)")
    analysis_results['model2']['train_metrics'] = train_metrics2
    analysis_results['model2']['test_metrics'] = test_metrics2
    analysis_results['model2']['overfitting'] = overfitting2
    
    # Generate recommendations
    print("\n" + "="*80)
    print("ENHANCEMENT RECOMMENDATIONS")
    print("="*80)
    
    recommendations_model1 = generate_recommendations(analysis_results['model1'])
    recommendations_model2 = generate_recommendations(analysis_results['model2'])
    
    print("\n📋 MODEL 1 (MUSCULAR) RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations_model1, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['issue']}")
        for suggestion in rec['recommendations']:
            print(f"   - {suggestion}")
    
    print("\n📋 MODEL 2 (SKELETAL) RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations_model2, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['issue']}")
        for suggestion in rec['recommendations']:
            print(f"   - {suggestion}")
    
    # Save analysis (convert DataFrames to dicts for JSON)
    analysis_results_json = {}
    for model_key in ['model1', 'model2']:
        if model_key in analysis_results:
            model_data = analysis_results[model_key].copy()
            if 'feature_importance' in model_data and isinstance(model_data['feature_importance'], pd.DataFrame):
                model_data['feature_importance'] = model_data['feature_importance'].to_dict('records')
            analysis_results_json[model_key] = model_data
    
    analysis_file = OUTPUT_DIR / 'performance_analysis.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results_json, f, indent=2, default=str)
    print(f"\n✅ Analysis saved to: {analysis_file}")
    
    return analysis_results

if __name__ == "__main__":
    main()
