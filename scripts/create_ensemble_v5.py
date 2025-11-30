#!/usr/bin/env python3
"""
Create Ensemble Model Script V5
Combines top N models into VotingClassifier or StackingClassifier
"""
import sys
import io
# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Reuse prepare_data from train_models_optimized_v5
def prepare_data(df, feature_list=None):
    """Prepare data with same encoding logic as V4, excluding week_5 features"""
    feature_columns = [col for col in df.columns 
                      if col not in ['player_id', 'reference_date', 'player_name', 'target']
                      and '_week_5' not in col]
    
    if feature_list:
        feature_columns = [col for col in feature_columns if col in feature_list]
    
    X = df[feature_columns]
    y = df['target']
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    for feature in categorical_features:
        X_encoded[feature] = X_encoded[feature].fillna('Unknown')
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    return X_encoded, y, feature_columns

def evaluate_model(model, X, y):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
    }
    metrics['gini'] = 2 * metrics['roc_auc'] - 1
    
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    return metrics

def load_top_models(models_dir, comparison_df, top_n=5):
    """Load top N models from comparison results"""
    top_models_info = comparison_df.nlargest(top_n, 'outsample_f1')
    
    models = []
    feature_sets = {}
    
    for _, row in top_models_info.iterrows():
        model_name = row['model_name']
        feature_set_name = row['feature_set_name']
        
        model_file = f"{models_dir}/{model_name}_{feature_set_name}.joblib"
        
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            models.append((f"{model_name}_{feature_set_name}", model))
            feature_sets[f"{model_name}_{feature_set_name}"] = feature_set_name
            print(f"   ✅ Loaded {model_name}_{feature_set_name}")
        else:
            print(f"   ⚠️  Model file not found: {model_file}")
    
    return models, feature_sets

def create_ensemble(models, ensemble_type='voting', voting='soft'):
    """Create ensemble from list of models"""
    if ensemble_type == 'voting':
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting,
            n_jobs=-1
        )
    elif ensemble_type == 'stacking':
        # Use Logistic Regression as meta-learner
        from sklearn.linear_model import LogisticRegression
        meta_learner = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        ensemble = StackingClassifier(
            estimators=models,
            final_estimator=meta_learner,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    return ensemble

def main():
    print("=" * 80)
    print("CREATE ENSEMBLE MODEL V5")
    print("=" * 80)
    
    # Load config
    config_file = 'config/model_selection_config.json'
    if not os.path.exists(config_file):
        config_file = f'scripts/{config_file}'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    models_dir = config['paths']['models_dir']
    analysis_dir = config['paths']['analysis_dir']
    
    # Load comparison results
    comparison_file = f"{analysis_dir}/model_comparison_results.csv"
    if not os.path.exists(comparison_file):
        print(f"❌ Comparison results not found: {comparison_file}")
        print("   Please run compare_optimized_models.py first")
        return
    
    print(f"\n📂 Loading comparison results from {comparison_file}")
    comparison_df = pd.read_csv(comparison_file)
    print(f"✅ Loaded {len(comparison_df)} model results")
    
    # Load top models
    top_n = 5  # Use top 5 models
    print(f"\n📂 Loading top {top_n} models...")
    models, feature_sets = load_top_models(models_dir, comparison_df, top_n=top_n)
    
    if len(models) == 0:
        print("❌ No models loaded. Cannot create ensemble.")
        return
    
    print(f"✅ Loaded {len(models)} models for ensemble")
    
    # Load data
    print("\n📂 Loading timelines...")
    train_file = 'timelines_35day_enhanced_balanced_v4_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_val.csv'
    
    if not os.path.exists(train_file):
        train_file = f'scripts/{train_file}'
    if not os.path.exists(val_file):
        val_file = f'scripts/{val_file}'
    
    df_train_full = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val_outsample = pd.read_csv(val_file, encoding='utf-8-sig')
    
    # Get union of all feature sets
    all_features = set()
    for feature_set_name in feature_sets.values():
        feature_set_file = f"feature_sets/{feature_set_name}.json"
        if os.path.exists(feature_set_file):
            with open(feature_set_file, 'r') as f:
                features = json.load(f)
                all_features.update(features)
    
    print(f"\n📊 Using {len(all_features)} features (union of all feature sets)")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train_full, y_train_full, _ = prepare_data(df_train_full, feature_list=list(all_features))
    X_val_outsample, y_val_outsample, _ = prepare_data(df_val_outsample, feature_list=list(all_features))
    
    X_train, X_val_insample, y_train, y_val_insample = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    # Align columns
    all_cols = sorted(set(X_train.columns) | set(X_val_insample.columns) | set(X_val_outsample.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_val_insample = X_val_insample.reindex(columns=all_cols, fill_value=0)
    X_val_outsample = X_val_outsample.reindex(columns=all_cols, fill_value=0)
    
    # Test different ensemble types
    ensemble_results = {}
    
    for ensemble_type in ['voting', 'stacking']:
        print(f"\n{'='*80}")
        print(f"Creating {ensemble_type.upper()} ensemble...")
        print(f"{'='*80}")
        
        try:
            if ensemble_type == 'voting':
                ensemble = create_ensemble(models, ensemble_type='voting', voting='soft')
            else:
                ensemble = create_ensemble(models, ensemble_type='stacking')
            
            print(f"   Training ensemble...")
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            train_metrics = evaluate_model(ensemble, X_train, y_train)
            insample_metrics = evaluate_model(ensemble, X_val_insample, y_val_insample)
            outsample_metrics = evaluate_model(ensemble, X_val_outsample, y_val_outsample)
            
            print(f"\n   Out-of-Sample Performance:")
            print(f"      F1-Score: {outsample_metrics['f1']:.4f}")
            print(f"      Precision: {outsample_metrics['precision']:.4f}")
            print(f"      Recall: {outsample_metrics['recall']:.4f}")
            print(f"      ROC AUC: {outsample_metrics['roc_auc']:.4f}")
            
            ensemble_results[ensemble_type] = {
                'model': ensemble,
                'train': train_metrics,
                'validation_insample': insample_metrics,
                'validation_outsample': outsample_metrics
            }
            
        except Exception as e:
            print(f"   ❌ Error creating {ensemble_type} ensemble: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare with individual models
    print(f"\n{'='*80}")
    print("ENSEMBLE vs INDIVIDUAL MODELS")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<40} {'F1':<10} {'Precision':<12} {'Recall':<12} {'ROC AUC':<10}")
    print("-" * 90)
    
    # Individual models
    for name, _ in models:
        model_info = comparison_df[
            (comparison_df['model_name'] == name.split('_')[0]) &
            (comparison_df['feature_set_name'] == '_'.join(name.split('_')[1:]))
        ]
        if len(model_info) > 0:
            row = model_info.iloc[0]
            print(f"{name:<40} {row['outsample_f1']:<10.4f} {row['outsample_precision']:<12.4f} "
                  f"{row['outsample_recall']:<12.4f} {row['outsample_roc_auc']:<10.4f}")
    
    # Ensembles
    for ensemble_type, results in ensemble_results.items():
        metrics = results['validation_outsample']
        print(f"{ensemble_type.capitalize() + ' Ensemble':<40} {metrics['f1']:<10.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['roc_auc']:<10.4f}")
    
    # Save best ensemble
    if ensemble_results:
        best_ensemble_type = max(ensemble_results.keys(), 
                                key=lambda k: ensemble_results[k]['validation_outsample']['f1'])
        best_ensemble = ensemble_results[best_ensemble_type]['model']
        
        ensemble_file = f"{models_dir}/ensemble_{best_ensemble_type}_v5.joblib"
        joblib.dump(best_ensemble, ensemble_file)
        print(f"\n✅ Saved best ensemble ({best_ensemble_type}) to {ensemble_file}")
        
        # Save metrics
        metrics_file = f"{models_dir}/ensemble_{best_ensemble_type}_v5_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(ensemble_results[best_ensemble_type], f, indent=2, default=str)
        print(f"✅ Saved ensemble metrics to {metrics_file}")
    
    print("\n🎉 ENSEMBLE CREATION COMPLETED!")

if __name__ == "__main__":
    main()


