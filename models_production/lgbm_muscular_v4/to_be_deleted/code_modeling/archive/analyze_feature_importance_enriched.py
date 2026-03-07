#!/usr/bin/env python3
"""
Feature Importance Analysis for Enriched Models (Layer 2 Features)

This script analyzes feature importances from the trained LGBM models to identify:
- Which Layer 2 enriched features are most valuable
- Which features may be causing overfitting
- Which features are redundant
- Recommendations for feature engineering improvements
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
MODEL_DIR = ROOT_DIR / 'models_production' / 'lgbm_muscular_v4' / 'models' / 'enriched_comparison'
OUTPUT_DIR = MODEL_DIR / 'feature_importance_analysis'
METRICS_FILE = MODEL_DIR / 'enriched_models_metrics.json'

# Layer 2 enriched feature categories
LAYER2_FEATURES = {
    'workload_minutes': [
        'minutes_last_3d', 'minutes_last_7d', 'minutes_last_14d', 
        'minutes_last_28d', 'minutes_last_35d'
    ],
    'workload_matches': [
        'matches_last_7d', 'matches_last_14d', 'matches_last_28d', 
        'matches_last_35d'
    ],
    'acwr': [
        'acwr_min_7_28'
    ],
    'season_normalized': [
        'minutes_season_to_date', 'minutes_last_7d_pct_season', 
        'minutes_last_28d_pct_season'
    ],
    'injury_history': [
        'injuries_last_90d', 'injuries_last_365d', 'injuries_season_to_date',
        'days_since_last_injury'
    ],
    'recovery_flags': [
        'is_back_to_back', 'short_rest_3_4d', 'long_rest_7d_plus',
        'days_since_last_match'
    ],
    'activity_flags': [
        'has_played_last_7d', 'has_played_last_28d', 'no_recent_activity_28d'
    ]
}

# Flatten Layer 2 features list
ALL_LAYER2_FEATURES = []
for category_features in LAYER2_FEATURES.values():
    ALL_LAYER2_FEATURES.extend(category_features)

def load_models_and_importances():
    """Load trained models and extract feature importances"""
    print("="*80)
    print("LOADING MODELS AND EXTRACTING FEATURE IMPORTANCES")
    print("="*80)
    
    models = {}
    importances = {}
    feature_names = {}
    
    model_files = {
        'target1': 'lgbm_target1_enriched.joblib',
        'target2': 'lgbm_target2_enriched.joblib',
        'combined': 'lgbm_combined_enriched.joblib'
    }
    
    for model_name, model_file in model_files.items():
        model_path = MODEL_DIR / model_file
        columns_path = MODEL_DIR / f"{model_file.replace('.joblib', '_columns.json')}"
        
        if not model_path.exists():
            print(f"⚠️  Model file not found: {model_path}")
            continue
        
        print(f"\n📂 Loading {model_name} model...")
        model = joblib.load(model_path)
        models[model_name] = model
        
        # Load feature names
        if columns_path.exists():
            with open(columns_path, 'r') as f:
                feature_list = json.load(f)
            feature_names[model_name] = feature_list
        else:
            # Fallback: use model feature names if available
            if hasattr(model, 'feature_name_'):
                feature_names[model_name] = model.feature_name_
            else:
                print(f"⚠️  Feature names not found for {model_name}")
                continue
        
        # Extract feature importances (using gain importance)
        if hasattr(model, 'booster_'):
            # LightGBM model
            importance_gain = model.booster_.feature_importance(importance_type='gain')
            importance_split = model.booster_.feature_importance(importance_type='split')
        else:
            # Fallback to sklearn-style importance
            importance_gain = model.feature_importances_
            importance_split = model.feature_importances_
        
        importances[model_name] = {
            'gain': importance_gain,
            'split': importance_split,
            'features': feature_names[model_name]
        }
        
        print(f"   ✅ Loaded {len(feature_names[model_name])} features")
    
    return models, importances, feature_names

def categorize_features(feature_name, all_layer2_features):
    """Categorize a feature as Layer 2 or Layer 1, and identify subcategory"""
    # Normalize feature name (remove one-hot encoding prefixes/suffixes)
    base_name = feature_name
    
    # Check if it's a Layer 2 feature (exact match or contains Layer 2 feature name)
    for layer2_feat in all_layer2_features:
        if layer2_feat in base_name or base_name == layer2_feat:
            # Identify subcategory
            for category, features in LAYER2_FEATURES.items():
                if any(f in base_name for f in features):
                    return 'Layer 2', category
            return 'Layer 2', 'unknown'
    
    return 'Layer 1', 'base'

def create_importance_dataframes(importances, feature_names):
    """Create DataFrames with feature importances and categories"""
    print("\n" + "="*80)
    print("CATEGORIZING FEATURES")
    print("="*80)
    
    importance_dfs = {}
    
    for model_name, imp_data in importances.items():
        features = imp_data['features']
        gain_imp = imp_data['gain']
        split_imp = imp_data['split']
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': features,
            'importance_gain': gain_imp,
            'importance_split': split_imp
        })
        
        # Categorize features
        df['layer'] = df['feature'].apply(
            lambda x: categorize_features(x, ALL_LAYER2_FEATURES)[0]
        )
        df['category'] = df['feature'].apply(
            lambda x: categorize_features(x, ALL_LAYER2_FEATURES)[1]
        )
        
        # Calculate percentile ranks
        df['percentile_gain'] = df['importance_gain'].rank(pct=True) * 100
        df['percentile_split'] = df['importance_split'].rank(pct=True) * 100
        
        # Sort by importance
        df = df.sort_values('importance_gain', ascending=False).reset_index(drop=True)
        
        importance_dfs[model_name] = df
        
        # Print summary
        layer2_count = (df['layer'] == 'Layer 2').sum()
        layer1_count = (df['layer'] == 'Layer 1').sum()
        print(f"\n{model_name}:")
        print(f"   Total features: {len(df)}")
        print(f"   Layer 2 features: {layer2_count}")
        print(f"   Layer 1 features: {layer1_count}")
    
    return importance_dfs

def analyze_category_importance(importance_dfs):
    """Analyze importance by category"""
    print("\n" + "="*80)
    print("ANALYZING IMPORTANCE BY CATEGORY")
    print("="*80)
    
    category_analysis = {}
    
    for model_name, df in importance_dfs.items():
        # Overall statistics
        layer2_avg = df[df['layer'] == 'Layer 2']['importance_gain'].mean()
        layer1_avg = df[df['layer'] == 'Layer 1']['importance_gain'].mean()
        layer2_total = df[df['layer'] == 'Layer 2']['importance_gain'].sum()
        layer1_total = df[df['layer'] == 'Layer 1']['importance_gain'].sum()
        
        # Category-wise statistics
        category_stats = df.groupby(['layer', 'category']).agg({
            'importance_gain': ['mean', 'sum', 'count'],
            'percentile_gain': 'mean'
        }).reset_index()
        
        category_stats.columns = ['layer', 'category', 'avg_importance', 'total_importance', 'count', 'avg_percentile']
        
        category_analysis[model_name] = {
            'layer2_avg': layer2_avg,
            'layer1_avg': layer1_avg,
            'layer2_total': layer2_total,
            'layer1_total': layer1_total,
            'layer2_ratio': layer2_total / (layer2_total + layer1_total) if (layer2_total + layer1_total) > 0 else 0,
            'category_stats': category_stats
        }
        
        print(f"\n{model_name}:")
        print(f"   Layer 2 avg importance: {layer2_avg:.2f}")
        print(f"   Layer 1 avg importance: {layer1_avg:.2f}")
        print(f"   Layer 2 total importance: {layer2_total:.2f} ({category_analysis[model_name]['layer2_ratio']*100:.1f}%)")
        print(f"   Layer 1 total importance: {layer1_total:.2f} ({(1-category_analysis[model_name]['layer2_ratio'])*100:.1f}%)")
    
    return category_analysis

def identify_top_features(importance_dfs, top_n=30):
    """Identify top N features for each model"""
    print(f"\n" + "="*80)
    print(f"IDENTIFYING TOP {top_n} FEATURES PER MODEL")
    print("="*80)
    
    top_features = {}
    
    for model_name, df in importance_dfs.items():
        top_df = df.head(top_n).copy()
        top_features[model_name] = top_df
        
        layer2_in_top = (top_df['layer'] == 'Layer 2').sum()
        layer1_in_top = (top_df['layer'] == 'Layer 1').sum()
        print(f"\n{model_name} - Top {top_n} features:")
        print(f"   Layer 1 features: {layer1_in_top} ({layer1_in_top/top_n*100:.1f}%)")
        print(f"   Layer 2 features: {layer2_in_top} ({layer2_in_top/top_n*100:.1f}%)")
        print(f"\n   Top 10 features:")
        for idx, row in top_df.head(10).iterrows():
            layer_marker = "🌟" if row['layer'] == 'Layer 2' else "  "
            print(f"   {layer_marker} {row['feature'][:60]:<60} {row['importance_gain']:>12.2f} ({row['percentile_gain']:>5.1f}%)")
    
    return top_features

def identify_bottom_features(importance_dfs, bottom_n=50):
    """Identify bottom N features (lowest importance) for each model"""
    print(f"\n" + "="*80)
    print(f"IDENTIFYING BOTTOM {bottom_n} FEATURES PER MODEL (LOWEST IMPORTANCE)")
    print("="*80)
    
    bottom_features = {}
    
    for model_name, df in importance_dfs.items():
        bottom_df = df.tail(bottom_n).copy()
        bottom_features[model_name] = bottom_df
        
        layer2_in_bottom = (bottom_df['layer'] == 'Layer 2').sum()
        layer1_in_bottom = (bottom_df['layer'] == 'Layer 1').sum()
        print(f"\n{model_name} - Bottom {bottom_n} features:")
        print(f"   Layer 1 features: {layer1_in_bottom} ({layer1_in_bottom/bottom_n*100:.1f}%)")
        print(f"   Layer 2 features: {layer2_in_bottom} ({layer2_in_bottom/bottom_n*100:.1f}%)")
        print(f"\n   Bottom 10 features (lowest importance):")
        for idx, row in bottom_df.head(10).iterrows():
            layer_marker = "🌟" if row['layer'] == 'Layer 2' else "  "
            print(f"   {layer_marker} {row['feature'][:60]:<60} {row['importance_gain']:>12.2f} ({row['percentile_gain']:>5.1f}%)")
    
    return bottom_features

def cross_model_analysis(importance_dfs):
    """Analyze feature importance consistency across models"""
    print("\n" + "="*80)
    print("CROSS-MODEL CONSISTENCY ANALYSIS")
    print("="*80)
    
    # Get all unique features across models
    all_features = set()
    for df in importance_dfs.values():
        all_features.update(df['feature'].tolist())
    
    # Create cross-model DataFrame
    cross_model_df = pd.DataFrame({'feature': list(all_features)})
    
    for model_name, df in importance_dfs.items():
        feature_dict = dict(zip(df['feature'], df['importance_gain']))
        cross_model_df[f'{model_name}_importance'] = cross_model_df['feature'].map(feature_dict).fillna(0)
        cross_model_df[f'{model_name}_percentile'] = cross_model_df['feature'].map(
            dict(zip(df['feature'], df['percentile_gain']))
        ).fillna(0)
    
    # Calculate statistics
    importance_cols = [col for col in cross_model_df.columns if col.endswith('_importance')]
    cross_model_df['avg_importance'] = cross_model_df[importance_cols].mean(axis=1)
    cross_model_df['std_importance'] = cross_model_df[importance_cols].std(axis=1)
    cross_model_df['max_importance'] = cross_model_df[importance_cols].max(axis=1)
    cross_model_df['min_importance'] = cross_model_df[importance_cols].min(axis=1)
    cross_model_df['stability'] = 1 - (cross_model_df['std_importance'] / (cross_model_df['avg_importance'] + 1e-6))
    
    # Categorize features
    cross_model_df['layer'] = cross_model_df['feature'].apply(
        lambda x: categorize_features(x, ALL_LAYER2_FEATURES)[0]
    )
    cross_model_df['category'] = cross_model_df['feature'].apply(
        lambda x: categorize_features(x, ALL_LAYER2_FEATURES)[1]
    )
    
    # Sort by average importance
    cross_model_df = cross_model_df.sort_values('avg_importance', ascending=False).reset_index(drop=True)
    
    # Print most consistent features
    print("\nTop 20 features by average importance across models:")
    for idx, row in cross_model_df.head(20).iterrows():
        layer_marker = "🌟" if row['layer'] == 'Layer 2' else "  "
        print(f"   {layer_marker} {row['feature'][:50]:<50} "
              f"Avg: {row['avg_importance']:>8.2f}  "
              f"Stability: {row['stability']:>5.2f}")
    
    return cross_model_df

def load_test_metrics():
    """Load test metrics for correlation analysis"""
    if METRICS_FILE.exists():
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        return metrics
    return None

def generate_recommendations(importance_dfs, category_analysis, cross_model_df, top_features, bottom_features, metrics):
    """Generate actionable recommendations for ALL features (Layer 1 and Layer 2)"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE RECOMMENDATIONS (ALL FEATURES)")
    print("="*80)
    
    recommendations = []
    
    # 1. Identify high-value features (both layers)
    print("\n1. HIGH-VALUE FEATURES (Top 20 overall per model):")
    for model_name, df in importance_dfs.items():
        top_20 = df.head(20)
        layer1_top = top_20[top_20['layer'] == 'Layer 1']
        layer2_top = top_20[top_20['layer'] == 'Layer 2']
        
        print(f"\n   {model_name}:")
        print(f"      Layer 1: {len(layer1_top)} features")
        print(f"      Layer 2: {len(layer2_top)} features")
        
        for idx, row in top_20.iterrows():
            layer_label = "Layer 2" if row['layer'] == 'Layer 2' else "Layer 1"
            recommendations.append({
                'model': model_name,
                'feature': row['feature'],
                'layer': layer_label,
                'action': 'KEEP',
                'reason': f'Top 20 feature (percentile: {row["percentile_gain"]:.1f}%)',
                'importance': row['importance_gain'],
                'percentile': row['percentile_gain']
            })
    
    # 2. Identify low-value features from BOTH layers (candidates for removal)
    print("\n2. LOW-VALUE FEATURES (Bottom 50 per model - candidates for removal):")
    for model_name, df in importance_dfs.items():
        bottom_50 = df.tail(50)
        layer1_low = bottom_50[bottom_50['layer'] == 'Layer 1']
        layer2_low = bottom_50[bottom_50['layer'] == 'Layer 2']
        
        print(f"\n   {model_name}:")
        print(f"      Layer 1 low-value: {len(layer1_low)} features")
        print(f"      Layer 2 low-value: {len(layer2_low)} features")
        
        # Show bottom 10 from each layer
        if len(layer1_low) > 0:
            print(f"      Bottom 5 Layer 1 features:")
            for idx, row in layer1_low.head(5).iterrows():
                print(f"         ⚠️  {row['feature'][:60]} (importance: {row['importance_gain']:.2f}, percentile: {row['percentile_gain']:.1f}%)")
                recommendations.append({
                    'model': model_name,
                    'feature': row['feature'],
                    'layer': 'Layer 1',
                    'action': 'INVESTIGATE_REMOVE',
                    'reason': f'Bottom 50 feature (percentile: {row["percentile_gain"]:.1f}%) - very low importance',
                    'importance': row['importance_gain'],
                    'percentile': row['percentile_gain']
                })
        
        if len(layer2_low) > 0:
            print(f"      Bottom 5 Layer 2 features:")
            for idx, row in layer2_low.head(5).iterrows():
                print(f"         ⚠️  {row['feature'][:60]} (importance: {row['importance_gain']:.2f}, percentile: {row['percentile_gain']:.1f}%)")
                recommendations.append({
                    'model': model_name,
                    'feature': row['feature'],
                    'layer': 'Layer 2',
                    'action': 'INVESTIGATE_REMOVE',
                    'reason': f'Bottom 50 feature (percentile: {row["percentile_gain"]:.1f}%) - very low importance',
                    'importance': row['importance_gain'],
                    'percentile': row['percentile_gain']
                })
    
    # 3. Zero-importance features (definite candidates for removal)
    print("\n3. ZERO-IMPORTANCE FEATURES (definite candidates for removal):")
    for model_name, df in importance_dfs.items():
        zero_imp = df[df['importance_gain'] == 0]
        if len(zero_imp) > 0:
            layer1_zero = zero_imp[zero_imp['layer'] == 'Layer 1']
            layer2_zero = zero_imp[zero_imp['layer'] == 'Layer 2']
            print(f"\n   {model_name}:")
            print(f"      Layer 1 zero-importance: {len(layer1_zero)} features")
            print(f"      Layer 2 zero-importance: {len(layer2_zero)} features")
            
            for idx, row in zero_imp.head(10).iterrows():
                layer_label = "Layer 2" if row['layer'] == 'Layer 2' else "Layer 1"
                recommendations.append({
                    'model': model_name,
                    'feature': row['feature'],
                    'layer': layer_label,
                    'action': 'REMOVE',
                    'reason': 'Zero importance across all models',
                    'importance': 0.0,
                    'percentile': 0.0
                })
    
    # 4. Most stable features across models (both layers)
    print("\n4. MOST STABLE FEATURES ACROSS MODELS (Top 20 by stability):")
    stable_all = cross_model_df.nlargest(20, 'avg_importance')
    layer1_stable = stable_all[stable_all['layer'] == 'Layer 1']
    layer2_stable = stable_all[stable_all['layer'] == 'Layer 2']
    
    print(f"   Layer 1 stable features: {len(layer1_stable)}")
    print(f"   Layer 2 stable features: {len(layer2_stable)}")
    
    for idx, row in stable_all.head(10).iterrows():
        layer_label = "Layer 2" if row['layer'] == 'Layer 2' else "Layer 1"
        print(f"   ✅ [{layer_label}] {row['feature'][:50]} (avg: {row['avg_importance']:.2f}, stability: {row['stability']:.2f})")
        recommendations.append({
            'model': 'all',
            'feature': row['feature'],
            'layer': layer_label,
            'action': 'KEEP',
            'reason': f'Stable across models (stability: {row["stability"]:.2f})',
            'importance': row['avg_importance'],
            'percentile': row['avg_importance']
        })
    
    # 5. Category-level recommendations (both layers)
    print("\n5. CATEGORY-LEVEL INSIGHTS:")
    for model_name, analysis in category_analysis.items():
        print(f"\n   {model_name}:")
        category_stats = analysis['category_stats']
        
        # Best Layer 1 category
        layer1_cats = category_stats[category_stats['layer'] == 'Layer 1']
        if len(layer1_cats) > 0:
            print(f"      Layer 1 (base): {len(layer1_cats)} categories, avg importance: {layer1_cats['avg_importance'].mean():.2f}")
        
        # Best Layer 2 category
        layer2_cats = category_stats[category_stats['layer'] == 'Layer 2']
        if len(layer2_cats) > 0:
            top_layer2_cat = layer2_cats.loc[layer2_cats['avg_importance'].idxmax()]
            print(f"      Best Layer 2 category: {top_layer2_cat['category']} "
                  f"(avg importance: {top_layer2_cat['avg_importance']:.2f})")
    
    # Convert to DataFrame
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        return rec_df
    return pd.DataFrame()

def create_visualizations(importance_dfs, cross_model_df, category_analysis, output_dir):
    """Create visualization charts"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # 1. Top features per model
    for model_name, df in importance_dfs.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        top_30 = df.head(30)
        
        colors = ['#2ecc71' if layer == 'Layer 2' else '#3498db' for layer in top_30['layer']]
        bars = ax.barh(range(len(top_30)), top_30['importance_gain'], color=colors)
        
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels([f[:50] for f in top_30['feature']], fontsize=8)
        ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
        ax.set_title(f'Top 30 Features - {model_name.upper()}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Layer 2 (Enriched)'),
            Patch(facecolor='#3498db', label='Layer 1 (Base)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'top_features_{model_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: top_features_{model_name}.png")
    
    # 2. Layer 2 vs Layer 1 comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_names = list(importance_dfs.keys())
    
    for idx, model_name in enumerate(model_names):
        df = importance_dfs[model_name]
        layer2_imp = df[df['layer'] == 'Layer 2']['importance_gain'].sum()
        layer1_imp = df[df['layer'] == 'Layer 1']['importance_gain'].sum()
        
        axes[idx].pie([layer2_imp, layer1_imp], 
                     labels=['Layer 2', 'Layer 1'],
                     autopct='%1.1f%%',
                     colors=['#2ecc71', '#3498db'],
                     startangle=90)
        axes[idx].set_title(f'{model_name.upper()}\nTotal Importance Distribution', 
                           fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: layer_comparison.png")
    
    # 3. Category-wise importance (both layers)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Layer 1 categories
    category_data_l1 = []
    for model_name, analysis in category_analysis.items():
        cat_stats = analysis['category_stats']
        for _, row in cat_stats.iterrows():
            if row['layer'] == 'Layer 1':
                category_data_l1.append({
                    'model': model_name,
                    'category': 'base',
                    'avg_importance': row['avg_importance']
                })
    
    if category_data_l1:
        cat_df_l1 = pd.DataFrame(category_data_l1)
        pivot_df_l1 = cat_df_l1.pivot(index='category', columns='model', values='avg_importance')
        pivot_df_l1.plot(kind='barh', ax=axes[0], width=0.8, color=['#3498db', '#2980b9', '#1f618d'])
        axes[0].set_xlabel('Average Importance', fontsize=12)
        axes[0].set_title('Layer 1 (Base) Average Importance by Model', fontsize=14, fontweight='bold')
        axes[0].legend(title='Model', fontsize=10)
    
    # Layer 2 categories
    category_data_l2 = []
    for model_name, analysis in category_analysis.items():
        cat_stats = analysis['category_stats']
        for _, row in cat_stats.iterrows():
            if row['layer'] == 'Layer 2':
                category_data_l2.append({
                    'model': model_name,
                    'category': row['category'],
                    'avg_importance': row['avg_importance']
                })
    
    if category_data_l2:
        cat_df_l2 = pd.DataFrame(category_data_l2)
        pivot_df_l2 = cat_df_l2.pivot(index='category', columns='model', values='avg_importance')
        pivot_df_l2.plot(kind='barh', ax=axes[1], width=0.8, color=['#2ecc71', '#27ae60', '#1e8449'])
        axes[1].set_xlabel('Average Importance', fontsize=12)
        axes[1].set_title('Layer 2 (Enriched) Category Importance by Model', fontsize=14, fontweight='bold')
        axes[1].legend(title='Model', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: category_importance.png")
    
    # 4. Cross-model stability
    fig, ax = plt.subplots(figsize=(12, 8))
    top_stable = cross_model_df.head(30)
    
    colors = ['#2ecc71' if layer == 'Layer 2' else '#3498db' for layer in top_stable['layer']]
    bars = ax.barh(range(len(top_stable)), top_stable['avg_importance'], color=colors)
    
    ax.set_yticks(range(len(top_stable)))
    ax.set_yticklabels([f[:50] for f in top_stable['feature']], fontsize=8)
    ax.set_xlabel('Average Importance Across Models', fontsize=12)
    ax.set_title('Top 30 Features by Average Importance (Cross-Model)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Layer 2 (Enriched)'),
        Patch(facecolor='#3498db', label='Layer 1 (Base)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_model_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: cross_model_stability.png")

def main():
    print("="*80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS - ALL FEATURES")
    print("="*80)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load models and extract importances
    models, importances, feature_names = load_models_and_importances()
    
    if not importances:
        print("❌ No models loaded. Exiting.")
        return 1
    
    # Step 2: Create importance DataFrames
    importance_dfs = create_importance_dataframes(importances, feature_names)
    
    # Step 3: Analyze by category
    category_analysis = analyze_category_importance(importance_dfs)
    
    # Step 4: Identify top features
    top_features = identify_top_features(importance_dfs, top_n=50)
    
    # Step 4b: Identify bottom features
    bottom_features = identify_bottom_features(importance_dfs, bottom_n=50)
    
    # Step 5: Cross-model analysis
    cross_model_df = cross_model_analysis(importance_dfs)
    
    # Step 6: Load metrics
    metrics = load_test_metrics()
    
    # Step 7: Generate comprehensive recommendations
    recommendations_df = generate_recommendations(
        importance_dfs, category_analysis, cross_model_df, top_features, bottom_features, metrics
    )
    
    # Step 8: Create visualizations
    create_visualizations(importance_dfs, cross_model_df, category_analysis, OUTPUT_DIR)
    
    # Step 9: Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save detailed importance DataFrames
    for model_name, df in importance_dfs.items():
        csv_path = OUTPUT_DIR / f'feature_importance_{model_name}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {csv_path.name}")
    
    # Save cross-model analysis
    cross_model_csv = OUTPUT_DIR / 'cross_model_analysis.csv'
    cross_model_df.to_csv(cross_model_csv, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {cross_model_csv.name}")
    
    # Save category analysis
    category_summary = []
    for model_name, analysis in category_analysis.items():
        for _, row in analysis['category_stats'].iterrows():
            category_summary.append({
                'model': model_name,
                'layer': row['layer'],
                'category': row['category'],
                'avg_importance': row['avg_importance'],
                'total_importance': row['total_importance'],
                'count': row['count'],
                'avg_percentile': row['avg_percentile']
            })
    
    category_df = pd.DataFrame(category_summary)
    category_csv = OUTPUT_DIR / 'category_analysis.csv'
    category_df.to_csv(category_csv, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved: {category_csv.name}")
    
    # Save recommendations
    if not recommendations_df.empty:
        rec_csv = OUTPUT_DIR / 'feature_recommendations.csv'
        recommendations_df.to_csv(rec_csv, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {rec_csv.name}")
        
        # Save separate recommendations by action
        for action in recommendations_df['action'].unique():
            action_df = recommendations_df[recommendations_df['action'] == action]
            if len(action_df) > 0:
                action_csv = OUTPUT_DIR / f'feature_recommendations_{action.lower()}.csv'
                action_df.to_csv(action_csv, index=False, encoding='utf-8-sig')
                print(f"   ✅ Saved: {action_csv.name}")
    
    # Save bottom features analysis
    for model_name, df in bottom_features.items():
        bottom_csv = OUTPUT_DIR / f'bottom_features_{model_name}.csv'
        df.to_csv(bottom_csv, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {bottom_csv.name}")
    
    # Save summary JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'models_analyzed': list(importance_dfs.keys()),
        'total_features': {name: int(len(df)) for name, df in importance_dfs.items()},
        'layer2_features_count': {
            name: int((df['layer'] == 'Layer 2').sum())
            for name, df in importance_dfs.items()
        },
        'layer2_importance_ratio': {
            name: float(analysis['layer2_ratio'])
            for name, analysis in category_analysis.items()
        }
    }
    
    summary_json = OUTPUT_DIR / 'analysis_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Saved: {summary_json.name}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
