#!/usr/bin/env python3
"""
Train baseline models on seasonal combined datasets - Muscular Injuries Only
- Training: All seasons (2000-2025) with 50% target ratio (combined, no validation split)
- Test: Season 2025-2026 (natural target ratio)
- Approach: Baseline with correlation filtering (threshold=0.8)
- Models: Random Forest, Gradient Boosting, XGBoost, and LightGBM
- Target: Muscular injuries only
- Optimizations: Caching for correlation filtering and preprocessed data
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import glob
import hashlib
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm

def serialize_column_names(columns):
    """Convert pandas Index/column names to JSON-serializable list of strings"""
    column_names = []
    for col in columns:
        col_str = str(col)
        # Handle bytes objects (shouldn't happen, but just in case)
        if isinstance(col_str, bytes):
            col_str = col_str.decode('utf-8', errors='replace')
        column_names.append(col_str)
    return column_names

def clean_categorical_value(value):
    """Clean categorical values to remove special characters that cause issues in feature names"""
    if pd.isna(value) or value is None:
        return 'Unknown'
    
    value_str = str(value).strip()
    
    # Handle data quality issues - common problematic values
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    
    # Replace special characters that cause issues in column names
    # These characters will become part of one-hot encoded column names
    # LightGBM doesn't like: :, ', ", \, /, &, spaces, and other special chars
    replacements = {
        ':': '_',
        "'": '_',
        ',': '_',
        '"': '_',
        ';': '_',
        '/': '_',
        '\\': '_',
        '{': '_',
        '}': '_',
        '[': '_',
        ']': '_',
        '(': '_',
        ')': '_',
        '|': '_',
        '&': '_',
        '?': '_',
        '!': '_',
        '*': '_',
        '+': '_',
        '=': '_',
        '@': '_',
        '#': '_',
        '$': '_',
        '%': '_',
        '^': '_',
        ' ': '_',  # Replace spaces with underscores (LightGBM doesn't like spaces)
    }
    
    for old_char, new_char in replacements.items():
        value_str = value_str.replace(old_char, new_char)
    
    # Remove any remaining control characters
    value_str = ''.join(char for char in value_str if ord(char) >= 32 or char in '\n\r\t')
    
    # Remove multiple consecutive underscores
    while '__' in value_str:
        value_str = value_str.replace('__', '_')
    
    # Remove leading/trailing underscores
    value_str = value_str.strip('_')
    
    # If empty after cleaning, return Unknown
    if not value_str:
        return 'Unknown'
    
    return value_str

def sanitize_feature_name(name):
    """Sanitize feature names to be JSON-safe for LightGBM"""
    # Convert to string
    name_str = str(name)
    # Replace special JSON characters with safe alternatives
    # JSON special chars: ", \, /, \b, \f, \n, \r, \t, and control characters
    # LightGBM also doesn't like: spaces, &, :, ', etc.
    replacements = {
        '"': '_quote_',
        '\\': '_backslash_',
        '/': '_slash_',
        '\b': '_bs_',
        '\f': '_ff_',
        '\n': '_nl_',
        '\r': '_cr_',
        '\t': '_tab_',
        ' ': '_',  # Replace spaces with underscores (LightGBM doesn't like spaces)
        # Also handle other problematic characters
        "'": '_apostrophe_',
        ':': '_colon_',
        ';': '_semicolon_',
        ',': '_comma_',
        '{': '_lbrace_',
        '}': '_rbrace_',
        '[': '_lbracket_',
        ']': '_rbracket_',
        '&': '_amp_',
    }
    for old, new in replacements.items():
        name_str = name_str.replace(old, new)
    # Remove any remaining control characters
    name_str = ''.join(char for char in name_str if ord(char) >= 32 or char in '\n\r\t')
    # Remove multiple consecutive underscores
    while '__' in name_str:
        name_str = name_str.replace('__', '_')
    # Remove leading/trailing underscores
    name_str = name_str.strip('_')
    return name_str

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    # Handle None first
    if obj is None:
        return None
    
    # Handle numpy scalars - use base classes for compatibility with NumPy 2.0+
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Check for NaN
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, 'item'):  # numpy scalar with .item() method (fallback for edge cases)
        try:
            return convert_numpy_types(obj.item())
        except (AttributeError, ValueError):
            pass
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    # Handle NaN (check after float check)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def get_features_hash(feature_names):
    """Create a hash of feature names for caching"""
    feature_str = '_'.join(sorted(feature_names))
    return hashlib.md5(feature_str.encode()).hexdigest()

def load_combined_seasonal_datasets(target_ratio=0.50, exclude_season='2025_2026', min_season=None):
    """
    Load and combine all seasonal datasets with specified target ratio.
    
    Args:
        target_ratio: Target ratio (0.10, 0.25, 0.50, or None for natural/unbalanced ratio)
        exclude_season: Season to exclude (default: '2025_2026' for test)
        min_season: Minimum season to include (e.g., '2017_2018'). If None, includes all seasons.
    
    Returns:
        Combined DataFrame
    """
    if target_ratio is None:
        # Natural ratio: files without _XXpc suffix
        pattern = 'timelines_35day_season_*_v4_muscular.csv'
        files = glob.glob(pattern)
        season_files = []
        
        for filepath in files:
            filename = os.path.basename(filepath)
            # Exclude files with ratio suffixes (10pc, 25pc, 50pc)
            if '_10pc_' in filename or '_25pc_' in filename or '_50pc_' in filename:
                continue
            if 'season_' in filename:
                parts = filename.split('season_')
                if len(parts) > 1:
                    # Extract season from pattern: timelines_35day_season_YYYY_YYYY_v4_muscular.csv
                    season_part = parts[1].split('_v4_muscular')[0]
                    if season_part != exclude_season:
                        # Filter by minimum season if specified
                        if min_season is None or season_part >= min_season:
                            season_files.append((season_part, filepath))
    else:
        # Balanced ratio: files with _XXpc suffix
        ratio_str = f"{int(target_ratio * 100):02d}pc"
        pattern = f'timelines_35day_season_*_{ratio_str}_v4_muscular.csv'
        
        files = glob.glob(pattern)
        season_files = []
        
        for filepath in files:
            filename = os.path.basename(filepath)
            if 'season_' in filename:
                parts = filename.split('season_')
                if len(parts) > 1:
                    season_part = parts[1].split(f'_{ratio_str}')[0]
                    if season_part != exclude_season:
                        # Filter by minimum season if specified
                        if min_season is None or season_part >= min_season:
                            season_files.append((season_part, filepath))
    
    # Sort chronologically
    season_files.sort(key=lambda x: x[0])
    
    if target_ratio is None:
        ratio_display = "natural (unbalanced)"
    else:
        ratio_display = f"{target_ratio:.0%}"
    
    print(f"\n📂 Loading {len(season_files)} season files with {ratio_display} target ratio...")
    if min_season:
        print(f"   (Filtering: Only seasons >= {min_season})")
    
    dfs = []
    total_records = 0
    total_positives = 0
    
    for season_id, filepath in season_files:
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            positives = df['target'].sum()
            if len(df) > 0 and positives > 0:  # Skip empty files AND files with 0 positives
                dfs.append(df)
                total_records += len(df)
                total_positives += positives
                print(f"   ✅ {season_id}: {len(df):,} records ({positives:,} positives)")
            elif len(df) > 0 and positives == 0:
                print(f"   ⚠️  {season_id}: {len(df):,} records (0 positives - skipping)")
        except Exception as e:
            print(f"   ⚠️  Error loading {season_id}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid season files found!")
    
    # Combine all dataframes
    print(f"\n📊 Combining {len(dfs)} season datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"✅ Combined dataset: {len(combined_df):,} records")
    print(f"   Total positives: {total_positives:,} ({combined_df['target'].mean():.2%})")
    print(f"   Total negatives: {len(combined_df) - total_positives:,}")
    
    return combined_df

def prepare_data(df, cache_file=None, use_cache=True):
    """Prepare data with basic preprocessing (no feature selection) and optional caching"""
    
    # Check cache
    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"   📦 Loading preprocessed data from cache: {cache_file}...")
        try:
            df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
            X = df_preprocessed.drop(columns=['target'])
            y = df_preprocessed['target']
            print(f"   ✅ Loaded preprocessed data from cache")
            return X, y
        except Exception as e:
            print(f"   ⚠️  Failed to load cache ({e}), preprocessing fresh...")
    
    # Drop non-feature columns
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
    ]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    if len(categorical_features) > 0:
        print(f"   Processing {len(categorical_features)} categorical features...")
        for feature in tqdm(categorical_features, desc="   Encoding categorical features", unit="feature", leave=False):
            # Fill NaN first
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            
            # Check for problematic values before cleaning (including spaces and &)
            problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
            problematic_count = problematic_mask.sum()
            if problematic_count > 0:
                problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
                print(f"\n⚠️  Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
            
            # Clean categorical values BEFORE one-hot encoding
            # This ensures column names created by get_dummies are safe
            X_encoded[feature] = X_encoded[feature].apply(clean_categorical_value)
            
            # Now one-hot encode (column names will be safe)
            dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
            
            # Double-check: sanitize dummy column names (belt and suspenders)
            dummies.columns = [sanitize_feature_name(col) for col in dummies.columns]
            
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded = X_encoded.drop(columns=[feature])
    
    # Fill numeric missing values
    if len(numeric_features) > 0:
        print(f"   Processing {len(numeric_features)} numeric features...")
        for feature in tqdm(numeric_features, desc="   Filling numeric missing values", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names (including original numeric features)
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    # Cache preprocessed data
    if use_cache and cache_file:
        print(f"   💾 Caching preprocessed data to {cache_file}...")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df_preprocessed = pd.concat([X_encoded, y], axis=1)
            df_preprocessed.to_csv(cache_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ Preprocessed data cached")
        except Exception as e:
            print(f"   ⚠️  Failed to cache ({e})")
    
    return X_encoded, y

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    # Get common features across both datasets
    common_features = list(set(X_train.columns) & set(X_test.columns))
    
    # Sort for consistency
    common_features = sorted(common_features)
    
    print(f"   Aligning features: {len(common_features)} common features")
    print(f"   Training: {X_train.shape[1]} -> {len(common_features)}")
    print(f"   Test: {X_test.shape[1]} -> {len(common_features)}")
    
    return X_train[common_features], X_test[common_features]

def apply_correlation_filter(X, threshold=0.8, cache_dir='cache', use_cache=True):
    """Drop one feature from each highly correlated pair with improved caching."""
    print(f"\n🔎 Applying correlation filter (threshold={threshold:.2f}) on training data...")
    print(f"   Computing correlation matrix for {X.shape[1]:,} features...")
    print("   (This may take a few minutes for large datasets)")
    
    # Create cache key based on feature names (not row count)
    features_hash = get_features_hash(X.columns.tolist())
    cache_file_corr = os.path.join(cache_dir, f'corr_matrix_{features_hash}_{threshold}.npy')
    cache_file_features = os.path.join(cache_dir, f'selected_features_{features_hash}_{threshold}.json')
    
    # Check if we have cached selected features (fastest path)
    if use_cache and os.path.exists(cache_file_features):
        print(f"   📦 Loading cached selected features from {cache_file_features}...")
        try:
            with open(cache_file_features, 'r') as f:
                kept = json.load(f)
            # Verify these features exist in current data
            kept = [f for f in kept if f in X.columns]
            if len(kept) > 0:
                print(f"   ✅ Loaded {len(kept)} cached features (skipping correlation computation)")
                return kept
        except Exception as e:
            print(f"   ⚠️  Failed to load cached features ({e}), computing fresh...")
    
    # Check for cached correlation matrix
    corr_matrix = None
    if use_cache and os.path.exists(cache_file_corr):
        print(f"   📦 Loading cached correlation matrix from {cache_file_corr}...")
        try:
            corr_matrix = np.load(cache_file_corr)
            corr_matrix = pd.DataFrame(corr_matrix, index=X.columns, columns=X.columns)
            print(f"   ✅ Loaded cached correlation matrix")
        except Exception as e:
            print(f"   ⚠️  Failed to load cache ({e}), computing fresh...")
            corr_matrix = None
    
    # Compute correlation matrix if not cached
    if corr_matrix is None:
        corr_start = datetime.now()
        if X.shape[1] > 1000:
            print(f"   Computing correlation matrix for {X.shape[1]:,} features (this may take several minutes)...")
        else:
            print(f"   Computing correlation matrix for {X.shape[1]:,} features...")
        corr_matrix = X.corr().abs()
        corr_time = datetime.now() - corr_start
        print(f"   ✅ Correlation matrix computed in {corr_time}")
        
        # Cache the correlation matrix
        if use_cache:
            print(f"   💾 Caching correlation matrix to {cache_file_corr}...")
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_file_corr, corr_matrix.values)
                print(f"   ✅ Correlation matrix cached")
            except Exception as e:
                print(f"   ⚠️  Failed to cache ({e})")
    
    print("   Identifying highly correlated pairs...")
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    
    print(f"   Removed {len(to_drop)} features due to correlation > {threshold}")
    print(f"   Remaining features: {len(kept)}")
    
    # Cache selected features list
    if use_cache:
        print(f"   💾 Caching selected features list to {cache_file_features}...")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file_features, 'w') as f:
                json.dump(kept, f, indent=2)
            print(f"   ✅ Selected features cached")
        except Exception as e:
            print(f"   ⚠️  Failed to cache features list ({e})")
    
    return kept

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    # For large datasets, show progress during prediction
    if len(X) > 100000:
        # Predict in chunks with progress bar
        chunk_size = 50000
        y_pred_list = []
        y_proba_list = []
        
        num_chunks = (len(X) + chunk_size - 1) // chunk_size
        for i in tqdm(range(0, len(X), chunk_size), 
                     desc=f"      Predicting {dataset_name}", 
                     unit="chunk",
                     total=num_chunks,
                     leave=False):
            chunk_X = X.iloc[i:i+chunk_size]
            y_pred_list.append(model.predict(chunk_X))
            y_proba_list.append(model.predict_proba(chunk_X)[:, 1])
        
        y_pred = np.concatenate(y_pred_list)
        y_proba = np.concatenate(y_proba_list)
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics (may contain numpy types)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        'gini': (2 * roc_auc_score(y, y_proba) - 1) if len(np.unique(y)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1]
        }
    else:
        # Handle edge case
        if len(cm) == 1:
            if y.sum() == 0:
                metrics['confusion_matrix'] = {'tn': cm[0, 0], 'fp': 0, 'fn': 0, 'tp': 0}
            else:
                metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': y.sum(), 'tp': 0}
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    # Convert all numpy types to Python native types recursively
    metrics = convert_numpy_types(metrics)
    
    print(f"\n   📊 {dataset_name} Results:")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1-Score: {metrics['f1']:.4f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"      Gini: {metrics['gini']:.4f}")
    print(f"      TP: {metrics['confusion_matrix']['tp']}, FP: {metrics['confusion_matrix']['fp']}, "
          f"TN: {metrics['confusion_matrix']['tn']}, FN: {metrics['confusion_matrix']['fn']}")
    
    return metrics

def train_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        max_features='sqrt'
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {rf_model.n_estimators}")
    print(f"   max_depth: {rf_model.max_depth}")
    print(f"   min_samples_split: {rf_model.min_samples_split}")
    print(f"   min_samples_leaf: {rf_model.min_samples_leaf}")
    print(f"   max_features: {rf_model.max_features}")
    print(f"   class_weight: {rf_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    rf_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on training and test sets
    train_metrics = evaluate_model(rf_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test")
    
    return rf_model, train_metrics, test_metrics

def train_gb(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {gb_model.n_estimators}")
    print(f"   max_depth: {gb_model.max_depth}")
    print(f"   learning_rate: {gb_model.learning_rate}")
    print(f"   min_samples_split: {gb_model.min_samples_split}")
    print(f"   min_samples_leaf: {gb_model.min_samples_leaf}")
    print(f"   subsample: {gb_model.subsample}")
    print(f"   max_features: {gb_model.max_features}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    gb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on training and test sets
    train_metrics = evaluate_model(gb_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(gb_model, X_test, y_test, "Test")
    
    return gb_model, train_metrics, test_metrics

def train_xgb(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING XGBOOST MODEL")
    print("=" * 70)
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=11.5,  # Approximate for 8% positive class (1/0.08 - 1)
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=1  # Show progress during training
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {xgb_model.n_estimators}")
    print(f"   max_depth: {xgb_model.max_depth}")
    print(f"   learning_rate: {xgb_model.learning_rate}")
    print(f"   min_child_weight: {xgb_model.min_child_weight}")
    print(f"   subsample: {xgb_model.subsample}")
    print(f"   colsample_bytree: {xgb_model.colsample_bytree}")
    print(f"   reg_alpha: {xgb_model.reg_alpha}")
    print(f"   reg_lambda: {xgb_model.reg_lambda}")
    print(f"   scale_pos_weight: {xgb_model.scale_pos_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    xgb_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on training and test sets
    train_metrics = evaluate_model(xgb_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(xgb_model, X_test, y_test, "Test")
    
    return xgb_model, train_metrics, test_metrics

def train_lgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING LIGHTGBM MODEL")
    print("=" * 70)
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1  # Show progress during training
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {lgbm_model.n_estimators}")
    print(f"   max_depth: {lgbm_model.max_depth}")
    print(f"   learning_rate: {lgbm_model.learning_rate}")
    print(f"   min_child_samples: {lgbm_model.min_child_samples}")
    print(f"   subsample: {lgbm_model.subsample}")
    print(f"   colsample_bytree: {lgbm_model.colsample_bytree}")
    print(f"   reg_alpha: {lgbm_model.reg_alpha}")
    print(f"   reg_lambda: {lgbm_model.reg_lambda}")
    print(f"   class_weight: {lgbm_model.class_weight}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    lgbm_model.fit(X_train, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate on training and test sets
    train_metrics = evaluate_model(lgbm_model, X_train, y_train, "Training")
    test_metrics = evaluate_model(lgbm_model, X_test, y_test, "Test")
    
    return lgbm_model, train_metrics, test_metrics

def main():
    # ========== CONFIGURATION ==========
    TARGET_RATIO = 0.50  # 50% target ratio for training data
    CORR_THRESHOLD = 0.5  # correlation threshold for feature filtering
    EXCLUDE_SEASON = '2025_2026'  # Test dataset season
    MIN_SEASON = None  # Include all seasons (2000-2025)
    USE_CACHE = True
    CACHE_DIR = 'cache'
    PREPROCESS_CACHE = True  # Set to False to skip preprocessing cache (saves disk space)
    # ===================================
    
    ratio_str = f"{int(TARGET_RATIO * 100):02d}pc"
    ratio_display = f"{TARGET_RATIO:.0%} (balanced)"
    ratio_title = f"{TARGET_RATIO:.0%} TARGET RATIO - SEASONAL COMBINED (ALL SEASONS)"
    
    print("="*80)
    print(f"TRAINING BASELINE MODELS - V4 MUSCULAR INJURIES ONLY ({ratio_title})")
    print("="*80)
    print("\n📋 Dataset Configuration:")
    if MIN_SEASON:
        print(f"   Training: Seasons {MIN_SEASON} onwards with {TARGET_RATIO:.0%} target ratio (combined, no validation split)")
    else:
        print(f"   Training: All seasons (2000-2025) with {TARGET_RATIO:.0%} target ratio (combined, no validation split)")
    print(f"   Validation: None (train on all data, test on 2025-2026)")
    print(f"   Test: Season 2025-2026 (natural target ratio)")
    print("   Target: Muscular injuries only")
    print(f"   Target ratio: {ratio_display}")
    print(f"   Approach: Baseline with correlation filtering (threshold={CORR_THRESHOLD})")
    print("   Models: Random Forest, Gradient Boosting, XGBoost, and LightGBM")
    print(f"   Caching: {'Enabled' if USE_CACHE else 'Disabled'}")
    print("="*80)
    
    start_time = datetime.now()
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Estimated total time: 2-3 hours")
    print(f"💡 Progress will be shown for each major step\n")
    
    # Load data
    print("\n📂 Loading timeline data...")
    
    # Load combined training data from all seasons (10% ratio)
    df_train_full = load_combined_seasonal_datasets(
        target_ratio=TARGET_RATIO, 
        exclude_season=EXCLUDE_SEASON,
        min_season=MIN_SEASON
    )
    
    # Load test data (2025-2026 season with natural ratio)
    test_file = 'timelines_35day_season_2025_2026_v4_muscular.csv'
    print(f"\n📂 Loading test dataset: {test_file}...")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    
    print(f"✅ Loaded combined training set: {len(df_train_full):,} records")
    print(f"   Injury ratio: {df_train_full['target'].mean():.1%}")
    print(f"✅ Loaded test set: {len(df_test):,} records")
    print(f"   Injury ratio: {df_test['target'].mean():.1%}")
    
    # Use all data for training (no validation split)
    df_train = df_train_full
    print(f"\n✅ Training set: {len(df_train):,} records ({df_train['target'].mean():.1%} injury ratio)")
    
    # Prepare data with optional caching
    print("\n📊 Preparing data...")
    prep_start = datetime.now()
    
    # Create cache filenames based on dataset hash
    train_hash = hashlib.md5(str(len(df_train)).encode()).hexdigest()[:8]
    test_hash = hashlib.md5(str(len(df_test)).encode()).hexdigest()[:8]
    
    train_cache = os.path.join(CACHE_DIR, f'preprocessed_train_seasonal_{train_hash}.csv') if PREPROCESS_CACHE else None
    test_cache = os.path.join(CACHE_DIR, f'preprocessed_test_seasonal_{test_hash}.csv') if PREPROCESS_CACHE else None
    
    print("   Preparing training set...")
    X_train, y_train = prepare_data(df_train, cache_file=train_cache, use_cache=USE_CACHE)
    print("   Preparing test set...")
    X_test, y_test = prepare_data(df_test, cache_file=test_cache, use_cache=USE_CACHE)
    prep_time = datetime.now() - prep_start
    print(f"✅ Data preparation completed in {prep_time}")
    print(f"   Training features: {X_train.shape[1]}")
    print(f"   Test features: {X_test.shape[1]}")
    
    # Align features (with timing)
    print("\n🔧 Aligning features across datasets...")
    align_start = datetime.now()
    X_train, X_test = align_features(X_train, X_test)
    align_time = datetime.now() - align_start
    print(f"✅ Feature alignment completed in {align_time}")
    
    initial_feature_count = X_train.shape[1]
    print(f"\n✅ Prepared data: {initial_feature_count} features (aligned across all datasets)")
    
    # ALWAYS sanitize feature names for LightGBM (even if JSON.dumps works, LightGBM is stricter)
    print("\n🔧 Sanitizing all feature names for LightGBM compatibility...")
    sanitize_start = datetime.now()
    X_train.columns = [sanitize_feature_name(col) for col in X_train.columns]
    X_test.columns = [sanitize_feature_name(col) for col in X_test.columns]
    sanitize_time = datetime.now() - sanitize_start
    print(f"✅ Sanitized {len(X_train.columns)} feature names in {sanitize_time}")
    
    # Apply correlation filter (with timing and improved caching)
    corr_start = datetime.now()
    selected_features = apply_correlation_filter(X_train, CORR_THRESHOLD, cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    corr_time = datetime.now() - corr_start
    print(f"\n✅ After correlation filtering: {len(selected_features)} features (removed {initial_feature_count - len(selected_features)} features)")
    print(f"   Total correlation filtering time: {corr_time}")
    
    # OPTIMIZATION: Serialize column names ONCE (instead of 4 times)
    print("\n💾 Serializing feature names (once for all models)...")
    serialize_start = datetime.now()
    column_names = serialize_column_names(X_train.columns)
    serialize_time = datetime.now() - serialize_start
    print(f"✅ Serialized {len(column_names)} feature names in {serialize_time}")
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    all_results = {}
    
    # Define models to train (for easier iteration)
    # Here we only retrain the Random Forest model for the 50% / corr=0.5 all-seasons setup
    models_to_train = [
        ('RF', train_rf, 'rf'),
    ]
    
    os.makedirs('models', exist_ok=True)
    
    for idx, (model_name, train_func, model_file_prefix) in enumerate(models_to_train, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {idx}/{len(models_to_train)}: {model_name}")
        print(f"{'='*80}")
        
        # Calculate elapsed time and estimate remaining
        elapsed = datetime.now() - start_time
        if idx > 1:
            avg_time_per_model = elapsed / (idx - 1)
            remaining_models = len(models_to_train) - idx + 1
            estimated_remaining = avg_time_per_model * remaining_models
            print(f"⏱️  Elapsed: {elapsed} | Estimated remaining: ~{estimated_remaining}")
        
        # Train model
        model, train_metrics, test_metrics = train_func(
            X_train, y_train,
            X_test, y_test
        )
        
        all_results[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        # Save model
        model_file = f'models/{model_file_prefix}_model_seasonal_50pc_v4_muscular_corr05.joblib'
        joblib.dump(model, model_file)
        
        # Save column names (REUSE the same serialized list - optimization!)
        columns_file = f'models/{model_file_prefix}_model_seasonal_50pc_v4_muscular_corr05_columns.json'
        with open(columns_file, 'w', encoding='utf-8') as f:
            json.dump(column_names, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {model_name} model to {model_file}")
    
    # Save metrics
    os.makedirs('experiments', exist_ok=True)
    metrics_file = f'experiments/seasonal_50pc_allseasons_corr05_metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved metrics to {metrics_file}")
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT - PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    print("\n" + "="*80)
    print("PERFORMANCE METRICS TABLE")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12} {'Gini':<12}")
    print("-" * 100)
    
    for model_name in ['RF']:
        results = all_results[model_name]
        print(f"{model_name:<10} {'Training':<30} {results['train']['precision']:<12.4f} {results['train']['recall']:<12.4f} {results['train']['f1']:<12.4f} {results['train']['roc_auc']:<12.4f} {results['train']['gini']:<12.4f}")
        print(f"{'':<10} {'Test':<30} {results['test']['precision']:<12.4f} {results['test']['recall']:<12.4f} {results['test']['f1']:<12.4f} {results['test']['roc_auc']:<12.4f} {results['test']['gini']:<12.4f}")
        print("-" * 100)
    
    # Create markdown summary
    summary_lines = [
        f"# Seasonal Combined Datasets - {TARGET_RATIO:.0%} Target Ratio - Model Performance (with Correlation Filtering, threshold={CORR_THRESHOLD})",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Split",
        "",
        f"- **Training:** All seasons (2000-2025) with {TARGET_RATIO:.0%} target ratio, combined ({len(df_train):,} records, {df_train['target'].mean():.1%} injury ratio)",
        f"- **Validation:** None (trained on all data)",
        f"- **Test:** Season 2025-2026 with natural target ratio ({len(df_test):,} records, {df_test['target'].mean():.1%} injury ratio)",
        "",
        "## Target",
        "",
        "- **Injury Type:** Muscular injuries only",
        "",
        "## Approach",
        "",
        f"- **Correlation filtering:** Threshold = {CORR_THRESHOLD} (removed one feature from each highly correlated pair)",
        f"- **Features:** {len(selected_features)} features (after correlation filtering, down from {initial_feature_count} initial features)",
        "- **Models:** Random Forest, Gradient Boosting, XGBoost, and LightGBM",
        "",
        "## Performance Metrics",
        ""
    ]
    
    # Create comparison table
    summary_lines.append("| Model | Dataset | Precision | Recall | F1-Score | ROC AUC | Gini |")
    summary_lines.append("|-------|---------|-----------|--------|----------|---------|------|")
    
    for model_name in ['RF']:
        results = all_results[model_name]
        summary_lines.append(f"| **{model_name}** | Training | {results['train']['precision']:.4f} | {results['train']['recall']:.4f} | {results['train']['f1']:.4f} | {results['train']['roc_auc']:.4f} | {results['train']['gini']:.4f} |")
        summary_lines.append(f"| | Test | {results['test']['precision']:.4f} | {results['test']['recall']:.4f} | {results['test']['f1']:.4f} | {results['test']['roc_auc']:.4f} | {results['test']['gini']:.4f} |")
        summary_lines.append("| | | | | | | |")
    
    # Add gap analysis
    summary_lines.extend([
        "",
        "## Performance Gaps",
        ""
    ])
    
    for model_name in ['RF']:
        results = all_results[model_name]
        train_f1 = results['train']['f1']
        test_f1 = results['test']['f1']
        
        summary_lines.append(f"### {model_name}")
        summary_lines.append(f"- **F1 Gap (Train → Test):** {train_f1 - test_f1:.4f} ({(train_f1 - test_f1)/train_f1*100:.1f}% relative)" if train_f1 > 0 else "- **F1 Gap (Train → Test):** N/A")
        summary_lines.append("")
    
    # Save summary
    summary_text = "\n".join(summary_lines)
    summary_file = f'experiments/seasonal_50pc_allseasons_corr05_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n✅ Saved summary report to {summary_file}")
    
    # Create detailed table with confusion matrix
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE TABLE (with Confusion Matrix)")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 135)
    
    for model_name in ['RF']:
        results = all_results[model_name]
        for dataset_name, metrics in [('Training', results['train']), 
                                       ('Test', results['test'])]:
            cm = metrics['confusion_matrix']
            print(f"{model_name:<10} {dataset_name:<30} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f} {cm['tp']:<8} {cm['fp']:<8} {cm['tn']:<8} {cm['fn']:<8}")
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

