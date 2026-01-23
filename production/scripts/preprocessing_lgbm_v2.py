#!/usr/bin/env python3
"""
Preprocessing functions for lgbm_muscular_v2 model.

Extracted from scripts/train_models_seasonal_combined.py to ensure
consistent preprocessing between training and inference.
"""

import os
import pandas as pd
from tqdm import tqdm


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


def prepare_data(df, cache_file=None, use_cache=True, timelines_file_mtime=None):
    """
    Prepare data with basic preprocessing (no feature selection) and optional caching.
    
    Args:
        df: Input DataFrame with timelines data
        cache_file: Optional path to cache file
        use_cache: Whether to use cache if available
        timelines_file_mtime: Modification time of timelines file to invalidate cache if changed
    """
    
    # Check cache with validation
    if use_cache and cache_file and os.path.exists(cache_file):
        # If timelines_file_mtime is provided, check if cache is stale
        if timelines_file_mtime is not None:
            cache_mtime = os.path.getmtime(cache_file)
            if cache_mtime < timelines_file_mtime:
                print(f"   [CACHE] Cache is stale (timelines file modified), removing cache...")
                try:
                    os.remove(cache_file)
                    print(f"   [CACHE] Removed stale cache file")
                except Exception as e:
                    print(f"   [WARNING] Failed to remove stale cache: {e}")
            else:
                # Cache is still valid, load it
                print(f"   [CACHE] Loading preprocessed data from cache: {cache_file}...")
                try:
                    df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
                    X = df_preprocessed.drop(columns=['target'])
                    y = df_preprocessed['target']
                    print(f"   [CACHE] Loaded preprocessed data from cache")
                    return X, y
                except Exception as e:
                    print(f"   [WARNING] Failed to load cache ({e}), preprocessing fresh...")
        else:
            # Load cache without validation (existing behavior for backward compatibility)
            print(f"   [CACHE] Loading preprocessed data from cache: {cache_file}...")
            try:
                df_preprocessed = pd.read_csv(cache_file, encoding='utf-8-sig', low_memory=False)
                X = df_preprocessed.drop(columns=['target'])
                y = df_preprocessed['target']
                print(f"   [CACHE] Loaded preprocessed data from cache")
                return X, y
            except Exception as e:
                print(f"   [WARNING] Failed to load cache ({e}), preprocessing fresh...")
    
    # Drop non-feature columns
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
    ]
    X = df[feature_columns].copy()
    
    # Handle target column (may not exist for prediction-only data)
    if 'target' in df.columns:
        y = df['target'].copy()
    else:
        y = None
    
    # Handle missing values and encode categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    # Encode categorical features
    if len(categorical_features) > 0:
        print(f"   [PREPROCESS] Processing {len(categorical_features)} categorical features...")
        for feature in tqdm(categorical_features, desc="   Encoding categorical features", unit="feature", leave=False):
            # Fill NaN first
            X_encoded[feature] = X_encoded[feature].fillna('Unknown')
            
            # Check for problematic values before cleaning (including spaces and &)
            problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
            problematic_count = problematic_mask.sum()
            if problematic_count > 0:
                problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
                print(f"\n   [WARNING] Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
            
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
        print(f"   [PREPROCESS] Processing {len(numeric_features)} numeric features...")
        for feature in tqdm(numeric_features, desc="   Filling numeric missing values", unit="feature", leave=False):
            X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names (including original numeric features)
    X_encoded.columns = [sanitize_feature_name(col) for col in X_encoded.columns]
    
    # Cache preprocessed data
    if use_cache and cache_file and y is not None:
        print(f"   [CACHE] Caching preprocessed data to {cache_file}...")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df_preprocessed = pd.concat([X_encoded, y], axis=1)
            df_preprocessed.to_csv(cache_file, index=False, encoding='utf-8-sig')
            print(f"   [CACHE] Preprocessed data cached")
        except Exception as e:
            print(f"   [WARNING] Failed to cache ({e})")
    
    return X_encoded, y


def align_features_to_model(X, model_columns):
    """
    Align features in X to match model_columns.
    
    Args:
        X: DataFrame with features
        model_columns: List of column names expected by the model
        
    Returns:
        X_aligned: DataFrame with features aligned to model_columns
    """
    # Get features that exist in both
    available_features = [col for col in model_columns if col in X.columns]
    missing_features = [col for col in model_columns if col not in X.columns]
    
    if missing_features:
        print(f"   [ALIGN] Missing {len(missing_features)} features from model (will be set to 0)")
        # Add missing features with zeros
        missing_df = pd.DataFrame(0, index=X.index, columns=missing_features)
        X = pd.concat([X, missing_df], axis=1)
    
    # Also check for extra features in X that aren't in model
    extra_features = [col for col in X.columns if col not in model_columns]
    if extra_features:
        print(f"   [ALIGN] Found {len(extra_features)} extra features in data (will be dropped)")
    
    # Select only model features in correct order
    X_aligned = X[model_columns].copy()
    
    print(f"   [ALIGN] Aligned features: {X_aligned.shape[1]} features")
    print(f"   [ALIGN] Samples: {X_aligned.shape[0]:,}")
    
    return X_aligned



