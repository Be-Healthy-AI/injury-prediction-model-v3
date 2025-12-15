#!/usr/bin/env python3
"""
Complete LightGBM training and update metrics file with all 4 models
This script will:
1. Load existing RF and GB models and re-evaluate them
2. Load existing XGBoost model and evaluate it
3. Train LightGBM model
4. Save all metrics together
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
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_lgbm_training_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def get_type_info(obj):
    """Get detailed type information for debugging"""
    obj_type = type(obj).__name__
    module = type(obj).__module__
    return f"{module}.{obj_type}"

def convert_numpy_types(obj, path="root", max_depth=10):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if max_depth <= 0:
        logger.warning(f"Max depth reached at path: {path}")
        return obj
    
    # Handle None first
    if obj is None:
        return None
    
    # Log the type we're processing (only for first level or problematic types)
    if path == "root" or path.count('.') < 2:
        logger.debug(f"Converting at {path}: {get_type_info(obj)}")
    
    # Handle numpy scalars - use base classes for compatibility with NumPy 2.0+
    if isinstance(obj, np.integer):
        result = int(obj)
        logger.debug(f"  Converted numpy integer at {path}: {obj} -> {result} ({type(result).__name__})")
        return result
    elif isinstance(obj, np.floating):
        # Check for NaN
        if np.isnan(obj):
            logger.debug(f"  Found NaN at {path}, converting to None")
            return None
        result = float(obj)
        logger.debug(f"  Converted numpy float at {path}: {obj} -> {result} ({type(result).__name__})")
        return result
    elif isinstance(obj, (np.bool_, bool)):
        result = bool(obj)
        logger.debug(f"  Converted numpy bool at {path}: {obj} -> {result} ({type(result).__name__})")
        return result
    elif isinstance(obj, np.ndarray):
        result = obj.tolist()
        logger.debug(f"  Converted numpy array at {path}: shape {obj.shape} -> list of length {len(result)}")
        return [convert_numpy_types(item, f"{path}[{i}]", max_depth-1) for i, item in enumerate(result)]
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        result = obj.tolist()
        logger.debug(f"  Converted pandas Series at {path}: length {len(obj)} -> list")
        return [convert_numpy_types(item, f"{path}[{i}]", max_depth-1) for i, item in enumerate(result)]
    elif isinstance(obj, pd.DataFrame):
        result = obj.to_dict('records')
        logger.debug(f"  Converted pandas DataFrame at {path}: shape {obj.shape} -> dict list")
        return [convert_numpy_types(item, f"{path}[{i}]", max_depth-1) for i, item in enumerate(result)]
    elif hasattr(obj, 'item'):  # numpy scalar with .item() method (fallback for edge cases)
        try:
            item_result = obj.item()
            logger.debug(f"  Using .item() method at {path}: {get_type_info(obj)} -> {get_type_info(item_result)}")
            return convert_numpy_types(item_result, path, max_depth-1)
        except (AttributeError, ValueError) as e:
            logger.warning(f"  .item() method failed at {path}: {e}")
            pass
    elif isinstance(obj, dict):
        logger.debug(f"  Processing dict at {path} with {len(obj)} keys")
        result = {}
        for key, value in obj.items():
            key_str = str(key)
            try:
                result[key_str] = convert_numpy_types(value, f"{path}.{key_str}", max_depth-1)
            except Exception as e:
                logger.error(f"  ERROR converting dict value at {path}.{key_str}: {e}")
                logger.error(f"    Value type: {get_type_info(value)}")
                logger.error(f"    Value: {repr(value)[:200]}")
                raise
        return result
    elif isinstance(obj, (list, tuple)):
        logger.debug(f"  Processing {type(obj).__name__} at {path} with {len(obj)} items")
        result = []
        for i, item in enumerate(obj):
            try:
                result.append(convert_numpy_types(item, f"{path}[{i}]", max_depth-1))
            except Exception as e:
                logger.error(f"  ERROR converting list item at {path}[{i}]: {e}")
                logger.error(f"    Item type: {get_type_info(item)}")
                logger.error(f"    Item: {repr(item)[:200]}")
                raise
        return result
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        result = str(obj)
        logger.debug(f"  Converted pandas timestamp/timedelta at {path}: {obj} -> {result}")
        return result
    # Handle NaN (check after float check)
    elif isinstance(obj, float) and np.isnan(obj):
        logger.debug(f"  Found float NaN at {path}, converting to None")
        return None
    else:
        # Check if it's a type we haven't handled
        if not isinstance(obj, (str, int, float, bool)):
            logger.debug(f"  Unhandled type at {path}: {get_type_info(obj)} - returning as-is")
        return obj

def clean_categorical_value(value, replace_spaces=True):
    """Clean categorical values to remove special characters that cause issues in feature names
    
    Args:
        value: The categorical value to clean
        replace_spaces: If True, replace spaces with underscores (for LightGBM compatibility)
                       If False, keep spaces (for compatibility with existing models)
    """
    if pd.isna(value) or value is None:
        return 'Unknown'
    
    value_str = str(value).strip()
    
    # Handle data quality issues - common problematic values
    problematic_values = ['tel:', 'tel', 'phone', 'n/a', 'na', 'null', 'none', '', 'nan', 'address:', 'website:']
    if value_str.lower() in problematic_values:
        return 'Unknown'
    
    # CRITICAL: Replace spaces FIRST if requested (for LightGBM)
    if replace_spaces:
        value_str = value_str.replace(' ', '_')
    
    # Replace special characters that cause issues in column names
    # These characters will become part of one-hot encoded column names
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

def sanitize_feature_name(name, replace_spaces=True):
    """Sanitize feature names to be JSON-safe for LightGBM
    
    Args:
        name: Feature name to sanitize
        replace_spaces: If True, replace spaces with underscores (for LightGBM).
                       If False, keep spaces (for compatibility with existing models).
    """
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
    # Only replace spaces if requested (for LightGBM compatibility)
    if replace_spaces:
        replacements[' '] = '_'
    
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

def prepare_data(df, replace_spaces_in_values=True):
    """Prepare data with basic preprocessing (same as main script)
    
    Args:
        df: Input dataframe
        replace_spaces_in_values: If True, replace spaces in categorical values (for LightGBM)
                                  If False, keep spaces (for compatibility with existing models)
    """
    feature_columns = [
        col for col in df.columns
        if col not in ['player_id', 'reference_date', 'player_name', 'target']
    ]
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_encoded = X.copy()
    
    for feature in categorical_features:
        # Fill NaN first
        X_encoded[feature] = X_encoded[feature].fillna('Unknown')
        
        # Check for problematic values before cleaning (including spaces and &)
        problematic_mask = X_encoded[feature].astype(str).str.contains(r'[:,\'"\\/;|&?!*+=@#$%^\s]', regex=True, na=False)
        problematic_count = problematic_mask.sum()
        if problematic_count > 0:
            problematic_values = X_encoded[feature][problematic_mask].unique()[:10]
            logger.warning(f"Found {problematic_count} problematic values in '{feature}': {list(problematic_values)}")
        
        # Clean categorical values BEFORE one-hot encoding
        # This ensures column names created by get_dummies are safe
        X_encoded[feature] = X_encoded[feature].apply(lambda x: clean_categorical_value(x, replace_spaces=replace_spaces_in_values))
        
        # CRITICAL: Verify no spaces remain in values before get_dummies (if replace_spaces_in_values is True)
        if replace_spaces_in_values:
            spaces_in_values = X_encoded[feature].astype(str).str.contains(' ', regex=False, na=False).sum()
            if spaces_in_values > 0:
                logger.error(f"ERROR: Found {spaces_in_values} values with spaces in '{feature}' after cleaning!")
                # Force clean again
                X_encoded[feature] = X_encoded[feature].apply(lambda x: str(x).replace(' ', '_'))
        
        # Now one-hot encode (column names will be safe if values were cleaned)
        dummies = pd.get_dummies(X_encoded[feature], prefix=feature, drop_first=True)
        
        # CRITICAL: Sanitize dummy column names IMMEDIATELY after get_dummies
        if replace_spaces_in_values:
            # For LightGBM: replace ALL spaces immediately
            dummies.columns = [str(col).replace(' ', '_') for col in dummies.columns]
            dummies.columns = [sanitize_feature_name(col, replace_spaces=True) for col in dummies.columns]
            # Final check: force replace any remaining spaces
            dummies.columns = [str(col).replace(' ', '_') for col in dummies.columns]
        else:
            # For existing models: only sanitize special chars, KEEP SPACES
            # This is critical - existing models were trained with spaces in feature names
            dummies.columns = [sanitize_feature_name(col, replace_spaces=False) for col in dummies.columns]
        
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(columns=[feature])
    
    for feature in numeric_features:
        X_encoded[feature] = X_encoded[feature].fillna(0)
    
    # Sanitize all column names (including original numeric features)
    if replace_spaces_in_values:
        # For LightGBM: replace ALL spaces
        X_encoded.columns = [str(col).replace(' ', '_') for col in X_encoded.columns]
        X_encoded.columns = [sanitize_feature_name(col, replace_spaces=True) for col in X_encoded.columns]
        # Final check: force replace any remaining spaces
        X_encoded.columns = [str(col).replace(' ', '_') for col in X_encoded.columns]
        
        # Verify no spaces remain
        spaces_found = [col for col in X_encoded.columns if ' ' in str(col)]
        if spaces_found:
            logger.error(f"ERROR: Found {len(spaces_found)} columns with spaces after sanitization in prepare_data!")
            logger.error(f"First 10: {spaces_found[:10]}")
            # Force fix
            X_encoded.columns = [str(col).replace(' ', '_') for col in X_encoded.columns]
    else:
        # For existing models: only sanitize special chars, KEEP SPACES
        # This is critical - existing models were trained with spaces in feature names
        X_encoded.columns = [sanitize_feature_name(col, replace_spaces=False) for col in X_encoded.columns]
    
    logger.info(f"Prepared data with {len(X_encoded.columns)} features")
    logger.debug(f"Sample feature names: {list(X_encoded.columns[:10])}")
    
    return X_encoded, y

def align_features(X_train, X_test):
    """Ensure all datasets have the same features"""
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features = sorted(common_features)
    return X_train[common_features], X_test[common_features]

def apply_correlation_filter(X, threshold=0.8):
    """Drop one feature from each highly correlated pair"""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [col for col in X.columns if col not in to_drop]
    return kept

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    logger.info(f"Starting prediction for {dataset_name} ({len(X):,} samples)...")
    y_pred = model.predict(X)
    logger.info(f"Prediction complete for {dataset_name}, starting probability prediction...")
    y_proba = model.predict_proba(X)[:, 1]
    logger.info(f"Probability prediction complete for {dataset_name}")
    
    # Calculate metrics (may contain numpy types)
    # Check if we have both classes
    unique_classes = np.unique(y)
    has_both_classes = len(unique_classes) > 1
    
    if has_both_classes:
        roc_auc = roc_auc_score(y, y_proba)
        gini = 2 * roc_auc - 1
    else:
        roc_auc = 0.0
        gini = 0.0
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc,
        'gini': gini
    }
    
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    else:
        if len(cm) == 1:
            y_sum = int(y.sum()) if hasattr(y, 'sum') else int(sum(y))
            if y_sum == 0:
                metrics['confusion_matrix'] = {'tn': int(cm[0, 0]), 'fp': 0, 'fn': 0, 'tp': 0}
            else:
                metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': y_sum, 'tp': 0}
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    # Convert all numpy types to Python native types recursively
    logger.info(f"Converting metrics for {dataset_name}...")
    logger.debug(f"Metrics before conversion: {list(metrics.keys())}")
    try:
        metrics = convert_numpy_types(metrics, path=f"metrics.{dataset_name}")
        logger.info(f"Successfully converted metrics for {dataset_name}")
    except Exception as e:
        logger.error(f"ERROR converting metrics for {dataset_name}: {e}")
        logger.error(f"Metrics structure: {json.dumps({k: str(type(v).__name__) for k, v in metrics.items()}, indent=2)}")
        raise
    
    print(f"\n   {dataset_name}:")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1-Score: {metrics['f1']:.4f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"      Gini: {metrics['gini']:.4f}")
    
    return metrics

def main():
    print("="*80)
    print("COMPLETING LIGHTGBM TRAINING AND UPDATING METRICS")
    print("="*80)
    logger.info("="*80)
    logger.info("Starting complete_lgbm_training.py")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    # Load data
    print("\n📂 Loading timeline data...")
    train_file = 'timelines_35day_enhanced_balanced_v4_muscular_train.csv'
    val_file = 'timelines_35day_enhanced_balanced_v4_muscular_val.csv'
    test_file = 'timelines_35day_enhanced_balanced_v4_muscular_test.csv'
    
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    
    print(f"✅ Loaded training set: {len(df_train):,} records")
    print(f"✅ Loaded validation set: {len(df_val):,} records")
    print(f"✅ Loaded test set: {len(df_test):,} records")
    
    # Combine train and validation
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)
    print(f"✅ Combined training set: {len(df_train_combined):,} records")
    
    # Prepare data WITHOUT replacing spaces (for compatibility with existing models)
    print("\n📊 Preparing data (keeping original feature names for existing models)...")
    logger.info("CRITICAL: Existing models (RF, GB, XGB) were trained with spaces in feature names (e.g., 'current_club_AC Milan')")
    X_train_orig, y_train = prepare_data(df_train_combined, replace_spaces_in_values=False)
    X_test_orig, y_test = prepare_data(df_test, replace_spaces_in_values=False)
    
    # Verify that spaces are preserved in feature names for existing models
    spaces_in_train = [col for col in X_train_orig.columns if ' ' in str(col)]
    spaces_in_test = [col for col in X_test_orig.columns if ' ' in str(col)]
    logger.info(f"Verification: Found {len(spaces_in_train)} features with spaces in train, {len(spaces_in_test)} in test")
    if spaces_in_train:
        logger.info(f"Sample features with spaces (first 5): {spaces_in_train[:5]}")
    if not spaces_in_train and not spaces_in_test:
        logger.warning("WARNING: No spaces found in feature names! This may cause issues with existing models!")
    
    # Align features
    print("\n🔧 Aligning features...")
    logger.info("Aligning features between train and test...")
    X_train_orig, X_test_orig = align_features(X_train_orig, X_test_orig)
    logger.info(f"Aligned features: {len(X_train_orig.columns)} common features")
    
    # Verify spaces are still present after alignment
    spaces_after_align = [col for col in X_train_orig.columns if ' ' in str(col)]
    logger.info(f"After alignment: {len(spaces_after_align)} features with spaces (should match original)")
    
    # Apply correlation filter (using original feature names)
    print("\n🔎 Applying correlation filter...")
    CORR_THRESHOLD = 0.8
    selected_features = apply_correlation_filter(X_train_orig, CORR_THRESHOLD)
    X_train_orig = X_train_orig[selected_features]
    X_test_orig = X_test_orig[selected_features]
    print(f"✅ Using {len(selected_features)} features after correlation filtering")
    
    all_results = {}
    
    # Option to skip existing model evaluation (set to True to skip and go straight to LightGBM)
    SKIP_EXISTING_MODELS = False  # Set to True to skip RF, GB, XGB evaluation and go straight to LightGBM
    
    if not SKIP_EXISTING_MODELS:
        # Load and evaluate RF model (using original feature names)
        print("\n" + "="*70)
        print("📊 EVALUATING EXISTING RF MODEL")
        print("="*70)
        logger.info("Loading and evaluating RF model...")
        rf_model = joblib.load('models/rf_model_v4_muscular_combined_corr.joblib')
        rf_train_metrics = evaluate_model(rf_model, X_train_orig, y_train, "Training (Train+Val Combined)")
        rf_test_metrics = evaluate_model(rf_model, X_test_orig, y_test, "Test (>= 2025-07-01)")
        logger.info("RF metrics collected, adding to all_results...")
        all_results['RF'] = {'train': rf_train_metrics, 'test': rf_test_metrics}
        logger.info("RF results added successfully")
        
        # Load and evaluate GB model (using original feature names)
        print("\n" + "="*70)
        print("📊 EVALUATING EXISTING GB MODEL")
        print("="*70)
        logger.info("Loading and evaluating GB model...")
        gb_model = joblib.load('models/gb_model_v4_muscular_combined_corr.joblib')
        gb_train_metrics = evaluate_model(gb_model, X_train_orig, y_train, "Training (Train+Val Combined)")
        gb_test_metrics = evaluate_model(gb_model, X_test_orig, y_test, "Test (>= 2025-07-01)")
        logger.info("GB metrics collected, adding to all_results...")
        all_results['GB'] = {'train': gb_train_metrics, 'test': gb_test_metrics}
        logger.info("GB results added successfully")
        
        # Load and evaluate XGBoost model (using original feature names)
        print("\n" + "="*70)
        print("📊 EVALUATING EXISTING XGBOOST MODEL")
        print("="*70)
        logger.info("Loading and evaluating XGBoost model...")
        xgb_model = joblib.load('models/xgb_model_v4_muscular_combined_corr.joblib')
        xgb_train_metrics = evaluate_model(xgb_model, X_train_orig, y_train, "Training (Train+Val Combined)")
        xgb_test_metrics = evaluate_model(xgb_model, X_test_orig, y_test, "Test (>= 2025-07-01)")
        logger.info("XGB metrics collected, adding to all_results...")
        all_results['XGB'] = {'train': xgb_train_metrics, 'test': xgb_test_metrics}
        logger.info("XGB results added successfully")
    else:
        logger.info("Skipping existing model evaluation (SKIP_EXISTING_MODELS=True)")
        print("\n⏭️  Skipping existing model evaluation - going straight to LightGBM training")
    
    # CRITICAL: Prepare data AGAIN with spaces replaced for LightGBM
    # This ensures LightGBM gets feature names without spaces
    print("\n📊 Preparing data for LightGBM (with spaces replaced)...")
    logger.info("Preparing data with spaces replaced for LightGBM compatibility...")
    X_train_lgbm, _ = prepare_data(df_train_combined, replace_spaces_in_values=True)
    X_test_lgbm, _ = prepare_data(df_test, replace_spaces_in_values=True)
    
    # Align features for LightGBM
    X_train_lgbm, X_test_lgbm = align_features(X_train_lgbm, X_test_lgbm)
    
    # Apply correlation filter directly on LightGBM data
    # (This ensures we get the same features, just with sanitized names)
    logger.info("Applying correlation filter to LightGBM data...")
    selected_features_lgbm = apply_correlation_filter(X_train_lgbm, CORR_THRESHOLD)
    X_train_lgbm = X_train_lgbm[selected_features_lgbm]
    X_test_lgbm = X_test_lgbm[selected_features_lgbm]
    logger.info(f"LightGBM data: {len(selected_features_lgbm)} features after correlation filtering")
    
    # Final sanitization for LightGBM - ensure NO spaces
    logger.info("Final sanitization of feature names for LightGBM...")
    X_train_lgbm.columns = [str(col).replace(' ', '_') for col in X_train_lgbm.columns]
    X_test_lgbm.columns = [str(col).replace(' ', '_') for col in X_test_lgbm.columns]
    X_train_lgbm.columns = [sanitize_feature_name(col, replace_spaces=True) for col in X_train_lgbm.columns]
    X_test_lgbm.columns = [sanitize_feature_name(col, replace_spaces=True) for col in X_test_lgbm.columns]
    X_train_lgbm.columns = [str(col).replace(' ', '_') for col in X_train_lgbm.columns]
    X_test_lgbm.columns = [str(col).replace(' ', '_') for col in X_test_lgbm.columns]
    
    # Verify no spaces remain
    spaces_remaining = [col for col in X_train_lgbm.columns if ' ' in str(col)]
    if spaces_remaining:
        logger.error(f"CRITICAL: Still found {len(spaces_remaining)} columns with spaces in LightGBM data!")
        logger.error(f"First 10: {spaces_remaining[:10]}")
        X_train_lgbm.columns = [str(col).replace(' ', '_') for col in X_train_lgbm.columns]
        X_test_lgbm.columns = [str(col).replace(' ', '_') for col in X_test_lgbm.columns]
    
    logger.info(f"LightGBM data prepared with {len(X_train_lgbm.columns)} features (all spaces removed)")
    logger.debug(f"Sample LightGBM feature names: {list(X_train_lgbm.columns[:10])}")
    
    # Train LightGBM model
    print("\n" + "="*70)
    print("🚀 TRAINING LIGHTGBM MODEL")
    print("="*70)
    
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
        verbose=-1
    )
    
    print("\n🔧 Model hyperparameters:")
    print(f"   n_estimators: {lgbm_model.n_estimators}")
    print(f"   max_depth: {lgbm_model.max_depth}")
    print(f"   learning_rate: {lgbm_model.learning_rate}")
    
    # Final check before training
    print("\n🔧 Final verification of feature names before training...")
    logger.info("Final verification of feature names before LightGBM training...")
    
    # Convert to list and verify
    train_cols = [str(col) for col in X_train_lgbm.columns]
    test_cols = [str(col) for col in X_test_lgbm.columns]
    
    # CRITICAL FIRST STEP: Replace ALL spaces immediately (LightGBM's main issue)
    # Do this FIRST before any other processing
    train_cols = [col.replace(' ', '_') for col in train_cols]
    test_cols = [col.replace(' ', '_') for col in test_cols]
    
    # Log any columns that still have spaces (should be none after this)
    spaces_before = [col for col in train_cols if ' ' in col]
    if spaces_before:
        logger.error(f"CRITICAL: Found {len(spaces_before)} columns with spaces after initial replacement!")
        logger.error(f"First 10: {spaces_before[:10]}")
        # Force fix
        train_cols = [col.replace(' ', '_') for col in train_cols]
        test_cols = [col.replace(' ', '_') for col in test_cols]
    
    # Check for any remaining problematic characters BEFORE sanitization
    problematic_chars = [' ', '&', ':', "'", '"', ',', ';', '/', '\\', '{', '}', '[', ']', '(', ')', '|', '?', '!', '*', '+', '=', '@', '#', '$', '%', '^']
    problematic_cols = []
    for i, col in enumerate(train_cols):
        if any(char in col for char in problematic_chars):
            problematic_cols.append((i, col))
    
    if problematic_cols:
        logger.warning(f"Found {len(problematic_cols)} columns with problematic characters BEFORE final sanitization")
        logger.warning(f"First 20 problematic columns: {[col[1] for col in problematic_cols[:20]]}")
    
    # Aggressively sanitize ALL column names - use a more comprehensive approach
    def ultra_sanitize(name):
        """Ultra-aggressive sanitization that removes ALL problematic characters"""
        name_str = str(name)
        # Replace all problematic characters with underscores
        for char in problematic_chars:
            name_str = name_str.replace(char, '_')
        # Remove multiple consecutive underscores
        while '__' in name_str:
            name_str = name_str.replace('__', '_')
        # Remove leading/trailing underscores
        name_str = name_str.strip('_')
        # Remove any non-ASCII characters (keep only alphanumeric, underscore, hyphen)
        name_str = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name_str)
        # Final cleanup
        while '__' in name_str:
            name_str = name_str.replace('__', '_')
        name_str = name_str.strip('_')
        return name_str if name_str else 'feature'
    
    train_cols_sanitized = [ultra_sanitize(col) for col in train_cols]
    test_cols_sanitized = [ultra_sanitize(col) for col in test_cols]
    
    # Create new DataFrames with sanitized column names to ensure clean data
    # CRITICAL: Replace spaces one more time before creating DataFrames
    train_cols_sanitized = [str(col).replace(' ', '_') for col in train_cols_sanitized]
    test_cols_sanitized = [str(col).replace(' ', '_') for col in test_cols_sanitized]
    
    X_train_lgbm_clean = X_train_lgbm.copy()
    X_train_lgbm_clean.columns = train_cols_sanitized
    X_test_lgbm_clean = X_test_lgbm.copy()
    X_test_lgbm_clean.columns = test_cols_sanitized
    
    # Replace DataFrames
    X_train_lgbm = X_train_lgbm_clean
    X_test_lgbm = X_test_lgbm_clean
    
    # FINAL VERIFICATION: Check for spaces one last time BEFORE training
    final_spaces = [col for col in X_train_lgbm.columns if ' ' in str(col)]
    if final_spaces:
        logger.error(f"CRITICAL ERROR: Still found {len(final_spaces)} columns with spaces after all sanitization!")
        logger.error(f"First 20: {final_spaces[:20]}")
        # Emergency fix: replace all spaces
        X_train_lgbm.columns = [str(col).replace(' ', '_') for col in X_train_lgbm.columns]
        X_test_lgbm.columns = [str(col).replace(' ', '_') for col in X_test_lgbm.columns]
        logger.warning("Applied emergency space replacement")
    
    # Final verification - check for ANY problematic characters
    final_problematic = []
    for col in X_train_lgbm.columns:
        col_str = str(col)
        if any(char in col_str for char in problematic_chars):
            final_problematic.append(col_str)
        # Also check for non-ASCII characters that might cause issues
        if not all(ord(c) < 128 and (c.isalnum() or c in ['_', '-']) for c in col_str):
            final_problematic.append(col_str)
    
    if final_problematic:
        logger.error(f"CRITICAL: Still found {len(final_problematic)} columns with problematic characters AFTER sanitization!")
        logger.error(f"First 20: {final_problematic[:20]}")
        # Force fix by replacing with safe names
        for i, col in enumerate(X_train_lgbm.columns):
            if str(col) in final_problematic:
                X_train_lgbm.columns.values[i] = f'feature_{i}'
        for i, col in enumerate(X_test_lgbm.columns):
            if str(col) in final_problematic:
                X_test_lgbm.columns.values[i] = f'feature_{i}'
        logger.warning("Forced replacement of problematic columns with safe names")
    
    # Ensure column names are unique (in case sanitization created duplicates)
    if len(X_train_lgbm.columns) != len(set(X_train_lgbm.columns)):
        logger.warning("Duplicate column names detected after sanitization, fixing...")
        seen = {}
        new_cols = []
        for col in X_train_lgbm.columns:
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                new_cols.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                new_cols.append(col_str)
        X_train_lgbm.columns = new_cols
        X_test_lgbm.columns = new_cols[:len(X_test_lgbm.columns)]  # Match test columns
    
    logger.info(f"✅ Final sanitization complete. Feature count: {len(X_train_lgbm.columns)}")
    logger.debug(f"Sample final feature names: {list(X_train_lgbm.columns[:10])}")
    
    # Final check - print any columns that still have issues
    for col in X_train_lgbm.columns[:50]:  # Check first 50
        col_str = str(col)
        if any(char in col_str for char in problematic_chars):
            logger.error(f"STILL PROBLEMATIC: {col_str}")
    
    print("\n🌳 Training started...")
    train_start = datetime.now()
    lgbm_model.fit(X_train_lgbm, y_train)
    train_time = datetime.now() - train_start
    print(f"✅ Training completed in {train_time}")
    
    # Evaluate LightGBM (using LightGBM-specific dataframes)
    logger.info("Evaluating LightGBM model...")
    lgbm_train_metrics = evaluate_model(lgbm_model, X_train_lgbm, y_train, "Training (Train+Val Combined)")
    lgbm_test_metrics = evaluate_model(lgbm_model, X_test_lgbm, y_test, "Test (>= 2025-07-01)")
    logger.info("LGBM metrics collected, adding to all_results...")
    all_results['LGBM'] = {'train': lgbm_train_metrics, 'test': lgbm_test_metrics}
    logger.info("LGBM results added successfully")
    
    # Save LightGBM model
    print("\n💾 Saving LightGBM model...")
    logger.info("Saving LightGBM model and column names...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(lgbm_model, 'models/lgbm_model_v4_muscular_combined_corr.joblib')
    logger.info("LightGBM model saved successfully")
    
    # Convert column names to strings and ensure JSON serializable
    try:
        logger.info(f"Serializing {len(X_train_lgbm.columns)} column names...")
        column_names = serialize_column_names(X_train_lgbm.columns)
        logger.debug(f"Column names after serialize_column_names: {len(column_names)} items")
        logger.debug(f"First 5 column names: {column_names[:5]}")
        
        # Double-check conversion
        logger.info("Converting column names with convert_numpy_types...")
        column_names_clean = convert_numpy_types(column_names, path="column_names")
        logger.info(f"Column names converted: {len(column_names_clean)} items")
        
        # Test JSON serialization before saving
        logger.info("Testing JSON serialization of column names...")
        test_json = json.dumps(column_names_clean, ensure_ascii=False)
        logger.info(f"JSON test successful: {len(test_json)} characters")
        
        with open('models/lgbm_model_v4_muscular_combined_corr_columns.json', 'w', encoding='utf-8') as f:
            json.dump(column_names_clean, f, indent=2, ensure_ascii=False)
        logger.info("Column names saved successfully")
        print(f"✅ Saved LightGBM model to models/lgbm_model_v4_muscular_combined_corr.joblib")
    except (TypeError, ValueError) as e:
        logger.error(f"ERROR saving column names: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Column names type: {type(column_names).__name__}")
        logger.error(f"Column names length: {len(column_names) if 'column_names' in locals() else 'N/A'}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save all metrics
    print("\n💾 Saving metrics...")
    logger.info("Saving all metrics to JSON...")
    os.makedirs('experiments', exist_ok=True)
    try:
        logger.info(f"all_results structure: {list(all_results.keys())}")
        for model_name in all_results.keys():
            logger.info(f"  Model {model_name}: {list(all_results[model_name].keys())}")
            for dataset_name in all_results[model_name].keys():
                logger.info(f"    Dataset {dataset_name}: {list(all_results[model_name][dataset_name].keys())}")
        
        # Double-check conversion before saving
        logger.info("Converting all_results with convert_numpy_types...")
        all_results_clean = convert_numpy_types(all_results, path="all_results")
        logger.info("Conversion completed successfully")
        
        # Test JSON serialization before saving
        logger.info("Testing JSON serialization of all_results...")
        try:
            test_json = json.dumps(all_results_clean, ensure_ascii=False)
            logger.info(f"JSON test successful: {len(test_json)} characters")
        except (TypeError, ValueError) as test_e:
            logger.error(f"JSON test FAILED: {test_e}")
            logger.error(f"Error type: {type(test_e).__name__}")
            # Try to find the problematic value
            logger.error("Attempting to identify problematic value...")
            for model_name in all_results_clean.keys():
                for dataset_name in all_results_clean[model_name].keys():
                    for metric_name in all_results_clean[model_name][dataset_name].keys():
                        try:
                            test_value = json.dumps(all_results_clean[model_name][dataset_name][metric_name])
                        except Exception as e:
                            logger.error(f"  PROBLEM FOUND: {model_name}.{dataset_name}.{metric_name}")
                            logger.error(f"    Type: {get_type_info(all_results_clean[model_name][dataset_name][metric_name])}")
                            logger.error(f"    Value: {repr(all_results_clean[model_name][dataset_name][metric_name])[:500]}")
                            raise test_e
            raise
        
        with open('experiments/v4_muscular_combined_corr_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(all_results_clean, f, indent=2, ensure_ascii=False)
        logger.info("Metrics saved successfully")
        print(f"✅ Saved metrics to experiments/v4_muscular_combined_corr_metrics.json")
    except (TypeError, ValueError) as e:
        logger.error(f"ERROR saving metrics: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"all_results type: {type(all_results).__name__}")
        logger.error(f"all_results keys: {list(all_results.keys()) if isinstance(all_results, dict) else 'N/A'}")
        # Try to identify the problematic value
        import traceback
        traceback.print_exc()
        raise
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS")
    print("="*80)
    print(f"\n{'Model':<10} {'Dataset':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12}")
    print("-" * 100)
    
    for model_name in ['RF', 'GB', 'XGB', 'LGBM']:
        results = all_results[model_name]
        print(f"{model_name:<10} {'Training (Train+Val Combined)':<30} {results['train']['precision']:<12.4f} {results['train']['recall']:<12.4f} {results['train']['f1']:<12.4f} {results['train']['roc_auc']:<12.4f}")
        print(f"{'':<10} {'Test (>= 2025-07-01)':<30} {results['test']['precision']:<12.4f} {results['test']['recall']:<12.4f} {results['test']['f1']:<12.4f} {results['test']['roc_auc']:<12.4f}")
        print("-" * 100)
    
    total_time = datetime.now() - start_time
    print(f"\n✅ Total execution time: {total_time}")
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

