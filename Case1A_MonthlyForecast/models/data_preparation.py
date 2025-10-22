"""Data preparation for Case1A Monthly Forecast.

This script loads the engineered features table, prepares train-test splits
based on temporal ordering, and validates data quality for model training.

Usage:
    python data_preparation.py --input ../data/processed/features_table.csv \
                               --output ../data/processed/

The script saves:
    - train_data.csv: Training set (2012-2018)
    - test_data.csv: Test/validation set (2019)
    - feature_metadata.json: Feature definitions and statistics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_features(path: str) -> pd.DataFrame:
    """Load features table and perform initial validation.
    
    Parameters
    ----------
    path : str
        Path to features_table.csv
        
    Returns
    -------
    pd.DataFrame
        Features dataframe with date column as datetime
    """
    df = pd.read_csv(path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Validation
    logger.info(f"Loaded features table: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Series count: {df.groupby(['Country', 'Product']).ngroups}")
    
    # Check for duplicates
    duplicates = df.groupby(['Country', 'Product', 'date']).size()
    if (duplicates > 1).any():
        logger.error(f"Found duplicate (Country, Product, date) combinations!")
        raise ValueError("Duplicate rows detected in features table")
    
    logger.info("✓ No duplicates found")
    
    return df


def define_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Define and categorize feature columns for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with categorized column names
    """
    # Columns to exclude from modeling
    exclude_cols = [
        'date',           # Used for splitting, not a feature
        'Country',        # Encoded as Country_code
        'Product',        # Encoded as Product_code
        'Value',          # Target variable
        'YEAR',           # Redundant with 'year' feature
        'MONTH',          # Redundant with 'month' feature
        'Country_Product' # Redundant with CountryProduct_code
    ]
    
    # All columns that should be used as features
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Categorize features for documentation
    time_features = [c for c in feature_cols if any(x in c for x in 
                    ['year', 'month', 'quarter', 'sin', 'cos', 'since_start'])]
    
    lag_features = [c for c in feature_cols if c.startswith('lag_')]
    
    roll_features = [c for c in feature_cols if c.startswith('roll_')]
    
    categorical_features = [c for c in feature_cols if any(x in c for x in 
                           ['code', 'CountryProduct'])]
    
    columns = {
        'exclude': exclude_cols,
        'all_features': feature_cols,
        'time': time_features,
        'lag': lag_features,
        'rolling': roll_features,
        'categorical': categorical_features,
        'target': 'Value'
    }
    
    logger.info(f"Feature breakdown:")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Time features: {len(time_features)}")
    logger.info(f"  Lag features: {len(lag_features)}")
    logger.info(f"  Rolling features: {len(roll_features)}")
    logger.info(f"  Categorical features: {len(categorical_features)}")
    
    return columns


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_end_date: str = '2019-01-01',
    test_start_date: str = '2019-01-01',
    test_end_date: str = '2020-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets based on temporal ordering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe
    feature_cols : List[str]
        List of feature column names
    train_end_date : str
        End date for training data (exclusive)
    test_start_date : str
        Start date for test data (inclusive)
    test_end_date : str
        End date for test data (exclusive)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    logger.info("=" * 80)
    logger.info("TEMPORAL TRAIN-TEST SPLIT")
    logger.info("=" * 80)
    
    # Split by date
    train_raw = df[df['date'] < train_end_date].copy()
    test_raw = df[(df['date'] >= test_start_date) & (df['date'] < test_end_date)].copy()
    
    logger.info(f"\nInitial split:")
    logger.info(f"  Training period: {train_raw['date'].min()} to {train_raw['date'].max()}")
    logger.info(f"  Test period: {test_raw['date'].min()} to {test_raw['date'].max()}")
    logger.info(f"  Training rows (before cleaning): {len(train_raw)}")
    logger.info(f"  Test rows (before cleaning): {len(test_raw)}")
    
    # Remove rows with missing values in feature columns
    # This is expected for first 12 months per series (lag_12 not available)
    train_clean = train_raw.dropna(subset=feature_cols)
    test_clean = test_raw.dropna(subset=feature_cols)
    
    rows_removed = len(train_raw) - len(train_clean)
    logger.info(f"\nAfter removing NaN values:")
    logger.info(f"  Training rows (clean): {len(train_clean)}")
    logger.info(f"  Test rows (clean): {len(test_clean)}")
    logger.info(f"  Rows removed from training: {rows_removed}")
    
    if rows_removed > 0:
        logger.info(f"  Note: Removed first ~12 months per series (lag_12 unavailable)")
    
    # Validate coverage
    train_series = train_clean.groupby(['Country', 'Product']).ngroups
    test_series = test_clean.groupby(['Country', 'Product']).ngroups
    
    logger.info(f"\nSeries coverage:")
    logger.info(f"  Training set: {train_series} country-product combinations")
    logger.info(f"  Test set: {test_series} country-product combinations")
    
    if train_series != test_series:
        logger.warning("⚠️  Train and test sets have different series coverage!")
    else:
        logger.info("  ✓ All series present in both sets")
    
    return train_clean, test_clean


def validate_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, any]:
    """Validate train-test splits and generate quality metrics.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    feature_cols : List[str]
        List of feature column names
        
    Returns
    -------
    Dict[str, any]
        Validation metrics and statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 80)
    
    metrics = {}
    
    # 1. Check for missing values
    train_missing = train_df[feature_cols].isnull().sum().sum()
    test_missing = test_df[feature_cols].isnull().sum().sum()
    
    logger.info(f"\n1. Missing Values:")
    logger.info(f"   Training set: {train_missing}")
    logger.info(f"   Test set: {test_missing}")
    
    if train_missing > 0 or test_missing > 0:
        logger.error("❌ Missing values found after cleaning!")
        raise ValueError("Unexpected missing values in feature columns")
    else:
        logger.info("   ✓ No missing values")
    
    metrics['missing_values'] = {'train': train_missing, 'test': test_missing}
    
    # 2. Check target variable
    train_target_missing = train_df['Value'].isnull().sum()
    test_target_missing = test_df['Value'].isnull().sum()
    
    logger.info(f"\n2. Target Variable:")
    logger.info(f"   Training missing: {train_target_missing}")
    logger.info(f"   Test missing: {test_target_missing}")
    logger.info(f"   Training range: [{train_df['Value'].min():.2f}, {train_df['Value'].max():.2f}]")
    logger.info(f"   Test range: [{test_df['Value'].min():.2f}, {test_df['Value'].max():.2f}]")
    
    if train_target_missing > 0 or test_target_missing > 0:
        logger.error("❌ Missing target values!")
        raise ValueError("Missing values in target variable")
    
    metrics['target'] = {
        'train_min': float(train_df['Value'].min()),
        'train_max': float(train_df['Value'].max()),
        'train_mean': float(train_df['Value'].mean()),
        'test_min': float(test_df['Value'].min()),
        'test_max': float(test_df['Value'].max()),
        'test_mean': float(test_df['Value'].mean())
    }
    
    # 3. Check feature statistics
    logger.info(f"\n3. Feature Statistics:")
    
    train_X = train_df[feature_cols]
    test_X = test_df[feature_cols]
    
    # Check for constant features (no variation)
    train_std = train_X.std()
    constant_features = train_std[train_std == 0].index.tolist()
    
    if constant_features:
        logger.warning(f"⚠️  Constant features (no variation): {constant_features}")
    else:
        logger.info("   ✓ All features have variation")
    
    # Feature range comparison
    logger.info("\n   Sample feature ranges (train vs test):")
    for feat in feature_cols[:5]:  # Show first 5 features
        train_range = (train_X[feat].min(), train_X[feat].max())
        test_range = (test_X[feat].min(), test_X[feat].max())
        logger.info(f"   {feat}: Train [{train_range[0]:.2f}, {train_range[1]:.2f}] | "
                   f"Test [{test_range[0]:.2f}, {test_range[1]:.2f}]")
    
    metrics['feature_stats'] = {
        'constant_features': constant_features,
        'train_shape': train_X.shape,
        'test_shape': test_X.shape
    }
    
    # 4. Final summary
    logger.info(f"\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Training set: {train_df.shape[0]} samples, {len(feature_cols)} features")
    logger.info(f"✓ Test set: {test_df.shape[0]} samples, {len(feature_cols)} features")
    logger.info(f"✓ No missing values in features or target")
    logger.info(f"✓ Data ready for model training")
    
    return metrics


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns_info: Dict,
    validation_metrics: Dict,
    output_dir: str
) -> None:
    """Save train-test splits and metadata to disk.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    columns_info : Dict
        Column categorization information
    validation_metrics : Dict
        Validation statistics
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data splits
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"\n✓ Saved training data to: {train_path}")
    logger.info(f"✓ Saved test data to: {test_path}")
    
    # Save feature metadata
    metadata = {
        'date_generated': pd.Timestamp.now().isoformat(),
        'train_period': {
            'start': train_df['date'].min().isoformat(),
            'end': train_df['date'].max().isoformat(),
            'n_samples': int(len(train_df)),
            'n_series': int(train_df.groupby(['Country', 'Product']).ngroups)
        },
        'test_period': {
            'start': test_df['date'].min().isoformat(),
            'end': test_df['date'].max().isoformat(),
            'n_samples': int(len(test_df)),
            'n_series': int(test_df.groupby(['Country', 'Product']).ngroups)
        },
        'features': {
            'total': len(columns_info['all_features']),
            'time_features': columns_info['time'],
            'lag_features': columns_info['lag'],
            'rolling_features': columns_info['rolling'],
            'categorical_features': columns_info['categorical']
        },
        'target_variable': columns_info['target'],
        'excluded_columns': columns_info['exclude'],
        'validation_metrics': validation_metrics
    }
    # Convert any numpy/pandas types to native Python types for JSON serialization
    def _to_builtin(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return o

    # Walk the metadata and coerce common pandas/numpy types
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        try:
            return _to_builtin(obj)
        except Exception:
            return obj

    metadata_clean = _sanitize(metadata)

    metadata_path = os.path.join(output_dir, 'feature_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_clean, f, indent=2)
    
    logger.info(f"✓ Saved feature metadata to: {metadata_path}")


def prepare_data(
    input_path: str,
    output_dir: str,
    train_end: str = '2019-01-01',
    test_start: str = '2019-01-01',
    test_end: str = '2020-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Main pipeline to prepare train-test splits.
    
    Parameters
    ----------
    input_path : str
        Path to features_table.csv
    output_dir : str
        Output directory for train/test splits
    train_end : str
        End date for training period
    test_start : str
        Start date for test period
    test_end : str
        End date for test period
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (train_df, test_df, columns_info)
    """
    logger.info("=" * 80)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 80)
    
    # Resolve input path robustly: accept absolute, cwd-relative, or script-relative paths
    input_p = Path(input_path)
    if not input_p.exists():
        logger.warning(f"Input path not found: {input_path}. Attempting fallbacks...")
        script_dir = Path(__file__).resolve().parent
        candidates = [
            input_p,  # as given
            script_dir / input_path,  # relative to script dir
            script_dir.parent / 'data' / 'processed' / Path(input_path).name,  # common fallback
            Path.cwd() / input_path,  # relative to current working dir
        ]
        tried = []
        for c in candidates:
            tried.append(str(c))
            if c.exists():
                input_p = c
                logger.info(f"Found input features file at: {c}")
                break
        else:
            logger.error("Input features file not found. Tried the following paths:\n" + "\n".join(tried))
            raise FileNotFoundError(f"Input features file not found. Tried: {tried}")

    # Step 1: Load features
    df = load_features(str(input_p))
    
    # Step 2: Define feature columns
    columns_info = define_feature_columns(df)
    
    # Step 3: Temporal train-test split
    train_df, test_df = temporal_train_test_split(
        df,
        columns_info['all_features'],
        train_end_date=train_end,
        test_start_date=test_start,
        test_end_date=test_end
    )
    
    # Step 4: Validate splits
    validation_metrics = validate_splits(
        train_df,
        test_df,
        columns_info['all_features']
    )
    
    # Step 5: Save to disk
    save_splits(train_df, test_df, columns_info, validation_metrics, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    
    return train_df, test_df, columns_info


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Prepare train-test splits for Case1A Monthly Forecast'
    )
    p.add_argument(
        '--input', '-i',
        default=os.path.join('..', 'data', 'processed', 'features_table.csv'),
        help='Path to features table CSV'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'data', 'processed'),
        help='Output directory for train/test splits'
    )
    p.add_argument(
        '--train-end',
        default='2019-01-01',
        help='End date for training period (YYYY-MM-DD)'
    )
    p.add_argument(
        '--test-start',
        default='2019-01-01',
        help='Start date for test period (YYYY-MM-DD)'
    )
    p.add_argument(
        '--test-end',
        default='2020-01-01',
        help='End date for test period (YYYY-MM-DD)'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    prepare_data(
        input_path=args.input,
        output_dir=args.output,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end
    )


if __name__ == '__main__':
    main()
