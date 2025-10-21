"""Data preparation and train-test split for Case1B Quarterly Forecast.

This script prepares the quarterly feature data for modeling by:
- Removing rows with missing lag features
- Splitting into train (2010-2018) and test (2019)
- Validating data quality
- Saving prepared datasets

Usage:
    python data_preparation_quarterly.py \
        --input ../data/processed/features_table_quarterly.csv \
        --output ../data/processed/

Outputs:
    - train_data_quarterly.csv: Training set (2010-2018)
    - test_data_quarterly.csv: Test set (2019)
    - data_split_metadata_quarterly.json: Split information
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_features(filepath: str) -> pd.DataFrame:
    """Load feature table.
    
    Parameters
    ----------
    filepath : str
        Path to features_table_quarterly.csv
        
    Returns
    -------
    pd.DataFrame
        Feature dataframe
    """
    logger.info("=" * 80)
    logger.info("LOADING FEATURE DATA")
    logger.info("=" * 80)
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Series: {df.groupby(['Country', 'Product']).ngroups}")
    
    return df


def remove_initial_nans(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Remove initial rows with NaN in lag features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (cleaned_dataframe, removal_stats)
    """
    logger.info("\n" + "=" * 80)
    logger.info("REMOVING INITIAL NaN VALUES")
    logger.info("=" * 80)
    
    initial_rows = len(df)
    
    # Identify lag features
    lag_cols = [c for c in df.columns if c.startswith('lag_')]
    
    logger.info(f"Lag features: {lag_cols}")
    
    # Remove rows with any NaN in lag features
    # This removes first 4 quarters per series (needed for lag_4)
    df_clean = df.dropna(subset=lag_cols).copy()
    
    rows_removed = initial_rows - len(df_clean)
    removal_pct = rows_removed / initial_rows * 100
    
    logger.info(f"Rows before: {initial_rows}")
    logger.info(f"Rows after: {len(df_clean)}")
    logger.info(f"Removed: {rows_removed} rows ({removal_pct:.1f}%)")
    
    # Verify no more NaN in features
    exclude_cols = ['date', 'Country', 'Product', 'Value', 'YEAR', 'QUARTER']
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
    
    remaining_nans = df_clean[feature_cols].isnull().sum().sum()
    
    if remaining_nans > 0:
        logger.warning(f"⚠ Warning: {remaining_nans} NaN values remain in features")
        nan_cols = df_clean[feature_cols].isnull().sum()
        logger.warning(f"Columns with NaN: {nan_cols[nan_cols > 0].to_dict()}")
    else:
        logger.info("✓ No NaN values in features")
    
    stats = {
        'initial_rows': initial_rows,
        'final_rows': len(df_clean),
        'rows_removed': rows_removed,
        'removal_percentage': removal_pct,
        'remaining_nans': int(remaining_nans)
    }
    
    return df_clean, stats


def create_train_test_split(
    df: pd.DataFrame,
    train_end_year: int = 2018,
    test_year: int = 2019
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets by year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean feature dataframe
    train_end_year : int
        Last year for training (inclusive)
    test_year : int
        Test year
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAIN-TEST SPLIT")
    logger.info("=" * 80)
    
    # Extract year from date
    df['year_split'] = df['date'].dt.year
    
    # Split by year
    train_df = df[df['year_split'] <= train_end_year].copy()
    test_df = df[df['year_split'] == test_year].copy()
    
    # Remove temporary column
    train_df = train_df.drop(columns=['year_split'])
    test_df = test_df.drop(columns=['year_split'])
    
    logger.info(f"\nTrain period: 2010-{train_end_year}")
    logger.info(f"  Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"  Rows: {len(train_df)}")
    logger.info(f"  Series: {train_df.groupby(['Country', 'Product']).ngroups}")
    
    logger.info(f"\nTest period: {test_year}")
    logger.info(f"  Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    logger.info(f"  Rows: {len(test_df)}")
    logger.info(f"  Series: {test_df.groupby(['Country', 'Product']).ngroups}")
    
    return train_df, test_df


def validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict:
    """Validate train-test split quality.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
        
    Returns
    -------
    Dict
        Validation metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING SPLIT")
    logger.info("=" * 80)
    
    # 1. Check series coverage
    train_series = set(train_df.groupby(['Country', 'Product']).groups.keys())
    test_series = set(test_df.groupby(['Country', 'Product']).groups.keys())
    
    common_series = train_series.intersection(test_series)
    missing_in_test = train_series - test_series
    
    logger.info(f"\n1. Series Coverage:")
    logger.info(f"   Train series: {len(train_series)}")
    logger.info(f"   Test series: {len(test_series)}")
    logger.info(f"   Common: {len(common_series)}")
    
    if missing_in_test:
        logger.warning(f"   ⚠ Missing in test: {len(missing_in_test)}")
    else:
        logger.info(f"   ✓ All train series present in test")
    
    # 2. Check for NaN values
    exclude_cols = ['date', 'Country', 'Product', 'Value', 'YEAR', 'QUARTER']
    train_features = [c for c in train_df.columns if c not in exclude_cols]
    test_features = [c for c in test_df.columns if c not in exclude_cols]
    
    train_nans = train_df[train_features].isnull().sum().sum()
    test_nans = test_df[test_features].isnull().sum().sum()
    
    logger.info(f"\n2. Missing Values:")
    logger.info(f"   Train NaNs: {train_nans}")
    logger.info(f"   Test NaNs: {test_nans}")
    
    if train_nans == 0 and test_nans == 0:
        logger.info("   ✓ No missing values")
    else:
        logger.warning("   ⚠ Missing values detected")
    
    # 3. Check feature consistency
    logger.info(f"\n3. Feature Consistency:")
    logger.info(f"   Train features: {len(train_features)}")
    logger.info(f"   Test features: {len(test_features)}")
    logger.info(f"   Match: {set(train_features) == set(test_features)}")
    
    # 4. Target variable statistics
    logger.info(f"\n4. Target Variable (Value):")
    logger.info(f"   Train - min: {train_df['Value'].min():.2f}, max: {train_df['Value'].max():.2f}, mean: {train_df['Value'].mean():.2f}")
    logger.info(f"   Test  - min: {test_df['Value'].min():.2f}, max: {test_df['Value'].max():.2f}, mean: {test_df['Value'].mean():.2f}")
    
    validation = {
        'train_series': len(train_series),
        'test_series': len(test_series),
        'common_series': len(common_series),
        'train_nans': int(train_nans),
        'test_nans': int(test_nans),
        'feature_count': len(train_features),
        'features_match': bool(set(train_features) == set(test_features)),
        'target_stats': {
            'train': {
                'min': float(train_df['Value'].min()),
                'max': float(train_df['Value'].max()),
                'mean': float(train_df['Value'].mean())
            },
            'test': {
                'min': float(test_df['Value'].min()),
                'max': float(test_df['Value'].max()),
                'mean': float(test_df['Value'].mean())
            }
        }
    }
    
    return validation


def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    removal_stats: Dict,
    validation: Dict,
    output_dir: str
) -> None:
    """Save train/test datasets and metadata.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    removal_stats : Dict
        NaN removal statistics
    validation : Dict
        Validation metrics
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING DATASETS")
    logger.info("=" * 80)
    
    # Save train data
    train_path = os.path.join(output_dir, 'train_data_quarterly.csv')
    train_df.to_csv(train_path, index=False)
    logger.info(f"✓ Saved training data to: {train_path}")
    
    # Save test data
    test_path = os.path.join(output_dir, 'test_data_quarterly.csv')
    test_df.to_csv(test_path, index=False)
    logger.info(f"✓ Saved test data to: {test_path}")
    
    # Save metadata
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_frequency': 'quarterly',
        'train_period': {
            'start': train_df['date'].min().isoformat(),
            'end': train_df['date'].max().isoformat(),
            'n_samples': len(train_df),
            'n_series': int(train_df.groupby(['Country', 'Product']).ngroups)
        },
        'test_period': {
            'start': test_df['date'].min().isoformat(),
            'end': test_df['date'].max().isoformat(),
            'n_samples': len(test_df),
            'n_series': int(test_df.groupby(['Country', 'Product']).ngroups)
        },
        'removal_stats': removal_stats,
        'validation': validation,
        'feature_columns': [c for c in train_df.columns 
                           if c not in ['date', 'Country', 'Product', 'Value', 'YEAR', 'QUARTER']]
    }
    
    metadata_path = os.path.join(output_dir, 'data_split_metadata_quarterly.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata to: {metadata_path}")


def main_pipeline(input_path: str, output_dir: str) -> None:
    """Main data preparation pipeline.
    
    Parameters
    ----------
    input_path : str
        Path to features_table_quarterly.csv
    output_dir : str
        Output directory
    """
    logger.info("=" * 80)
    logger.info("CASE 1B: DATA PREPARATION & TRAIN-TEST SPLIT")
    logger.info("=" * 80)
    
    # Step 1: Load features
    df = load_features(input_path)
    
    # Step 2: Remove initial NaN values
    df_clean, removal_stats = remove_initial_nans(df)
    
    # Step 3: Create train-test split
    train_df, test_df = create_train_test_split(df_clean, train_end_year=2018, test_year=2019)
    
    # Step 4: Validate split
    validation = validate_split(train_df, test_df)
    
    # Step 5: Save datasets
    save_datasets(train_df, test_df, removal_stats, validation, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Train: {len(train_df)} samples, {validation['train_series']} series")
    logger.info(f"Test: {len(test_df)} samples, {validation['test_series']} series")
    logger.info(f"Features: {validation['feature_count']}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Prepare quarterly data for modeling'
    )
    p.add_argument(
        '--input', '-i',
        default=os.path.join('..', 'data', 'processed', 'features_table_quarterly.csv'),
        help='Path to feature table'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'data', 'processed'),
        help='Output directory'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(args.input, args.output)


if __name__ == '__main__':
    main()
