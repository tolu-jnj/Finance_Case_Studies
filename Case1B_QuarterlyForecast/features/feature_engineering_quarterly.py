"""Feature engineering for Case1B Quarterly Forecast.

This script creates features adapted for quarterly time series data including:
- Time features (year, quarter, cyclical encodings)
- Lag features (lag_1, lag_2, lag_4 for year-over-year)
- Rolling statistics (2, 4, 8 quarters)
- Categorical encodings

Usage:
    python feature_engineering_quarterly.py \
        --input ../data/processed/quarterly_data_long.csv \
        --valid_series ../data/processed/valid_series_list.csv \
        --output ../data/processed/

Outputs:
    - features_table_quarterly.csv: Full feature matrix
    - feature_metadata_quarterly.json: Feature definitions
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data(data_path: str, valid_series_path: str = None) -> pd.DataFrame:
    """Load quarterly data and filter to valid series only.
    
    Parameters
    ----------
    data_path : str
        Path to quarterly_data_long.csv
    valid_series_path : str, optional
        Path to valid_series_list.csv (series without >40% zeros)
        
    Returns
    -------
    pd.DataFrame
        Filtered quarterly data
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Total series: {df.groupby(['Country', 'Product']).ngroups}")
    
    # Filter to valid series if list provided
    if valid_series_path and os.path.exists(valid_series_path):
        valid = pd.read_csv(valid_series_path)
        df = df.merge(valid, on=['Country', 'Product'], how='inner')
        logger.info(f"Filtered to {len(valid)} valid series (excluded series with >40% zeros)")
        logger.info(f"Filtered data: {df.shape}")
    
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features for quarterly data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date column
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added time features
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TIME FEATURES")
    logger.info("=" * 80)
    
    df = df.copy()
    
    # Basic time features
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Quarters since start (from 2010 Q1)
    start_date = pd.Timestamp('2010-01-01')
    df['quarters_since_start'] = ((df['date'].dt.year - start_date.year) * 4 + 
                                  (df['date'].dt.quarter - start_date.quarter))
    
    # Cyclical encoding for quarters (period=4)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    time_features = ['year', 'quarter', 'quarter_sin', 'quarter_cos', 'quarters_since_start']
    logger.info(f"Created {len(time_features)} time features: {time_features}")
    
    return df


def create_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 4]) -> pd.DataFrame:
    """Create lag features for each series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    lags : List[int]
        List of lag periods (quarters)
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added lag features
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING LAG FEATURES")
    logger.info("=" * 80)
    
    df = df.sort_values(['Country', 'Product', 'date']).copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['Country', 'Product'])['Value'].shift(lag)
        logger.info(f"Created lag_{lag}")
    
    logger.info(f"Created {len(lags)} lag features")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [2, 4, 8],
    stats: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """Create rolling window statistics for each series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    windows : List[int]
        List of window sizes (quarters)
    stats : List[str]
        List of statistics to compute
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added rolling features
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING ROLLING FEATURES")
    logger.info("=" * 80)
    
    df = df.sort_values(['Country', 'Product', 'date']).copy()
    feature_count = 0
    
    for window in windows:
        for stat in stats:
            col_name = f'roll_{stat}_{window}'
            
            if stat == 'mean':
                df[col_name] = df.groupby(['Country', 'Product'])['Value'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            elif stat == 'std':
                df[col_name] = df.groupby(['Country', 'Product'])['Value'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            elif stat == 'min':
                df[col_name] = df.groupby(['Country', 'Product'])['Value'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
            elif stat == 'max':
                df[col_name] = df.groupby(['Country', 'Product'])['Value'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            
            feature_count += 1
            logger.info(f"Created {col_name}")
    
    logger.info(f"Created {feature_count} rolling features ({len(windows)} windows × {len(stats)} stats)")
    return df


def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create categorical encoding features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added categorical features
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING CATEGORICAL FEATURES")
    logger.info("=" * 80)
    
    df = df.copy()
    
    # Label encoding for Country
    country_map = {c: i for i, c in enumerate(sorted(df['Country'].unique()))}
    df['Country_code'] = df['Country'].map(country_map)
    logger.info(f"Country encoding: {country_map}")
    
    # Label encoding for Product
    product_map = {p: i for i, p in enumerate(sorted(df['Product'].unique()))}
    df['Product_code'] = df['Product'].map(product_map)
    logger.info(f"Product encoding: {len(product_map)} products")
    
    # Interaction: Country-Product combination
    df['CountryProduct_code'] = df.groupby(['Country', 'Product']).ngroup()
    
    logger.info("Created 3 categorical features: Country_code, Product_code, CountryProduct_code")
    
    return df


def validate_features(df: pd.DataFrame) -> Dict:
    """Validate feature matrix and return quality metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe
        
    Returns
    -------
    Dict
        Validation metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING FEATURES")
    logger.info("=" * 80)
    
    # Identify feature columns
    exclude_cols = ['date', 'Country', 'Product', 'Value', 'YEAR', 'QUARTER']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Check for missing values
    missing = df[feature_cols].isnull().sum()
    missing_cols = missing[missing > 0]
    
    logger.info(f"\n1. Missing Values:")
    if len(missing_cols) > 0:
        logger.info(f"   Columns with missing values:")
        for col, count in missing_cols.items():
            logger.info(f"   - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    else:
        logger.info("   ✓ No missing values in feature columns")
    
    # Check for infinite values
    inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    logger.info(f"\n2. Infinite Values: {inf_count}")
    
    # Feature statistics
    logger.info(f"\n3. Feature Statistics:")
    logger.info(f"   Total features: {len(feature_cols)}")
    logger.info(f"   Time features: {len([c for c in feature_cols if any(x in c for x in ['year', 'quarter', 'since'])])}")
    logger.info(f"   Lag features: {len([c for c in feature_cols if c.startswith('lag_')])}")
    logger.info(f"   Rolling features: {len([c for c in feature_cols if c.startswith('roll_')])}")
    logger.info(f"   Categorical features: {len([c for c in feature_cols if 'code' in c])}")
    
    validation = {
        'total_features': len(feature_cols),
        'missing_values': int(missing.sum()),
        'infinite_values': int(inf_count),
        'total_observations': len(df),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        }
    }
    
    return validation


def save_features(
    df: pd.DataFrame,
    validation: Dict,
    output_dir: str
) -> None:
    """Save feature matrix and metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe
    validation : Dict
        Validation metrics
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING FEATURES")
    logger.info("=" * 80)
    
    # Save features table
    features_path = os.path.join(output_dir, 'features_table_quarterly.csv')
    df.to_csv(features_path, index=False)
    logger.info(f"✓ Saved features to: {features_path}")
    
    # Define feature categories
    exclude_cols = ['date', 'Country', 'Product', 'Value', 'YEAR', 'QUARTER']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    time_features = [c for c in feature_cols if any(x in c for x in ['year', 'quarter', 'since'])]
    lag_features = [c for c in feature_cols if c.startswith('lag_')]
    rolling_features = [c for c in feature_cols if c.startswith('roll_')]
    categorical_features = [c for c in feature_cols if 'code' in c]
    
    # Save metadata
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_frequency': 'quarterly',
        'total_observations': len(df),
        'total_series': int(df.groupby(['Country', 'Product']).ngroups),
        'date_range': validation['date_range'],
        'features': {
            'total': len(feature_cols),
            'time_features': time_features,
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'categorical_features': categorical_features
        },
        'target_variable': 'Value',
        'excluded_columns': exclude_cols,
        'validation': validation
    }
    
    metadata_path = os.path.join(output_dir, 'feature_metadata_quarterly.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata to: {metadata_path}")


def main_pipeline(
    data_path: str,
    output_dir: str,
    valid_series_path: str = None
) -> None:
    """Main feature engineering pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to quarterly_data_long.csv
    output_dir : str
        Output directory
    valid_series_path : str, optional
        Path to valid_series_list.csv
    """
    logger.info("=" * 80)
    logger.info("CASE 1B: QUARTERLY FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    df = load_data(data_path, valid_series_path)
    
    # Step 2: Create time features
    df = create_time_features(df)
    
    # Step 3: Create lag features (1, 2, 4 quarters)
    df = create_lag_features(df, lags=[1, 2, 4])
    
    # Step 4: Create rolling features (2, 4, 8 quarter windows)
    df = create_rolling_features(df, windows=[2, 4, 8], stats=['mean', 'std', 'min', 'max'])
    
    # Step 5: Create categorical features
    df = create_categorical_features(df)
    
    # Step 6: Validate features
    validation = validate_features(df)
    
    # Step 7: Save outputs
    save_features(df, validation, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Total features: {validation['total_features']}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Feature engineering for Case1B quarterly data'
    )
    p.add_argument(
        '--input', '-i',
        default=os.path.join('..', 'data', 'processed', 'quarterly_data_long.csv'),
        help='Path to quarterly data (long format)'
    )
    p.add_argument(
        '--valid_series',
        default=os.path.join('..', 'data', 'processed', 'valid_series_list.csv'),
        help='Path to valid series list (optional)'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'data', 'processed'),
        help='Output directory for features'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(
        data_path=args.input,
        output_dir=args.output,
        valid_series_path=args.valid_series if os.path.exists(args.valid_series) else None
    )


if __name__ == '__main__':
    main()
