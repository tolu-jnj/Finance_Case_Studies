"""Data loading and reshaping for Case1B Quarterly Forecast.

This script loads the wide-format quarterly data, reshapes it to long format,
extracts country and product information, and validates data quality.

Usage:
    python data_loader_quarterly.py --input ../data/raw/Case1B_QuarterlyData.csv \
                                     --output ../data/processed/

Outputs:
    - quarterly_data_long.csv: Reshaped data in long format
    - data_validation_report.json: Data quality metrics
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


def load_wide_format(filepath: str) -> pd.DataFrame:
    """Load quarterly data from wide format CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the wide-format CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw data in wide format
    """
    logger.info("=" * 80)
    logger.info("LOADING QUARTERLY DATA (WIDE FORMAT)")
    logger.info("=" * 80)
    
    df = pd.read_csv(filepath)
    
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Date column: {df.columns[0]}")
    logger.info(f"Product columns: {df.columns[1:].tolist()[:5]}...")
    
    return df


def reshape_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide format to long format.
    
    Wide format: Date | CountryA_Y1 | CountryA_Y2 | ...
    Long format: Date | Country | Product | Value
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format dataframe
        
    Returns
    -------
    pd.DataFrame
        Long format dataframe
    """
    logger.info("\n" + "=" * 80)
    logger.info("RESHAPING TO LONG FORMAT")
    logger.info("=" * 80)
    
    # Parse date column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
    
    # Melt to long format
    df_long = df.melt(
        id_vars=[date_col],
        var_name='Country_Product',
        value_name='Value'
    )
    
    # Rename date column to standard 'date'
    df_long = df_long.rename(columns={date_col: 'date'})
    
    logger.info(f"Long format shape: {df_long.shape}")
    logger.info(f"Sample Country_Product values: {df_long['Country_Product'].unique()[:5].tolist()}")
    
    return df_long


def extract_country_product(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Country and Product from combined column.
    
    Example: CountryA_Y1 → Country=A, Product=Y1
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with Country_Product column
        
    Returns
    -------
    pd.DataFrame
        Dataframe with separate Country and Product columns
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTING COUNTRY AND PRODUCT")
    logger.info("=" * 80)
    
    # Extract Country (letter after 'Country')
    df['Country'] = df['Country_Product'].str.extract(r'Country([A-C])')[0]
    
    # Extract Product (Y followed by number)
    df['Product'] = df['Country_Product'].str.extract(r'(Y\d+)')[0]
    
    # Drop the combined column
    df = df.drop(columns=['Country_Product'])
    
    # Reorder columns
    df = df[['date', 'Country', 'Product', 'Value']]
    
    logger.info(f"Unique countries: {sorted(df['Country'].unique())}")
    logger.info(f"Unique products: {sorted(df['Product'].unique())}")
    logger.info(f"Country-Product combinations: {df.groupby(['Country', 'Product']).ngroups}")
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and quarter columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with date column
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added YEAR and QUARTER columns
    """
    df['YEAR'] = df['date'].dt.year
    df['QUARTER'] = df['date'].dt.quarter
    
    logger.info(f"\nTime range: {df['YEAR'].min()} Q{df['QUARTER'].min()} to {df['YEAR'].max()} Q{df['QUARTER'].max()}")
    logger.info(f"Total quarters: {len(df['date'].unique())}")
    
    return df


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """Validate data quality and generate report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe
        
    Returns
    -------
    Dict
        Validation metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 80)
    
    validation = {}
    
    # 1. Missing values
    missing = df.isnull().sum()
    validation['missing_values'] = {
        'total': int(missing.sum()),
        'by_column': missing[missing > 0].to_dict()
    }
    logger.info(f"\n1. Missing values: {missing.sum()}")
    
    # 2. Negative values
    negative_count = (df['Value'] < 0).sum()
    validation['negative_values'] = int(negative_count)
    
    if negative_count > 0:
        neg_series = df[df['Value'] < 0].groupby(['Country', 'Product']).size()
        logger.info(f"\n2. Negative values: {negative_count}")
        logger.info(f"   Affected series:")
        for (country, product), count in neg_series.items():
            logger.info(f"   - Country {country}, Product {product}: {count} negatives")
    else:
        logger.info(f"\n2. Negative values: 0")
    
    # 3. Series completeness
    series_counts = df.groupby(['Country', 'Product']).size()
    expected_quarters = len(df['date'].unique())
    complete_series = (series_counts == expected_quarters).sum()
    
    validation['series_completeness'] = {
        'total_series': len(series_counts),
        'complete_series': int(complete_series),
        'expected_quarters': expected_quarters
    }
    
    logger.info(f"\n3. Series completeness:")
    logger.info(f"   Total series: {len(series_counts)}")
    logger.info(f"   Complete series: {complete_series}/{len(series_counts)}")
    logger.info(f"   Expected quarters per series: {expected_quarters}")
    
    if complete_series < len(series_counts):
        incomplete = series_counts[series_counts != expected_quarters]
        logger.warning(f"   Incomplete series:")
        for (country, product), count in incomplete.items():
            logger.warning(f"   - Country {country}, Product {product}: {count} quarters")
    
    # 4. Value statistics
    validation['value_stats'] = {
        'min': float(df['Value'].min()),
        'max': float(df['Value'].max()),
        'mean': float(df['Value'].mean()),
        'median': float(df['Value'].median()),
        'std': float(df['Value'].std())
    }
    
    logger.info(f"\n4. Value statistics:")
    logger.info(f"   Min: {df['Value'].min():.2f}")
    logger.info(f"   Max: {df['Value'].max():.2f}")
    logger.info(f"   Mean: {df['Value'].mean():.2f}")
    logger.info(f"   Median: {df['Value'].median():.2f}")
    logger.info(f"   Std: {df['Value'].std():.2f}")
    
    # 5. Duplicates
    duplicates = df.duplicated(subset=['date', 'Country', 'Product']).sum()
    validation['duplicates'] = int(duplicates)
    
    logger.info(f"\n5. Duplicates: {duplicates}")
    
    if duplicates == 0 and missing.sum() == 0 and complete_series == len(series_counts):
        logger.info("\n✓ DATA QUALITY: EXCELLENT")
    else:
        logger.warning("\n⚠ DATA QUALITY: ISSUES DETECTED (see above)")
    
    return validation


def save_outputs(
    df: pd.DataFrame,
    validation: Dict,
    output_dir: str
) -> None:
    """Save processed data and validation report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed long format dataframe
    validation : Dict
        Validation metrics
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 80)
    
    # Save long format data
    data_path = os.path.join(output_dir, 'quarterly_data_long.csv')
    df.to_csv(data_path, index=False)
    logger.info(f"✓ Saved long format data to: {data_path}")
    
    # Save validation report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'time_period': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat(),
            'total_quarters': int(len(df['date'].unique()))
        },
        'series_info': {
            'total_series': int(df.groupby(['Country', 'Product']).ngroups),
            'countries': sorted(df['Country'].unique().tolist()),
            'products': sorted(df['Product'].unique().tolist())
        },
        'validation': validation
    }
    
    report_path = os.path.join(output_dir, 'data_validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✓ Saved validation report to: {report_path}")


def main_pipeline(input_path: str, output_dir: str) -> None:
    """Main data loading and reshaping pipeline.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file (wide format)
    output_dir : str
        Output directory for processed data
    """
    logger.info("=" * 80)
    logger.info("CASE 1B: DATA LOADING AND RESHAPING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load wide format data
    df_wide = load_wide_format(input_path)
    
    # Step 2: Reshape to long format
    df_long = reshape_to_long_format(df_wide)
    
    # Step 3: Extract Country and Product
    df_long = extract_country_product(df_long)
    
    # Step 4: Add time features
    df_long = add_time_features(df_long)
    
    # Step 5: Validate data quality
    validation = validate_data_quality(df_long)
    
    # Step 6: Save outputs
    save_outputs(df_long, validation, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DATA LOADING AND RESHAPING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nFinal data shape: {df_long.shape}")
    logger.info(f"Total observations: {len(df_long)}")
    logger.info(f"Series: {df_long.groupby(['Country', 'Product']).ngroups}")
    logger.info(f"Time period: {df_long['YEAR'].min()} to {df_long['YEAR'].max()}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Load and reshape quarterly data for Case1B'
    )
    p.add_argument(
        '--input', '-i',
        default=os.path.join('..', 'data', 'raw', 'Case1B_QuarterlyData.csv'),
        help='Path to input CSV file (wide format)'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'data', 'processed'),
        help='Output directory for processed data'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(args.input, args.output)


if __name__ == '__main__':
    main()
