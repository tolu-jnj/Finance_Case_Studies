#!/usr/bin/env python3

"""Feature engineering for Case1A Monthly Forecast.

This script reads the raw monthly data, constructs time-based features,
lag and rolling features per Country-Product series, encodes categorical
variables, and writes the resulting feature table as CSV to
`../data/processed/features_table.csv` by default.

CORRECTED VERSION: Fixes merge issue to maintain 1,620 rows (one per Country-Product-Month).

Usage:
    python feature_engineering.py --input ../data/raw/Case1A_MonthlyData.csv \
        --output ../data/processed/features_table.csv

The script is written to be importable; call `build_and_save_features()` from
other code if you want to integrate into pipelines.
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Sequence
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data(path: str) -> pd.DataFrame:
    """Load dataset and ensure a monthly `date` column exists.

    The raw data in this repo includes YEAR and MONTH columns. If a `date`
    column exists it will be used. Otherwise we construct it as the first
    day of the month.
    
    Data Quality Checks:
    - Verifies no duplicates per (Country, Product, Month)
    - Confirms presence of required columns
    """
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        # handle YEAR and MONTH
        if {'YEAR', 'MONTH'}.issubset(df.columns):
            df['DAY'] = 1
            df['date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
            df = df.drop(columns=['DAY'])
        else:
            raise ValueError('Input file must contain a `date` column or YEAR and MONTH columns')
    df['date'] = pd.to_datetime(df['date'])
    
    # ensure Value column exists
    if 'Value' not in df.columns:
        raise ValueError('Input file must contain a `Value` column with target values')
    
    # DATA QUALITY CHECK: Verify no duplicates
    duplicate_check = df.groupby(['Country', 'Product', 'date']).size()
    if (duplicate_check > 1).any():
        n_dupes = (duplicate_check > 1).sum()
        logger.warning(f"Found {n_dupes} duplicate (Country, Product, date) combinations. "
                      f"Aggregating by mean to ensure one row per series-month.")
        # Aggregate by mean if duplicates found
        df = df.groupby(['Country', 'Product', 'date'], as_index=False).agg({
            'Value': 'mean',
            **{col: 'first' for col in df.columns if col not in ['Country', 'Product', 'date', 'Value']}
        })
    
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Add time-based features (year, month, quarter, cyclical encodings).
    
    Features created:
    - year, month, quarter: Basic temporal components
    - month_sin/cos: Cyclical encoding (Dec ≈ Jan in feature space)
    - quarter_sin/cos: Cyclical quarterly encoding
    - months_since_start: Linear trend proxy (0, 1, 2, ...)
    
    Expects df to contain a `date` column (datetime).
    """
    d = df.copy()
    d['year'] = d[date_col].dt.year
    d['month'] = d[date_col].dt.month
    d['quarter'] = d[date_col].dt.quarter
    
    # Cyclical encoding for month (1..12)
    # Why: Treats December (12) and January (1) as close neighbors
    d['month_sin'] = np.sin(2 * np.pi * (d['month'] / 12))
    d['month_cos'] = np.cos(2 * np.pi * (d['month'] / 12))
    
    # Cyclical for quarter (1..4)
    d['quarter_sin'] = np.sin(2 * np.pi * (d['quarter'] / 4))
    d['quarter_cos'] = np.cos(2 * np.pi * (d['quarter'] / 4))
    
    # Months since start for linear trend capture
    d = d.sort_values(['Country', 'Product', date_col])
    d['months_since_start'] = d.groupby(['Country', 'Product'])[date_col].transform(
        lambda x: ((x.dt.year - x.dt.year.min()) * 12 + (x.dt.month - x.dt.month.min()))
    )
    
    logger.info(f"Created {9} time-based features")
    return d


def add_lag_features(df: pd.DataFrame, value_col: str = 'Value', 
                     lags: Sequence[int] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    """Add lag features per Country-Product series.
    
    Features created:
    - lag_1: Previous month value (autoregressive AR(1))
    - lag_2: 2 months ago (short-term memory)
    - lag_3: 3 months ago (quarterly lookback)
    - lag_6: 6 months ago (semi-annual comparison)
    - lag_12: 12 months ago (year-over-year comparison)
    
    The function assumes monthly frequency and will reindex each series to
    monthly start frequency to ensure shifts align with months.
    """
    out = []
    for (country, product), g in df.groupby(['Country', 'Product']):
        s = g.set_index('date').sort_index()
        # Ensure monthly frequency index for proper shifting
        s = s.asfreq('MS')
        for lag in lags:
            s[f'lag_{lag}'] = s[value_col].shift(lag)
        s = s.reset_index()
        s['Country'] = country
        s['Product'] = product
        out.append(s)
    
    result = pd.concat(out, axis=0, ignore_index=True)
    logger.info(f"Created {len(lags)} lag features")
    return result


def add_rolling_features(df: pd.DataFrame, value_col: str = 'Value', 
                        windows: Sequence[int] = (3, 6, 12)) -> pd.DataFrame:
    """Add rolling statistics (mean, std, min, max) per series.
    
    Features created per window:
    - roll_mean_W: Moving average over W months (trend smoothing)
    - roll_std_W: Rolling volatility (risk indicator)
    - roll_min_W: Recent minimum (support level)
    - roll_max_W: Recent maximum (resistance level)
    
    Windows are specified in months; values will be aligned to the right
    (i.e., include current row). min_periods=1 ensures early months have values.
    """
    out = []
    for (country, product), g in df.groupby(['Country', 'Product']):
        s = g.set_index('date').sort_index()
        s = s.asfreq('MS')
        for w in windows:
            s[f'roll_mean_{w}'] = s[value_col].rolling(window=w, min_periods=1).mean()
            s[f'roll_std_{w}'] = s[value_col].rolling(window=w, min_periods=1).std()
            s[f'roll_min_{w}'] = s[value_col].rolling(window=w, min_periods=1).min()
            s[f'roll_max_{w}'] = s[value_col].rolling(window=w, min_periods=1).max()
        s = s.reset_index()
        s['Country'] = country
        s['Product'] = product
        out.append(s)
    
    result = pd.concat(out, axis=0, ignore_index=True)
    logger.info(f"Created {len(windows) * 4} rolling window features")
    return result


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Country and Product as integer codes and add interaction code.
    
    Features created:
    - Country_code: Integer encoding (0-4 for 5 countries)
    - Product_code: Integer encoding (0-2 for 3 products)
    - CountryProduct_code: Unique code per combination (0-14 for 15 series)
    
    Why: Tree-based models use these to create splits, learning series-specific patterns.
    """
    out = df.copy()
    out['Country_code'] = pd.Categorical(out['Country']).codes
    out['Product_code'] = pd.Categorical(out['Product']).codes
    # Simple interaction code (use tuple mapping for clarity)
    out['Country_Product'] = out['Country'].astype(str) + '|' + out['Product'].astype(str)
    out['CountryProduct_code'] = pd.Categorical(out['Country_Product']).codes
    
    logger.info(f"Created 3 categorical encoding features")
    return out


def build_features(
    df_raw: pd.DataFrame,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
):
    """Full pipeline: time features, lags, rolling, encodings.

    Returns a DataFrame with feature columns merged alongside the original
    `Value` target.
    
    CRITICAL FIX: Merge on (Country, Product, date) to avoid cartesian product.
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Step 1: Time features (applied to full dataset)
    df_time = time_features(df_raw, date_col='date')
    
    # Step 2: Lag features (per Country-Product series)
    df_lags = add_lag_features(
        df_time[['Country', 'Product', 'date', 'Value']], 
        value_col='Value', 
        lags=lags
    )
    
    # Step 3: Rolling features (per Country-Product series)
    df_roll = add_rolling_features(
        df_time[['Country', 'Product', 'date', 'Value']], 
        value_col='Value', 
        windows=rolling_windows
    )
    
    # Step 4: Merge everything on (Country, Product, date)
    # CRITICAL: Must merge on ALL THREE keys to maintain 1:1 mapping
    logger.info("Merging feature dataframes...")
    
    # First merge: time features + lag features
    df_merged = df_time.merge(
        df_lags[['Country', 'Product', 'date'] + [c for c in df_lags.columns if c.startswith('lag_')]],
        on=['Country', 'Product', 'date'],
        how='left'
    )
    
    # Second merge: add rolling features
    df_merged = df_merged.merge(
        df_roll[['Country', 'Product', 'date'] + [c for c in df_roll.columns if c.startswith('roll_')]],
        on=['Country', 'Product', 'date'],
        how='left'
    )
    
    # Verify no duplication occurred
    expected_rows = len(df_raw)
    actual_rows = len(df_merged)
    if actual_rows != expected_rows:
        logger.error(f"Merge created {actual_rows} rows, expected {expected_rows}. Check merge keys!")
        raise ValueError(f"Merge error: Expected {expected_rows} rows, got {actual_rows}")
    
    logger.info(f"✓ Merge successful: {actual_rows} rows maintained")
    
    # Step 5: Encode categoricals
    df_final = encode_categoricals(df_merged)
    
    # Step 6: Sort columns for readability
    cols = [c for c in ['date', 'Country', 'Product', 'Value'] if c in df_final.columns]
    other_cols = sorted([c for c in df_final.columns if c not in cols])
    df_final = df_final[cols + other_cols]
    
    # Final feature count
    feature_cols = [c for c in df_final.columns if c not in ['date', 'Country', 'Product', 'Value', 'YEAR', 'MONTH']]
    logger.info(f"✓ Feature engineering complete: {len(feature_cols)} features created")
    
    return df_final


def build_and_save_features(input_path: str, output_path: str, force: bool = False) -> str:
    """Run the pipeline and save the feature table to CSV.

    Returns the path to the written file.
    
    Quality checks:
    - Verifies output row count matches input
    - Logs feature count and data dimensions
    """
    if os.path.exists(output_path) and not force:
        logger.info('Output already exists at %s (use force=True to overwrite)', output_path)
        return output_path

    df = load_data(input_path)
    input_rows = len(df)
    
    features = build_features(df)
    output_rows = len(features)
    
    # Quality check
    if input_rows != output_rows:
        logger.warning(f"Row count changed: {input_rows} → {output_rows}")
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    features.to_csv(output_path, index=False)
    logger.info('✓ Wrote features to %s (rows=%d, cols=%d)', output_path, features.shape[0], features.shape[1])
    
    # Summary statistics
    feature_cols = [c for c in features.columns if c not in ['date', 'Country', 'Product', 'Value', 'YEAR', 'MONTH']]
    logger.info(f"Feature summary: {len(feature_cols)} features, {features['Country'].nunique()} countries, "
               f"{features['Product'].nunique()} products")
    
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate feature table for Case1A Monthly Forecast')
    p.add_argument('--input', '-i', default=os.path.join('..', 'data', 'raw', 'Case1A_MonthlyData.csv'))
    p.add_argument('--output', '-o', default=os.path.join('..', 'data', 'processed', 'features_table.csv'))
    p.add_argument('--force', action='store_true', help='Overwrite existing output')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_and_save_features(args.input, args.output, force=args.force)


if __name__ == '__main__':
    main()
