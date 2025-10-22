"""Forecast generation for Case1A Monthly Forecast (2020 predictions).

This script loads the trained gradientboosting model and generates recursive multi-step
forecasts for all 15 country-product combinations for the year 2020.

Usage:
    python forecast_generation.py --model ../models/all_models/gradientboosting_model.pkl \
                                   --features ../data/processed/features_table.csv \
                                   --metadata ../data/processed/feature_metadata.json \
                                   --output ../forecasts/

The script uses recursive forecasting where:
- Month 1 uses actual 2019 data for lag features
- Month 2 uses Month 1 forecast for lag_1, actual 2019 for others
- Month 12 uses all previous forecasts in lag features

Outputs:
    - forecasts_2020.csv: All 180 forecasts (15 series × 12 months)
    - forecast_summary.csv: Summary statistics by series
    - forecast_report.json: Metadata and quality metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_model(model_path: str) -> object:
    """Load the trained model from pickle file.
    
    Parameters
    ----------
    model_path : str
        Path to the serialized model (.pkl)
        
    Returns
    -------
    object
        Trained model object
    """
    logger.info(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"✓ Model loaded: {type(model).__name__}")
    return model


def load_historical_data(
    features_path: str,
    metadata_path: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Load historical data and feature metadata.
    
    Parameters
    ----------
    features_path : str
        Path to features_table.csv
    metadata_path : str
        Path to feature_metadata.json
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (historical_data, feature_columns)
    """
    logger.info("=" * 80)
    logger.info("LOADING HISTORICAL DATA")
    logger.info("=" * 80)
    
    # Load features table
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded features table: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Load metadata to get feature columns
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_cols = (metadata['features']['time_features'] +
                   metadata['features']['lag_features'] +
                   metadata['features']['rolling_features'] +
                   metadata['features']['categorical_features'])
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    return df, feature_cols


def create_forecast_features(
    country: str,
    product: str,
    forecast_date: pd.Timestamp,
    history_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Create feature vector for a single forecast.
    
    Parameters
    ----------
    country : str
        Country name
    product : str
        Product name
    forecast_date : pd.Timestamp
        Date to forecast
    history_df : pd.DataFrame
        Historical data for this series (includes actual + previous forecasts)
    feature_cols : List[str]
        List of feature column names
        
    Returns
    -------
    pd.DataFrame
        Single-row dataframe with all features
    """
    # Time-based features
    year = forecast_date.year
    month = forecast_date.month
    quarter = (month - 1) // 3 + 1
    day_of_year = forecast_date.dayofyear
    
    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)
    
    # Months since start (relative to 2011-01)
    months_since_start = (year - 2011) * 12 + month
    
    # Lag features (from history)
    sorted_history = history_df.sort_values('date')
    lag_1 = sorted_history['Value'].iloc[-1] if len(sorted_history) >= 1 else np.nan
    lag_2 = sorted_history['Value'].iloc[-2] if len(sorted_history) >= 2 else np.nan
    lag_3 = sorted_history['Value'].iloc[-3] if len(sorted_history) >= 3 else np.nan
    lag_6 = sorted_history['Value'].iloc[-6] if len(sorted_history) >= 6 else np.nan
    lag_12 = sorted_history['Value'].iloc[-12] if len(sorted_history) >= 12 else np.nan
    
    # Rolling features (from history)
    last_3 = sorted_history['Value'].tail(3)
    last_6 = sorted_history['Value'].tail(6)
    last_12 = sorted_history['Value'].tail(12)
    
    roll_mean_3 = last_3.mean() if len(last_3) > 0 else np.nan
    roll_std_3 = last_3.std() if len(last_3) > 1 else np.nan
    roll_min_3 = last_3.min() if len(last_3) > 0 else np.nan
    roll_max_3 = last_3.max() if len(last_3) > 0 else np.nan
    
    roll_mean_6 = last_6.mean() if len(last_6) > 0 else np.nan
    roll_std_6 = last_6.std() if len(last_6) > 1 else np.nan
    roll_min_6 = last_6.min() if len(last_6) > 0 else np.nan
    roll_max_6 = last_6.max() if len(last_6) > 0 else np.nan
    
    roll_mean_12 = last_12.mean() if len(last_12) > 0 else np.nan
    roll_std_12 = last_12.std() if len(last_12) > 1 else np.nan
    roll_min_12 = last_12.min() if len(last_12) > 0 else np.nan
    roll_max_12 = last_12.max() if len(last_12) > 0 else np.nan
    
    # Categorical features (from history)
    country_code = sorted_history['Country_code'].iloc[-1]
    product_code = sorted_history['Product_code'].iloc[-1]
    country_product_code = sorted_history['CountryProduct_code'].iloc[-1]
    
    # Create feature dictionary
    features = {
        'year': year,
        'month': month,
        'quarter': quarter,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'quarter_sin': quarter_sin,
        'quarter_cos': quarter_cos,
        'months_since_start': months_since_start,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'lag_6': lag_6,
        'lag_12': lag_12,
        'roll_mean_3': roll_mean_3,
        'roll_std_3': roll_std_3,
        'roll_min_3': roll_min_3,
        'roll_max_3': roll_max_3,
        'roll_mean_6': roll_mean_6,
        'roll_std_6': roll_std_6,
        'roll_min_6': roll_min_6,
        'roll_max_6': roll_max_6,
        'roll_mean_12': roll_mean_12,
        'roll_std_12': roll_std_12,
        'roll_min_12': roll_min_12,
        'roll_max_12': roll_max_12,
        'Country_code': country_code,
        'Product_code': product_code,
        'CountryProduct_code': country_product_code
    }
    
    # Create dataframe with features in correct order
    feature_df = pd.DataFrame([features])[feature_cols]
    
    return feature_df


def generate_recursive_forecasts(
    model: object,
    historical_df: pd.DataFrame,
    feature_cols: List[str],
    forecast_year: int = 2020,
    n_months: int = 12
) -> pd.DataFrame:
    """Generate recursive multi-step forecasts for 2020.
    
    Parameters
    ----------
    model : object
        Trained forecasting model
    historical_df : pd.DataFrame
        Complete historical data (2011-2019)
    feature_cols : List[str]
        List of feature column names
    forecast_year : int
        Year to forecast
    n_months : int
        Number of months to forecast
        
    Returns
    -------
    pd.DataFrame
        Forecast results with columns: date, Country, Product, forecast
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING 2020 FORECASTS (RECURSIVE)")
    logger.info("=" * 80)
    
    all_forecasts = []
    
    # Get all country-product combinations
    combinations = historical_df[['Country', 'Product']].drop_duplicates()
    
    for idx, row in combinations.iterrows():
        country = row['Country']
        product = row['Product']
        
        logger.info(f"\nForecasting: {country} - {product}")
        
        # Get historical data for this series
        series_history = historical_df[
            (historical_df['Country'] == country) &
            (historical_df['Product'] == product)
        ].copy()
        
        series_history = series_history.sort_values('date')
        
        # Generate forecasts month by month
        for month in range(1, n_months + 1):
            forecast_date = pd.Timestamp(f'{forecast_year}-{month:02d}-01')
            
            # Create features for this forecast
            X_forecast = create_forecast_features(
                country, product, forecast_date, series_history, feature_cols
            )
            
            # Make prediction
            forecast_value = model.predict(X_forecast)[0]
            
            # Store forecast
            all_forecasts.append({
                'date': forecast_date,
                'Country': country,
                'Product': product,
                'forecast': forecast_value,
                'month': month,
                'year': forecast_year
            })
            
            # Add forecast to history for next iteration
            new_row = pd.DataFrame({
                'date': [forecast_date],
                'Country': [country],
                'Product': [product],
                'Value': [forecast_value],
                'Country_code': [series_history['Country_code'].iloc[-1]],
                'Product_code': [series_history['Product_code'].iloc[-1]],
                'CountryProduct_code': [series_history['CountryProduct_code'].iloc[-1]]
            })
            
            series_history = pd.concat([series_history, new_row], ignore_index=True)
        
        logger.info(f"  ✓ Generated {n_months} forecasts")
    
    forecast_df = pd.DataFrame(all_forecasts)
    
    logger.info("\n" + "=" * 80)
    logger.info("FORECAST GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total forecasts: {len(forecast_df)}")
    logger.info(f"Series: {forecast_df.groupby(['Country', 'Product']).ngroups}")
    logger.info(f"Date range: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
    
    return forecast_df


def generate_forecast_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by series.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast results
        
    Returns
    -------
    pd.DataFrame
        Summary statistics by Country-Product
    """
    summary = forecast_df.groupby(['Country', 'Product'])['forecast'].agg([
        ('mean_forecast', 'mean'),
        ('std_forecast', 'std'),
        ('min_forecast', 'min'),
        ('max_forecast', 'max'),
        ('range_forecast', lambda x: x.max() - x.min())
    ]).reset_index()
    
    return summary


def save_forecasts(
    forecast_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_name: str,
    output_dir: str
) -> None:
    """Save forecast results and metadata.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast results
    summary_df : pd.DataFrame
        Summary statistics
    model_name : str
        Name of the model used
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING FORECAST RESULTS")
    logger.info("=" * 80)
    
    # Save main forecasts
    forecast_path = os.path.join(output_dir, 'forecasts_2020.csv')
    forecast_df.to_csv(forecast_path, index=False)
    logger.info(f"✓ Saved forecasts to: {forecast_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'forecast_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Saved summary to: {summary_path}")
    
    # Save metadata report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model': model_name,
        'forecast_year': 2020,
        'n_forecasts': len(forecast_df),
        'n_series': int(forecast_df.groupby(['Country', 'Product']).ngroups),
        'forecast_range': {
            'min': float(forecast_df['forecast'].min()),
            'max': float(forecast_df['forecast'].max()),
            'mean': float(forecast_df['forecast'].mean()),
            'median': float(forecast_df['forecast'].median())
        },
        'series_summary': summary_df.to_dict('records')
    }
    
    report_path = os.path.join(output_dir, 'forecast_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✓ Saved report to: {report_path}")


def main_pipeline(
    model_path: str,
    features_path: str,
    metadata_path: str,
    output_dir: str
) -> None:
    """Main forecasting pipeline.
    
    Parameters
    ----------
    model_path : str
        Path to trained model
    features_path : str
        Path to features table
    metadata_path : str
        Path to feature metadata
    output_dir : str
        Output directory
    """
    logger.info("=" * 80)
    logger.info("CASE 1A: 2020 FORECAST GENERATION")
    logger.info("=" * 80)
    
    # Step 1: Load model
    model = load_model(model_path)
    model_name = type(model).__name__
    
    # Step 2: Load historical data
    historical_df, feature_cols = load_historical_data(features_path, metadata_path)
    
    # Step 3: Generate forecasts
    forecast_df = generate_recursive_forecasts(
        model, historical_df, feature_cols, forecast_year=2020, n_months=12
    )
    
    # Step 4: Generate summary
    summary_df = generate_forecast_summary(forecast_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("FORECAST SUMMARY BY SERIES")
    logger.info("=" * 80)
    print("\n" + summary_df.to_string(index=False))
    
    # Step 5: Save results
    save_forecasts(forecast_df, summary_df, model_name, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ FORECAST GENERATION COMPLETE")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Generate 2020 forecasts for Case1A'
    )
    p.add_argument(
        '--model',
        default=os.path.join('..', 'models', 'all_models', 'lasso_model.pkl'),
        help='Path to trained model pickle file'
    )
    p.add_argument(
        '--features',
        default=os.path.join('..', 'data', 'processed', 'features_table.csv'),
        help='Path to features table CSV'
    )
    p.add_argument(
        '--metadata',
        default=os.path.join('..', 'data', 'processed', 'feature_metadata.json'),
        help='Path to feature metadata JSON'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'forecasts'),
        help='Output directory for forecasts'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(
        model_path=args.model,
        features_path=args.features,
        metadata_path=args.metadata,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
