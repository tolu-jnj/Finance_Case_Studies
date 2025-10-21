"""Forecast generation for Case1B Quarterly using Gradient Boosting.

This script generates 2020 Q1-Q4 forecasts using recursive multi-step forecasting.

Usage:
    python forecast_generation_quarterly.py \
        --model ../models/all_models/gradientboosting_model_quarterly.pkl \
        --features ../data/processed/features_table_quarterly.csv \
        --metadata ../data/processed/feature_metadata_quarterly.json \
        --output ../forecasts/

Outputs:
    - forecasts_2020_quarterly.csv: 136 forecasts (34 series × 4 quarters)
    - forecast_summary_quarterly.csv: Summary by series
    - forecast_report_quarterly.json: Metadata and statistics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_model(model_path: str):
    """Load trained model.
    
    Parameters
    ----------
    model_path : str
        Path to .pkl model file
        
    Returns
    -------
    model
        Trained sklearn model
    """
    logger.info("=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"✓ Model loaded: {type(model).__name__}")
    logger.info(f"Model parameters: {model.get_params()}")
    
    return model


def load_features_and_metadata(
    features_path: str,
    metadata_path: str
) -> tuple[pd.DataFrame, Dict]:
    """Load historical features and metadata.
    
    Parameters
    ----------
    features_path : str
        Path to features_table_quarterly.csv
    metadata_path : str
        Path to feature_metadata_quarterly.json
        
    Returns
    -------
    tuple[pd.DataFrame, Dict]
        (features_df, metadata_dict)
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING HISTORICAL DATA")
    logger.info("=" * 80)
    
    # Load features
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded features: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['features']
    logger.info(f"Feature columns: {len(feature_cols['time_features']) + len(feature_cols['lag_features']) + len(feature_cols['rolling_features']) + len(feature_cols['categorical_features'])}")
    
    return df, metadata


def create_forecast_features(
    series_history: pd.DataFrame,
    forecast_date: pd.Timestamp,
    feature_cols: List[str],
    quarters_since_start_base: int
) -> np.ndarray:
    """Create features for a single forecast quarter.
    
    Parameters
    ----------
    series_history : pd.DataFrame
        Historical data for the series
    forecast_date : pd.Timestamp
        Date to forecast
    feature_cols : List[str]
        List of feature column names
    quarters_since_start_base : int
        Base value for quarters_since_start
        
    Returns
    -------
    np.ndarray
        Feature vector for prediction
    """
    features = {}
    
    # Time features
    features['year'] = forecast_date.year
    features['quarter'] = forecast_date.quarter
    features['quarters_since_start'] = quarters_since_start_base + ((forecast_date.year - 2010) * 4 + forecast_date.quarter - 1)
    features['quarter_sin'] = np.sin(2 * np.pi * forecast_date.quarter / 4)
    features['quarter_cos'] = np.cos(2 * np.pi * forecast_date.quarter / 4)
    
    # Get recent values for lag and rolling features
    recent_values = series_history['Value'].values[-12:]  # Last 12 quarters
    
    # Lag features
    if len(recent_values) >= 1:
        features['lag_1'] = recent_values[-1]
    if len(recent_values) >= 2:
        features['lag_2'] = recent_values[-2]
    if len(recent_values) >= 4:
        features['lag_4'] = recent_values[-4]
    
    # Rolling features (2, 4, 8 quarters)
    for window in [2, 4, 8]:
        if len(recent_values) >= window:
            window_vals = recent_values[-window:]
            features[f'roll_mean_{window}'] = np.mean(window_vals)
            features[f'roll_std_{window}'] = np.std(window_vals)
            features[f'roll_min_{window}'] = np.min(window_vals)
            features[f'roll_max_{window}'] = np.max(window_vals)
    
    # Categorical features (from series_history)
    features['Country_code'] = series_history['Country_code'].iloc[-1]
    features['Product_code'] = series_history['Product_code'].iloc[-1]
    features['CountryProduct_code'] = series_history['CountryProduct_code'].iloc[-1]
    
    # Create feature vector in correct order
    feature_vector = []
    for col in feature_cols:
        feature_vector.append(features.get(col, 0))  # Default to 0 if missing
    
    return np.array(feature_vector).reshape(1, -1)


def generate_forecasts_for_series(
    model,
    series_history: pd.DataFrame,
    feature_cols: List[str],
    country: str,
    product: str,
    forecast_year: int = 2020
) -> pd.DataFrame:
    """Generate 4 quarterly forecasts for a single series.
    
    Parameters
    ----------
    model
        Trained model
    series_history : pd.DataFrame
        Historical data for the series (through 2019)
    feature_cols : List[str]
        Feature column names
    country : str
        Country identifier
    product : str
        Product identifier
    forecast_year : int
        Year to forecast (default 2020)
        
    Returns
    -------
    pd.DataFrame
        Forecasts for 4 quarters
    """
    forecasts = []
    
    # Get quarters_since_start base from last historical observation
    quarters_since_start_base = series_history['quarters_since_start'].iloc[-1]
    
    # Start with historical data
    forecast_history = series_history.copy()
    
    # Generate forecasts for 4 quarters
    for quarter in range(1, 5):
        forecast_date = pd.Timestamp(f'{forecast_year}-{quarter*3-2:02d}-01')
        
        # Create features
        X_forecast = create_forecast_features(
            forecast_history,
            forecast_date,
            feature_cols,
            quarters_since_start_base
        )
        
        # Generate forecast
        forecast_value = model.predict(X_forecast)[0]
        
        # Store forecast
        forecasts.append({
            'date': forecast_date,
            'Country': country,
            'Product': product,
            'forecast': forecast_value,
            'quarter': quarter,
            'year': forecast_year
        })
        
        # Add forecast to history for next iteration (recursive forecasting)
        new_row = pd.DataFrame({
            'date': [forecast_date],
            'Value': [forecast_value],
            'quarters_since_start': [quarters_since_start_base + (quarter - 1) + (forecast_year - 2010) * 4 + 1],
            'Country_code': [forecast_history['Country_code'].iloc[-1]],
            'Product_code': [forecast_history['Product_code'].iloc[-1]],
            'CountryProduct_code': [forecast_history['CountryProduct_code'].iloc[-1]]
        })
        forecast_history = pd.concat([forecast_history, new_row], ignore_index=True)
    
    return pd.DataFrame(forecasts)


def generate_all_forecasts(
    model,
    features_df: pd.DataFrame,
    metadata: Dict,
    forecast_year: int = 2020
) -> pd.DataFrame:
    """Generate forecasts for all series.
    
    Parameters
    ----------
    model
        Trained model
    features_df : pd.DataFrame
        Historical features
    metadata : Dict
        Feature metadata
    forecast_year : int
        Year to forecast
        
    Returns
    -------
    pd.DataFrame
        All forecasts
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"GENERATING {forecast_year} FORECASTS (RECURSIVE)")
    logger.info("=" * 80)
    
    # Get feature columns in correct order
    feature_meta = metadata['features']
    feature_cols = (feature_meta['time_features'] + 
                   feature_meta['lag_features'] + 
                   feature_meta['rolling_features'] + 
                   feature_meta['categorical_features'])
    
    all_forecasts = []
    
    # Get unique series
    series_list = features_df.groupby(['Country', 'Product']).groups.keys()
    
    logger.info(f"Forecasting for {len(series_list)} series...\n")
    
    for country, product in series_list:
        logger.info(f"Forecasting: Country {country} - Product {product}")
        
        # Get historical data for this series
        series_history = features_df[
            (features_df['Country'] == country) & 
            (features_df['Product'] == product)
        ].copy()
        series_history = series_history.sort_values('date')
        
        # Generate forecasts
        series_forecasts = generate_forecasts_for_series(
            model,
            series_history,
            feature_cols,
            country,
            product,
            forecast_year
        )
        
        all_forecasts.append(series_forecasts)
        logger.info(f"  ✓ Generated {len(series_forecasts)} quarterly forecasts")
    
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    
    logger.info(f"\n✓ Total forecasts generated: {len(forecasts_df)}")
    
    return forecasts_df


def create_summary_statistics(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by series.
    
    Parameters
    ----------
    forecasts_df : pd.DataFrame
        All forecasts
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("FORECAST SUMMARY BY SERIES")
    logger.info("=" * 80)
    
    summary = forecasts_df.groupby(['Country', 'Product'])['forecast'].agg([
        'mean', 'std', 'min', 'max'
    ]).reset_index()
    summary['range'] = summary['max'] - summary['min']
    summary.columns = ['Country', 'Product', 'mean_forecast', 'std_forecast', 
                      'min_forecast', 'max_forecast', 'range_forecast']
    
    print("\n" + summary.to_string(index=False))
    
    return summary


def save_forecasts(
    forecasts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_name: str,
    output_dir: str
) -> None:
    """Save forecast results.
    
    Parameters
    ----------
    forecasts_df : pd.DataFrame
        All forecasts
    summary_df : pd.DataFrame
        Summary statistics
    model_name : str
        Name of the model used
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING FORECAST RESULTS")
    logger.info("=" * 80)
    
    # Save forecasts
    forecasts_path = os.path.join(output_dir, 'forecasts_2020_quarterly.csv')
    forecasts_df.to_csv(forecasts_path, index=False)
    logger.info(f"✓ Saved forecasts to: {forecasts_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'forecast_summary_quarterly.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Saved summary to: {summary_path}")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'forecast_year': 2020,
        'n_forecasts': len(forecasts_df),
        'n_series': int(forecasts_df.groupby(['Country', 'Product']).ngroups),
        'n_quarters': 4,
        'forecast_range': {
            'min': float(forecasts_df['forecast'].min()),
            'max': float(forecasts_df['forecast'].max()),
            'mean': float(forecasts_df['forecast'].mean()),
            'median': float(forecasts_df['forecast'].median())
        },
        'series_summary': summary_df.to_dict('records')
    }
    
    report_path = os.path.join(output_dir, 'forecast_report_quarterly.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✓ Saved report to: {report_path}")


def main_pipeline(
    model_path: str,
    features_path: str,
    metadata_path: str,
    output_dir: str
) -> None:
    """Main forecast generation pipeline.
    
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
    logger.info("CASE 1B: 2020 QUARTERLY FORECAST GENERATION")
    logger.info("=" * 80)
    
    # Step 1: Load model
    model = load_model(model_path)
    model_name = type(model).__name__
    
    # Step 2: Load features and metadata
    features_df, metadata = load_features_and_metadata(features_path, metadata_path)
    
    # Step 3: Generate forecasts
    forecasts_df = generate_all_forecasts(model, features_df, metadata, forecast_year=2020)
    
    # Step 4: Create summary
    summary_df = create_summary_statistics(forecasts_df)
    
    # Step 5: Save results
    save_forecasts(forecasts_df, summary_df, model_name, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ FORECAST GENERATION COMPLETE")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Generate 2020 quarterly forecasts'
    )
    p.add_argument(
        '--model',
        default=os.path.join('..', 'models', 'all_models', 'gradientboosting_model_quarterly.pkl'),
        help='Path to trained model (.pkl)'
    )
    p.add_argument(
        '--features',
        default=os.path.join('..', 'data', 'processed', 'features_table_quarterly.csv'),
        help='Path to features table'
    )
    p.add_argument(
        '--metadata',
        default=os.path.join('..', 'data', 'processed', 'feature_metadata_quarterly.json'),
        help='Path to feature metadata'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'forecasts'),
        help='Output directory'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(
        args.model,
        args.features,
        args.metadata,
        args.output
    )


if __name__ == '__main__':
    main()
