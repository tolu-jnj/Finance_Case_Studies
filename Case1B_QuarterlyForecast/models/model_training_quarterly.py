"""Model training and evaluation for Case1B Quarterly Forecast.

This script trains multiple models on quarterly data and compares performance.

Usage:
    python model_training_quarterly.py \
        --train ../data/processed/train_data_quarterly.csv \
        --test ../data/processed/test_data_quarterly.csv \
        --output ../models/

Outputs:
    - model_comparison_quarterly.csv: Performance metrics for all models
    - all_models/: Directory with trained models (.pkl files)
    - predictions_test_quarterly.csv: Test predictions
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets.
    
    Parameters
    ----------
    train_path : str
        Path to train_data_quarterly.csv
    test_path : str
        Path to test_data_quarterly.csv
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train data: {train_df.shape}")
    logger.info(f"Test data: {test_df.shape}")
    
    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare feature matrices and target vectors.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, y_train, X_test, y_test)
    """
    logger.info("\n" + "=" * 80)
    logger.info("PREPARING FEATURES")
    logger.info("=" * 80)
    
    # Define columns to exclude
    exclude_cols = ['date', 'Country', 'Product', 'Value', 'YEAR', 'QUARTER']
    
    # Get feature columns
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")
    
    # Extract features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['Value'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Value'].values
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # WAPE (Weighted Absolute Percentage Error)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'WAPE': wape,
        'R2': r2
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict, pd.DataFrame]:
    """Train multiple models and evaluate performance.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
        
    Returns
    -------
    Tuple[Dict, pd.DataFrame]
        (trained_models, comparison_df)
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODELS")
    logger.info("=" * 80)
    
    # Define models with adjusted hyperparameters for quarterly data
    models = {
        'Ridge': Ridge(alpha=10.0),  # Higher regularization
        'Lasso': Lasso(alpha=1.0, max_iter=5000),  # Higher regularization
        'DecisionTree': DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)
        
        # Store results
        results.append({
            'Model': name,
            'Split': 'Train',
            **train_metrics
        })
        results.append({
            'Model': name,
            'Split': 'Test',
            **test_metrics
        })
        
        # Store trained model
        trained_models[name] = model
        
        logger.info(f"  Train MAPE: {train_metrics['MAPE']:.2f}%")
        logger.info(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")
        logger.info(f"  Test R²: {test_metrics['R2']:.4f}")
    
    comparison_df = pd.DataFrame(results)
    
    return trained_models, comparison_df


def display_results(comparison_df: pd.DataFrame) -> None:
    """Display model comparison results.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Identify best model
    test_results = comparison_df[comparison_df['Split'] == 'Test'].copy()
    best_model = test_results.loc[test_results['MAPE'].idxmin()]
    
    logger.info(f"\n🏆 BEST MODEL: {best_model['Model']}")
    logger.info(f"   Test MAPE: {best_model['MAPE']:.2f}%")
    logger.info(f"   Test R²: {best_model['R2']:.4f}")


def save_outputs(
    trained_models: Dict,
    comparison_df: pd.DataFrame,
    output_dir: str
) -> None:
    """Save models and results.
    
    Parameters
    ----------
    trained_models : Dict
        Dictionary of trained models
    comparison_df : pd.DataFrame
        Comparison results
    output_dir : str
        Output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model comparison
    comparison_path = os.path.join(output_dir, 'model_comparison_quarterly.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"✓ Saved comparison to: {comparison_path}")
    
    # Save models
    models_dir = os.path.join(output_dir, 'all_models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        model_path = os.path.join(models_dir, f'{name.lower()}_model_quarterly.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved {name} model to: {model_path}")
    
    # Identify and save best model
    test_results = comparison_df[comparison_df['Split'] == 'Test']
    best_model_name = test_results.loc[test_results['MAPE'].idxmin(), 'Model']
    best_model = trained_models[best_model_name]
    
    best_model_path = os.path.join(output_dir, 'best_model_quarterly.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"✓ Saved best model ({best_model_name}) to: {best_model_path}")


def main_pipeline(
    train_path: str,
    test_path: str,
    output_dir: str
) -> None:
    """Main training pipeline.
    
    Parameters
    ----------
    train_path : str
        Path to training data
    test_path : str
        Path to test data
    output_dir : str
        Output directory
    """
    logger.info("=" * 80)
    logger.info("CASE 1B: MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Step 2: Prepare features
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)
    
    # Step 3: Train and evaluate models
    trained_models, comparison_df = train_and_evaluate_models(
        X_train, y_train, X_test, y_test
    )
    
    # Step 4: Display results
    display_results(comparison_df)
    
    # Step 5: Save outputs
    save_outputs(trained_models, comparison_df, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ MODEL TRAINING COMPLETE")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Train models for Case1B quarterly forecasting'
    )
    p.add_argument(
        '--train',
        default=os.path.join('..', 'data', 'processed', 'train_data_quarterly.csv'),
        help='Path to training data'
    )
    p.add_argument(
        '--test',
        default=os.path.join('..', 'data', 'processed', 'test_data_quarterly.csv'),
        help='Path to test data'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'models'),
        help='Output directory for models'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(args.train, args.test, args.output)


if __name__ == '__main__':
    main()
