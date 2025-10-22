"""Model training and evaluation for Case1A Monthly Forecast.

This script trains multiple forecasting models, evaluates their performance,
and selects the best model for generating 2020 forecasts.

Usage:
    python model_training.py --train ../data/processed/train_data.csv \
                            --test ../data/processed/test_data.csv \
                            --output ../models/

The script trains:
    - Ridge Regression (baseline linear model)
    - Lasso Regression (sparse linear model)
    - Decision Tree (non-linear, interpretable)
    - Random Forest (ensemble, robust)
    - Gradient Boosting (sequential ensemble, often best for tabular data)

Outputs:
    - model_comparison.csv: Performance metrics for all models
    - best_model.pkl: Serialized best-performing model
    - predictions_test.csv: Test set predictions from all models
    - training_report.json: Comprehensive training metadata
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_train_test_data(
    train_path: str,
    test_path: str,
    metadata_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load train and test datasets with feature metadata.
    
    Parameters
    ----------
    train_path : str
        Path to train_data.csv
    test_path : str
        Path to test_data.csv
    metadata_path : str, optional
        Path to feature_metadata.json
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        (train_df, test_df, feature_columns)
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Training data: {train_df.shape}")
    logger.info(f"Test data: {test_df.shape}")
    
    # Load feature metadata if available
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata['features']['time_features'] + \
                      metadata['features']['lag_features'] + \
                      metadata['features']['rolling_features'] + \
                      metadata['features']['categorical_features']
        logger.info(f"Loaded {len(feature_cols)} features from metadata")
    else:
        # Define features manually
        exclude_cols = ['date', 'Country', 'Product', 'Value', 'YEAR', 'MONTH', 'Country_Product']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        logger.info(f"Defined {len(feature_cols)} features from column names")
    
    return train_df, test_df, feature_cols


def prepare_X_y(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare feature matrices and target vectors.
    
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
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, y_train, X_test, y_test)
    """
    logger.info("\n" + "=" * 80)
    logger.info("PREPARING FEATURE MATRICES")
    logger.info("=" * 80)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['Value'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Value'].values
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    
    # Validate no NaN
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("NaN values detected in training data!")
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        raise ValueError("NaN values detected in test data!")
    
    logger.info("✓ No NaN values in feature matrices")
    
    return X_train, y_train, X_test, y_test


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Name of the model
    split : str
        'train' or 'test'
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE with zero protection
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    
    # WAPE (Weighted Absolute Percentage Error)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    return {
        'Model': model_name,
        'Split': split,
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'WAPE': float(wape),
        'R2': float(r2)
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Train multiple models and evaluate performance.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
        
    Returns
    -------
    Tuple[Dict, pd.DataFrame, Dict]
        (trained_models, results_df, predictions)
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL TRAINING AND EVALUATION")
    logger.info("=" * 80)
    
    # Define models with optimized hyperparameters
    models = {
        'Ridge': Ridge(
            alpha=1.0,
            random_state=42
        ),
        'Lasso': Lasso(
            alpha=1.0,
            max_iter=5000,
            random_state=42
        ),
        'DecisionTree': DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    results = []
    predictions = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*60}")
        
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Store predictions
        predictions[f'{name}_train'] = y_train_pred
        predictions[f'{name}_test'] = y_test_pred
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred, name, 'Train')
        test_metrics = calculate_metrics(y_test, y_test_pred, name, 'Test')
        
        results.append(train_metrics)
        results.append(test_metrics)
        
        # Log performance
        logger.info(f"Train - MAE: {train_metrics['MAE']:.2f}, "
                   f"RMSE: {train_metrics['RMSE']:.2f}, "
                   f"MAPE: {train_metrics['MAPE']:.2f}%, "
                   f"R²: {train_metrics['R2']:.4f}")
        logger.info(f"Test  - MAE: {test_metrics['MAE']:.2f}, "
                   f"RMSE: {test_metrics['RMSE']:.2f}, "
                   f"MAPE: {test_metrics['MAPE']:.2f}%, "
                   f"R²: {test_metrics['R2']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info("\n" + results_df.to_string(index=False))
    
    return trained_models, results_df, predictions


def select_best_model(
    results_df: pd.DataFrame,
    trained_models: Dict,
    metric: str = 'MAPE'
) -> Tuple[str, object, Dict]:
    """Select the best performing model based on test set metric.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all model metrics
    trained_models : Dict
        Dictionary of trained model objects
    metric : str
        Metric to use for selection ('MAPE', 'MAE', 'RMSE')
        
    Returns
    -------
    Tuple[str, object, Dict]
        (best_model_name, best_model_object, best_metrics)
    """
    logger.info("\n" + "=" * 80)
    logger.info("BEST MODEL SELECTION")
    logger.info("=" * 80)
    
    # Filter to test results only
    test_results = results_df[results_df['Split'] == 'Test'].copy()
    
    # Find best model (lowest metric value)
    best_idx = test_results[metric].idxmin()
    best_model_name = test_results.loc[best_idx, 'Model']
    best_metrics = test_results.loc[best_idx].to_dict()
    
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"   Selection Metric: {metric}")
    logger.info(f"   Test {metric}: {best_metrics[metric]:.2f}{'%' if metric in ['MAPE', 'WAPE'] else ''}")
    logger.info(f"   Test MAE: {best_metrics['MAE']:.2f}")
    logger.info(f"   Test RMSE: {best_metrics['RMSE']:.2f}")
    logger.info(f"   Test R²: {best_metrics['R2']:.4f}")
    
    # Compare to industry standards
    if best_metrics['MAPE'] < 10:
        logger.info(f"\n✅ Excellent performance! (Industry standard: 10-15% MAPE)")
    elif best_metrics['MAPE'] < 15:
        logger.info(f"\n✓ Good performance (within industry standard: 10-15% MAPE)")
    else:
        logger.warning(f"\n⚠️  Performance below industry standard (10-15% MAPE)")
    
    return best_model_name, trained_models[best_model_name], best_metrics


def save_results(
    trained_models: Dict,
    results_df: pd.DataFrame,
    predictions: Dict,
    best_model_name: str,
    best_metrics: Dict,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str
) -> None:
    """Save all training results, models, and predictions.
    
    Parameters
    ----------
    trained_models : Dict
        Dictionary of trained models
    results_df : pd.DataFrame
        Model comparison results
    predictions : Dict
        Dictionary of predictions
    best_model_name : str
        Name of best performing model
    best_metrics : Dict
        Best model metrics
    test_df : pd.DataFrame
        Test dataframe with metadata
    feature_cols : List[str]
        List of feature columns
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    # 1. Save model comparison
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    results_df.to_csv(comparison_path, index=False)
    logger.info(f"✓ Saved model comparison to: {comparison_path}")
    
    # 2. Save best model
    best_model_path = os.path.join(output_dir, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(trained_models[best_model_name], f)
    logger.info(f"✓ Saved best model ({best_model_name}) to: {best_model_path}")
    
    # 3. Save all models (optional)
    models_dir = os.path.join(output_dir, 'all_models')
    os.makedirs(models_dir, exist_ok=True)
    for name, model in trained_models.items():
        model_path = os.path.join(models_dir, f'{name.lower()}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    logger.info(f"✓ Saved all models to: {models_dir}/")
    
    # 4. Save test predictions
    pred_df = test_df[['date', 'Country', 'Product', 'Value']].copy()
    for name in trained_models.keys():
        pred_df[f'{name}_prediction'] = predictions[f'{name}_test']
    
    pred_path = os.path.join(output_dir, 'predictions_test.csv')
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"✓ Saved test predictions to: {pred_path}")
    
    # 5. Save training report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'models_trained': list(trained_models.keys()),
        'train_samples': len(predictions['Ridge_train']),
        'test_samples': len(predictions['Ridge_test'])
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✓ Saved training report to: {report_path}")


def main_pipeline(
    train_path: str,
    test_path: str,
    output_dir: str,
    metadata_path: str = None,
    selection_metric: str = 'MAPE'
) -> None:
    """Main training pipeline.
    
    Parameters
    ----------
    train_path : str
        Path to training data CSV
    test_path : str
        Path to test data CSV
    output_dir : str
        Output directory for results
    metadata_path : str, optional
        Path to feature metadata JSON
    selection_metric : str
        Metric to use for model selection
    """
    logger.info("=" * 80)
    logger.info("CASE 1A: MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load data
    train_df, test_df, feature_cols = load_train_test_data(
        train_path, test_path, metadata_path
    )
    
    # Step 2: Prepare X, y
    X_train, y_train, X_test, y_test = prepare_X_y(
        train_df, test_df, feature_cols
    )
    
    # Step 3: Train and evaluate models
    trained_models, results_df, predictions = train_and_evaluate_models(
        X_train, y_train, X_test, y_test
    )
    
    # Step 4: Select best model
    best_model_name, best_model, best_metrics = select_best_model(
        results_df, trained_models, selection_metric
    )
    
    # Step 5: Save results
    save_results(
        trained_models, results_df, predictions,
        best_model_name, best_metrics,
        test_df, feature_cols, output_dir
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ MODEL TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Test MAPE: {best_metrics['MAPE']:.2f}%")
    logger.info(f"Test MAE: {best_metrics['MAE']:.2f}")
    logger.info(f"Test R²: {best_metrics['R2']:.4f}")
    logger.info(f"\nResults saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description='Train forecasting models for Case1A'
    )
    p.add_argument(
        '--train',
        default=os.path.join('..', 'data', 'processed', 'train_data.csv'),
        help='Path to training data CSV'
    )
    p.add_argument(
        '--test',
        default=os.path.join('..', 'data', 'processed', 'test_data.csv'),
        help='Path to test data CSV'
    )
    p.add_argument(
        '--metadata',
        default=os.path.join('..', 'data', 'processed', 'feature_metadata.json'),
        help='Path to feature metadata JSON'
    )
    p.add_argument(
        '--output', '-o',
        default=os.path.join('..', 'models'),
        help='Output directory for models and results'
    )
    p.add_argument(
        '--metric',
        default='MAPE',
        choices=['MAPE', 'MAE', 'RMSE'],
        help='Metric for model selection'
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    main_pipeline(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output,
        metadata_path=args.metadata if os.path.exists(args.metadata) else None,
        selection_metric=args.metric
    )


if __name__ == '__main__':
    main()
