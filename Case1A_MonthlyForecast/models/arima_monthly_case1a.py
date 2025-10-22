"""ARIMA Modeling for Case 1A Monthly Data - Final Working Version.

Uses original Case1A_MonthlyData.csv (YEAR, MONTH, Country, Product, Value format).

Usage:
    python arima_case1a_final.py

Outputs:
    - arima_monthly_forecasts.csv
    - arima_monthly_summary.json
    - arima_monthly_diagnostics.png
    - arima_monthly_acf_pacf.png
"""
import json
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = '../data/raw/Case1A_MonthlyData.csv'
OUTPUT_DIR = '../models/arima/'

def main():
    logger.info("="*80)
    logger.info("CASE 1A: SARIMA MODELING (Monthly)")
    logger.info("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    logger.info("\nLoading data from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Select first series: Country_A, Product_X
    series_df = df[(df['Country'] == 'Country_A') & (df['Product'] == 'Product_X')].copy()
    series_df = series_df.sort_values(['YEAR', 'MONTH'])
    
    # Split train (2011-2018) and test (2019)
    train = series_df[series_df['YEAR'] <= 2018]['Value'].reset_index(drop=True)
    test = series_df[series_df['YEAR'] == 2019]['Value'].reset_index(drop=True)
    
    logger.info("Series: Country_A, Product_X")
    logger.info("Train: %d months (2011-2018)", len(train))
    logger.info("Test: %d months (2019)", len(test))
    
    # Step 1: Stationarity
    logger.info("\n" + "="*80)
    logger.info("STEP 1: STATIONARITY TESTING")
    logger.info("="*80)
    adf = adfuller(train, autolag='AIC')
    kpss_result = kpss(train, regression='ct', nlags='auto')
    logger.info("ADF: statistic=%.3f, p=%.3f → %s", adf[0], adf[1], 
                "STATIONARY" if adf[1] < 0.05 else "NON-STATIONARY")
    logger.info("KPSS: statistic=%.3f, p=%.3f → %s", kpss_result[0], kpss_result[1],
                "STATIONARY" if kpss_result[1] > 0.05 else "NON-STATIONARY")
    
    # Step 2: ACF/PACF
    logger.info("\n" + "="*80)
    logger.info("STEP 2: ACF/PACF PLOTS")
    logger.info("="*80)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train, lags=40, ax=axes[0])
    axes[0].set_title('ACF - Monthly Data')
    plot_pacf(train, lags=40, ax=axes[1], method='ywm')
    axes[1].set_title('PACF - Monthly Data')
    plt.tight_layout()
    acf_path = os.path.join(OUTPUT_DIR, 'arima_monthly_acf_pacf.png')
    plt.savefig(acf_path, dpi=150)
    logger.info("Saved: %s", acf_path)
    plt.close()
    
    # Step 3: Fit SARIMA(1,1,1)(1,1,1)12
    logger.info("\n" + "="*80)
    logger.info("STEP 3: FITTING SARIMA(1,1,1)(1,1,1)12")
    logger.info("="*80)
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    logger.info("Model fitted: AIC=%.2f, BIC=%.2f", fitted.aic, fitted.bic)
    
    # Step 4: Diagnostics
    logger.info("\n" + "="*80)
    logger.info("STEP 4: RESIDUAL DIAGNOSTICS")
    logger.info("="*80)
    residuals = fitted.resid
    lb = acorr_ljungbox(residuals, lags=20, return_df=True)
    lb_pass = len(lb[lb['lb_pvalue'] < 0.05]) == 0
    jb_stat, jb_p = stats.jarque_bera(residuals.dropna())
    logger.info("Ljung-Box: %s (p=%.3f at lag 10)", "PASS" if lb_pass else "FAIL",
                lb['lb_pvalue'].iloc[9])
    logger.info("Jarque-Bera: %s (p=%.3f)", "PASS" if jb_p > 0.05 else "FAIL", jb_p)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='r', linestyle='--')
    axes[0,0].set_title('Residuals Over Time')
    axes[0,1].hist(residuals.dropna(), bins=30, density=True, alpha=0.7)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    axes[0,1].set_title('Distribution')
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1,0])
    plot_acf(residuals.dropna(), lags=20, ax=axes[1,1])
    axes[1,1].set_title('Residual ACF')
    plt.tight_layout()
    diag_path = os.path.join(OUTPUT_DIR, 'arima_monthly_diagnostics.png')
    plt.savefig(diag_path, dpi=150)
    logger.info("Saved: %s", diag_path)
    plt.close()
    
    # Step 5: Forecast 2020 (12 months)
    logger.info("\n" + "="*80)
    logger.info("STEP 5: 2020 FORECAST (12 months)")
    logger.info("="*80)
    forecast_result = fitted.get_forecast(steps=12)
    forecasts = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    
    forecast_df = pd.DataFrame({
        'month': range(1, 13),
        'forecast': forecasts.values,
        'lower_95': conf_int.iloc[:, 0].values,
        'upper_95': conf_int.iloc[:, 1].values
    })
    forecast_df['interval_width'] = forecast_df['upper_95'] - forecast_df['lower_95']
    logger.info("Mean forecast: %.2f", forecasts.mean())
    logger.info("Avg interval width: %.2f", forecast_df['interval_width'].mean())
    
    # Evaluate on test
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION (2019 test)")
    logger.info("="*80)
    test_forecast = fitted.get_forecast(steps=len(test))
    test_pred = test_forecast.predicted_mean
    errors = test.values - test_pred.values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / test.values)) * 100
    r2 = 1 - (np.sum(errors**2) / np.sum((test.values - test.values.mean())**2))
    logger.info("MAE: %.2f", mae)
    logger.info("RMSE: %.2f", rmse)
    logger.info("MAPE: %.2f%%", mape)
    logger.info("R²: %.4f", r2)
    
    # Save
    forecast_path = os.path.join(OUTPUT_DIR, 'arima_monthly_forecasts.csv')
    forecast_df.to_csv(forecast_path, index=False)
    logger.info("\nSaved: %s", forecast_path)
    
    summary = {
        'case': '1A_Monthly',
        'model': 'SARIMA(1,1,1)(1,1,1)12',
        'stationarity': {'adf_pvalue': float(adf[1]), 'kpss_pvalue': float(kpss_result[1])},
        'diagnostics': {'ljung_box_pass': lb_pass, 'jarque_bera_pvalue': float(jb_p)},
        'performance': {'test_mape': float(mape), 'test_r2': float(r2), 
                       'test_mae': float(mae), 'test_rmse': float(rmse)},
        'aic': float(fitted.aic),
        'bic': float(fitted.bic)
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'arima_monthly_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", summary_path)
    
    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE - SARIMA MAPE: %.2f%%", mape)
    logger.info("="*80)
    logger.info("Compare with ML models (Lasso 1.36%%, GB 6.32%%)")

if __name__ == '__main__':
    main()
