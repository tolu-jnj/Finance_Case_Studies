"""ARIMA Modeling for Case 1B Quarterly Data - Final Working Version.

Uses original Case1B_QuarterlyData.csv (wide format: Date, CountryA_Y1, CountryA_Y2, etc.).

Usage:
    python arima_case1b_final.py

Outputs:
    - arima_quarterly_forecasts.csv
    - arima_quarterly_summary.json
    - arima_quarterly_diagnostics.png
    - arima_quarterly_acf_pacf.png
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
DATA_PATH = '../data/raw/Case1B_QuarterlyData.csv'
OUTPUT_DIR = '../models/arima/'

def main():
    logger.info("="*80)
    logger.info("CASE 1B: SARIMA MODELING (Quarterly)")
    logger.info("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    logger.info("\nLoading data from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    
    # Select first series: CountryA_Y1
    series = df['CountryA_Y1']
    
    # Split train (2010-2018) and test (2019)
    train = series[df['Year'] <= 2018].reset_index(drop=True)
    test = series[df['Year'] == 2019].reset_index(drop=True)
    
    logger.info("Series: CountryA_Y1")
    logger.info("Train: %d quarters (2010-2018)", len(train))
    logger.info("Test: %d quarters (2019)", len(test))
    
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
    plot_acf(train, lags=16, ax=axes[0])
    axes[0].set_title('ACF - Quarterly Data')
    plot_pacf(train, lags=16, ax=axes[1], method='ywm')
    axes[1].set_title('PACF - Quarterly Data')
    plt.tight_layout()
    acf_path = os.path.join(OUTPUT_DIR, 'arima_quarterly_acf_pacf.png')
    plt.savefig(acf_path, dpi=150)
    logger.info("Saved: %s", acf_path)
    plt.close()
    
    # Step 3: Fit SARIMA(0,1,1)(0,1,1)4
    logger.info("\n" + "="*80)
    logger.info("STEP 3: FITTING SARIMA(0,1,1)(0,1,1)4")
    logger.info("="*80)
    logger.info("Using conservative model for limited data (36 quarters)")
    model = SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,4),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    logger.info("Model fitted: AIC=%.2f, BIC=%.2f", fitted.aic, fitted.bic)
    
    # Step 4: Diagnostics
    logger.info("\n" + "="*80)
    logger.info("STEP 4: RESIDUAL DIAGNOSTICS")
    logger.info("="*80)
    residuals = fitted.resid
    lb = acorr_ljungbox(residuals, lags=8, return_df=True)
    lb_pass = len(lb[lb['lb_pvalue'] < 0.05]) == 0
    jb_stat, jb_p = stats.jarque_bera(residuals.dropna())
    logger.info("Ljung-Box: %s (p=%.3f at lag 8)", "PASS" if lb_pass else "WARN",
                lb['lb_pvalue'].iloc[-1])
    logger.info("Jarque-Bera: %s (p=%.3f)", "PASS" if jb_p > 0.05 else "WARN", jb_p)
    logger.info("(Warnings expected with only 36 training observations)")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='r', linestyle='--')
    axes[0,0].set_title('Residuals Over Time')
    axes[0,1].hist(residuals.dropna(), bins=15, density=True, alpha=0.7)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    axes[0,1].set_title('Distribution')
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1,0])
    plot_acf(residuals.dropna(), lags=12, ax=axes[1,1])
    axes[1,1].set_title('Residual ACF')
    plt.tight_layout()
    diag_path = os.path.join(OUTPUT_DIR, 'arima_quarterly_diagnostics.png')
    plt.savefig(diag_path, dpi=150)
    logger.info("Saved: %s", diag_path)
    plt.close()
    
    # Step 5: Forecast 2020 (4 quarters)
    logger.info("\n" + "="*80)
    logger.info("STEP 5: 2020 FORECAST (4 quarters)")
    logger.info("="*80)
    forecast_result = fitted.get_forecast(steps=4)
    forecasts = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    
    forecast_df = pd.DataFrame({
        'quarter': range(1, 5),
        'forecast': forecasts.values,
        'lower_95': conf_int.iloc[:, 0].values,
        'upper_95': conf_int.iloc[:, 1].values
    })
    forecast_df['interval_width'] = forecast_df['upper_95'] - forecast_df['lower_95']
    logger.info("Mean forecast: %.2f", forecasts.mean())
    logger.info("Avg interval width: %.2f", forecast_df['interval_width'].mean())
    logger.info("Q1 interval: ±%.0f", forecast_df['interval_width'].iloc[0]/2)
    logger.info("Q4 interval: ±%.0f (wider, uncertainty compounds)", 
                forecast_df['interval_width'].iloc[3]/2)
    
    # Evaluate on test
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION (2019 test)")
    logger.info("="*80)
    test_forecast = fitted.get_forecast(steps=len(test))
    test_pred = test_forecast.predicted_mean
    errors = test.values - test_pred.values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    wape = np.sum(np.abs(errors)) / np.sum(np.abs(test.values)) * 100
    r2 = 1 - (np.sum(errors**2) / np.sum((test.values - test.values.mean())**2))
    logger.info("MAE: %.2f", mae)
    logger.info("RMSE: %.2f", rmse)
    logger.info("WAPE: %.2f%%", wape)
    logger.info("R²: %.4f", r2)
    
    # Save
    forecast_path = os.path.join(OUTPUT_DIR, 'arima_quarterly_forecasts.csv')
    forecast_df.to_csv(forecast_path, index=False)
    logger.info("\nSaved: %s", forecast_path)
    
    summary = {
        'case': '1B_Quarterly',
        'model': 'SARIMA(0,1,1)(0,1,1)4',
        'stationarity': {'adf_pvalue': float(adf[1]), 'kpss_pvalue': float(kpss_result[1])},
        'diagnostics': {'ljung_box_pass': lb_pass, 'jarque_bera_pvalue': float(jb_p)},
        'performance': {'test_wape': float(wape), 'test_r2': float(r2), 
                       'test_mae': float(mae), 'test_rmse': float(rmse)},
        'aic': float(fitted.aic),
        'bic': float(fitted.bic)
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'arima_quarterly_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", summary_path)
    
    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE - SARIMA WAPE: %.2f%%", wape)
    logger.info("="*80)
    logger.info("Compare with ML models (GB 5.59%% WAPE)")

if __name__ == '__main__':
    main()
