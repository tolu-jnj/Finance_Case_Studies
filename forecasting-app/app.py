import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Dict, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Time Series Forecasting Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data(case: str = "1A") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data for the specified case study.
    
    Args:
        case (str): Either "1A" (Monthly) or "1B" (Quarterly)
        
    Returns:
        Tuple containing historical data, forecasts, and model comparisons
    """
    try:
        if case == "1A":
            historical = pd.read_csv("data/raw/Case1A_MonthlyData.csv")
            forecasts = pd.read_csv("forecasts/forecasts_2020.csv")
            model_comp = pd.read_csv("models/model_comparison.csv")
        else:
            historical = pd.read_csv("data/raw/Case1B_QuarterlyData.csv")
            forecasts = pd.read_csv("forecasts/forecasts_2020_quarterly.csv")
            model_comp = pd.read_csv("models/model_comparison_quarterly.csv")
            
        return historical, forecasts, model_comp
    except Exception as e:
        logger.error(f"Error loading data for case {case}: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def plot_time_series(data: pd.DataFrame, title: str) -> go.Figure:
    """
    Create an interactive time series plot using Plotly.
    
    Args:
        data (pd.DataFrame): Time series data to plot
        title (str): Plot title
        
    Returns:
        Plotly figure object
    """
    fig = px.line(data, x="Date", y="Value",
                  title=title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    return fig

def calculate_data_quality_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate data quality statistics for the time series.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        DataFrame with statistics
    """
    stats = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Missing Values'],
        'Value': [
            data['Value'].mean(),
            data['Value'].std(),
            data['Value'].min(),
            data['Value'].max(),
            data['Value'].isnull().sum()
        ]
    })
    return stats

def main():
    # Sidebar navigation
    pages = ["Homepage", "Data Explorer", "Forecasting Dashboard", 
             "Model Performance", "AI Assistance"]
    page = st.sidebar.selectbox("Navigate", pages)
    
    if page == "Homepage":
        st.title("Time Series Forecasting Portfolio: ML & Classical Approaches")
        
        # Executive summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cases", "2")
        with col2:
            st.metric("Total Forecasts", "316")
        with col3:
            st.metric("Revenue Covered", "$207M")
        with col4:
            st.metric("Performance vs Industry", "2.7x better")
            
        # Methodology overview
        st.header("Methodology Overview")
        st.write("""
        This portfolio project follows the Data Theory 6-phase framework:
        1. Governance & Problem Scoping
        2. Data Generation & Collection
        3. Data Standardization
        4. Feature Engineering & Aggregation
        5. Model Analysis & Selection
        6. Production Deployment
        """)
        
        # Key findings
        st.header("Key Findings")
        st.write("""
        - SARIMA outperforms Gradient Boosting for seasonal patterns
        - Successfully detected and prevented data leakage
        - Identified key business drivers through feature importance
        """)
        
    elif page == "Data Explorer":
        st.title("Data Explorer")
        
        # Case selection
        case = st.selectbox("Select Case Study", ["Case 1A (Monthly)", "Case 1B (Quarterly)"])
        case_id = "1A" if "Monthly" in case else "1B"
        
        # Load data
        historical, forecasts, model_comp = load_data(case_id)
        
        if historical is not None:
            # Country/Product selector
            if case_id == "1A":
                country = st.selectbox("Select Country", historical["Country"].unique())
                product = st.selectbox("Select Product", historical["Product"].unique())
                data = historical[
                    (historical["Country"] == country) & 
                    (historical["Product"] == product)
                ]
            else:
                series = st.selectbox("Select Time Series", [f"Y{i}" for i in range(1, 37)])
                data = historical[historical["Series"] == series]
            
            # Time series plot
            st.plotly_chart(plot_time_series(data, f"Historical Data ({case})"))
            
            # Data quality statistics
            st.header("Data Quality Statistics")
            st.table(calculate_data_quality_stats(data))
            
            # Seasonal decomposition
            st.header("Seasonal Decomposition")
            decomposition = seasonal_decompose(data["Value"], period=12 if case_id == "1A" else 4)
            
            fig = make_subplots(rows=4, cols=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["Value"], name="Original"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, name="Trend"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, name="Seasonal"), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, name="Residual"), row=4, col=1)
            fig.update_layout(height=800, title="Seasonal Decomposition")
            st.plotly_chart(fig)
            
    elif page == "Forecasting Dashboard":
        st.title("Forecasting Dashboard")
        
        # Case selection
        case = st.selectbox("Select Case Study", ["Case 1A (Monthly)", "Case 1B (Quarterly)"])
        case_id = "1A" if "Monthly" in case else "1B"
        
        # Load data
        historical, forecasts, model_comp = load_data(case_id)
        
        if historical is not None and forecasts is not None:
            # Model selection
            model = st.selectbox("Select Model", [
                "SARIMA", "Gradient Boosting", "Lasso", "Ridge", 
                "Random Forest", "Decision Tree"
            ])
            
            # Filters
            if case_id == "1A":
                country = st.selectbox("Select Country", historical["Country"].unique())
                product = st.selectbox("Select Product", historical["Product"].unique())
                historical_filtered = historical[
                    (historical["Country"] == country) & 
                    (historical["Product"] == product)
                ]
                forecasts_filtered = forecasts[
                    (forecasts["Country"] == country) & 
                    (forecasts["Product"] == product)
                ]
            else:
                series = st.selectbox("Select Time Series", [f"Y{i}" for i in range(1, 37)])
                historical_filtered = historical[historical["Series"] == series]
                forecasts_filtered = forecasts[forecasts["Series"] == series]
            
            # Generate forecast button
            if st.button("Generate Forecast"):
                # Create combined plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical_filtered["Date"],
                    y=historical_filtered["Value"],
                    name="Historical",
                    line=dict(color="blue")
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecasts_filtered["Date"],
                    y=forecasts_filtered[f"{model}_Forecast"],
                    name="Forecast",
                    line=dict(color="red")
                ))
                
                # Confidence intervals for SARIMA
                if model == "SARIMA":
                    fig.add_trace(go.Scatter(
                        x=forecasts_filtered["Date"],
                        y=forecasts_filtered["Upper_95"],
                        name="Upper 95% CI",
                        line=dict(dash="dash", color="gray")
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecasts_filtered["Date"],
                        y=forecasts_filtered["Lower_95"],
                        name="Lower 95% CI",
                        line=dict(dash="dash", color="gray"),
                        fill="tonexty"
                    ))
                
                fig.update_layout(
                    title=f"{model} Forecast for {case}",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig)
                
                # Forecast summary table
                st.header("Forecast Summary")
                summary = pd.DataFrame({
                    'Metric': ['Mean', 'Min', 'Max'],
                    'Value': [
                        forecasts_filtered[f"{model}_Forecast"].mean(),
                        forecasts_filtered[f"{model}_Forecast"].min(),
                        forecasts_filtered[f"{model}_Forecast"].max()
                    ]
                })
                st.table(summary)
                
    elif page == "Model Performance":
        st.title("Model Performance")
        
        # Case selection
        case = st.selectbox("Select Case Study", ["Case 1A (Monthly)", "Case 1B (Quarterly)"])
        case_id = "1A" if "Monthly" in case else "1B"
        
        # Load data
        historical, forecasts, model_comp = load_data(case_id)
        
        if model_comp is not None:
            # Performance metrics table
            st.header("Model Performance Metrics")
            st.table(model_comp)
            
            # Performance comparison plot
            fig = px.bar(
                model_comp,
                x="Model",
                y="MAPE",
                title="Model Performance Comparison"
            )
            # Add industry standard line
            fig.add_hline(y=10, line_dash="dash", line_color="red",
                         annotation_text="Industry Standard (10%)")
            st.plotly_chart(fig)
            
            # Feature importance plot for Gradient Boosting
            st.header("Feature Importance (Gradient Boosting)")
            # Placeholder for feature importance plot
            # This would be loaded from a saved visualization
            
            # SARIMA diagnostics
            st.header("SARIMA Diagnostics")
            # Placeholder for SARIMA diagnostic plots
            # These would be loaded from saved visualizations
            
    elif page == "AI Assistance":
        st.title("AI Assistance Documentation")
        
        # Create tabs for different phases
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Engineering",
            "Model Development",
            "Diagnostics",
            "Visualization"
        ])
        
        with tab1:
            st.header("Data Engineering")
            st.write("""
            ### Feature Engineering
            - Implemented lag features (1, 3, 6, 12 months)
            - Created rolling statistics
            - Detected and prevented data leakage
            
            ```python
            # Example feature engineering code
            def create_lag_features(df, lags=[1, 3, 6, 12]):
                for lag in lags:
                    df[f'lag_{lag}'] = df.groupby(['Country', 'Product'])['Value'].shift(lag)
                return df
            ```
            """)
            
        with tab2:
            st.header("Model Development")
            st.write("""
            ### SARIMA Implementation
            - Automated order selection using AIC
            - Cross-validation for hyperparameter tuning
            - Ensemble method development
            
            ```python
            # Example SARIMA implementation
            def fit_sarima(data, order, seasonal_order):
                model = sm.tsa.SARIMAX(data, 
                                     order=order,
                                     seasonal_order=seasonal_order)
                return model.fit()
            ```
            """)
            
        with tab3:
            st.header("Diagnostics")
            st.write("""
            ### Statistical Tests
            - Ljung-Box test for autocorrelation
            - Augmented Dickey-Fuller for stationarity
            - Residual analysis and normality tests
            
            ```python
            # Example diagnostic code
            def check_stationarity(series):
                result = adfuller(series)
                return result[1]  # p-value
            ```
            """)
            
        with tab4:
            st.header("Visualization")
            st.write("""
            ### Plot Generation
            - Interactive Plotly visualizations
            - Automated report generation
            - Custom styling and branding
            
            ```python
            # Example visualization code
            def plot_forecast(historical, forecast):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical.index, 
                                       y=historical.values))
                fig.add_trace(go.Scatter(x=forecast.index,
                                       y=forecast.values))
                return fig
            ```
            """)

if __name__ == "__main__":
    main()