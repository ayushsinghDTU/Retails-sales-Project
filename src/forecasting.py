"""
Demand Forecasting Module using Prophet
Author: Your Name
Date: January 2024

This module implements time-series forecasting for retail demand prediction
using Facebook's Prophet library.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Prophet is an optional dependency used for forecasting. Import it lazily
# so the package can be imported (e.g., for parts of the app that don't
# require forecasting) even when prophet isn't installed in the environment.
try:
    from prophet import Prophet  # type: ignore
    from prophet.diagnostics import cross_validation, performance_metrics  # type: ignore
    _PROPHET_AVAILABLE = True
except Exception:
    Prophet = Any  # type: ignore
    cross_validation = None  # type: ignore
    performance_metrics = None  # type: ignore
    _PROPHET_AVAILABLE = False
from typing import Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_for_prophet(df: pd.DataFrame, 
                             date_col: str = 'transaction_date',
                             value_col: str = 'amount',
                             aggregation: str = 'D') -> pd.DataFrame:
    """
    Prepare data in Prophet's required format (ds, y columns).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
    date_col : str
        Name of date column
    value_col : str
        Name of value column to forecast
    aggregation : str
        Aggregation frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
    
    Returns:
    --------
    pd.DataFrame
        Data in Prophet format with 'ds' and 'y' columns
    """
    logger.info(f"Preparing data for Prophet with {aggregation} aggregation")
    
    # Aggregate data
    df_agg = df.groupby(pd.Grouper(key=date_col, freq=aggregation))[value_col].sum().reset_index()
    
    # Rename columns for Prophet
    df_prophet = df_agg.rename(columns={
        date_col: 'ds',
        value_col: 'y'
    })
    
    # Remove zero values
    df_prophet = df_prophet[df_prophet['y'] > 0]
    
    logger.info(f"Prepared {len(df_prophet)} data points for forecasting")
    
    return df_prophet


def create_prophet_model(seasonality_mode: str = 'multiplicative',
                        changepoint_prior_scale: float = 0.05,
                        seasonality_prior_scale: float = 10.0,
                        yearly_seasonality: bool = True,
                        weekly_seasonality: bool = True,
                        daily_seasonality: bool = False) -> Any:
    """
    Create and configure a Prophet model.
    
    Parameters:
    -----------
    seasonality_mode : str
        'additive' or 'multiplicative'
    changepoint_prior_scale : float
        Flexibility of trend (higher = more flexible)
    seasonality_prior_scale : float
        Strength of seasonality (higher = stronger)
    yearly_seasonality : bool
        Include yearly seasonality
    weekly_seasonality : bool
        Include weekly seasonality
    daily_seasonality : bool
        Include daily seasonality
    
    Returns:
    --------
    Prophet
        Configured Prophet model
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "The 'prophet' package is required for forecasting. "
            "Install it with `pip install prophet` (or see requirements.txt)."
        )

    logger.info("Creating Prophet model")

    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        interval_width=0.95
    )
    
    # Add custom seasonalities
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    logger.info("Prophet model configured")
    
    return model


def train_prophet_model(df: pd.DataFrame, 
                       model: Optional[Any] = None) -> Any:
    """
    Train a Prophet model on historical data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data in Prophet format (ds, y columns)
    model : Prophet, optional
        Pre-configured model (creates default if None)
    
    Returns:
    --------
    Prophet
        Trained Prophet model
    """
    logger.info("Training Prophet model")
    
    if model is None:
        model = create_prophet_model()
    
    # Train model
    model.fit(df)
    
    logger.info("Model training complete")
    
    return model


def generate_forecast(model: Any, 
                     periods: int = 180,
                     freq: str = 'D') -> pd.DataFrame:
    """
    Generate future predictions using trained Prophet model.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    periods : int
        Number of periods to forecast
    freq : str
        Frequency of forecast ('D', 'W', 'M')
    
    Returns:
    --------
    pd.DataFrame
        Forecast with predictions and confidence intervals
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "The 'prophet' package is required for forecasting. "
            "Install it with `pip install prophet` (or see requirements.txt)."
        )

    logger.info(f"Generating {periods} period forecast")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Select relevant columns
    forecast_output = forecast[[
        'ds', 'yhat', 'yhat_lower', 'yhat_upper',
        'trend', 'yearly', 'weekly'
    ]]
    
    logger.info("Forecast generation complete")
    
    return forecast_output


def evaluate_forecast(model: Any, 
                     df: pd.DataFrame,
                     horizon: str = '30 days',
                     period: str = '90 days',
                     initial: str = '365 days') -> pd.DataFrame:
    """
    Evaluate forecast accuracy using cross-validation.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    df : pd.DataFrame
        Historical data for validation
    horizon : str
        Forecast horizon for evaluation
    period : str
        Spacing between cutoff dates
    initial : str
        Initial training period
    
    Returns:
    --------
    pd.DataFrame
        Performance metrics (MAPE, RMSE, MAE)
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "The 'prophet' package is required for forecasting evaluation. "
            "Install it with `pip install prophet` (or see requirements.txt)."
        )

    logger.info("Evaluating forecast performance")

    # Perform cross-validation
    df_cv = cross_validation(
        model, 
        initial=initial,
        period=period,
        horizon=horizon
    )
    
    # Calculate performance metrics
    df_metrics = performance_metrics(df_cv)
    
    logger.info("Forecast evaluation complete")
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  MAPE: {df_metrics['mape'].mean():.2%}")
    logger.info(f"  RMSE: {df_metrics['rmse'].mean():.2f}")
    logger.info(f"  MAE: {df_metrics['mae'].mean():.2f}")
    
    return df_metrics


def calculate_forecast_accuracy(actual: pd.Series, 
                               predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate various forecast accuracy metrics.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of accuracy metrics
    """
    # Remove NaN values
    mask = ~(actual.isna() | predicted.isna())
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Calculate metrics
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Calculate R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': 100 - mape
    }
    
    return metrics


def detect_anomalies(df: pd.DataFrame, 
                    threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalies in sales data using statistical methods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with 'ds' and 'y' columns
    threshold : float
        Number of standard deviations for anomaly detection
    
    Returns:
    --------
    pd.DataFrame
        Data with anomaly flags
    """
    logger.info("Detecting anomalies in sales data")
    
    df_anomaly = df.copy()
    
    # Calculate rolling statistics
    df_anomaly['rolling_mean'] = df_anomaly['y'].rolling(window=7, center=True).mean()
    df_anomaly['rolling_std'] = df_anomaly['y'].rolling(window=7, center=True).std()
    
    # Calculate z-score
    df_anomaly['z_score'] = np.abs(
        (df_anomaly['y'] - df_anomaly['rolling_mean']) / df_anomaly['rolling_std']
    )
    
    # Flag anomalies
    df_anomaly['is_anomaly'] = df_anomaly['z_score'] > threshold
    
    num_anomalies = df_anomaly['is_anomaly'].sum()
    logger.info(f"Detected {num_anomalies} anomalies ({num_anomalies/len(df)*100:.1f}%)")
    
    return df_anomaly


def calculate_inventory_recommendations(forecast: pd.DataFrame,
                                       safety_stock_days: int = 7,
                                       service_level: float = 0.95) -> pd.DataFrame:
    """
    Calculate inventory recommendations based on demand forecast.
    
    Parameters:
    -----------
    forecast : pd.DataFrame
        Forecast data with predictions
    safety_stock_days : int
        Number of days of safety stock
    service_level : float
        Target service level (0-1)
    
    Returns:
    --------
    pd.DataFrame
        Inventory recommendations
    """
    logger.info("Calculating inventory recommendations")
    
    inventory = forecast.copy()
    
    # Calculate safety stock
    z_score = 1.96 if service_level == 0.95 else 1.65  # Z-score for service level
    inventory['safety_stock'] = (
        inventory['yhat'] * safety_stock_days * z_score / 
        np.sqrt(safety_stock_days)
    )
    
    # Calculate reorder point
    inventory['reorder_point'] = inventory['yhat'] + inventory['safety_stock']
    
    # Calculate optimal order quantity (EOQ approximation)
    inventory['order_quantity'] = inventory['yhat'] * 30  # Monthly demand
    
    # Stock status
    inventory['stock_status'] = inventory.apply(
        lambda x: 'Critical' if x['yhat'] > x['yhat_upper'] else
                 'Low' if x['yhat'] > x['yhat'] * 1.2 else
                 'Normal', axis=1
    )
    
    logger.info("Inventory recommendations calculated")
    
    return inventory


def forecast_by_category(df: pd.DataFrame,
                        category_col: str = 'category',
                        periods: int = 90) -> Dict[str, pd.DataFrame]:
    """
    Generate forecasts for each product category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with category column
    category_col : str
        Name of category column
    periods : int
        Number of periods to forecast
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of forecasts by category
    """
    logger.info("Generating category-level forecasts")
    
    forecasts = {}
    categories = df[category_col].unique()
    
    for category in categories:
        logger.info(f"Forecasting for category: {category}")
        
        # Filter data for category
        df_category = df[df[category_col] == category]
        
        # Prepare data
        df_prophet = prepare_data_for_prophet(df_category)
        
        # Train model
        model = train_prophet_model(df_prophet)
        
        # Generate forecast
        forecast = generate_forecast(model, periods=periods)
        
        forecasts[category] = forecast
    
    logger.info(f"Generated forecasts for {len(categories)} categories")
    
    return forecasts


def complete_forecasting_pipeline(df: pd.DataFrame,
                                 output_dir: str = 'data/processed/',
                                 forecast_periods: int = 180) -> Tuple[Any, pd.DataFrame]:
    """
    Complete demand forecasting pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data
    output_dir : str
        Directory to save outputs
    forecast_periods : int
        Number of days to forecast
    
    Returns:
    --------
    Tuple[Prophet, pd.DataFrame]
        Trained model and forecast results
    """
    logger.info("Starting complete forecasting pipeline")
    
    # Prepare data
    df_prophet = prepare_data_for_prophet(df)
    
    # Detect and handle anomalies
    df_clean = detect_anomalies(df_prophet)
    df_clean = df_clean[~df_clean['is_anomaly']]
    df_clean = df_clean[['ds', 'y']]
    
    # Train model
    model = train_prophet_model(df_clean)
    
    # Generate forecast
    forecast = generate_forecast(model, periods=forecast_periods)
    
    # Calculate accuracy on historical data
    historical = forecast[forecast['ds'] <= df_clean['ds'].max()].copy()
    historical = historical.merge(df_clean, on='ds', how='left')
    
    if not historical.empty:
        accuracy_metrics = calculate_forecast_accuracy(
            historical['y'], 
            historical['yhat']
        )
        
        logger.info("\n=== Forecast Accuracy Metrics ===")
        for metric, value in accuracy_metrics.items():
            logger.info(f"{metric}: {value:.2f}")
    
    # Calculate inventory recommendations
    inventory = calculate_inventory_recommendations(forecast)
    
    # Save outputs
    forecast.to_csv(f'{output_dir}sales_forecast.csv', index=False)
    inventory.to_csv(f'{output_dir}inventory_recommendations.csv', index=False)
    
    logger.info("Forecasting pipeline complete!")
    
    return model, forecast


# Example usage
if __name__ == "__main__":
    # Load sample data
    from data_processing import load_and_clean_data
    
    df = load_and_clean_data('data/raw/transactions.csv')
    
    # Run forecasting pipeline
    model, forecast = complete_forecasting_pipeline(df, forecast_periods=180)
    
    print("\n=== Demand Forecasting Complete ===")
    print(f"\nForecast generated for {len(forecast)} periods")
    print(f"Date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
    print(f"\nNext 30 days forecast:")
    future_30 = forecast[forecast['ds'] > df['transaction_date'].max()].head(30)
    print(future_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])