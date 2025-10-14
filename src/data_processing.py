"""
Data Processing Module for Retail Sales Optimization
Author: Your Name
Date: January 2024

This module handles all data loading, cleaning, and preprocessing tasks.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load transaction data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv
    
    Returns:
    --------
    pd.DataFrame
        Raw transaction data
    """
    logger.info(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully loaded {len(df):,} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess transaction data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned transaction data
    """
    logger.info("Starting data cleaning process")
    
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df_clean)} duplicate records")
    
    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.dropna(subset=['customer_id', 'transaction_date', 'amount'])
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"Handled {missing_before - missing_after} missing values")
    
    # Convert date column to datetime
    if 'transaction_date' in df_clean.columns:
        df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'])
    
    # Remove negative amounts (returns/refunds handled separately)
    df_clean = df_clean[df_clean['amount'] > 0]
    
    # Remove outliers using IQR method
    Q1 = df_clean['amount'].quantile(0.25)
    Q3 = df_clean['amount'].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[
        (df_clean['amount'] >= Q1 - 3 * IQR) & 
        (df_clean['amount'] <= Q3 + 3 * IQR)
    ]
    
    logger.info(f"Data cleaning complete. Final records: {len(df_clean):,}")
    
    return df_clean


def add_temporal_features(df: pd.DataFrame, date_col: str = 'transaction_date') -> pd.DataFrame:
    """
    Add temporal features for time series analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with date column
    date_col : str
        Name of the date column
    
    Returns:
    --------
    pd.DataFrame
        Data with additional temporal features
    """
    logger.info("Adding temporal features")
    
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    
    # Extract temporal features
    df_temp['year'] = df_temp[date_col].dt.year
    df_temp['month'] = df_temp[date_col].dt.month
    df_temp['day'] = df_temp[date_col].dt.day
    df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
    df_temp['day_name'] = df_temp[date_col].dt.day_name()
    df_temp['week'] = df_temp[date_col].dt.isocalendar().week
    df_temp['quarter'] = df_temp[date_col].dt.quarter
    df_temp['is_weekend'] = df_temp['day_of_week'].isin([5, 6]).astype(int)
    df_temp['is_month_start'] = df_temp[date_col].dt.is_month_start.astype(int)
    df_temp['is_month_end'] = df_temp[date_col].dt.is_month_end.astype(int)
    
    logger.info("Temporal features added successfully")
    
    return df_temp


def aggregate_daily_sales(df: pd.DataFrame, 
                         date_col: str = 'transaction_date',
                         amount_col: str = 'amount') -> pd.DataFrame:
    """
    Aggregate sales data by day.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data
    date_col : str
        Name of the date column
    amount_col : str
        Name of the amount column
    
    Returns:
    --------
    pd.DataFrame
        Daily aggregated sales
    """
    logger.info("Aggregating daily sales")
    
    daily_sales = df.groupby(date_col).agg({
        amount_col: ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).reset_index()
    
    daily_sales.columns = [date_col, 'total_sales', 'avg_transaction', 
                           'num_transactions', 'unique_customers']
    
    logger.info(f"Created daily sales data with {len(daily_sales)} days")
    
    return daily_sales


def create_customer_transaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for each customer.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data
    
    Returns:
    --------
    pd.DataFrame
        Customer-level summary statistics
    """
    logger.info("Creating customer transaction summary")
    
    summary = df.groupby('customer_id').agg({
        'transaction_date': ['min', 'max', 'count'],
        'amount': ['sum', 'mean', 'std'],
        'transaction_id': 'count'
    }).reset_index()
    
    summary.columns = ['customer_id', 'first_purchase', 'last_purchase', 
                       'purchase_count', 'total_spent', 'avg_order_value', 
                       'std_order_value', 'num_transactions']
    
    # Calculate customer lifetime in days
    summary['customer_lifetime_days'] = (
        summary['last_purchase'] - summary['first_purchase']
    ).dt.days
    
    logger.info(f"Created summary for {len(summary):,} customers")
    
    return summary


def generate_synthetic_retail_data(num_customers: int = 5000,
                                   num_transactions: int = 2000000,
                                   start_date: str = '2023-01-01',
                                   end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Generate synthetic retail transaction data for testing.
    
    Parameters:
    -----------
    num_customers : int
        Number of unique customers
    num_transactions : int
        Total number of transactions
    start_date : str
        Start date for transactions
    end_date : str
        End date for transactions
    
    Returns:
    --------
    pd.DataFrame
        Synthetic transaction data
    """
    logger.info(f"Generating {num_transactions:,} synthetic transactions")
    
    np.random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = range(1, num_transactions + 1)
    
    # Generate customer IDs (some customers have multiple transactions)
    customer_ids = np.random.choice(
        [f'CUST{str(i).zfill(5)}' for i in range(1, num_customers + 1)],
        size=num_transactions,
        replace=True
    )
    
    # Generate dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = np.random.choice(date_range, size=num_transactions)
    
    # Generate amounts (log-normal distribution for realistic sales)
    amounts = np.random.lognormal(mean=4, sigma=1, size=num_transactions)
    amounts = np.clip(amounts, 10, 5000).round(2)
    
    # Generate categories
    categories = np.random.choice(
        ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books'],
        size=num_transactions,
        p=[0.25, 0.30, 0.20, 0.15, 0.07, 0.03]
    )
    
    # Generate product IDs
    product_ids = np.random.randint(1000, 9999, size=num_transactions)
    
    # Generate quantities
    quantities = np.random.choice([1, 2, 3, 4, 5], size=num_transactions, 
                                 p=[0.60, 0.25, 0.10, 0.03, 0.02])
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'customer_id': customer_ids,
        'transaction_date': dates,
        'amount': amounts,
        'category': categories,
        'product_id': product_ids,
        'quantity': quantities
    })
    
    # Sort by date
    df = df.sort_values('transaction_date').reset_index(drop=True)
    
    logger.info(f"Generated {len(df):,} synthetic transactions for {df['customer_id'].nunique():,} customers")
    
    return df


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed data to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed data
    filepath : str
        Path to save the file
    """
    logger.info(f"Saving processed data to {filepath}")
    
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {len(df):,} records")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Convenience function to load and clean data in one step.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Cleaned transaction data
    """
    df = load_data(filepath)
    df_clean = clean_data(df)
    df_with_features = add_temporal_features(df_clean)
    
    return df_with_features


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    df = generate_synthetic_retail_data(
        num_customers=5000,
        num_transactions=2000000
    )
    
    # Save to file
    save_processed_data(df, 'data/raw/transactions.csv')
    
    # Load and clean
    df_clean = load_and_clean_data('data/raw/transactions.csv')
    
    # Create customer summary
    customer_summary = create_customer_transaction_summary(df_clean)
    save_processed_data(customer_summary, 'data/processed/customer_summary.csv')
    
    # Create daily sales
    daily_sales = aggregate_daily_sales(df_clean)
    save_processed_data(daily_sales, 'data/processed/daily_sales.csv')
    
    print("\n=== Data Processing Complete ===")
    print(f"Total Transactions: {len(df_clean):,}")
    print(f"Unique Customers: {df_clean['customer_id'].nunique():,}")
    print(f"Date Range: {df_clean['transaction_date'].min()} to {df_clean['transaction_date'].max()}")
    print(f"Total Sales: ${df_clean['amount'].sum():,.2f}")