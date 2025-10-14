# src/rfm_segmentation.py

"""
RFM Segmentation Module for Customer Analysis.
"""

import pandas as pd
from datetime import datetime
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary metrics for each customer.

    This function is written to be tolerant of slightly different column
    names used by the dashboard. It expects a transactions dataframe with at
    least the following columns: 'customer_id', 'transaction_date', 'amount'.
    Frequency is computed as the number of transactions per customer (nrows)
    unless a specific transaction id column is present.
    """
    logger.info("Calculating RFM metrics...")

    # Determine analysis date (most recent transaction date in the data)
    if 'transaction_date' not in df.columns:
        raise ValueError("Expected column 'transaction_date' in transactions dataframe")

    analysis_date = df['transaction_date'].max()

    # Choose grouping keys / column names with fallbacks
    customer_col = 'customer_id' if 'customer_id' in df.columns else ('CustomerID' if 'CustomerID' in df.columns else None)
    if customer_col is None:
        raise ValueError("Expected 'customer_id' or 'CustomerID' column in transactions dataframe")

    tx_id_col = None
    for candidate in ['transaction_id', 'invoice_no', 'InvoiceNo', 'transaction_reference']:
        if candidate in df.columns:
            tx_id_col = candidate
            break

    monetary_col = 'amount' if 'amount' in df.columns else ('TotalPrice' if 'TotalPrice' in df.columns else None)
    if monetary_col is None:
        raise ValueError("Expected 'amount' or 'TotalPrice' column in transactions dataframe")

    # Aggregations: recency (days since last purchase), frequency (# transactions), monetary (total spend)
    if tx_id_col:
        agg_dict = {
            'transaction_date': lambda d: (analysis_date - d.max()).days,
            tx_id_col: 'nunique',
            monetary_col: 'sum'
        }
    else:
        # If no unique transaction id is available, frequency = count of rows
        agg_dict = {
            'transaction_date': lambda d: (analysis_date - d.max()).days,
            monetary_col: 'sum'
        }

    rfm = df.groupby(customer_col).agg(agg_dict).reset_index()

    # Normalize column names
    if tx_id_col:
        rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
    else:
        rfm.columns = [customer_col, 'recency', 'monetary']
        # insert frequency as count per customer
        freq_series = df.groupby(customer_col).size().reset_index(name='frequency')
        rfm = rfm.merge(freq_series, on=customer_col)

    # Rename customer id column to a consistent name
    rfm = rfm.rename(columns={customer_col: 'customer_id'})

    logger.info(f"Calculated RFM metrics for {len(rfm):,} customers.")
    return rfm


def calculate_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RFM scores (1-5 scale) using percentile-based scoring for robustness.
    """
    logger.info("Calculating RFM scores...")
    rfm_scored = rfm_df.copy()

    # Defensive: if there are too few unique values for proper quantiles, use rank-based or cut-based fallback
    def score_recency(series: pd.Series) -> pd.Series:
        # lower recency (fewer days) is better
        try:
            q1 = series.quantile(0.2)
            q2 = series.quantile(0.4)
            q3 = series.quantile(0.6)
            q4 = series.quantile(0.8)
            return series.apply(lambda x: 5 if x <= q1 else 4 if x <= q2 else 3 if x <= q3 else 2 if x <= q4 else 1)
        except Exception:
            # fallback to rank-based scoring
            ranks = series.rank(method='average', pct=True)
            return ranks.apply(lambda p: 5 if p <= 0.2 else 4 if p <= 0.4 else 3 if p <= 0.6 else 2 if p <= 0.8 else 1)

    def score_higher_better(series: pd.Series) -> pd.Series:
        # higher frequency/monetary is better
        try:
            q1 = series.quantile(0.2)
            q2 = series.quantile(0.4)
            q3 = series.quantile(0.6)
            q4 = series.quantile(0.8)
            return series.apply(lambda x: 5 if x >= q4 else 4 if x >= q3 else 3 if x >= q2 else 2 if x >= q1 else 1)
        except Exception:
            ranks = series.rank(method='average', pct=True)
            return ranks.apply(lambda p: 5 if p >= 0.8 else 4 if p >= 0.6 else 3 if p >= 0.4 else 2 if p >= 0.2 else 1)

    rfm_scored['r_score'] = score_recency(rfm_scored['recency'])
    rfm_scored['f_score'] = score_higher_better(rfm_scored['frequency'])
    rfm_scored['m_score'] = score_higher_better(rfm_scored['monetary'])

    # Convert scores to integer type
    for col in ['r_score', 'f_score', 'm_score']:
        rfm_scored[col] = rfm_scored[col].astype(int)

    rfm_scored['rfm_score'] = rfm_scored['r_score'] + rfm_scored['f_score'] + rfm_scored['m_score']
    rfm_scored['rfm_segment'] = rfm_scored['r_score'].astype(str) + rfm_scored['f_score'].astype(str)

    logger.info("RFM scores calculated successfully.")
    return rfm_scored


def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign customer segments based on RFM scores using a regex map.
    """
    logger.info("Segmenting customers...")
    rfm_segmented = rfm_df.copy()
    
    segment_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Cannot Lose',
        r'3[1-2]': 'About to Sleep',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising',
        r'51': 'New Customers',
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }

    rfm_segmented['customer_segment'] = rfm_segmented['rfm_segment'].replace(segment_map, regex=True)
    logger.info("Customer segmentation complete.")
    return rfm_segmented


def get_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate metrics for each customer segment.
    """
    logger.info("Calculating segment summary metrics...")
    
    segment_summary = rfm_df.groupby('customer_segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'count', 'sum']
    }).round(1)
    
    segment_summary.columns = ['avg_recency', 'avg_frequency', 'avg_monetary', 'customer_count', 'total_monetary']
    total_customers = segment_summary['customer_count'].sum()
    total_revenue = segment_summary['total_monetary'].sum()
    segment_summary['pct_customers'] = (segment_summary['customer_count'] / total_customers * 100).round(1)
    segment_summary['pct_revenue'] = (segment_summary['total_monetary'] / total_revenue * 100).round(1)
    
    return segment_summary.sort_values('total_monetary', ascending=False)


def calculate_segment_metrics(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-compatible wrapper expected by the dashboard.
    """
    return get_segment_summary(rfm_df)