"""
Analytics and statistics utilities for transaction data.

This module provides functions to generate statistics, detect outliers,
and perform analysis on transaction DataFrames.
"""

import pandas as pd


def get_category_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics by category.
    
    Calculates total, average, median, count, and standard deviation
    for each transaction category.
    
    Args:
        df: Processed transaction dataframe
        
    Returns:
        Summary statistics dataframe
    """
    if df.empty or 'Category' not in df.columns:
        return pd.DataFrame()
    
    stats = df.groupby('Category').agg({
        'Amount (₹)': ['sum', 'mean', 'median', 'count', 'std']
    }).round(2)
    
    stats.columns = ['Total', 'Average', 'Median', 'Count', 'Std Dev']
    stats = stats.sort_values('Total', ascending=False)
    
    return stats


def detect_unusual_transactions(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect unusual transactions based on statistical outliers.
    
    Uses standard deviation to identify transactions that are
    significantly larger than the mean.
    
    Args:
        df: Transaction dataframe
        threshold: Number of standard deviations to consider unusual
        
    Returns:
        Dataframe of unusual transactions
    """
    if df.empty or 'Amount (₹)' not in df.columns:
        return pd.DataFrame()
    
    mean_amount = df['Amount (₹)'].mean()
    std_amount = df['Amount (₹)'].std()
    
    unusual = df[df['Amount (₹)'] > mean_amount + (threshold * std_amount)]
    
    return unusual.sort_values('Amount (₹)', ascending=False)
