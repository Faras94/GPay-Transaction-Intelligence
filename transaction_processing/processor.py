"""
Main transaction data processing pipeline.

This module orchestrates the data processing flow by composing
functions from other modules (categories, merchant_utils, enrichment).
"""

import pandas as pd
from .categories import categorize_transaction
from .merchant_utils import extract_merchant_name
from .enrichment import add_spending_insights


def process_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main processing function: cleans, validates, and categorizes transaction data.
    
    Pipeline:
    1. Clean and validate amount column
    2. Extract merchant names
    3. Categorize transactions
    4. Add analytical insights
    5. Remove duplicates
    
    Args:
        df: Raw dataframe from PDF extraction
        
    Returns:
        Processed and categorized dataframe
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Clean and validate amount column
    if 'Amount (₹)' in df_copy.columns:
        df_copy['Amount (₹)'] = pd.to_numeric(df_copy['Amount (₹)'], errors='coerce')
        # Remove rows with invalid amounts
        df_copy = df_copy[df_copy['Amount (₹)'] > 0]
    
    # Clean description
    if 'Description' in df_copy.columns:
        df_copy['Description'] = df_copy['Description'].apply(extract_merchant_name)
    
    # Categorize transactions
    if 'Description' in df_copy.columns:
        df_copy['Category'] = df_copy['Description'].apply(categorize_transaction)
    
    # Add insights
    df_copy = add_spending_insights(df_copy)
    
    # Remove duplicates based on Date, Time, Amount, and Description
    if all(col in df_copy.columns for col in ['Date', 'Time', 'Amount (₹)', 'Description']):
        df_copy = df_copy.drop_duplicates(subset=['Date', 'Time', 'Amount (₹)', 'Description'])
    
    return df_copy
