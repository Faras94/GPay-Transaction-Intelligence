"""
Data enrichment utilities for transaction DataFrames.

This module adds analytical columns like time categories, amount ranges,
day of week, etc. to enhance transaction data for analysis.
"""

import pandas as pd
from .categories import get_category_emoji


def add_spending_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional analytical columns to the dataframe.
    
    Enriches data with:
    - Day of week, month name, week of year
    - Time of day category (Morning/Afternoon/Evening/Night)
    - Amount category (Small/Medium/Large/Very Large)
    - Category emoji icons
    
    Args:
        df: Input dataframe with transactions
        
    Returns:
        Enhanced dataframe with insights
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Add day of week
    if 'Date' in df_copy.columns:
        df_copy['Day_of_Week'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.day_name()
        df_copy['Month_Name'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.strftime('%B %Y')
        df_copy['Week_of_Year'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.isocalendar().week
    
    # Add time of day category
    if 'Time' in df_copy.columns:
        df_copy['Time_of_Day'] = df_copy['Time'].apply(_categorize_time)
    
    # Add amount categories
    if 'Amount (₹)' in df_copy.columns:
        df_copy['Amount_Category'] = df_copy['Amount (₹)'].apply(_categorize_amount)
    
    # Add category emoji
    if 'Category' in df_copy.columns:
        df_copy['Category_Icon'] = df_copy['Category'].apply(get_category_emoji)
    
    return df_copy


def _categorize_time(time_str: str) -> str:
    """Categorize time into Morning/Afternoon/Evening/Night."""
    try:
        hour = pd.to_datetime(time_str, format='%I:%M %p').hour
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    except:
        return "Unknown"


def _categorize_amount(amount: float) -> str:
    """Categorize amount into size ranges."""
    if amount < 100:
        return "Small (< ₹100)"
    elif amount < 500:
        return "Medium (₹100-500)"
    elif amount < 2000:
        return "Large (₹500-2000)"
    else:
        return "Very Large (> ₹2000)"
