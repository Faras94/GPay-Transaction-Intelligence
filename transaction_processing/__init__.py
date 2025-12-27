"""
Transaction Processing Package

This package contains modular components for processing transaction data:
- categories: Transaction categorization logic
- merchant_utils: Merchant name extraction
- enrichment: Data enrichment with analytical columns
- analytics: Statistics and outlier detection
- processor: Main processing pipeline
"""

from .categories import (
    TRANSACTION_CATEGORIES,
    categorize_transaction,
    get_category_emoji
)
from .merchant_utils import extract_merchant_name
from .enrichment import add_spending_insights
from .analytics import (
    get_category_statistics,
    detect_unusual_transactions
)
from .processor import process_csv_data

__all__ = [
    'TRANSACTION_CATEGORIES',
    'categorize_transaction',
    'get_category_emoji',
    'extract_merchant_name',
    'add_spending_insights',
    'get_category_statistics',
    'detect_unusual_transactions',
    'process_csv_data'
]
