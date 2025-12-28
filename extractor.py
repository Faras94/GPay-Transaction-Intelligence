"""
DEPRECATED: This module is kept for backward compatibility only.

All functionality has been moved to the transaction_processing package:
- transaction_processing.categories: Transaction categorization logic
- transaction_processing.merchant_utils: Merchant name extraction
- transaction_processing.enrichment: Data enrichment
- transaction_processing.analytics: Statistics and analysis
- transaction_processing.processor: Main processing pipeline

Please import from transaction_processing instead of this file.
"""

# Re-export everything for backward compatibility
from transaction_processing import (
    TRANSACTION_CATEGORIES,
    categorize_transaction,
    get_category_emoji,
    extract_merchant_name,
    add_spending_insights,
    get_category_statistics,
    detect_unusual_transactions,
    process_csv_data,
)

__all__ = [
    "TRANSACTION_CATEGORIES",
    "categorize_transaction",
    "get_category_emoji",
    "extract_merchant_name",
    "add_spending_insights",
    "get_category_statistics",
    "detect_unusual_transactions",
    "process_csv_data",
]
