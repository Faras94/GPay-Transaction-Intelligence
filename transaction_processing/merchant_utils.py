"""
Merchant name extraction and cleaning utilities.

This module provides functions to extract and clean merchant names
from transaction descriptions.
"""

import re


def extract_merchant_name(description: str) -> str:
    """
    Extracts clean merchant name from transaction description.
    
    Removes common prefixes, suffixes, and cleans up the text.
    
    Args:
        description: Transaction description
        
    Returns:
        Cleaned merchant name
    """
    desc = str(description).strip()
    
    # Remove common prefixes/suffixes
    patterns_to_remove = [
        r'^To\s+',
        r'^From\s+',
        r'^UPI\s+',
        r'\s+-\s+.*$',  # Remove everything after dash
        r'\s+\d{10,}$',  # Remove phone numbers at end
        r'\s+@\w+$',  # Remove UPI handles
    ]
    
    for pattern in patterns_to_remove:
        desc = re.sub(pattern, '', desc, flags=re.IGNORECASE)
    
    # Limit length
    return desc[:50].strip()
