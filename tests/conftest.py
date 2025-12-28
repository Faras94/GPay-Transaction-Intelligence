
import pytest
import pandas as pd
import os
import sys

# Ensure the root directory is in the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_transactions_df():
    """Fixture for a sample transactions DataFrame."""
    return pd.DataFrame([
        {
            "Date": "01Jan,2024",
            "Time": "10:00AM",
            "Description": "Uber Ride",
            "Amount (₹)": 500.0,
            "Type": "Spent",
            "Category": "Travel",
            "UPI ID": "123456"
        },
        {
            "Date": "02Jan,2024",
            "Time": "02:00PM",
            "Description": "Payment Result",
            "Amount (₹)": 1000.0,
            "Type": "Received",
            "Category": "Income",
            "UPI ID": "789012"
        }
    ])

@pytest.fixture
def mock_pdf_text():
    return """
    01Jan,2024 10:00AM
    Uber Ride
    UPI Transaction ID: 123456
    ₹500.00
    
    02Jan,2024 02:00PM
    Payment Result
    UPI Transaction ID: 789012
    ₹1,000.00
    """
