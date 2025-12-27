"""
Transaction categorization logic.

This module provides pure functions for categorizing transactions
based on merchant names and keywords. No external dependencies.
"""

import re
from typing import Dict, List

# Comprehensive India-specific merchant mapping
TRANSACTION_CATEGORIES: Dict[str, List[str]] = {
    "Food & Dining": [
        "ZOMATO", "SWIGGY", "RESTAURANT", "CAFE", "KITCHEN", "CHEF", 
        "BIRIYANI", "BIRYANI", "HUT", "DOSA", "ALREEM", "RESTO", "SHAPPU", 
        "BAAWRCHI", "EATS", "BAKES", "HOTEL", "ZAMZAM", "ARABIAN", "ALTAZA", 
        "PALACE", "KOTTAYAM", "LITTLE", "MCDONALD", "KFC", "DOMINO", "PIZZA",
        "BURGER", "FOOD", "MEALS", "TIFFIN", "PARCEL", "CANTEEN", "DINING",
        "CHAAT", "JUICE", "SWEET", "BAKERY", "TEA", "COFFEE", "STARBUCKS",
        "CCD", "BARISTA"
    ],
    "Travel & Transport": [
        "FUEL", "PETROL", "PUMP", "SHELL", "HPCL", "BPCL", "IOCL", "ESSAR",
        "RAILWAY", "IRCTC", "KSRTC", "UTS", "AUTO", "METRO", "FASTAG", 
        "MANAKATTIL", "SUNFUELS", "TOLL", "INDIANRAILWAY", "YATHRA", "RANI", 
        "TRAVELLS", "TRAVELS", "OLA", "UBER", "RAPIDO", "TAXI", "CAB",
        "DIESEL", "BUS", "TRAIN", "FLIGHT", "INDIGO", "SPICEJET", "AIRINDIA",
        "PARKING", "VALET"
    ],
    "Shopping & Retail": [
        "LULU", "MART", "MALL", "SUPER", "MARKET", "STORE", "SHOPPE", 
        "AMAZON", "FLIPKART", "BLINKIT", "ZEPTO", "LENSMAGIC", "FASHION", 
        "HYPER", "DECATHLON", "KKSTORE", "CENTURY", "VADUTHALA", "RELIANCE",
        "DMart", "BIGBAZAAR", "MORE", "SPENCERS", "LIFESTYLE", "WESTSIDE",
        "PANTALOONS", "CLOTHING", "FOOTWEAR", "ELECTRONICS", "MOBILE",
        "GROCERY", "VEGETABLE", "MYNTRA", "AJIO", "MEESHO", "NYKAA"
    ],
    "Bills & Services": [
        "RECHARGE", "JIO", "AIRTEL", "VI", "VODAFONE", "BSNL", "ASIANET", 
        "FIBER", "ELECTRIC", "ELECTRICITY", "KSEB", "WATER", "OPENAI", "BILL", 
        "GOOGLE", "INSURANCE", "JIOFIBER", "THOMSUN", "BROADBAND", "WIFI",
        "INTERNET", "DTH", "TATASKY", "DISH", "NETFLIX", "PRIME", "HOTSTAR",
        "SPOTIFY", "YOUTUBE", "SUBSCRIPTION", "PREPAID", "POSTPAID",
        "MAINTENANCE", "SOCIETY", "APARTMENT"
    ],
    "Health & Medical": [
        "HOSPITAL", "CLINIC", "PHARMACY", "MEDICAL", "MEDICINE", "DOCTOR",
        "HEALTH", "MEDPLUS", "APOLLO", "NETMEDS", "1MG", "PHARMA",
        "DIAGNOSTICS", "LAB", "TEST", "SCAN", "CHECKUP", "DENTAL",
        "WELLNESS", "FITNESS", "GYM"
    ],
    "Entertainment": [
        "MOVIE", "CINEMA", "PVR", "INOX", "THEATRE", "GAME", "GAMING",
        "PLAYSTATION", "XBOX", "STEAM", "BOOK", "MUSIC", "CONCERT",
        "EVENT", "TICKET", "BOOKMYSHOW", "PAYTM INSIDER"
    ],
    "Education": [
        "STUDY", "COURSE", "COLLEGE", "SCHOOL", "UNIVERSITY", "TUITION",
        "COACHING", "CLASS", "TRAINING", "EXAM", "SEMESTER", "UDEMY", "COURSERA",
        "BRILLIANT", "BYJU", "UNACADEMY", "LIBRARY", "BOOK"
    ],
    "Govt & Taxes": [
        "MISSION", "KERALA", "GOVT", "GOVERNMENT", "TREASURY", "PANCHAYAT", 
        "MVD", "TAX", "CHALLAN", "FINE", "LICENSE", "PASSPORT", "AADHAAR",
        "PAN", "MUNICIPALITY", "CORPORATION", "INFORMATIONKERALA"
    ],
    "Personal Care": [
        "SALON", "PARLOR", "PARLOUR", "SPA", "BEAUTY", "GROOMING",
        "HAIRCUT", "MASSAGE", "COSMETIC", "SKINCARE"
    ],
    "Home & Utilities": [
        "FURNITURE", "DECOR", "FURNISH", "INTERIOR", "HARDWARE",
        "PLUMBER", "ELECTRICIAN", "REPAIR", "CLEANING", "LAUNDRY",
        "GAS", "CYLINDER", "IKEA", "PEPPERFRY", "URBANLADDER"
    ],
    "Investments & Savings": [
        "MUTUAL", "FUND", "SIP", "STOCK", "SHARE", "ZERODHA", "GROWW",
        "UPSTOX", "INVESTMENT", "SAVING", "DEPOSIT", "FD", "RD"
    ],
    "Donations & Charity": [
        "DONATION", "CHARITY", "NGO", "TEMPLE", "CHURCH", "MOSQUE",
        "RELIGIOUS", "CONTRIBUTION", "ZAKAT", "SADAQAH"
    ]
}


def categorize_transaction(description: str) -> str:
    """
    Categorizes transactions based on merchant names and keywords.
    
    Uses hybrid matching:
    - Short keywords (<=3 chars): strict word boundary matching
    - Long keywords (>3 chars): flexible substring matching
    
    Args:
        description: Transaction description string
        
    Returns:
        Category name as string
    """
    desc = str(description).upper().strip()
    
    # Try to match with category keywords
    for category, tags in TRANSACTION_CATEGORIES.items():
        for tag in tags:
            # For short keywords (<=3 chars), use word boundaries to avoid false matches
            # For longer keywords, use substring matching for flexibility
            if len(tag) <= 3:
                pattern = r'\b' + re.escape(tag) + r'\b'
                if re.search(pattern, desc):
                    return category
            else:
                if tag in desc:
                    return category
    
    # Personal name detection (1-3 words, no digits, not generic)
    clean_words = re.sub(r'[^A-Z\s]', '', desc).split()
    
    # Check if it looks like a person's name
    if 1 <= len(clean_words) <= 3:
        # Exclude if contains numbers
        if not any(char.isdigit() for char in desc):
            # Exclude generic terms
            generic_terms = [
                "GENERAL", "TRANSACTION", "PAYMENT", "TRANSFER", "UPI",
                "MONEY", "SENT", "RECEIVED", "ACCOUNT", "BANK"
            ]
            if not any(term in desc for term in generic_terms):
                return "Personal Transfers"
    
    # Default category
    return "Miscellaneous"


def get_category_emoji(category: str) -> str:
    """
    Returns an emoji icon for each category.
    
    Args:
        category: Category name
        
    Returns:
        Emoji string
    """
    emoji_map = {
        "Food & Dining": "🍽️",
        "Travel & Transport": "🚗",
        "Shopping & Retail": "🛍️",
        "Bills & Services": "📱",
        "Health & Medical": "🏥",
        "Entertainment": "🎬",
        "Education": "📚",
        "Govt & Taxes": "🏛️",
        "Personal Care": "💇",
        "Home & Utilities": "🏠",
        "Investments & Savings": "💰",
        "Donations & Charity": "🙏",
        "Personal Transfers": "👤",
        "Miscellaneous": "📦"
    }
    return emoji_map.get(category, "📦")
