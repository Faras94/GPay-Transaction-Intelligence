from extractor import categorize_transaction

cases = [
    ("VI RECHARGE", "Bills & Services"),
    ("VISHNU S", "Personal Transfers"),
    ("COLLEGE FEE PAYMENT", "Education"),
    ("THAUFEEQ S", "Personal Transfers"),
    ("ZOMATO ORDER", "Food & Dining"),
    ("AMAZON INDIA", "Shopping & Retail")
]

print("Running categorization tests...")
for desc, expected in cases:
    result = categorize_transaction(desc)
    status = "✅" if result == expected else f"❌ (Got: {result})"
    print(f"{status} '{desc}' -> {expected}")
