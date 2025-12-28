import fitz  # PyMuPDF
import re
import pandas as pd

def clean_pdf_text(text):
    """Normalize text and fix common PDF extraction artifacts."""
    text = re.sub(r'\s+', ' ', text)
    # Fix joined words - reduced need with PyMuPDF but kept for safety
    text = text.replace("Receivedfrom", "Received from ")
    text = text.replace("Paidto", "Paid to ")
    text = text.replace("Paidby", "Paid by ")
    return text.strip()

def extract_gpay_transactions(pdf_path, output_csv):
    transactions = []
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    doc.close()

    # Pre-process text
    full_text = clean_pdf_text(full_text)

    # Patterns for extraction
    date_pattern = r'(\d{1,2}\s*[A-Z][a-z]{2},\s*\d{4})'
    time_pattern = r'(\d{2}:\d{2}\s*[AP]M)'
    upi_pattern = r'UPI\s*Transaction\s*ID:\s*(\d+)'
    amount_pattern = r'₹\s*([\d,]+(?:\.\d{2})?)'

    # Split text by date blocks
    date_blocks = list(re.finditer(date_pattern, full_text))
    
    for i in range(len(date_blocks)):
        start = date_blocks[i].start()
        end = date_blocks[i+1].start() if i+1 < len(date_blocks) else len(full_text)
        current_date = date_blocks[i].group(0)
        content = full_text[start:end]

        # Find individual transactions within the date block (marked by time)
        time_matches = list(re.finditer(time_pattern, content))
        for j in range(len(time_matches)):
            t_start = time_matches[j].start()
            t_end = time_matches[j+1].start() if j+1 < len(time_matches) else len(content)
            txn_segment = content[t_start:t_end]
            
            time_val = time_matches[j].group(0)
            
            # 1. Extract UPI ID
            upi_match = re.search(upi_pattern, txn_segment)
            upi_id = upi_match.group(1) if upi_match else ""
            
            # 2. Extract Amount
            amount_match = re.search(amount_pattern, txn_segment)
            amount = amount_match.group(1).replace(',', '') if amount_match else "0.00"
            
            # 3. Extract and Clean Description
            # Start with the raw segment and remove known data points
            desc = txn_segment.replace(time_val, "")
            if upi_match:
                desc = desc.replace(upi_match.group(0), "")
            if amount_match:
                # Remove the symbol and the value
                desc = desc.replace(f"₹{amount_match.group(1)}", "").replace("₹", "")
            
            # Remove "Paid to", "Received from", and Bank footer notes
            desc = re.sub(r'Paid\s*by\s*.*Bank.*', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'Paid\s*to\s*.*Bank.*', '', desc, flags=re.IGNORECASE)
            desc = desc.replace("Received from", "").replace("Paid to", "").strip()
            
            transactions.append({
                'Date': current_date,
                'Time': time_val,
                'Description': desc,
                'UPI ID': upi_id,
                'Amount (₹)': amount
            })

    # Save to CSV
    df = pd.DataFrame(transactions)
    df.to_csv(output_csv, index=False)
    print(f"Project Complete: {len(df)} transactions extracted to {output_csv}")
    return df

# Run the project
if __name__ == "__main__":
    extract_gpay_transactions('gpay_latest.pdf', 'gpay_transactions_fixed.csv')