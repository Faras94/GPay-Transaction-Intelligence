import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pypdf
import re
import tempfile
import os
from datetime import datetime
from transaction_processing import process_csv_data

# ================== OPTIONAL RAG ==================
try:
    from rag import initialize_rag, query_rag
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="GPay Transaction Intelligence v1.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== MODERN CSS STYLING ==================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding but keep header for sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;}  <-- This hides the sidebar toggle! */
    
    /* Main container */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar Styling Compact */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio {
        margin-bottom: -1rem !important; /* Pull next item closer */
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        margin-bottom: 0.5rem;
        font-weight: 500;
        color: #1e293b;
    }
    
    /* Ensure visible text in sidebar */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #1e293b;
    }
    
    /* Header Section */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #1e293b;
    }
    
    [data-testid="stSidebar"] .element-container {
        padding: 0.25rem 0;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Metric Cards - Enhanced */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    
    [data-testid="stMetric"]:hover::before {
        opacity: 1;
    }
    
    [data-testid="stMetric"] label {
        font-size: 0.8125rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Info boxes */
    .stAlert {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* File uploader - Enhanced */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #1e293b;
        position: relative;
    }

    [data-testid="stFileUploader"] label {
        color: #1e293b;
        font-weight: 500;
    }
    
    [data-testid="stFileUploader"] small {
        color: #64748b;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Buttons - Enhanced */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.9375rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.35);
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 500;
        color: #1e293b;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.625rem;
        font-size: 0.9375rem;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        color: #166534;
    }
    
    .stError {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    
    .stWarning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    
    .stInfo {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }
    
    /* Welcome screen */
    .welcome-card {
        background: white;
        border-radius: 12px;
        padding: 3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .welcome-card h2 {
        color: #1e293b;
        font-size: 1.875rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-card p {
        color: #64748b;
        font-size: 1.125rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .feature-list {
        text-align: left;
        display: inline-block;
        margin: 2rem 0;
    }
    
    .feature-list li {
        color: #475569;
        font-size: 1rem;
        line-height: 2;
        padding-left: 0.5rem;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("""
<div class="main-header">
    <h1>GPay Transaction Intelligence v1.0</h1>
    <p>Enterprise-grade financial analytics and insights platform</p>
</div>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def clean_pdf_text(text):
    """Clean extracted PDF text"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@st.cache_data(show_spinner=False)
def extract_gpay_transactions_from_file(path):
    """Extract GPay transactions from PDF - handles concatenated text format"""
    rows = []
    
    try:
        reader = pypdf.PdfReader(path)
        
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        original_text = full_text
        
        with st.expander("View extracted PDF content"):
            st.text(full_text[:2000])
        
        pattern = r'(\d{1,2}[A-Za-z]{3},\d{4})\s*(\d{1,2}:\d{2}[AP]M)'
        matches = list(re.finditer(pattern, full_text))
        
        if not matches:
            st.error("Couldn't find any transactions in this PDF. The file might be in a different format.")
            return pd.DataFrame()
        
        for i, match in enumerate(matches):
            date_str = match.group(1)
            time_str = match.group(2)
            
            start_pos = match.end()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(full_text)
            
            transaction_text = full_text[start_pos:end_pos].strip()
            
            # Extract UPI Transaction ID
            upi_pattern = r'UPI\s*Transaction\s*ID:\s*(\d+)'
            upi_match = re.search(upi_pattern, transaction_text)
            upi_id = upi_match.group(1) if upi_match else None
            
            amount_match = re.search(r'₹([\d,]+(?:\.\d{2})?)', transaction_text)
            
            if not amount_match:
                continue
            
            amount = float(amount_match.group(1).replace(",", ""))
            desc_text = transaction_text[:amount_match.start()].strip()
            
            if 'Receivedfrom' in desc_text or 'Received from' in desc_text:
                txn_type = "Received"
                desc = re.sub(r'.*Receivedfrom\s*', '', desc_text, flags=re.IGNORECASE)
            elif 'Paidto' in desc_text or 'Paid to' in desc_text:
                txn_type = "Spent"
                desc = re.sub(r'.*Paidto\s*', '', desc_text, flags=re.IGNORECASE)
            else:
                txn_type = "Spent"
                desc = desc_text
            
            desc = re.sub(r'UPITransactionID:\d+', '', desc)
            desc = re.sub(r'Paid\s*(to|by)\s*[A-Z]+\s*Bank\d+', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'[A-Z]{4}Bank\d+', '', desc)
            
            desc_parts = re.split(r'(?:UPI|Paid|Transaction)', desc, maxsplit=1)
            desc = desc_parts[0].strip()
            desc = re.sub(r'\s+', ' ', desc).strip()
            
            if not desc or desc.isdigit() or len(desc) < 2:
                desc = "Unknown Transaction"
            
            rows.append({
                "Date": date_str,
                "Time": time_str,
                "Description": desc[:100],
                "Amount (₹)": amount,
                "Type": txn_type,
                "UPI ID": upi_id
            })
        
        if rows:
            df = pd.DataFrame(rows)
            with st.expander(f"Preview: Found {len(rows)} transactions"):
                st.dataframe(df.head(10), width="stretch")
            return df
        else:
            st.error("No transactions could be extracted. Try using the CSV upload option instead.")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Something went wrong while reading the PDF: {str(e)}")
        with st.expander("Technical details"):
            import traceback
            st.code(traceback.format_exc())
        return pd.DataFrame()

def validate_dataframe(df):
    """Validate required columns in dataframe"""
    required_cols = ["Date", "Amount (₹)", "Type", "Description"]
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        st.error(f"The file is missing these required columns: {', '.join(missing)}")
        return False
    return True

def create_category_chart(data):
    """Create enhanced pie chart for categories"""
    fig = px.pie(
        data,
        values="Amount (₹)",
        names="Category",
        hole=0.4,
        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#30cfd0']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
    fig.update_layout(
        showlegend=True,
        height=400,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    return fig

def create_monthly_trend(data):
    """Create enhanced monthly trend chart"""
    fig = px.bar(
        data,
        x="Month",
        y="Amount (₹)",
        color="Type",
        barmode="group",
        color_discrete_map={"Spent": "#ef4444", "Received": "#10b981"}
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Amount",
        height=400,
        hovermode='x unified',
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_daily_trend(data):
    """Create daily spending line chart"""
    daily = data.groupby(['Date', 'Type'])['Amount (₹)'].sum().reset_index()
    
    fig = px.line(
        daily,
        x='Date',
        y='Amount (₹)',
        color='Type',
        markers=True,
        color_discrete_map={"Spent": "#ef4444", "Received": "#10b981"}
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Amount",
        height=400,
        hovermode='x unified',
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_top_expenses_chart(data, top_n=10):
    """Create bar chart of top expenses"""
    top_expenses = data.nlargest(top_n, 'Amount (₹)')
    
    fig = px.bar(
        top_expenses,
        x='Amount (₹)',
        y='Description',
        orientation='h',
        color='Category',
        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Amount",
        yaxis_title="",
        height=400,
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    return fig

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 1.5rem;'>Upload Data</h3>", unsafe_allow_html=True)
    
    
    # Check for existing file
    UPLOAD_DIR = "uploads"
    PDF_DIR = os.path.join(UPLOAD_DIR, "pdf")
    CSV_DIR = os.path.join(UPLOAD_DIR, "csv")
    
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    
    pdf_files = [f"pdf/{f}" for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    # csv_files = [f"csv/{f}" for f in os.listdir(CSV_DIR) if f.lower().endswith('.csv')] # Hidden from frontend
    
    existing_files = pdf_files # + csv_files
    existing_files.sort(reverse=True)
    
    # 1. Prepare options
    input_options = ["PDF Statement", "Manual Entry"]
    if existing_files:
        input_options.insert(0, "Select Recent File")
    
    # 2. Unified Radio
    upload_method = st.radio(
        "Choose input method",
        input_options,
        help="Select how you want to add your transactions"
    )
    
    pdf_path = None
    uploaded_file = None
    
    # 3. Handle selection
    if upload_method == "Select Recent File":
        selected_file = st.selectbox("Select file", existing_files)
        # Construct full path based on relative path
        pdf_path = os.path.join(UPLOAD_DIR, selected_file)
        uploaded_file = "LOCAL_FILE"
        st.info(f"Using stored file: {selected_file}")
        
    elif upload_method == "PDF Statement":
        uploaded_file = st.file_uploader(
            "Upload your Google Pay statement",
            type="pdf",
            help="Upload a PDF export from Google Pay"
        )
        
        # Don't save yet - will save after successful extraction
        if uploaded_file:
            # Store temporarily for processing
            pdf_path = os.path.join(PDF_DIR, uploaded_file.name)
            st.info(f"📄 Processing: {uploaded_file.name}")

    elif upload_method == "CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type="csv",
            help="Must include: Date, Description, Amount (₹), Type"
        )
        st.caption("Required columns: Date, Time, Description, Amount (₹), Type")
        
        if uploaded_file:
            # Save uploaded CSV
            csv_path = os.path.join(CSV_DIR, uploaded_file.name)
            if not os.path.exists(csv_path):
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved CSV to {csv_path}")
            else:
                st.info(f"CSV already exists in storage: {csv_path}")
            
            # Allow logic to proceed
            pdf_path = csv_path
    
    st.markdown("---")
    
    # Budget Settings (Hidden)
    monthly_budget = 20000

# ================== MAIN ==================
if uploaded_file or pdf_path or upload_method == "Manual Entry":
    # Preserving pdf_path from sidebar
    pass
    
    try:
        if upload_method == "Manual Entry":
            st.markdown("<h2 class='section-header'>Add Transaction</h2>", unsafe_allow_html=True)
            
            with st.form("manual_entry"):
                col1, col2 = st.columns(2)
                
                with col1:
                    manual_date = st.date_input("Date", datetime.now())
                    manual_desc = st.text_input("Description", placeholder="What was this for?")
                    manual_amount = st.number_input("Amount", min_value=0.0, step=10.0)
                
                with col2:
                    manual_time = st.time_input("Time", datetime.now().time())
                    manual_type = st.selectbox("Type", ["Spent", "Received"])
                
                submitted = st.form_submit_button("Add Transaction")
                
                if submitted and manual_amount > 0:
                    df_raw = pd.DataFrame([{
                        "Date": manual_date.strftime("%d%b,%Y"),
                        "Time": manual_time.strftime("%I:%M%p"),
                        "Description": manual_desc,
                        "Amount (₹)": manual_amount,
                        "Type": manual_type,
                        "UPI ID": None
                    }])
                    st.success("Transaction added successfully")
                else:
                    st.stop()
        
        if upload_method == "CSV File" or (pdf_path and pdf_path.endswith('.csv')):
            if upload_method == "CSV File" and uploaded_file:
                 # CSV already read or accessible
                 df_raw = pd.read_csv(uploaded_file)
            elif pdf_path and os.path.exists(pdf_path):
                 df_raw = pd.read_csv(pdf_path)
                 
            required_cols = ["Date", "Description", "Amount (₹)", "Type"]
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            
            if missing_cols:
                st.error(f"Your CSV is missing these columns: {', '.join(missing_cols)}")
                st.info("Make sure your file has: Date, Time (optional), Description, Amount (₹), Type")
                st.stop()
            
            if "Time" not in df_raw.columns:
                df_raw["Time"] = "00:00AM"
            
            if "UPI ID" not in df_raw.columns:
                df_raw["UPI ID"] = None
            
            st.success(f"Loaded {len(df_raw)} transactions from your file")
        
        else:
            # Check if this is a local file override or just uploaded
            if not pdf_path and uploaded_file:
                 # This handles the case where it wasn't saved in the sidebar logic for some reason
                 # typically sidebar logic handles saving for PDF. 
                 # But if user enters Manual Entry or CSV, pdf_path is None.
                 pass
            
            # If we have a pdf_path (either selected from existing or just uploaded/saved)
            if pdf_path and os.path.exists(pdf_path):
                 pass # path is ready
            if pdf_path and os.path.exists(pdf_path):
                 pass # path is ready
            elif uploaded_file and upload_method == "PDF Statement" and uploaded_file != "LOCAL_FILE":
                 # Fallback if sidebar didn't save it (shouldn't happen with new logic, but safe)
                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = tmp.name

            # RAG initialization moved to after processing to use structured data
            pass
            
            if pdf_path and os.path.exists(pdf_path):
                with st.spinner(f"Reading {os.path.basename(pdf_path)}..."):
                    df_raw = extract_gpay_transactions_from_file(pdf_path)
                    
                    # Only save PDF if extraction was successful
                    if not df_raw.empty and uploaded_file and uploaded_file != "LOCAL_FILE":
                        final_pdf_path = os.path.join(PDF_DIR, uploaded_file.name)
                        if not os.path.exists(final_pdf_path):
                            with open(final_pdf_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.success(f"✅ Saved PDF to {final_pdf_path}")
                        else:
                            st.info(f"File already exists in storage")
                    
                    # Auto-save CSV for persistence
                    if not df_raw.empty:
                        csv_name = os.path.basename(pdf_path).replace('.pdf', '.csv').replace('.PDF', '.csv')
                        # Ensure we save to the dedicated CSV folder
                        csv_path = os.path.join(os.path.dirname(os.path.dirname(pdf_path)), "csv", csv_name)
                        # Fallback if path manipulation fails (e.g. strict UPLOAD_DIR usage)
                        if "uploads" not in csv_path:
                             csv_path = os.path.join("uploads", "csv", csv_name)
                        
                        df_raw.to_csv(csv_path, index=False)
                        st.toast(f"✅ Data saved to {csv_path}")
            else:
                st.error("No valid PDF file found. Please upload a file or select one from the list.")
                st.stop()
        
        if df_raw.empty:
            st.warning("No transactions found. Try a different file or use manual entry.")
            st.stop()
        
        if not validate_dataframe(df_raw):
            st.stop()
        
        with st.spinner("Analyzing your transactions..."):
            df = process_csv_data(df_raw)
            
            def parse_gpay_date(date_str):
                try:
                    return pd.to_datetime(date_str, format='%d%b,%Y')
                except:
                    try:
                        return pd.to_datetime(date_str, errors='coerce')
                    except:
                        return pd.NaT
            
            df["Date"] = df["Date"].apply(parse_gpay_date)
            df = df.dropna(subset=['Date'])
            
            # Initialize RAG with the structured data
            if RAG_AVAILABLE:
                # Check if cache exists first to show appropriate message
                cache_exists = False
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        h = file_hash(pdf_path)
                        cache_file = cache_path(h)
                        cache_exists = os.path.exists(cache_file)
                    except:
                        pass
                
                spinner_msg = "⚡ Loading embeddings from cache..." if cache_exists else "🔄 Processing and generating embeddings..."
                
                with st.spinner(spinner_msg):
                    # Pass pdf_path as source_file for caching
                    success, status_info = initialize_rag(df=df, source_file=pdf_path if pdf_path else None)
                    if success:
                        if isinstance(status_info, dict) and "message" in status_info:
                            st.info(f"🤖 **AI Assistant**: {status_info['message']}")
                            with st.expander("📊 Embedding Details"):
                                st.write(f"**Chunks**: {status_info.get('chunk_count', 'N/A')}")
                                st.write(f"**Source**: {status_info.get('source', 'N/A')}")
                                if 'file_hash' in status_info:
                                    st.write(f"**Cache ID**: `{status_info['file_hash'][:12]}...`")
                    else:
                        msg = status_info.get("message", str(status_info)) if isinstance(status_info, dict) else str(status_info)
                        st.warning(f"AI Assistant warning: {msg}")
        
        st.success(f"All done! Processed {len(df)} transactions")
        
        # ================== FILTERS ==================
        with st.sidebar:
            st.markdown("---")
            st.markdown("<h3 style='margin-bottom: 1.5rem;'>Filters</h3>", unsafe_allow_html=True)
            
            txn_types = st.multiselect(
                "Transaction type",
                ["Spent", "Received"],
                default=["Spent", "Received"]
            )
            
            min_date = df["Date"].min().date()
            max_date = df["Date"].max().date()
            
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if "Category" in df.columns:
                categories = st.multiselect(
                    "Categories",
                    options=sorted(df["Category"].unique()),
                    default=sorted(df["Category"].unique())
                )
            else:
                categories = []
        
        filtered = df[df["Type"].isin(txn_types)].copy()
        
        if len(date_range) == 2:
            filtered = filtered[
                (filtered['Date'].dt.date >= date_range[0]) & 
                (filtered['Date'].dt.date <= date_range[1])
            ]
        
        if categories and "Category" in filtered.columns:
            filtered = filtered[filtered["Category"].isin(categories)]
        
        # ================== SUMMARY ==================
        st.markdown("<h2 class='section-header'>Overview</h2>", unsafe_allow_html=True)
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.info(f"{min_date.strftime('%d %b %Y')} to {max_date.strftime('%d %b %Y')}")
        with info_col2:
            st.info(f"{len(filtered)} transactions")
        with info_col3:
            st.info(f"{filtered['Category'].nunique() if 'Category' in filtered.columns else 0} categories")
        
        # ================== METRICS ==================
        spent = filtered[filtered["Type"] == "Spent"]
        received = filtered[filtered["Type"] == "Received"]
        
        total_spent = spent["Amount (₹)"].sum()
        total_income = received["Amount (₹)"].sum()
        net = total_income - total_spent
        
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_month_spent = spent[
            (spent['Date'].dt.month == current_month) & 
            (spent['Date'].dt.year == current_year)
        ]['Amount (₹)'].sum()
        
        budget_remaining = monthly_budget - current_month_spent
        budget_percentage = (current_month_spent / monthly_budget * 100) if monthly_budget > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Spent", f"₹{total_spent:,.2f}")
        
        with col2:
            st.metric("Total Received", f"₹{total_income:,.2f}")
        
        with col3:
            st.metric("Net Balance", f"₹{net:,.2f}", delta=f"{net:,.2f}")
            
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
        col4, col5 = st.columns(2)
        
        with col4:
            avg = total_spent / len(spent) if len(spent) > 0 else 0
            st.metric("Avg Transaction", f"₹{avg:,.2f}")
        
        with col5:
            st.metric("Budget Left", f"₹{budget_remaining:,.2f}", delta=f"{budget_percentage:.0f}% used")
        
        if budget_percentage > 100:
            st.error(f"You're over budget by ₹{abs(budget_remaining):,.2f} this month")
        elif budget_percentage > 80:
            st.warning(f"You've used {budget_percentage:.0f}% of your monthly budget")
        
        # ================== CHARTS ==================
        st.markdown("<h2 class='section-header'>Analytics</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Trends", "Top Expenses"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Spending by Category")
                if "Category" in spent.columns and not spent.empty:
                    fig = create_category_chart(spent)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No spending data available")
            
            with col2:
                st.markdown("#### Monthly Comparison")
                if not filtered.empty:
                    filtered["Month"] = filtered["Date"].dt.to_period("M").astype(str)
                    monthly_data = filtered.groupby(["Month", "Type"])["Amount (₹)"].sum().reset_index()
                    fig = create_monthly_trend(monthly_data)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No data available")
        
        with tab2:
            st.markdown("#### Daily Pattern")
            if not filtered.empty:
                fig = create_daily_trend(filtered)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available")
            
            if not spent.empty and "Category" in spent.columns:
                st.markdown("#### Category Trends")
                spent_copy = spent.copy()
                spent_copy["Month"] = spent_copy["Date"].dt.to_period("M").astype(str)
                category_trend = spent_copy.groupby(["Month", "Category"])["Amount (₹)"].sum().reset_index()
                
                fig = px.line(
                    category_trend,
                    x="Month",
                    y="Amount (₹)",
                    color="Category",
                    markers=True
                )
                fig.update_layout(
                    height=400,
                    hovermode='x unified',
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif"),
                    xaxis_title="",
                    yaxis_title="Amount"
                )
                st.plotly_chart(fig, width="stretch")
        
        with tab3:
            st.markdown("#### Top 10 Expenses")
            if not spent.empty:
                fig = create_top_expenses_chart(spent, top_n=10)
                st.plotly_chart(fig, width="stretch")
                
                st.markdown("#### Frequent Merchants")
                top_merchants = spent.groupby("Description")["Amount (₹)"].sum().nlargest(5).reset_index()
                for idx, row in top_merchants.iterrows():
                    st.text(f"{idx + 1}. {row['Description']} - ₹{row['Amount (₹)']:,.2f}")
            else:
                st.info("No spending data available")
        
        # ================== CATEGORY INSIGHTS ==================
        if not spent.empty and "Category" in spent.columns:
            st.markdown("<h2 class='section-header'>Category Breakdown</h2>", unsafe_allow_html=True)
            
            category_summary = spent.groupby("Category").agg({
                "Amount (₹)": ["sum", "mean", "count"]
            }).round(2)
            category_summary.columns = ["Total", "Average", "Count"]
            category_summary = category_summary.sort_values("Total", ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(category_summary, width="stretch")
            
            with col2:
                st.markdown("#### Quick Stats")
                top_category = category_summary.index[0]
                top_amount = category_summary.iloc[0]["Total"]
                
                st.text(f"Top category: {top_category}")
                st.text(f"Amount: ₹{top_amount:,.2f}")
                st.text(f"Share: {(top_amount/total_spent*100):.0f}% of spending")
                st.text("")
                st.text(f"Categories: {len(category_summary)}")
                st.text(f"Transactions: {int(category_summary['Count'].sum())}")
        
        # ================== TABLE ==================
        st.markdown("<h2 class='section-header'>All Transactions</h2>", unsafe_allow_html=True)
        
        search_term = st.text_input("Search", placeholder="Filter by description")
        
        display_df = filtered.copy()
        if search_term:
            display_df = display_df[display_df["Description"].str.contains(search_term, case=False, na=False)]
        
        display_df = display_df.sort_values("Date", ascending=False)
        
        st.dataframe(
            display_df,
            width="stretch",
            height=400,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Amount (₹)": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                "UPI ID": st.column_config.TextColumn("UPI Transaction ID"),
            }
        )
        
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # ================== RAG ==================
        if RAG_AVAILABLE:
            st.markdown("<h2 class='section-header'>Ask Questions</h2>", unsafe_allow_html=True)
            
            # Initialize RAG silently if needed (and we have a valid source)
            if "rag_initialized" not in st.session_state and (pdf_path or not df.empty):
                # We can try to initialize. In the original flow, it might have been initialized.
                # But here we ensure it.
                if pdf_path:
                    initialize_rag(df=df, source_file=pdf_path)
                else:
                    initialize_rag(df=df)
                st.session_state.rag_initialized = True

            # Initialize Messages History
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hi! I'm your financial assistant. Ask me anything about your spending found in the statement."}
                ]
            
            # Display Chat History
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Chat Input
            if prompt := st.chat_input("Ask a question about your transactions..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        response = query_rag(prompt)
                        
                        if isinstance(response, dict):
                            ans = response.get("answer", "No answer generated.")
                            sources = response.get("sources", [])
                        else:
                            ans = str(response)
                            sources = []
                        
                        st.markdown(ans)
                        
                        # Show sources in an expander if available
                        if sources:
                            with st.expander("View Sources & Scores"):
                                for i, src in enumerate(sources):
                                    score = src.get('cosine_score', 0.0)
                                    rerank = src.get('rerank_score', 0.0)
                                    match_type = src.get('type', 'semantic')
                                    text = src.get('text', '')
                                    
                                    st.markdown(f"**Source {i+1}** (Type: `{match_type}`)")
                                    st.markdown(f"Cosine Similarity: `{score:.4f}` | Rerank Score: `{rerank:.4f}`")
                                    st.code(text, language="text")
                                    st.divider()

                        # Add assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": ans})
    
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.exception(e)
    
    finally:
        # Only cleanup if it was a truly temporary file, but currently our logic relies on 
        # either uploads/ or tempfiles. Since we moved to persistent uploads/ for even new files,
        # we generally don't want to delete them immediately if we want "recent files" to work.
        # If we really utilized tempfile for "manual" uploads that shouldn't be saved, we would check.
        # For now, safe to remove this deletion to verify persistence.
        pass

else:
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
<div class="welcome-card">
    <h2>Get started with your finances</h2>
    <p>Upload your transaction data to see insights, track spending, and manage your budget</p>
    
    <div class="feature-list">
        <ul>
            <li>Automatic categorization of expenses</li>
            <li>Visual charts and spending patterns</li>
            <li>Budget tracking and alerts</li>
            <li>Export and analyze your data</li>
            <li>AI-powered insights (optional)</li>
        </ul>
    </div>
    
    <p style="color: #94a3b8; font-size: 0.9375rem; margin-top: 2rem;">
        Use the sidebar to upload a PDF or CSV file
    </p>
</div>
""", unsafe_allow_html=True)