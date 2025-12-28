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

# ================== RAG SETUP ==================
try:
    from rag.rag_pipeline import initialize_rag, query_rag
    RAG_AVAILABLE = True
except ImportError:
    # Fallback/Mock for UI dev if modules missing
    RAG_AVAILABLE = False
    def initialize_rag(*args, **kwargs): return False, "RAG module not found"
    def query_rag(*args, **kwargs): return {"answer": "RAG not available", "sources": []}

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Expense Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== GOOGLE PAY STYLE CSS ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Product+Sans:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Product Sans', 'Roboto', sans-serif;
        color: #3c4043;
        background-color: #f8f9fa; /* Light grey background like GPay app */
    }

    /* Hide Streamlit Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */ /* Unhidden to allow sidebar toggle > to show */

    /* Main Container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Metric Cards (GPay Style) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 24px;
        padding: 20px 24px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.1), 0 2px 6px 2px rgba(60,64,67,0.05); /* Soft GPay shadow */
        border: none;
        transition: box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 8px 3px rgba(60,64,67,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #5f6368; /* Google Grey 700 */
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'Product Sans', sans-serif;
        font-weight: 400; /* GPay uses lighter weights for large numbers */
        font-size: 1.8rem;
        color: #202124; /* Google Grey 900 */
    }

    /* Custom Cards (Using standard markdown divs) */
    .gpay-card {
        background-color: white;
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(60,64,67,0.1), 0 4px 8px 3px rgba(60,64,67,0.05);
        border: 1px solid #e8eaed;
    }
    .gpay-card h3 {
        margin-top: 0;
        font-size: 1.1rem;
        color: #202124;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    /* Primary Headers */
    h1 {
        font-family: 'Product Sans', sans-serif;
        font-weight: 400; /* Google Logo style */
        color: #202124;
    }
    h2, h3 {
        font-family: 'Product Sans', sans-serif;
        color: #202124;
    }

    /* Monthly Filter Dropdown Styling */
    div[data-testid="stSelectbox"] > div > div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #dadce0;
    }

    /* AI Chat Styling */
    .stChatMessage {
        background-color: transparent;
    }
    .stChatMessage[data-testid="user-message"] {
        background-color: #f1f3f4; /* Google Grey 100 */
        border-radius: 20px;
        padding: 10px 20px;
    }
    div[data-testid="stChatMessageAvatarUser"] {
        display: none; /* Cleaner look without avatars often */
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1a73e8; /* Google Blue */
        color: white;
        border-radius: 24px;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 1px 2px rgba(60,64,67,0.3);
    }
    .stButton button:hover {
        background-color: #1765cc;
        box-shadow: 0 2px 4px rgba(60,64,67,0.3);
    }
    
    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 12px;
        border: 1px dashed #dadce0;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)

# ================== HELPER FUNCTIONS ==================

@st.cache_data(show_spinner=False)
def extract_gpay_transactions_from_file(path):
    """Extract GPay transactions from PDF (Preserved Logic)"""
    rows = []
    try:
        reader = pypdf.PdfReader(path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
            
        pattern = r'(\d{1,2}[A-Za-z]{3},\d{4})\s*(\d{1,2}:\d{2}[AP]M)'
        matches = list(re.finditer(pattern, full_text))
        
        if not matches:
            return pd.DataFrame()
        
        for i, match in enumerate(matches):
            date_str = match.group(1)
            time_str = match.group(2)
            
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
            
            transaction_text = full_text[start_pos:end_pos].strip()
            
            # UPI ID
            upi_match = re.search(r'UPI\s*Transaction\s*ID:\s*(\d+)', transaction_text)
            upi_id = upi_match.group(1) if upi_match else None
            
            # Amount
            amount_match = re.search(r'₹([\d,]+(?:\.\d{2})?)', transaction_text)
            if not amount_match: continue
            
            amount = float(amount_match.group(1).replace(",", ""))
            desc_text = transaction_text[:amount_match.start()].strip()
            
            # Type Logic
            if 'Receivedfrom' in desc_text or 'Received from' in desc_text:
                txn_type = "Received"
                desc = re.sub(r'.*Receivedfrom\s*', '', desc_text, flags=re.IGNORECASE)
            elif 'Paidto' in desc_text or 'Paid to' in desc_text:
                txn_type = "Spent"
                desc = re.sub(r'.*Paidto\s*', '', desc_text, flags=re.IGNORECASE)
            else:
                txn_type = "Spent"
                desc = desc_text
            
            # Cleanup Description
            desc = re.sub(r'UPITransactionID:\d+', '', desc)
            desc = re.sub(r'Paid\s*(to|by)\s*[A-Z]+\s*Bank\d+', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'[A-Z]{4}Bank\d+', '', desc)
            desc = re.split(r'(?:UPI|Paid|Transaction)', desc)[0].strip()
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
            
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame() # Soft fail

def load_data(uploaded_file):
    """Load and process data from uploaded file"""
    if uploaded_file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        df_raw = extract_gpay_transactions_from_file(tmp_path)
        return df_raw, tmp_path
    elif uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
        return df_raw, None
    return pd.DataFrame(), None

# GLOBAL DATA STATE
if "df" not in st.session_state:
    st.session_state.df = None
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None
if "rag_info" not in st.session_state:
    st.session_state.rag_info = None

# 1. SIDEBAR (Data Upload)
with st.sidebar:
    st.title("Expense Intelligence")
    st.markdown("---")
    
    st.subheader("📂 Data Management")
    sidebar_file = st.file_uploader("Upload Statement", type=['pdf', 'csv'], help="GPay PDF statement", key="sidebar_uploader")
    st.caption("Supported: GPay PDF Export, CSV")
    
    # Download Button Logic
    if st.session_state.df is not None:
        st.write("") # Spacer
        csv_buffer = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Transaction DB (CSV)",
            data=csv_buffer,
            file_name=f"gpay_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Export processed transactions (including UPI IDs) as a CSV database."
        )
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.info("Additional preferences coming soon.")
    
    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("**Version:** 1.0.0")
        st.markdown("Secure, local processing of your financial data.")


# PROCESS UPLOAD
# Check both sidebar and main uploader
uploaded_file = sidebar_file or st.session_state.get("main_uploader")

if uploaded_file:
    # Detect if file changed
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_file_id") != file_id:
        st.session_state.rag_ready = False
        st.session_state.last_file_id = file_id
        st.session_state.rag_info = None # Clear info
        # Clear chat history for new file
        st.session_state.messages = []

    with st.spinner("Processing statement..."):
        raw_df, file_path_for_rag = load_data(uploaded_file)
        
        if not raw_df.empty:
            # Process using the existing robust logic
            processed_df = process_csv_data(raw_df)
            
            # Simple Date Parsing cleanup
            processed_df["Date"] = pd.to_datetime(processed_df["Date"], errors='coerce')
            processed_df = processed_df.dropna(subset=['Date'])
            
            st.session_state.df = processed_df
            
            # Initialize RAG Logic
            if RAG_AVAILABLE:
                # We attempt initialization if not ready OR if we want to ensure we have value (e.g. after clear)
                # But we only run the heavy initialize_rag if we really need to.
                # Since rag_pipeline handles caching efficiently, calling it again is okay for "same file" checks
                # if we want to get the "Loaded from cache" message.
                
                if not st.session_state.rag_ready:
                    with st.status("🧠 Initializing AI...", expanded=True) as status:
                        st.write("Chunking and Embedding data...")
                        if file_path_for_rag:
                            success, info = initialize_rag(source_file=file_path_for_rag, df=processed_df)
                        else:
                            success, info = initialize_rag(df=processed_df)
                        
                        if success:
                            st.session_state.rag_ready = True
                            st.session_state.rag_info = info  # Persist info
                            status.update(label="✅ AI Ready!", state="complete", expanded=False)
                        else:
                            status.update(label="❌ AI Failed", state="error")
                            st.error(info.get("message"))
            
        else:
            st.error("Could not extract transactions. Please try another file.")



# MAIN CONTENT
if st.session_state.df is not None:
    df = st.session_state.df
    
    # 2. HEADER SECTION
    col_head_1, col_head_2 = st.columns([3, 1])
    with col_head_1:
        st.markdown("# Expense Intelligence")
        st.markdown("### Smart insights from your GPay data")
    
    with col_head_2:
        # Month Selector Logic
        df['Month_Year'] = df['Date'].dt.strftime('%B %Y')
        available_months = sorted(df['Month_Year'].unique(), key=lambda x: datetime.strptime(x, "%B %Y"), reverse=True)
        available_months.insert(0, "All Time")
        
        selected_month = st.selectbox("Select Period", available_months)

    # Filter Data by Month
    if selected_month != "All Time":
        mask = df['Month_Year'] == selected_month
        df_view = df[mask]
    else:
        df_view = df

    if df_view.empty:
        st.warning(f"No transactions found for {selected_month}")
    else:
        # 3. METRICS CARDS (GPay Style)
        # Calculate Metrics
        total_spent = df_view[df_view['Type'] == 'Spent']['Amount (₹)'].sum()
        total_received = df_view[df_view['Type'] == 'Received']['Amount (₹)'].sum()
        
        # Calculate Category Top
        if 'Category' in df_view.columns:
            top_cat_row = df_view[df_view['Type'] == 'Spent'].groupby('Category')['Amount (₹)'].sum().reset_index().nlargest(1, 'Amount (₹)')
            top_category = top_cat_row.iloc[0]['Category'] if not top_cat_row.empty else "N/A"
            top_cat_amt = top_cat_row.iloc[0]['Amount (₹)'] if not top_cat_row.empty else 0
        else:
            top_category = "Uncategorized"
            top_cat_amt = 0

        st.write("") # Spacer
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("Total Spent", f"₹{total_spent:,.0f}")
        with m_col2:
            st.metric("Money In", f"₹{total_received:,.0f}")
        with m_col3:
            st.metric(f"Top: {top_category}", f"₹{top_cat_amt:,.0f}")

        st.write("---") # Subtle divider

        # 4. ANALYTICS SECTION
        c_col1, c_col2 = st.columns([1, 2])
        
        with c_col1:
            st.markdown("#### Spending by Category")
            if 'Category' in df_view.columns:
                cat_data = df_view[df_view['Type'] == 'Spent'].groupby('Category')['Amount (₹)'].sum().reset_index()
                
                # Google Colors Palette
                gpay_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8AB4F8', '#F6AEA9', '#FDE293', '#81C995']
                
                fig_donut = px.pie(cat_data, values='Amount (₹)', names='Category', hole=0.6, 
                                   color_discrete_sequence=gpay_colors)
                fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
                fig_donut.update_traces(textinfo='percent+label', textposition='inside')
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Category data not available.")

        with c_col2:
            st.markdown("#### Spending Trend")
            # Daily or Monthly trend depending on view
            if selected_month == "All Time":
                # Monthly Trend
                trend_df = df_view[df_view['Type'] == 'Spent'].groupby(df_view['Date'].dt.to_period('M').astype(str))['Amount (₹)'].sum().reset_index()
                trend_df.columns = ['Date', 'Amount']
                x_axis = 'Date'
            else:
                # Daily Trend
                trend_df = df_view[df_view['Type'] == 'Spent'].groupby('Date')['Amount (₹)'].sum().reset_index()
                trend_df.columns = ['Date', 'Amount']
                x_axis = 'Date'
            
            fig_line = px.bar(trend_df, x=x_axis, y='Amount', 
                               color_discrete_sequence=['#1a73e8']) # Google Blue
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                xaxis_title="",
                yaxis_title=""
            )
            # Remove grid lines for cleaner look
            fig_line.update_xaxes(showgrid=False)
            fig_line.update_yaxes(showgrid=True, gridcolor='#f1f3f4')
            st.plotly_chart(fig_line, use_container_width=True)

        # 5. TRANSACTIONS TABLE
        st.markdown("### Recent Activity")
        
        # --- SIDEBAR FILTERS ---
        with st.sidebar:
            st.markdown("---")
            st.subheader("🔍 Table Filters")
            
            # 1. Type Filter
            type_filter = st.radio("Transaction Type", ["All", "Spent", "Received"], horizontal=True)
            
            # 2. Category Filter (Dynamic)
            if 'Category' in df_view.columns:
                unique_cats = sorted(df_view['Category'].dropna().unique())
                selected_cats = st.multiselect("Category", unique_cats, default=unique_cats[:5]) # Select first 5 by default or None? Better empty means all or all selected?
                # Let's make default empty = All
            else:
                selected_cats = []

            # 3. Amount Slider
            min_amt = float(df_view['Amount (₹)'].min()) if not df_view.empty else 0.0
            max_amt = float(df_view['Amount (₹)'].max()) if not df_view.empty else 1000.0
            amount_range = st.slider("Amount Range (₹)", min_amt, max_amt, (min_amt, max_amt))
            
        # --- FILTERING LOGIC ---
        table_df = df_view.copy()
        
        # Apply Type Filter
        if type_filter != "All":
            table_df = table_df[table_df['Type'] == type_filter]
            
        # Apply Category Filter
        if selected_cats:
            table_df = table_df[table_df['Category'].isin(selected_cats)]
            
        # Apply Amount Filter
        table_df = table_df[
            (table_df['Amount (₹)'] >= amount_range[0]) & 
            (table_df['Amount (₹)'] <= amount_range[1])
        ]
        
        # Search Box (Existing)
        search_q = st.text_input("", placeholder="Search transactions by merchant, amount...", label_visibility="collapsed")
        
        if search_q:
            table_df = table_df[table_df['Description'].str.contains(search_q, case=False, na=False)]
            
        st.dataframe(
            table_df[['Date', 'Description', 'Category', 'Amount (₹)', 'Type', 'UPI ID']].sort_values(by='Date', ascending=False),
            column_config={
                "Date": st.column_config.DateColumn("Date", format="D MMM"),
                "Amount (₹)": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                "Category": st.column_config.TextColumn("Category"),
                "UPI ID": st.column_config.TextColumn("UPI ID"),
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )

    # 6. AI ASSISTANT SECTION
    st.markdown("---")
    st.markdown("### Ask Your Expenses")
    st.caption("Powered by RAG • Ask simple questions about your spending habits")

    if not RAG_AVAILABLE:
        st.warning("⚠️ AI Assistant is currently unavailable (Libraries missing).")
    else:
        # Pushed down RAG Info Block
        if st.session_state.rag_ready and st.session_state.rag_info:
            info = st.session_state.rag_info
            with st.expander("🔍 Index & Chunk Details", expanded=False):
                st.success(f"{info.get('message', 'Active')}")
                cols = st.columns(3)
                with cols[0]: st.metric("Total Chunks", info.get('chunk_count', 'N/A'))
                with cols[1]: st.metric("Source", info.get('source', 'N/A').split()[0]) 
                with cols[2]: st.metric("Status", info.get('status', 'N/A').title())

        # Chat Container
        rag_container = st.container()
        
        with rag_container:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("How much did I spend on Swiggy last month?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.spinner("Analyzing..."):
                    response = query_rag(prompt)
                    ans = response.get("answer", "I couldn't find an answer.")
                    
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)
                
                # SOURCES DISPLAY
                sources = response.get("sources", [])
                if sources:
                    st.session_state.messages.append({"role": "assistant-source", "content": sources})
                    with st.expander("📚 Sources & context"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}**")
                            # Display scores
                            col_s1, col_s2 = st.columns(2)
                            with col_s1: st.caption(f"🔹 Cosine: `{src.get('cosine_score', 0):.4f}`")
                            with col_s2: st.caption(f"🔸 Rerank: `{src.get('rerank_score', 0):.4f}`")
                            
                            st.text(src.get('text', ''))
                            st.divider()

else:
    # WELCOME / EMPTY STATE
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h1>Expense Intelligence</h1>
        <p style="font-size: 1.2rem; color: #5f6368;">Upload your Google Pay PDF statement to get started.</p>
        <div style="background: white; padding: 2rem; border-radius: 20px; max-width: 600px; margin: 2rem auto; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <ul style="text-align: left; list-style: none; padding: 0; color: #3c4043;">
                <li style="margin-bottom: 1rem;">✅ <strong>Financial Overview</strong>: Track total spending and income</li>
                <li style="margin-bottom: 1rem;">📊 <strong>Smart Analytics</strong>: Visualize category breakdowns</li>
                <li style="margin-bottom: 1rem;">🤖 <strong>AI Assistant</strong>: Chat with your transaction history</li>
                <li>🔒 <strong>Private</strong>: Data processes locally in memory</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a prominent uploader for the empty state
    st.markdown("<h3 style='text-align: center; color: #5f6368; margin-bottom: 1rem;'>Get Started</h3>", unsafe_allow_html=True)
    cols = st.columns([1, 2, 1])
    with cols[1]:
        main_uploaded_file = st.file_uploader("Upload PDF Statement (Center)", type=['pdf', 'csv'], label_visibility="collapsed", key="main_uploader")
        
        if main_uploaded_file:
            # Force rerun to let the top-level logic handle the global state update
            # based on the now-populated 'main_uploader' key
            st.rerun()