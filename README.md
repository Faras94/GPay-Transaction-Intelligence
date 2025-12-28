# GPay Transaction Intelligence 📊

An enterprise-grade financial analytics dashboard for Google Pay PDF statements. Unlock insights from your transaction history with interactive charts, categorization, and AI-powered Q&A.

## ✨ Features

- **📄 PDF Extraction**: Automatically extracts transactions from GPay PDF statements.
- **📊 Interactive Dashboard**: Visualizes spending habits with Plotly charts.
- **🤖 AI Assistant (RAG)**: Chat with your financial data using Retrieval Augmented Generation. Ask questions like *"How much did I spend on food?"*.
- **🏷️ Auto-Categorization**: Intelligently categories transactions (Food, Travel, Bills, etc.).
- **📈 Trend Analysis**: Monthly and daily spending trends.
- **📦 Data Export**: Export processed data to CSV.

## 🚀 Getting Started

### Option 1: Docker (Recommended)

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd gpay-analytics
    ```

2.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the App**:
    Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2: Local Installation

1.  **Prerequisites**: Python 3.10+ installed.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run dashboard.py
    ```

## 💡 Usage

1.  **Upload**: Use the sidebar to upload your Google Pay PDF statement (or a CSV file).
2.  **View**: Explore the "Overview", "Trends", and "Category Breakdown" tabs.
3.  **Ask**: Use the "Ask Questions" section at the bottom to query your data using AI.

## 📂 Project Structure

- `dashboard.py`: Main Streamlit application.
- `rag/`: RAG (Retrieval Augmented Generation) module for AI features.
- `transaction_processing/`: Core logic for data cleaning and categorization.
- `tests/`: Unit and Integration tests.
- `uploads/`: Directory for storing uploaded files (persisted in Docker).

## 🧪 Running Tests

To verify the system integrity:

```bash
pip install pytest
pytest tests/ -v
```

## 📝 License

MIT License
