# GPay Transaction Intelligence 📊

[![CI - Tests & Code Quality](https://github.com/Faras94/GPay-Transaction-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Faras94/GPay-Transaction-Intelligence/actions/workflows/ci.yml)
[![Docker Build & Push](https://github.com/Faras94/GPay-Transaction-Intelligence/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Faras94/GPay-Transaction-Intelligence/actions/workflows/docker-build.yml)
[![codecov](https://codecov.io/gh/Faras94/GPay-Transaction-Intelligence/branch/main/graph/badge.svg)](https://codecov.io/gh/Faras94/GPay-Transaction-Intelligence)

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

## 🔄 CI/CD Pipeline

This project uses **100% free** CI/CD services:

### Automated Workflows

- **Continuous Integration** (`ci.yml`)
  - Runs on every push and pull request
  - Tests across Python 3.10 and 3.11
  - Code quality checks (flake8, black, isort)
  - Security scanning with Trivy
  - Coverage reporting to Codecov

- **Docker Build & Push** (`docker-build.yml`)
  - Builds multi-platform images (amd64, arm64)
  - Publishes to GitHub Container Registry (GHCR) - **FREE**
  - Automatic versioning with tags
  - Triggered on main branch pushes and version tags

### Running CI Checks Locally

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Check code formatting
black --check .
isort --check-only .

# Run linter
flake8 .
```

### Using the Docker Image

Pull and run the pre-built image from GitHub Container Registry:

```bash
# Pull the latest image (FREE - no Docker Hub account needed)
docker pull ghcr.io/faras94/gpay-transaction-intelligence:latest

# Run the container
docker run -p 8501:8501 ghcr.io/faras94/gpay-transaction-intelligence:latest
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting locally (see above)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

The CI pipeline will automatically run tests and checks on your PR!

## 📝 License

MIT License
