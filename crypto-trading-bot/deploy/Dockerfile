# LIMITLESS Trading Bot — Docker Image
# Usage:
#   docker build -t limitless-bot .
#   docker run --env-file .env limitless-bot

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create required directories
RUN mkdir -p logs state models/saved data/historical

# Streamlit config — bind to all interfaces, disable browser auto-open
RUN mkdir -p /root/.streamlit
COPY deploy/streamlit_config.toml /root/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Default command: run the trading bot
# Override with: docker run ... limitless-bot streamlit run dashboard/app.py
CMD ["python", "main.py"]
