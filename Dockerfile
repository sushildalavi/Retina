FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (data/models are mounted at runtime)
COPY app/ ./app/
COPY configs/ ./configs/
COPY evaluation/ ./evaluation/
COPY retrieval/ ./retrieval/
COPY scripts/ ./scripts/
COPY serving/ ./serving/
COPY training/ ./training/

EXPOSE 8000

CMD ["python", "-m", "scripts.run_api", "--config", "/app/configs/retina.yaml", "--host", "0.0.0.0", "--port", "8000"]
