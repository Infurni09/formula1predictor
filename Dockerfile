# Formula1-AI — Production Docker Image
# Build: docker build -t formula1-ai .
# Run:   docker-compose up

FROM python:3.12-slim

LABEL maintainer="Infurni09"
LABEL description="Formula1-AI — Production ML Strategy Platform"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY dashboards/ ./dashboards/
COPY scripts/ ./scripts/
COPY .env.example .env

# Create data directories
RUN mkdir -p data/raw data/processed data/models models/trained mlruns

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000 8050

# Default: run FastAPI (override in docker-compose for dashboard)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
