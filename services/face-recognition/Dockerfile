FROM python:3.11-slim

WORKDIR /app

# Install necessary system packages for OpenCV and hnswlib
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    build-essential \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Ensure models directory exists (models are mounted as a volume in docker compose)
RUN mkdir -p ./app/models

# (Optional) Copy models to image if needed
# COPY models/ ./app/models/

WORKDIR /app/app

ENV MODELS_PATH=/app/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "face_recog_api:app", "--host", "0.0.0.0", "--port", "8000"]
