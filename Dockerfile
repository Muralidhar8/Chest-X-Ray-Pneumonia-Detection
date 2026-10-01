# Production Dockerfile for Chest X-Ray Pneumonia AI Decision Support System
FROM python:3.12-slim

# System dependencies for OpenCV, libgl, font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Generate benchmark cases on container build
RUN python scripts/generate_sample_data.py
RUN python scripts/make_placeholder.py

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
