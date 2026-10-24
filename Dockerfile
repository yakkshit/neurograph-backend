# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/neurograph-q4_k_m.gguf \
    CPU_THREADS=2

# Install system dependencies for llama-cpp-python (CPU optimized)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    # Force CPU compilation for maximum compatibility and speed on Azure VM
    CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_METAL=OFF -DLLAMA_CUDA=OFF -DLLAMA_OPENMP=ON" \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create models directory (we will mount the actual model here via docker-compose)
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Run with Uvicorn (4 workers for async concurrency)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]