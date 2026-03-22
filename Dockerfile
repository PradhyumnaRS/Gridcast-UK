# Dockerfile — HuggingFace Spaces deployment
# Serves FastAPI (port 7860) + React static files from a single container
# Redis is not available on HuggingFace free tier — caching is gracefully disabled

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ ./api/

# Copy model and features
COPY Models/ ./Models/
COPY Data/processed/features.parquet ./Data/processed/features.parquet

# Copy React production build as static files
COPY frontend/build/ ./frontend/build/

# Expose HuggingFace Spaces port
EXPOSE 7860

# Run uvicorn on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]