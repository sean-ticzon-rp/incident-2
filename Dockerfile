# Dockerfile (production-friendly, small image)
FROM python:3.11-slim

# Avoid running as root in production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for some libraries (qdrant-client may need openssl/cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Railway will use this)
EXPOSE 8000

# Use Railway's PORT env variable with better proxy settings
CMD uvicorn simple_api_upgraded:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'