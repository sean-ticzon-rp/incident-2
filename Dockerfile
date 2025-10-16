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

# Make start script executable
RUN chmod +x start.sh

# Expose port (Railway will use this)
EXPOSE 8000

# Use startup script
CMD ["./start.sh"]