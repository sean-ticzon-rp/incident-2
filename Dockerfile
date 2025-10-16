FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python packages with minimal cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rag_system.py .
COPY simple_api_upgraded.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "simple_api_upgraded.py"]