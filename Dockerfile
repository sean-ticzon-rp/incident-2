FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rag_system.py simple_api_upgraded.py ./

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "simple_api_upgraded.py"]
