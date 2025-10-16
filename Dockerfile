FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rag_system.py .
COPY simple_api_upgraded.py .

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "simple_api_upgraded.py"]
