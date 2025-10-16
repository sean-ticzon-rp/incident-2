# Use a small Python base image
FROM python:3.11-slim

# Set environment
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port (Cloudflare will map automatically)
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "simple_api_upgraded:app", "--host", "0.0.0.0", "--port", "8000"]
