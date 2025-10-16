# Use lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose the port Railway uses
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "simple_api_upgraded:app", "--host", "0.0.0.0", "--port", "8000"]
