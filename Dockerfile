FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install other packages
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_system.py simple_api_upgraded.py ./

EXPOSE 8000

CMD ["python", "simple_api_upgraded.py"]
