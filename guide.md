# Complete Setup Guide: Incident Response AI with RAG

## Prerequisites

### Required Software
1. **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop)
2. **Python 3.11+** - [Download](https://www.python.org/downloads/)
3. **Text Editor** - VS Code, Sublime, or any editor

### System Requirements
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space
- macOS, Windows, or Linux

---

## Step 1: Install Docker & Python

### macOS
```bash
# Install Docker Desktop from website, then verify:
docker --version

# Python (usually pre-installed)
python3 --version
```

### Windows
```bash
# Install Docker Desktop from website
# Install Python from python.org
# Verify in Command Prompt:
docker --version
python --version
```

---

## Step 2: Start Required Services

### Start Ollama (AI Model Runtime)
```bash
# Create and start Ollama container
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama

# Wait 5 seconds
sleep 5

# Download AI model (takes 3-5 minutes)
docker exec ollama ollama pull llama3.2:3b-instruct-q8_0

# Verify
curl http://localhost:11434/api/tags
```

### Start Qdrant (Vector Database)
```bash
# Create and start Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant

# Wait 5 seconds
sleep 5

# Verify
curl http://localhost:6333/
```

You should see JSON responses from both.

---

## Step 3: Create Project Files

### Create Project Directory
```bash
mkdir ~/incident-ai
cd ~/incident-ai
```

### Create File 1: `rag_system.py`

Copy the complete RAG system code and save as: `rag_system.py`

### Create File 2: `simple_api.py`

Copy the complete API code with streaming and save as: `simple_api.py`

### Verify Files
```bash
ls -la
# You should see:
# rag_system.py
# simple_api.py
```

---

## Step 4: Install Python Dependencies

```bash
pip3 install fastapi uvicorn httpx qdrant-client sentence-transformers
```

This installs:
- `fastapi` - Web framework
- `uvicorn` - Server
- `httpx` - HTTP client
- `qdrant-client` - Vector database client
- `sentence-transformers` - Text embeddings

---

## Step 5: Start the API

```bash
python3 simple_api.py
```

You should see:
```
Loading embedding model...
RAG system initialized
RAG initialized with 0 incidents
RAG: Enabled
Incidents: 0

Uvicorn running on http://0.0.0.0:8000
```

---

## Step 6: Seed Example Data

Open a new terminal (keep API running):

```bash
# Using curl
curl -X POST http://localhost:8000/incidents/seed

# Or open browser
# http://localhost:8000/docs
# Click on POST /incidents/seed -> Try it out -> Execute
```

Response:
```json
{
  "message": "Example incidents added",
  "total_incidents": 5
}
```

---

## Step 7: Test the System

### Test 1: Health Check
```bash
curl http://localhost:8000/
```

### Test 2: Analyze an Incident
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "High CPU usage on api-gateway",
    "description": "CPU at 95%, users reporting timeouts",
    "service": "api-gateway"
  }'
```

You should get a response referencing past incident INC-00234.

### Test 3: View API Documentation
Open browser: `http://localhost:8000/docs`

---

## Step 8: Stopping & Restarting

### Stop Everything
```bash
# Stop API (in terminal running simple_api.py)
Ctrl+C

# Stop containers
docker stop ollama qdrant
```

### Start Again Later
```bash
# Start containers
docker start ollama qdrant

# Wait 5 seconds
sleep 5

# Start API
cd ~/incident-ai
python3 simple_api.py
```

Your data persists - all incidents are still there.

---

## Verification Checklist

- [ ] Docker Desktop running
- [ ] Ollama container running: `docker ps | grep ollama`
- [ ] Qdrant container running: `docker ps | grep qdrant`
- [ ] Model downloaded: `docker exec ollama ollama list`
- [ ] API responding: `curl http://localhost:8000/`
- [ ] 5 incidents seeded: `curl http://localhost:8000/incidents/count`

---

## Common Issues & Fixes

### "Cannot connect to Ollama"
```bash
docker start ollama
sleep 5
```

### "RAG not available"
```bash
docker start qdrant
pip3 install qdrant-client sentence-transformers
```

### "Address already in use"
```bash
# Find process on port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in simple_api.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### "Model not found"
```bash
docker exec ollama ollama pull llama3.2:3b-instruct-q8_0
```

---

## Complete Fresh Start (If Everything Fails)

```bash
# Remove everything
docker stop ollama qdrant
docker rm ollama qdrant
docker volume rm ollama-data qdrant-data

# Start from Step 2
```

---

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/analyze` | POST | Analyze incident (non-streaming) |
| `/analyze/stream` | POST | Analyze incident (streaming) |
| `/incidents/seed` | POST | Add example data |
| `/incidents/add` | POST | Add resolved incident |
| `/incidents/search` | GET | Search past incidents |
| `/incidents/count` | GET | Count incidents |

---

## File Structure Summary

```
incident-ai/
├── rag_system.py          # RAG logic & vector DB
├── simple_api.py          # FastAPI server
└── (Docker containers)
    ├── ollama             # AI model runtime
    └── qdrant            # Vector database
```

---

## Tech Stack Used

### Backend
- **Python 3** - Programming language
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server

### AI/ML
- **Ollama** - Local AI runtime
- **Llama 3.2 3B (Q8)** - AI model
- **Sentence Transformers** - Text embeddings
- **all-MiniLM-L6-v2** - Embedding model

### Database
- **Qdrant** - Vector database for RAG

### Infrastructure
- **Docker** - Containerization
- **Docker Volumes** - Data persistence

---

## Next Steps

1. Test all endpoints in Postman or browser
2. Build a frontend UI
3. Integrate with your alert system (PagerDuty, etc.)
4. Add real incidents as you resolve them

---

## Support

If you encounter issues:
1. Check all containers are running: `docker ps`
2. Check API logs in terminal
3. Verify ports 8000, 11434, 6333 are not in use
4. Review common issues section above

---

## Project Summary

This system provides AI-powered incident analysis using:
- Local AI (no external API calls)
- RAG (learns from your past incidents)
- Fast responses (6-10 seconds)
- Streaming support for chatbot UIs
- Complete data ownership

Total setup time: 15-20 minutes