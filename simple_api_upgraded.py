"""
INCIDENT AI WITH RAG - ENHANCED COMPETITION EDITION
Modified to use GROQ API instead of local Ollama
No download needed - just set your API key!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
import httpx
import time
import json
import os
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import asyncio

from dotenv import load_dotenv

load_dotenv()  # Load .env variables

# Define app first
app = FastAPI(
    title="Incident AI - Groq API Edition",
    description="🏆 Advanced AI-powered incident analysis with RAG - using Groq API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://localhost:5173",
        "http://localhost:3000",
        "https://your-frontend-domain.com",  # Add your production frontend URL here
        "*"  # Allow all for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# RAG SYSTEM SETUP
# ============================================================================

# RAG system setup
RAG_AVAILABLE = False
rag = None

try:
    from rag_system import IncidentRAG, seed_example_data_async
    RAG_AVAILABLE = True
    logger.info("✅ RAG module imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ RAG system not available: {e}")

if RAG_AVAILABLE:
    try:
        # FIX: IncidentRAG only accepts collection_name parameter
        # It reads QDRANT_URL from environment variables internally
        rag = IncidentRAG(collection_name="incidents")
        logger.info(f"✅ RAG initialized with {rag.count_incidents()} incidents")
    except Exception as e:
        logger.error(f"⚠️ Could not initialize RAG: {e}")
        import traceback
        logger.error(traceback.format_exc())

# ============================================================================
# CONFIGURATION - Set your API key here or in environment variable
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Choose model (all FREE on Groq!)
MODEL_NAME = "llama-3.1-8b-instant"  # ⭐⭐⭐⭐⭐ Fast & smart

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Analytics data storage
analytics_data = {
    "total_analyses": 0,
    "successful_analyses": 0,
    "failed_analyses": 0,
    "response_times": [],
    "incidents_by_service": defaultdict(int),
    "incidents_by_hour": defaultdict(int),
    "confidence_scores": [],
    "rag_matches": 0,
    "rag_no_matches": 0,
    "tools_used": defaultdict(int),
    "agentic_calls": 0,
    "quality_scores": [],
    "last_updated": datetime.now().isoformat()
}

# ============================================================================
# CONVERSATION MANAGER - FIXES REPETITION
# ============================================================================

class ConversationManager:
    """Manages conversation history to prevent repetitive responses"""
    
    def __init__(self):
        self.conversations = {}
        self.max_history = 10
        logger.info("💬 Conversation Manager initialized")
    
    def add_message(self, incident_id: str, role: str, content: str):
        if incident_id not in self.conversations:
            self.conversations[incident_id] = []
        
        self.conversations[incident_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversations[incident_id]) > self.max_history:
            self.conversations[incident_id] = self.conversations[incident_id][-self.max_history:]
    
    def get_history(self, incident_id: str) -> List[Dict]:
        return self.conversations.get(incident_id, [])
    
    def clear_history(self, incident_id: str):
        if incident_id in self.conversations:
            del self.conversations[incident_id]

# ============================================================================
# ADAPTIVE LLM CONFIGURATION SYSTEM
# ============================================================================

class AdaptiveLLMConfig:
    """Auto-tunes LLM parameters based on response quality"""
    
    def __init__(self):
        self.config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 600,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2
        }
        self.response_history = deque(maxlen=20)
        self.total_adjustments = 0
        logger.info("🤖 Adaptive LLM Config initialized")
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def analyze_quality(self, text: str) -> float:
        if len(text) < 30:
            return 0.3
        
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        word_count = len(words)
        length_score = 1.0 if 80 <= word_count <= 400 else 0.7
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        structure_score = 1.0 if len(sentences) >= 3 else 0.6
        
        repetitive_phrases = ["root cause assessment", "immediate action steps", "prevention measure"]
        phrase_count = sum(text.lower().count(phrase) for phrase in repetitive_phrases)
        repetition_penalty = 1.0 if phrase_count <= 1 else 0.8
        
        quality = (unique_ratio * 0.4 + length_score * 0.25 + structure_score * 0.25 + repetition_penalty * 0.1)
        return min(1.0, quality)
    
    def update(self, response_text: str) -> Dict[str, Any]:
        quality = self.analyze_quality(response_text)
        
        self.response_history.append({
            "timestamp": datetime.now().isoformat(),
            "quality": quality,
            "config": self.config.copy()
        })
        
        adjustments = {}
        
        if quality < 0.6:
            old_freq = self.config["frequency_penalty"]
            self.config["frequency_penalty"] = min(0.5, old_freq + 0.05)
            adjustments["frequency_penalty"] = f"{old_freq:.2f} → {self.config['frequency_penalty']:.2f}"
            self.total_adjustments += 1
            logger.warning(f"⚠️ Low quality ({quality:.2f}), adjusted parameters")
        
        return {
            "quality": quality,
            "adjustments": adjustments,
            "config": self.config
        }
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.response_history:
            return {
                "total_responses": 0,
                "average_quality": 0.0,
                "total_adjustments": self.total_adjustments,
                "current_config": self.config
            }
        
        qualities = [r["quality"] for r in self.response_history]
        return {
            "total_responses": len(self.response_history),
            "average_quality": round(sum(qualities) / len(qualities), 2),
            "min_quality": round(min(qualities), 2),
            "max_quality": round(max(qualities), 2),
            "total_adjustments": self.total_adjustments,
            "current_config": self.config,
            "trend": "improving" if len(qualities) > 5 and qualities[-1] > qualities[0] else "stable"
        }

# ============================================================================
# CONTEXT GATHERING SYSTEM
# ============================================================================

class ContextGatherer:
    """Gathers context for better AI analysis"""
    
    async def gather(self, title: str, description: str, service: str, similar_incidents: List[Dict]) -> Dict[str, Any]:
        return {
            "rag_context": similar_incidents,
            "service_info": self._get_service_info(service),
            "time_context": self._get_time_context(),
            "keywords": self._extract_keywords(title, description)
        }
    
    def _get_service_info(self, service: str) -> Dict[str, Any]:
        services = {
            "api-gateway": {
                "replicas": 3,
                "cpu": "45%",
                "memory": "1.2GB",
                "common_issues": ["timeout", "rate limiting", "502 errors"]
            },
            "database": {
                "replicas": 2,
                "cpu": "65%",
                "memory": "8GB",
                "common_issues": ["slow queries", "connection pool", "deadlocks"]
            },
            "auth-service": {
                "replicas": 2,
                "cpu": "30%",
                "memory": "512MB",
                "common_issues": ["token expiry", "redis cache", "rate limiting"]
            },
            "payment-service": {
                "replicas": 4,
                "cpu": "55%",
                "memory": "2GB",
                "common_issues": ["timeout", "api errors", "webhook failures"]
            },
            "system": {
                "replicas": "N/A",
                "cpu": "N/A",
                "memory": "N/A",
                "common_issues": ["general system errors", "configuration issues"]
            }
        }
        return services.get(service.lower(), services["system"])
    
    def _get_time_context(self) -> Dict[str, Any]:
        now = datetime.now()
        hour = now.hour
        day = now.strftime("%A")
        
        return {
            "current_time": now.isoformat(),
            "hour": hour,
            "day": day,
            "is_deployment_window": hour in [9, 10, 14, 15],
            "is_business_hours": 9 <= hour <= 17,
            "risk_level": "high" if hour in [9, 10, 14, 15] else "normal"
        }
    
    def _extract_keywords(self, title: str, description: str) -> List[str]:
        text = f"{title} {description}".lower()
        terms = [
            "timeout", "503", "502", "500", "404", "memory", "cpu",
            "disk", "latency", "slow", "error", "crash", "restart",
            "database", "redis", "cache", "queue", "deployment",
            "connection", "pool", "exhausted", "high load"
        ]
        return [term for term in terms if term in text]

# ============================================================================
# INITIALIZE SYSTEMS
# ============================================================================

adaptive_config = AdaptiveLLMConfig()
context_gatherer = ContextGatherer()
conversation_manager = ConversationManager()
webhook_subscribers = []

# ============================================================================
# GROQ API HELPER FUNCTIONS
# ============================================================================

async def call_groq_api(messages: List[Dict], stream: bool = False):
    """Call Groq API"""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not set. Get free key at https://console.groq.com"
        )
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    llm_config = adaptive_config.get_config()
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": llm_config["temperature"],
        "max_tokens": llm_config["max_tokens"],
        "top_p": llm_config["top_p"],
        "frequency_penalty": llm_config.get("frequency_penalty", 0.3),
        "presence_penalty": llm_config.get("presence_penalty", 0.2),
        "stream": stream
    }
    
    return headers, payload

def build_messages(prompt: str, conversation_history: List[Dict] = None) -> List[Dict]:
    """Build messages for Groq API"""
    messages = [
        {"role": "system", "content": "You are an expert SRE assistant helping analyze incidents."}
    ]
    
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    messages.append({"role": "user", "content": prompt})
    
    return messages

# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_conversation_aware_prompt(
    title: str,
    description: str,
    service: str,
    context: Dict[str, Any],
    user_question: str,
    conversation_history: List[Dict]
) -> str:
    """Build prompts that understand conversation context"""
    
    similar = context["rag_context"]
    
    already_mentioned = {
        "commands": set(),
        "topics": set(),
        "incident_ids": set()
    }
    
    for msg in conversation_history:
        if msg["role"] == "assistant":
            content_lower = msg["content"].lower()
            
            if "pg_stat_activity" in content_lower:
                already_mentioned["commands"].add("pg_stat_activity")
            if "pg_settings" in content_lower:
                already_mentioned["commands"].add("pg_settings")
            if "ps aux" in content_lower:
                already_mentioned["commands"].add("ps aux")
            
            if "root cause" in content_lower:
                already_mentioned["topics"].add("root_cause")
            if "prevention" in content_lower:
                already_mentioned["topics"].add("prevention")
            
            for inc in similar:
                if inc["incident_id"].lower() in content_lower:
                    already_mentioned["incident_ids"].add(inc["incident_id"])
    
    question_lower = user_question.lower()
    
    asking_about_similar = any(word in question_lower for word in 
                               ["similar", "past", "previous", "related", "other", "cases", "guide"])
    
    asking_without_similar = any(phrase in question_lower for phrase in 
                                 ["without", "instead of", "other than", "different", "alternative"])
    
    if asking_without_similar and similar:
        prompt = f"""You're helping with: {title}

The user asked: "{user_question}"

**Important:** The user explicitly wants to solve this WITHOUT using the similar incident approach.

**Previous conversation:**
"""
        for msg in conversation_history[-4:]:
            role = "User" if msg["role"] == "user" else "You"
            prompt += f"{role}: {msg['content'][:120]}...\n"
        
        prompt += f"""

**Commands already suggested (DON'T repeat these):**
{', '.join(already_mentioned['commands']) if already_mentioned['commands'] else 'None yet'}

**Provide COMPLETELY DIFFERENT troubleshooting:**

1. Alternative diagnostic (different from what you said before)
2. Different root cause angle
3. Quick fixes

Keep under 150 words. Be specific."""

    elif asking_about_similar and similar:
        best = similar[0]
        prompt = f"""User asked: "{user_question}"

**Found: Incident {best['incident_id']} (73% match)**

What happened: {best['title']}
Root cause: {best['root_cause']}
How we fixed it: {best['resolution']}

Answer the user's question about this similar case. Be conversational. Under 150 words."""

    else:
        prompt = f"""Conversation about: {title}

**Recent chat:**
"""
        for msg in conversation_history[-4:]:
            role = "User" if msg["role"] == "user" else "You"
            prompt += f"{role}: {msg['content'][:100]}...\n"
        
        prompt += f"""

User asked: "{user_question}"

**Already covered (don't repeat):**
- Commands: {', '.join(already_mentioned['commands']) if already_mentioned['commands'] else 'none'}
- Topics: {', '.join(already_mentioned['topics']) if already_mentioned['topics'] else 'none'}

Answer directly. Reference what was discussed. Add NEW information. Under 120 words."""
    
    return prompt

def build_enhanced_prompt(
    title: str, 
    description: str, 
    service: str, 
    context: Dict[str, Any],
    user_question: Optional[str] = None
) -> str:
    """Enhanced prompt for first message"""
    
    similar = context["rag_context"]
    
    prompt = f"""You're an expert SRE assistant analyzing this incident.

**Incident:**
Service: {service}
Title: {title}
Description: {description}

"""
    
    if similar:
        best = similar[0]
        sim_pct = int(best['similarity_score'] * 100)
        
        if sim_pct >= 75:
            prompt += f"""**Found incident {best['incident_id']} ({sim_pct}% match):**
Cause: {best['root_cause']}
Fix: {best['resolution']}

Start with: "This looks very similar to incident {best['incident_id']} we had before."
"""
        elif sim_pct >= 60:
            prompt += f"""**Found incident {best['incident_id']} ({sim_pct}% similar):**
Cause: {best['root_cause']}
Fix: {best['resolution']}

Start with: "I found a related incident ({best['incident_id']})."
"""
        else:
            prompt += f"""**Past incidents only {sim_pct}% similar.**

Start with: "I found some loosely related incidents, but this looks like a new pattern."
"""
    else:
        prompt += """**No similar past incidents.**

Start with: "This appears to be a new type of incident."
"""
    
    service_info = context.get("service_info", {})
    if service_info.get("common_issues"):
        prompt += f"\nNote: {service} often has {', '.join(service_info['common_issues'][:2])}.\n"
    
    keywords = context.get("keywords", [])
    if keywords:
        prompt += f"Detected: {', '.join(keywords[:4])}\n"
    
    prompt += """
Give:
1. Quick diagnosis (2 sentences)
2. Immediate steps (3-4 specific commands)
3. Prevention tip (1 sentence)

Conversational style. Under 200 words."""
    
    return prompt

# ============================================================================
# API ENDPOINTS - HOME & HEALTH
# ============================================================================

@app.get("/")
async def home():
    """Home endpoint"""
    try:
        incident_count = rag.count_incidents() if rag else 0
        
        return {
            "status": "🚀 Incident AI - Groq API Edition",
            "version": "2.0 - No Ollama Download Needed!",
            "model": MODEL_NAME,
            "api_provider": "Groq (FREE)",
            "services": {
                "groq_api": "✅ ready" if GROQ_API_KEY else "❌ API key not set",
                "rag": "✅ enabled" if rag else "❌ disabled"
            },
            "knowledge_base": {
                "incidents": incident_count,
                "auto_learning": True
            },
            "endpoints": {
                "analysis": ["/analyze", "/analyze/stream", "/analyze/agentic-stream"],
                "incidents": ["/incidents/add", "/incidents/search", "/incidents/list"],
                "analytics": ["/api/analytics/overview", "/api/analytics/export"],
                "conversation": ["/conversation/{incident_id}"]
            }
        }
    except Exception as e:
        logger.error(f"Error in home endpoint: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/debug/qdrant")
async def debug_qdrant():
    """Debug Qdrant connection"""
    qdrant_url = os.getenv("QDRANT_URL", "not set")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(qdrant_url)
            return {
                "qdrant_url": qdrant_url,
                "status": "reachable",
                "status_code": response.status_code,
                "response": response.json() if response.text else "no content"
            }
    except Exception as e:
        return {
            "qdrant_url": qdrant_url,
            "status": "unreachable",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check Groq API
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                response = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
                health_status["services"]["groq"] = {
                    "status": "✅ connected" if response.status_code == 200 else "⚠️ error",
                    "reachable": True
                }
        except Exception as e:
            health_status["services"]["groq"] = {
                "status": "⚠️ connection error",
                "reachable": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["groq"] = {
            "status": "❌ API key not set",
            "message": "Set GROQ_API_KEY environment variable"
        }
        health_status["status"] = "degraded"
    
    # Check RAG
    health_status["services"]["rag"] = {
        "status": "✅ enabled" if rag else "❌ disabled",
        "incidents_count": rag.count_incidents() if rag else 0
    }
    
    return health_status
# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/analyze")
async def analyze_basic(incident: dict):
    """Basic non-streaming analysis"""
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    
    if not title or not description:
        raise HTTPException(status_code=400, detail="Missing title or description")
    
    start_time = time.time()
    analytics_data["total_analyses"] += 1
    analytics_data["incidents_by_service"][service] += 1
    
    # Search similar incidents
    similar_incidents = []
    if rag:
        try:
            similar_incidents = rag.search_similar(
                query=f"{title}. {description}",
                service=service,
                limit=3
            )
            if similar_incidents:
                analytics_data["rag_matches"] += 1
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
    
    context = await context_gatherer.gather(title, description, service, similar_incidents)
    prompt = build_enhanced_prompt(title, description, service, context)
    messages = build_messages(prompt)
    
    try:
        headers, payload = await call_groq_api(messages, stream=False)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                elapsed = time.time() - start_time
                
                quality_report = adaptive_config.update(analysis_text)
                
                analytics_data["successful_analyses"] += 1
                analytics_data["response_times"].append(elapsed)
                analytics_data["quality_scores"].append(quality_report["quality"])
                analytics_data["last_updated"] = datetime.now().isoformat()
                
                return {
                    "incident_title": title,
                    "service": service,
                    "analysis": analysis_text,
                    "quality_score": quality_report["quality"],
                    "similar_incidents": [
                        {
                            "id": inc["incident_id"],
                            "similarity": inc["similarity_score"],
                            "title": inc["title"]
                        }
                        for inc in similar_incidents[:3]
                    ],
                    "response_time": round(elapsed, 2),
                    "model": MODEL_NAME,
                    "mode": "basic"
                }
            
            error_data = response.json() if response.text else {"error": "Unknown error"}
            raise HTTPException(status_code=response.status_code, detail=str(error_data))
    
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Groq API")
    except Exception as e:
        analytics_data["failed_analyses"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/stream")
async def analyze_stream(incident: dict):
    """Streaming analysis"""
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    
    if not title or not description:
        async def error_gen():
            yield f"data: {json.dumps({'error': 'Missing title or description'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    async def generate_stream():
        try:
            start_time = time.time()
            analytics_data["total_analyses"] += 1
            
            # Search similar
            similar_incidents = []
            if rag:
                try:
                    similar_incidents = rag.search_similar(
                        query=f"{title}. {description}",
                        service=service,
                        limit=3
                    )
                except Exception as e:
                    logger.error(f"RAG search: {e}")
            
            context = await context_gatherer.gather(title, description, service, similar_incidents)
            prompt = build_enhanced_prompt(title, description, service, context)
            messages = build_messages(prompt)
            
            # Send metadata
            metadata = {
                "type": "metadata",
                "incident": title,
                "service": service,
                "similar_count": len(similar_incidents)
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Call Groq API with streaming
            headers, payload = await call_groq_api(messages, stream=True)
            
            full_response = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", GROQ_URL, headers=headers, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if delta := data['choices'][0]['delta'].get('content'):
                                    full_response += delta
                                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                                    await asyncio.sleep(0.01)
                            except json.JSONDecodeError:
                                continue
            
            elapsed = time.time() - start_time
            quality_report = adaptive_config.update(full_response)
            
            analytics_data["successful_analyses"] += 1
            analytics_data["response_times"].append(elapsed)
            analytics_data["quality_scores"].append(quality_report["quality"])
            
            completion = {
                "type": "done",
                "response_time": round(elapsed, 2),
                "quality": quality_report["quality"]
            }
            yield f"data: {json.dumps(completion)}\n\n"
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/analyze/agentic-stream")
async def analyze_agentic_stream(incident: dict):
    """Conversation-aware streaming - FIXES REPETITION"""
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    incident_id = incident.get('incident_id', title)
    
    # Extract user question
    user_question = ""
    if "\n\nUser Question: " in description:
        parts = description.split("\n\nUser Question: ")
        description = parts[0]
        user_question = parts[1] if len(parts) > 1 else ""
    
    if not title:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Missing title'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    async def generate_agentic_stream():
        try:
            start_time = time.time()
            logger.info(f"🚀 Agentic analysis: {title}")
            
            analytics_data["total_analyses"] += 1
            analytics_data["agentic_calls"] += 1
            
            # Get conversation history
            conversation_history = conversation_manager.get_history(incident_id)
            
            if user_question:
                conversation_manager.add_message(incident_id, "user", user_question)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing context...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Search similar
            similar_incidents = []
            if rag:
                try:
                    similar_incidents = rag.search_similar(
                        query=f"{title}. {description}",
                        service=service,
                        limit=3
                    )
                    if similar_incidents:
                        analytics_data["rag_matches"] += 1
                except Exception as e:
                    logger.error(f"RAG error: {e}")
            
            context = await context_gatherer.gather(title, description, service, similar_incidents)
            
            # Build conversation-aware prompt
            if user_question and conversation_history:
                prompt = build_conversation_aware_prompt(
                    title, description, service, context,
                    user_question, conversation_history
                )
            else:
                prompt = build_enhanced_prompt(title, description, service, context, user_question)
            
            messages = build_messages(prompt, conversation_history)
            
            # Send metadata
            metadata = {
                "type": "metadata",
                "incident_title": title,
                "service": service,
                "conversation_length": len(conversation_history),
                "similar_past_incidents": [
                    {
                        "id": inc["incident_id"],
                        "similarity": round(inc["similarity_score"], 2),
                        "title": inc["title"]
                    }
                    for inc in similar_incidents[:2]
                ]
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            
            # Call Groq API with streaming
            headers, payload = await call_groq_api(messages, stream=True)
            
            full_response = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", GROQ_URL, headers=headers, json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if delta := data['choices'][0]['delta'].get('content'):
                                    full_response += delta
                                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                                    await asyncio.sleep(0.01)
                            except json.JSONDecodeError:
                                continue
            
            elapsed = time.time() - start_time
            
            # Add response to history
            conversation_manager.add_message(incident_id, "assistant", full_response)
            
            quality_report = adaptive_config.update(full_response)
            
            analytics_data["successful_analyses"] += 1
            analytics_data["response_times"].append(elapsed)
            analytics_data["quality_scores"].append(quality_report["quality"])
            
            completion = {
                "type": "done",
                "response_time": round(elapsed, 2),
                "quality_score": quality_report["quality"],
                "conversation_messages": len(conversation_history) + 2
            }
            yield f"data: {json.dumps(completion)}\n\n"
        
        except Exception as e:
            logger.error(f"Agentic stream error: {e}")
            analytics_data["failed_analyses"] += 1
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_agentic_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# CONVERSATION ENDPOINTS
# ============================================================================

@app.get("/conversation/{incident_id}")
async def get_conversation(incident_id: str):
    history = conversation_manager.get_history(incident_id)
    return {
        "incident_id": incident_id,
        "message_count": len(history),
        "history": history
    }

@app.delete("/conversation/{incident_id}")
async def clear_conversation(incident_id: str):
    conversation_manager.clear_history(incident_id)
    return {
        "message": "✅ Conversation history cleared",
        "incident_id": incident_id
    }

# ============================================================================
# INCIDENT MANAGEMENT (keeping existing endpoints)
# ============================================================================

@app.post("/incidents/seed")
async def seed_incidents():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        seed_example_data(rag)
        return {
            "message": "✅ Incidents seeded successfully",
            "total": rag.count_incidents()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/incidents/add")
async def add_incident(incident: dict):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    required = ['incident_id', 'title', 'description', 'service']
    if not all(k in incident for k in required):
        raise HTTPException(status_code=400, detail=f"Missing required fields: {required}")
    
    try:
        doc_id = rag.add_incident(
            incident_id=incident['incident_id'],
            title=incident['title'],
            description=incident['description'],
            service=incident['service'],
            root_cause=incident.get('root_cause', ''),
            resolution=incident.get('resolution', ''),
            severity=incident.get('severity', 'medium'),
            tags=incident.get('tags', [])
        )
        return {
            "message": "✅ Incident added",
            "document_id": doc_id,
            "total_incidents": rag.count_incidents()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/search")
async def search_incidents(query: str, service: Optional[str] = None, limit: int = 5):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        results = rag.search_similar(query=query, service=service, limit=limit)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/count")
async def count_incidents():
    if not rag:
        return {"count": 0, "rag_available": False}
    return {"count": rag.count_incidents(), "rag_available": True}

# ============================================================================
# ANALYTICS (keeping key endpoints)
# ============================================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    total = analytics_data["total_analyses"]
    avg_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"]) if analytics_data["response_times"] else 0
    success_rate = (analytics_data["successful_analyses"] / total * 100) if total > 0 else 0
    avg_quality = sum(analytics_data["quality_scores"]) / len(analytics_data["quality_scores"]) if analytics_data["quality_scores"] else 0
    
    return {
        "total_analyses": total,
        "successful_analyses": analytics_data["successful_analyses"],
        "failed_analyses": analytics_data["failed_analyses"],
        "success_rate": round(success_rate, 1),
        "average_response_time": round(avg_time, 2),
        "average_quality_score": round(avg_quality, 2),
        "knowledge_base_size": rag.count_incidents() if rag else 0,
        "rag_match_rate": round(analytics_data["rag_matches"] / total * 100, 1) if total > 0 else 0,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/analytics/export")
async def export_all_analytics():
    total = analytics_data["total_analyses"]
    avg_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"]) if analytics_data["response_times"] else 0
    success_rate = (analytics_data["successful_analyses"] / total * 100) if total > 0 else 0
    avg_quality = sum(analytics_data["quality_scores"]) / len(analytics_data["quality_scores"]) if analytics_data["quality_scores"] else 0
    
    return {
        "export_timestamp": datetime.now().isoformat(),
        "overview": {
            "total_analyses": total,
            "successful": analytics_data["successful_analyses"],
            "failed": analytics_data["failed_analyses"],
            "success_rate": round(success_rate, 1),
            "avg_quality": round(avg_quality, 2)
        },
        "performance": {
            "avg_response_time": round(avg_time, 2),
            "response_times": [round(t, 2) for t in analytics_data["response_times"][-100:]]
        },
        "model": {
            "name": MODEL_NAME,
            "provider": "Groq",
            "features": ["RAG", "Streaming", "Adaptive Config", "Anti-Repetition"]
        }
    }

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 INCIDENT AI - GROQ API EDITION")
    print("="*70)
    print(f"\n🤖 Model: {MODEL_NAME}")
    print(f"🌐 API: Groq (FREE - no download!)")
    print(f"🔑 API Key: {'✅ Set' if GROQ_API_KEY else '❌ Not set'}")
    print(f"🔍 RAG: {'✅ Enabled' if rag else '❌ Disabled'}")
    print(f"💬 Conversation Tracking: ✅ Enabled")
    
    if not GROQ_API_KEY:
        print("\n⚠️  WARNING: GROQ_API_KEY not set!")
        print("Get your FREE key at: https://console.groq.com")
        print("Then set it: export GROQ_API_KEY='your-key-here'")
    
    print("\n📍 Main Endpoints:")
    print("  • POST /analyze                - Basic analysis")
    print("  • POST /analyze/stream         - Streaming")
    print("  • POST /analyze/agentic-stream - Conversation-aware")
    print("  • POST /incidents/seed         - Add examples")
    print("  • GET  /api/analytics/overview - Analytics")
    
    print("\n" + "="*70 + "\n")
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")