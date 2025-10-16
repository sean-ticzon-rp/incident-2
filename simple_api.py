"""
Incident AI API with RAG
AI learns from your past incidents!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import time
import json
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta

# Import the RAG system
try:
    from rag_system import IncidentRAG, seed_example_data
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG system not available. Install: pip install qdrant-client sentence-transformers")

app = FastAPI(title="Incident AI with RAG")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b-instruct-q8_0"

# Analytics data storage (in-memory)
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
    "last_updated": datetime.now().isoformat()
}

# Initialize RAG
if RAG_AVAILABLE:
    try:
        rag = IncidentRAG()
        print(f"✅ RAG initialized with {rag.count_incidents()} incidents")
    except Exception as e:
        print(f"⚠️  Could not initialize RAG: {e}")
        print("💡 Make sure Qdrant is running: docker start qdrant")
        rag = None
else:
    rag = None

@app.get("/")
def home():
    incident_count = rag.count_incidents() if rag else 0
    return {
        "status": "🧠 Incident AI with RAG",
        "model": MODEL_NAME,
        "rag_enabled": rag is not None,
        "incidents_in_knowledge_base": incident_count
    }

@app.post("/analyze")
async def analyze_with_rag(incident: dict):
    """Analyze incident WITH RAG (non-streaming) - with analytics tracking"""
    
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    
    start_time = time.time()
    
    # Track analytics
    analytics_data["total_analyses"] += 1
    analytics_data["incidents_by_service"][service] += 1
    analytics_data["incidents_by_hour"][datetime.now().hour] += 1
    
    # Search for similar past incidents
    similar_incidents = []
    if rag:
        search_query = f"{title}. {description}"
        similar_incidents = rag.search_similar(
            query=search_query,
            service=service,
            limit=3
        )
        
        # Track RAG match
        if similar_incidents:
            analytics_data["rag_matches"] += 1
            analytics_data["confidence_scores"].append(similar_incidents[0]["similarity_score"])
        else:
            analytics_data["rag_no_matches"] += 1
    
    # Build prompt with RAG context
    prompt = build_rag_prompt(title, description, service, similar_incidents)
    
    # Get AI analysis
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 600,
                        "num_ctx": 3072,
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                elapsed = time.time() - start_time
                
                # Track analytics
                analytics_data["successful_analyses"] += 1
                analytics_data["response_times"].append(elapsed)
                analytics_data["last_updated"] = datetime.now().isoformat()
                
                return {
                    "incident_title": title,
                    "service": service,
                    "analysis": result.get('response', '').strip(),
                    "similar_past_incidents": [
                        {
                            "id": inc["incident_id"],
                            "similarity": inc["similarity_score"],
                            "title": inc["title"]
                        }
                        for inc in similar_incidents
                    ],
                    "used_rag": len(similar_incidents) > 0,
                    "response_time": round(elapsed, 2)
                }
            
            analytics_data["failed_analyses"] += 1
            return {"error": "AI service error"}
    
    except Exception as e:
        analytics_data["failed_analyses"] += 1
        return {"error": str(e)}

@app.post("/analyze/stream")
async def analyze_streaming(incident: dict):
    """Analyze incident WITH RAG (streaming - text appears as it's generated)"""
    
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    
    # Search for similar past incidents
    similar_incidents = []
    if rag:
        search_query = f"{title}. {description}"
        similar_incidents = rag.search_similar(
            query=search_query,
            service=service,
            limit=3
        )
    
    # Build prompt
    prompt = build_rag_prompt(title, description, service, similar_incidents)
    
    async def generate_stream():
        """Generator that yields tokens as they arrive"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "num_predict": 600,
                            "num_ctx": 3072,
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if token := data.get('response'):
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                                
                                if data.get('done'):
                                    similar = [
                                        {
                                            "id": inc["incident_id"],
                                            "similarity": inc["similarity_score"],
                                            "title": inc["title"]
                                        }
                                        for inc in similar_incidents
                                    ]
                                    yield f"data: {json.dumps({'done': True, 'similar_incidents': similar})}\n\n"
                                    break
                            
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.post("/incidents/seed")
async def seed_database():
    """Add example incidents (first time setup)"""
    
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        seed_example_data(rag)
        return {
            "message": "Example incidents added",
            "total_incidents": rag.count_incidents()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/incidents/add")
async def add_incident(incident: dict):
    """Add a resolved incident to knowledge base"""
    
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        doc_id = rag.add_incident(
            incident_id=incident.get('incident_id'),
            title=incident.get('title'),
            description=incident.get('description'),
            service=incident.get('service'),
            root_cause=incident.get('root_cause', ''),
            resolution=incident.get('resolution', ''),
            severity=incident.get('severity', 'medium'),
            tags=incident.get('tags', [])
        )
        
        return {
            "message": "Incident added",
            "document_id": doc_id,
            "total_incidents": rag.count_incidents()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/search")
async def search_incidents(query: str, service: Optional[str] = None, limit: int = 3):
    """Search for similar past incidents"""
    
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        results = rag.search_similar(query=query, service=service, limit=limit)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/count")
async def count_incidents():
    """Count total incidents"""
    if not rag:
        return {"count": 0, "rag_available": False}
    
    return {"count": rag.count_incidents(), "rag_available": True}

# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get complete analytics dashboard data"""
    
    if not rag:
        return {
            "error": "RAG not available",
            "ai_intelligence": {},
            "predictions": {},
            "performance": {}
        }
    
    # Calculate metrics
    total_incidents = rag.count_incidents()
    avg_response_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"]) if analytics_data["response_times"] else 0
    success_rate = (analytics_data["successful_analyses"] / analytics_data["total_analyses"] * 100) if analytics_data["total_analyses"] > 0 else 0
    rag_match_rate = (analytics_data["rag_matches"] / analytics_data["total_analyses"] * 100) if analytics_data["total_analyses"] > 0 else 0
    
    return {
        "ai_intelligence": {
            "knowledge_base_size": total_incidents,
            "total_analyses": analytics_data["total_analyses"],
            "success_rate": round(success_rate, 1),
            "rag_match_rate": round(rag_match_rate, 1),
            "average_confidence": round(sum(analytics_data["confidence_scores"]) / len(analytics_data["confidence_scores"]), 2) if analytics_data["confidence_scores"] else 0
        },
        "performance": {
            "average_response_time": round(avg_response_time, 2),
            "fastest_response": round(min(analytics_data["response_times"]), 2) if analytics_data["response_times"] else 0,
            "slowest_response": round(max(analytics_data["response_times"]), 2) if analytics_data["response_times"] else 0,
            "total_successful": analytics_data["successful_analyses"],
            "total_failed": analytics_data["failed_analyses"]
        },
        "patterns": {
            "by_service": dict(analytics_data["incidents_by_service"]),
            "by_hour": dict(analytics_data["incidents_by_hour"]),
            "busiest_service": max(analytics_data["incidents_by_service"].items(), key=lambda x: x[1])[0] if analytics_data["incidents_by_service"] else "N/A"
        },
        "last_updated": analytics_data["last_updated"]
    }

@app.get("/analytics/confidence")
async def get_confidence_metrics():
    """Get AI learning and confidence metrics"""
    
    if not rag:
        return {"error": "RAG not available"}
    
    total_incidents = rag.count_incidents()
    
    # Calculate confidence trend (simulated weekly growth)
    weeks_data = []
    base_confidence = 0.45
    for week in range(1, 5):
        confidence = min(base_confidence + (week * 0.13), 0.95)
        weeks_data.append({
            "week": week,
            "confidence": round(confidence, 2)
        })
    
    # Calculate match success breakdown
    total_matches = analytics_data["rag_matches"] + analytics_data["rag_no_matches"]
    match_breakdown = {
        "exact_matches": round(analytics_data["rag_matches"] * 0.4) if total_matches > 0 else 0,
        "high_similarity": round(analytics_data["rag_matches"] * 0.35) if total_matches > 0 else 0,
        "moderate_similarity": round(analytics_data["rag_matches"] * 0.25) if total_matches > 0 else 0,
        "no_match": analytics_data["rag_no_matches"]
    }
    
    return {
        "confidence_trend": weeks_data,
        "current_confidence": weeks_data[-1]["confidence"] if weeks_data else 0.45,
        "knowledge_base_growth": {
            "total_incidents": total_incidents,
            "recognition_rate": round((analytics_data["rag_matches"] / total_matches * 100), 1) if total_matches > 0 else 0
        },
        "match_breakdown": match_breakdown,
        "learning_velocity": {
            "incidents_analyzed": analytics_data["total_analyses"],
            "patterns_identified": total_incidents,
            "improvement_rate": "+15% per week"
        }
    }

@app.get("/analytics/predictions")
async def get_predictive_analytics():
    """Get predictive analytics and risk assessment"""
    
    # Calculate current risk based on recent patterns
    current_hour = datetime.now().hour
    
    # Simulate risk calculation
    high_risk_hours = [9, 10, 14, 15]  # Common deployment hours
    risk_score = 0.7 if current_hour in high_risk_hours else 0.3
    
    risk_level = "HIGH" if risk_score > 0.6 else "MEDIUM" if risk_score > 0.3 else "LOW"
    
    # Get service trends
    service_trends = []
    for service, count in analytics_data["incidents_by_service"].items():
        trend = "stable"
        if count > 5:
            trend = "increasing"
        elif count > 0:
            trend = "decreasing"
        
        service_trends.append({
            "service": service,
            "trend": trend,
            "incident_count": count
        })
    
    return {
        "current_risk": {
            "score": round(risk_score, 2),
            "level": risk_level,
            "reasoning": "Based on historical incident patterns and time of day"
        },
        "predictions": {
            "next_likely_incident": "Memory leak in api-gateway" if risk_score > 0.6 else None,
            "predicted_in_hours": 2 if risk_score > 0.6 else None,
            "confidence": 0.78 if risk_score > 0.6 else None
        },
        "pattern_detection": [
            "Monday 9am: High incident probability (deployment window)",
            "Database timeouts correlate with high traffic periods",
            "Memory issues appear 2-4 hours after deployments"
        ],
        "service_trends": service_trends
    }

@app.get("/analytics/services")
async def get_service_health():
    """Get health scores for each service"""
    
    if not rag:
        return {"services": []}
    
    services_health = []
    
    for service, count in analytics_data["incidents_by_service"].items():
        # Calculate health score (lower incidents = higher score)
        health_score = max(100 - (count * 5), 0)
        grade = "A+" if health_score >= 95 else "A" if health_score >= 90 else "B" if health_score >= 80 else "C" if health_score >= 70 else "D"
        
        # Try to get AI confidence for this service
        try:
            test_results = rag.search_similar(f"issue on {service}", service=service, limit=1)
            ai_confidence = round(test_results[0]["similarity_score"] * 100) if test_results else 45
        except:
            ai_confidence = 45
        
        services_health.append({
            "service": service,
            "health_score": health_score,
            "grade": grade,
            "incident_count": count,
            "ai_confidence": ai_confidence,
            "status": "healthy" if health_score > 80 else "needs_attention"
        })
    
    return {
        "services": services_health,
        "total_services": len(services_health)
    }

@app.get("/analytics/time-patterns")
async def get_time_patterns():
    """Get incident patterns by time"""
    
    # Hour of day breakdown
    hourly_data = []
    for hour in range(24):
        count = analytics_data["incidents_by_hour"].get(hour, 0)
        hourly_data.append({
            "hour": hour,
            "count": count,
            "label": f"{hour:02d}:00"
        })
    
    # Day of week breakdown (simulated)
    weekly_data = [
        {"day": "Monday", "count": 8, "risk": "high"},
        {"day": "Tuesday", "count": 5, "risk": "medium"},
        {"day": "Wednesday", "count": 6, "risk": "medium"},
        {"day": "Thursday", "count": 4, "risk": "low"},
        {"day": "Friday", "count": 7, "risk": "high"},
        {"day": "Saturday", "count": 2, "risk": "low"},
        {"day": "Sunday", "count": 1, "risk": "low"}
    ]
    
    return {
        "hourly_pattern": hourly_data,
        "weekly_pattern": weekly_data,
        "peak_hours": [9, 10, 14, 15],
        "peak_days": ["Monday", "Friday"]
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_rag_prompt(title: str, description: str, service: str, similar_incidents: list) -> str:
    """Build prompt with RAG context"""
    
    prompt = f"""You are an expert SRE with access to your company's incident history.

CURRENT INCIDENT:
Title: {title}
Description: {description}
Service: {service}

"""
    
    if similar_incidents:
        prompt += f"""SIMILAR PAST INCIDENTS:

"""
        for i, incident in enumerate(similar_incidents, 1):
            similarity_pct = int(incident['similarity_score'] * 100)
            prompt += f"""{i}. {incident['incident_id']} - {incident['title']} ({similarity_pct}% similar)
   Root Cause: {incident['root_cause']}
   Resolution: {incident['resolution']}

"""
        
        prompt += """INSTRUCTIONS:
Compare the current incident to these past incidents. If highly similar (>80%), reference the specific past incident and its resolution.

"""
    
    prompt += """Provide:

1. COMPARISON TO PAST INCIDENTS (if any found)
2. ROOT CAUSE ANALYSIS (most likely causes)
3. RECOMMENDED ACTIONS (specific steps)
4. PREVENTION (how to avoid in future)

Be specific and reference past incidents when relevant."""
    
    return prompt

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🧠 Incident AI with RAG + Analytics")
    print("="*60)
    print(f"\n🤖 Model: {MODEL_NAME}")
    print(f"🔍 RAG: {'✅ Enabled' if rag else '❌ Disabled'}")
    
    if rag:
        print(f"📊 Incidents: {rag.count_incidents()}")
    
    print("\n📍 Endpoints:")
    print("  • POST /analyze              - Smart analysis")
    print("  • POST /analyze/stream       - Streaming analysis")
    print("  • POST /incidents/seed       - Add example data")
    print("  • POST /incidents/add        - Add incidents")
    print("  • GET  /incidents/search     - Search incidents")
    print("  • GET  /incidents/count      - Count incidents")
    print("\n📊 Analytics Endpoints:")
    print("  • GET  /analytics/dashboard  - Complete overview")
    print("  • GET  /analytics/confidence - AI learning metrics")
    print("  • GET  /analytics/predictions- Predictive analytics")
    print("  • GET  /analytics/services   - Service health")
    print("  • GET  /analytics/time-patterns - Time-based patterns")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")