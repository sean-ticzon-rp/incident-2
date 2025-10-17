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
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import asyncio

from dotenv import load_dotenv

load_dotenv()  # Load .env variables

# ============================================================================
# PYDANTIC MODELS FOR API DOCUMENTATION
# ============================================================================

class IncidentAdd(BaseModel):
    incident_id: str
    title: str
    description: str
    service: str
    root_cause: Optional[str] = ""
    resolution: Optional[str] = ""
    severity: Optional[str] = "medium"
    tags: Optional[List[str]] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_id": "INC-2025-001",
                "title": "High CPU usage on api-gateway",
                "description": "CPU spiked to 95% at 2:15pm, causing slow response times",
                "service": "api-gateway",
                "root_cause": "Memory leak after v2.3.4 deployment",
                "resolution": "Rolled back to v2.3.3 and deployed v2.3.5 with proper connection pooling",
                "severity": "high",
                "tags": ["cpu", "memory-leak", "performance"]
            }
        }

# ============================================================================
# APP SETUP
# ============================================================================

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
        
        # Debug: Log available methods
        rag_methods = [method for method in dir(rag) if not method.startswith('_')]
        logger.info(f"✅ RAG initialized with {rag.count_incidents()} incidents")
        logger.info(f"🔍 Available RAG methods: {rag_methods}")
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
MODEL_NAME = "llama-3.3-70b-versatile"  # ⭐⭐⭐⭐⭐ BEST - Smartest & most capable
# MODEL_NAME = "llama-3.1-70b-versatile"  # ⭐⭐⭐⭐⭐ Very smart alternative
# MODEL_NAME = "mixtral-8x7b-32768"  # ⭐⭐⭐⭐ Good balance, longer context
# MODEL_NAME = "llama-3.1-8b-instant"  # ⭐⭐⭐ Fast but less intelligent

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
            "temperature": 0.5,  # Lower for more focused, less random responses
            "top_p": 0.85,  # More deterministic
            "max_tokens": 800,  # More room for detailed answers
            "frequency_penalty": 0.4,  # Stronger penalty for repetition
            "presence_penalty": 0.3  # Encourage diverse topics
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

# ============================================================================
# INTELLIGENT MESSAGE CLASSIFICATION
# ============================================================================

async def classify_message_intent(text: str) -> Dict[str, Any]:
    """
    Use the LLM itself to classify the user's intent dynamically.
    No hardcoded lists - fully intelligent classification!
    """
    classification_prompt = f"""Analyze this user message and classify its intent.

User message: "{text}"

Classify into ONE of these categories:
1. GREETING - Casual greetings, small talk, asking how you are
2. CAPABILITY_INQUIRY - Asking what you can do, your features, capabilities
3. INCIDENT_REPORT - Describing a technical problem, error, or incident
4. FOLLOW_UP - Follow-up question on an ongoing conversation
5. GENERAL_QUESTION - General technical question not about a specific incident

Respond ONLY with this JSON format:
{{
  "category": "GREETING|CAPABILITY_INQUIRY|INCIDENT_REPORT|FOLLOW_UP|GENERAL_QUESTION",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  # Fast model for classification
            "messages": [{"role": "user", "content": classification_prompt}],
            "temperature": 0.1,  # Very deterministic
            "max_tokens": 150
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(GROQ_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                    logger.info(f"🎯 Message classified as: {classification['category']} (confidence: {classification['confidence']})")
                    return classification
    except Exception as e:
        logger.warning(f"Classification failed, using fallback: {e}")
    
    # Fallback: Simple keyword detection
    text_lower = text.lower()
    if len(text) < 30 and any(word in text_lower for word in ["hi", "hello", "hey"]):
        return {"category": "GREETING", "confidence": 0.7, "reasoning": "fallback detection"}
    elif any(word in text_lower for word in ["error", "issue", "problem", "incident", "down", "slow"]):
        return {"category": "INCIDENT_REPORT", "confidence": 0.8, "reasoning": "fallback detection"}
    else:
        return {"category": "GENERAL_QUESTION", "confidence": 0.5, "reasoning": "fallback detection"}

# ============================================================================
# IMPROVED PROMPT BUILDERS
# ============================================================================

def build_messages(prompt: str, conversation_history: List[Dict] = None) -> List[Dict]:
    """Build messages for Groq API with comprehensive system understanding"""
    messages = [
        {
            "role": "system", 
            "content": """You are an expert Site Reliability Engineer (SRE) assistant with deep experience in incident response, system architecture, and troubleshooting.

**YOUR CORE ABILITIES:**
- Analyze incidents with precision and provide actionable solutions
- Search through past incident knowledge base to find similar cases
- Provide step-by-step diagnostic commands with expected outputs
- Explain complex technical concepts clearly
- Maintain context across conversations

**YOUR PERSONALITY:**
- Professional yet approachable - like a senior colleague
- Direct and practical - no fluff or corporate speak
- Patient with beginners, challenging with experts
- Honest about limitations - you don't make up information

**RESPONSE GUIDELINES:**

When user is greeting or chatting casually:
- Be warm and friendly (1-2 sentences)
- Briefly mention you help with incidents
- Ask what they need help with

When user asks about your capabilities:
- Explain you analyze incidents and find solutions
- Mention you can search past incidents for patterns
- Keep it under 100 words

When user reports an incident:
- Immediately shift to expert mode
- Provide structured analysis: Assessment → Diagnostics → Actions → Next Steps
- Use specific commands, thresholds, and values
- Reference similar past incidents ONLY if they exist in the provided data
- Be specific about confidence levels (e.g., "80% confident this is...")

When continuing a conversation:
- Build on what was already discussed
- Don't repeat previous suggestions
- Answer the specific question asked
- Maintain conversational flow

**CRITICAL RULES:**
- NEVER invent past incidents - only reference ones explicitly provided
- If no similar incidents exist, say so clearly
- Be specific with numbers, commands, and thresholds
- Adjust response length to match the situation (short for greetings, detailed for incidents)

You're here to help, whether that's a quick hello or saving production at 3am."""
        }
    ]
    
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    messages.append({"role": "user", "content": prompt})
    
    return messages

def build_enhanced_prompt(
    title: str, 
    description: str, 
    service: str, 
    context: Dict[str, Any],
    user_question: Optional[str] = None,
    message_classification: Optional[Dict] = None
) -> str:
    """Enhanced prompt that adapts to conversation type"""
    
    full_text = f"{title} {description}".strip()
    similar = context["rag_context"]
    
    # Use intelligent classification
    if message_classification and message_classification["category"] in ["GREETING", "CAPABILITY_INQUIRY"]:
        return f"""The user said: "{full_text}"

This is a casual conversation, not an incident report.

**Classification:** {message_classification['category']}
**Context:** {message_classification.get('reasoning', 'casual chat')}

Respond naturally and conversationally:
- If greeting: Greet warmly, briefly mention you help with incidents, ask what they need
- If asking capabilities: Explain you analyze incidents, search past cases, provide solutions
- Keep it friendly and brief (2-4 sentences)
- DON'T launch into incident analysis

Example tone: "Hi! I'm your SRE assistant. I help analyze incidents, find root causes, and suggest fixes based on past experience. What can I help you troubleshoot today?" """
    
    # INCIDENT ANALYSIS MODE - Show them what we found!
    prompt = f"""**INCIDENT ANALYSIS REQUEST**

Service: {service}
Issue: {title}
Description: {description}
"""
    
    # IMPORTANT: Show incident matches clearly
    if similar and len(similar) > 0:
        prompt += f"\n**🔍 KNOWLEDGE BASE SEARCH RESULTS:**\n"
        prompt += f"Found {len(similar)} potentially related incident(s) in our database:\n\n"
        
        for idx, inc in enumerate(similar[:3], 1):
            sim_pct = int(inc['similarity_score'] * 100)
            
            # Color code by similarity
            if sim_pct >= 70:
                confidence = "🟢 HIGH MATCH"
            elif sim_pct >= 50:
                confidence = "🟡 MODERATE MATCH"
            else:
                confidence = "🔴 LOW MATCH"
            
            prompt += f"""{idx}. **Incident {inc['incident_id']}** - {confidence} ({sim_pct}% similar)
   Title: {inc['title']}
   Service: {inc['service']}
   Root Cause: {inc['root_cause']}
   Resolution: {inc['resolution']}
   Severity: {inc.get('severity', 'unknown')}

"""
        
        best_match = similar[0]
        best_sim = int(best_match['similarity_score'] * 100)
        
        if best_sim >= 70:
            prompt += f"""**✅ STRONG PATTERN MATCH**
The best match (Incident {best_match['incident_id']}) has {best_sim}% similarity.
Start your response by acknowledging: "I found a very similar incident ({best_match['incident_id']}) where..."
Then explain how the previous solution applies to this case.
"""
        elif best_sim >= 50:
            prompt += f"""**⚠️ POSSIBLE PATTERN**
The best match has {best_sim}% similarity - not conclusive but worth noting.
Mention it as: "There's a possibly related incident ({best_match['incident_id']}), though not identical..."
Then provide your own fresh analysis.
"""
        else:
            prompt += f"""**ℹ️ LIMITED SIMILARITY**
Found {len(similar)} past incidents, but highest similarity is only {best_sim}%.
State clearly: "I found some past incidents but none are very similar to this. Let me analyze this as a new pattern."
"""
    else:
        prompt += """
**🆕 NEW INCIDENT PATTERN**
No similar incidents found in the knowledge base.

**IMPORTANT:** Start by clearly stating: "This appears to be a new incident pattern - I don't have similar cases to reference. Let me analyze this fresh..."
"""
    
    # Add context
    keywords = context.get("keywords", [])
    if keywords:
        prompt += f"\n**Detected Keywords:** {', '.join(keywords[:5])}"
    
    service_info = context.get("service_info", {})
    if service_info:
        prompt += f"\n**Service Context:** {service_info.get('common_issues', [])}"
    
    prompt += """

**YOUR RESPONSE STRUCTURE:**

**1. Opening (1-2 sentences):**
- Acknowledge any similar incidents found (or explicitly state none exist)
- Quick assessment of the situation

**2. Root Cause Analysis (3-4 sentences):**
- What's likely happening based on symptoms
- Confidence level (e.g., "75% confident this is...")
- Why you think this (evidence from description or past incidents)

**3. Diagnostic Commands (3-5 commands):**
Present as a numbered list with specific guidance:
```
1. `command here` - What to check, expected normal values
2. `another command` - What indicates a problem
```

**4. Immediate Actions (if critical):**
One-line fix to stabilize right now (only if urgent)

**5. Next Steps (2-3 sentences):**
What to investigate after diagnostics

**RESPONSE REQUIREMENTS:**
- Be conversational but precise
- Use actual command examples, not placeholders
- Give specific thresholds and values
- Length: 200-400 words (adjust for streaming readability)
- Never fabricate past incidents
- If uncertain, say so

Remember: The user is watching this stream live during an outage. Be helpful, specific, and quick."""
    
    return prompt

def build_conversation_aware_prompt(
    title: str,
    description: str,
    service: str,
    context: Dict[str, Any],
    user_question: str,
    conversation_history: List[Dict]
) -> str:
    """Build prompts for follow-up questions in ongoing conversations"""
    
    similar = context["rag_context"]
    
    # Track what's been discussed
    already_mentioned = {
        "commands": set(),
        "topics": set(),
        "incident_ids": set()
    }
    
    for msg in conversation_history:
        if msg["role"] == "assistant":
            content_lower = msg["content"].lower()
            
            # Extract mentioned commands
            import re
            commands = re.findall(r'`([^`]+)`', msg["content"])
            already_mentioned["commands"].update(commands)
            
            # Track topics
            if "root cause" in content_lower:
                already_mentioned["topics"].add("root_cause")
            if "prevention" in content_lower or "prevent" in content_lower:
                already_mentioned["topics"].add("prevention")
            if "monitor" in content_lower:
                already_mentioned["topics"].add("monitoring")
            
            # Track incident IDs
            for inc in similar:
                if inc["incident_id"].lower() in content_lower:
                    already_mentioned["incident_ids"].add(inc["incident_id"])
    
    question_lower = user_question.lower()
    
    # Detect question type
    asking_about_similar = any(word in question_lower for word in 
                               ["similar", "past", "previous", "related", "other", "history", "before"])
    
    asking_for_alternatives = any(phrase in question_lower for phrase in 
                                 ["without", "instead", "other than", "different", "alternative", "else"])
    
    asking_for_prevention = any(word in question_lower for word in
                               ["prevent", "avoid", "stop", "future", "again"])
    
    # Build context-aware prompt
    if asking_about_similar:
        if similar and len(similar) > 0:
            prompt = f"""**FOLLOW-UP QUESTION ABOUT SIMILAR INCIDENTS**

User asked: "{user_question}"

**Available Past Incidents:**
"""
            for idx, inc in enumerate(similar[:3], 1):
                sim_pct = int(inc['similarity_score'] * 100)
                already_discussed = "✅ Already discussed" if inc['incident_id'] in already_mentioned['incident_ids'] else "🆕 Not yet mentioned"
                
                prompt += f"""
{idx}. **{inc['incident_id']}** ({sim_pct}% similar) - {already_discussed}
   Title: {inc['title']}
   Root Cause: {inc['root_cause']}
   Resolution: {inc['resolution']}
"""
            
            prompt += f"""

Answer their question about these incidents:
- Focus on incidents NOT yet discussed if possible
- Be honest about similarity percentages
- If asking "how many" or "what happened", give factual answers
- Keep response under 200 words"""
        else:
            prompt = f"""User asked: "{user_question}"

**IMPORTANT:** No similar incidents exist in the database.

Be honest: "I don't have any similar incidents in the knowledge base for this pattern yet."
Then offer to help with current incident analysis instead. (Under 100 words)"""
    
    elif asking_for_prevention:
        prompt = f"""**PREVENTION STRATEGIES REQUEST**

Current incident: {title}
User asked: "{user_question}"

**Already discussed:**
Topics covered: {', '.join(already_mentioned['topics']) if already_mentioned['topics'] else 'Initial analysis'}

Provide 3-5 specific prevention strategies:
1. Monitoring/alerting improvements (specific metrics and thresholds)
2. Configuration changes (actual settings)
3. Process improvements (specific practices)

Focus on actionable, implementable steps. Under 250 words."""
    
    elif asking_for_alternatives:
        prompt = f"""**ALTERNATIVE APPROACH REQUEST**

User wants different solutions than what was discussed.

**Already suggested:**
Commands: {', '.join(list(already_mentioned['commands'])[:5]) if already_mentioned['commands'] else 'none yet'}

Provide COMPLETELY DIFFERENT approaches:
- Different diagnostic tools/commands
- Alternative hypotheses for root cause
- Different resolution strategies

Be specific and practical. Under 200 words."""
    
    else:
        # General follow-up
        prompt = f"""**FOLLOW-UP QUESTION**

Ongoing incident: {title}

**Recent conversation context:**
"""
        for msg in conversation_history[-2:]:
            role = "User" if msg["role"] == "user" else "You"
            snippet = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            prompt += f"{role}: {snippet}\n\n"
        
        prompt += f"""
User's new question: "{user_question}"

**What we've covered:**
- Commands mentioned: {', '.join(list(already_mentioned['commands'])[:3])}
- Topics: {', '.join(already_mentioned['topics'])}

Answer their specific question:
- Build on previous discussion naturally
- Don't repeat what was already said
- Provide new information or clarification
- Be conversational and helpful

Keep under 200 words."""
    
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
            "version": "2.0 - Intelligent Conversation Mode",
            "model": MODEL_NAME,
            "api_provider": "Groq (FREE)",
            "services": {
                "groq_api": "✅ ready" if GROQ_API_KEY else "❌ API key not set",
                "rag": "✅ enabled" if rag else "❌ disabled",
                "intelligent_classification": "✅ enabled"
            },
            "knowledge_base": {
                "incidents": incident_count,
                "auto_learning": True
            },
            "features": [
                "🤖 Dynamic message classification (no hardcoded greetings)",
                "🔍 Smart incident matching with confidence scores",
                "💬 Context-aware conversations",
                "📊 Detailed similarity explanations"
            ],
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
    
    # Classify message intent
    full_text = f"{title} {description}"
    classification = await classify_message_intent(full_text)
    
    # Search similar incidents
    similar_incidents = []
    if rag:
        try:
            similar_incidents = await rag.search_similar_async(
                query=f"{title}. {description}",
                service=service,
                limit=3
            )
            if similar_incidents:
                analytics_data["rag_matches"] += 1
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
    
    context = await context_gatherer.gather(title, description, service, similar_incidents)
    prompt = build_enhanced_prompt(title, description, service, context, message_classification=classification)
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
                    "message_classification": classification,
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
    """Streaming analysis with intelligent classification"""
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
            
            # Classify message intent
            full_text = f"{title} {description}"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Classifying message intent...'})}\n\n"
            
            classification = await classify_message_intent(full_text)
            
            yield f"data: {json.dumps({'type': 'classification', 'category': classification['category'], 'confidence': classification['confidence']})}\n\n"
            await asyncio.sleep(0.1)
            
            # Search similar
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
            
            similar_incidents = []
            if rag and classification['category'] in ['INCIDENT_REPORT', 'FOLLOW_UP']:
                try:
                    similar_incidents = await rag.search_similar_async(
                        query=full_text,
                        service=service,
                        limit=3
                    )
                    if similar_incidents:
                        analytics_data["rag_matches"] += 1
                except Exception as e:
                    logger.error(f"RAG search: {e}")
            
            context = await context_gatherer.gather(title, description, service, similar_incidents)
            prompt = build_enhanced_prompt(title, description, service, context, message_classification=classification)
            messages = build_messages(prompt)
            
            # Send metadata with match details
            metadata = {
                "type": "metadata",
                "incident": title,
                "service": service,
                "classification": classification['category'],
                "similar_incidents_found": len(similar_incidents),
                "matches": [
                    {
                        "id": inc["incident_id"],
                        "similarity_pct": round(inc["similarity_score"] * 100),
                        "title": inc["title"][:50] + "..." if len(inc["title"]) > 50 else inc["title"],
                        "confidence": "high" if inc["similarity_score"] >= 0.7 else "medium" if inc["similarity_score"] >= 0.5 else "low"
                    }
                    for inc in similar_incidents[:3]
                ]
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating analysis...'})}\n\n"
            
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
                "quality": quality_report["quality"],
                "word_count": len(full_response.split())
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
    """Conversation-aware streaming - Intelligent and context-aware"""
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
            
            # Classify intent
            full_text = f"{title} {description} {user_question}"
            yield f"data: {json.dumps({'type': 'status', 'message': '🤖 Understanding your message...'})}\n\n"
            
            classification = await classify_message_intent(full_text)
            
            yield f"data: {json.dumps({'type': 'classification', 'category': classification['category']})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'type': 'status', 'message': '🔍 Searching knowledge base...'})}\n\n"
            
            # Search similar
            similar_incidents = []
            if rag and classification['category'] in ['INCIDENT_REPORT', 'FOLLOW_UP']:
                try:
                    similar_incidents = await rag.search_similar_async(
                        query=full_text,
                        service=service,
                        limit=3
                    )
                    if similar_incidents:
                        analytics_data["rag_matches"] += 1
                except Exception as e:
                    logger.error(f"RAG error: {e}")
            
            context = await context_gatherer.gather(title, description, service, similar_incidents)
            
            # Build appropriate prompt
            if user_question and conversation_history:
                prompt = build_conversation_aware_prompt(
                    title, description, service, context,
                    user_question, conversation_history
                )
            else:
                prompt = build_enhanced_prompt(title, description, service, context, user_question, classification)
            
            messages = build_messages(prompt, conversation_history)
            
            # Send rich metadata
            metadata = {
                "type": "metadata",
                "incident_title": title,
                "service": service,
                "conversation_turn": len(conversation_history) + 1,
                "intent": classification['category'],
                "knowledge_base": {
                    "total_incidents": rag.count_incidents() if rag else 0,
                    "matches_found": len(similar_incidents),
                    "best_match": {
                        "id": similar_incidents[0]["incident_id"],
                        "similarity": round(similar_incidents[0]["similarity_score"] * 100),
                        "confidence": "🟢 HIGH" if similar_incidents[0]["similarity_score"] >= 0.7 else "🟡 MODERATE" if similar_incidents[0]["similarity_score"] >= 0.5 else "🔴 LOW"
                    } if similar_incidents else None
                }
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': '💭 Analyzing and generating response...'})}\n\n"
            
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
                "conversation_messages": len(conversation_history) + 2,
                "word_count": len(full_response.split())
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
# INCIDENT MANAGEMENT
# ============================================================================

@app.post("/incidents/seed")
async def seed_incidents():
    """Seed example incidents into the knowledge base"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        await seed_example_data_async(rag)
        return {
            "message": "✅ Incidents seeded successfully",
            "total": rag.count_incidents()
        }
    except Exception as e:
        logger.error(f"Error seeding incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/incidents/add")
async def add_incident(incident: IncidentAdd):
    """Add a resolved incident to the AI knowledge base"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        incident_dict = incident.model_dump()
        
        doc_id = await rag.add_incident_async(
            incident_id=incident_dict['incident_id'],
            title=incident_dict['title'],
            description=incident_dict['description'],
            service=incident_dict['service'],
            root_cause=incident_dict.get('root_cause', ''),
            resolution=incident_dict.get('resolution', ''),
            severity=incident_dict.get('severity', 'medium'),
            tags=incident_dict.get('tags', [])
        )
        return {
            "message": "✅ Incident added to knowledge base",
            "document_id": doc_id,
            "total_incidents": rag.count_incidents()
        }
    except Exception as e:
        logger.error(f"Error adding incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/search")
async def search_incidents(query: str, service: Optional[str] = None, limit: int = 5):
    """Search for similar past incidents"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        results = await rag.search_similar_async(query=query, service=service, limit=limit)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/count")
async def count_incidents():
    """Get total count of incidents in knowledge base"""
    if not rag:
        return {"count": 0, "rag_available": False}
    return {"count": rag.count_incidents(), "rag_available": True}

# ============================================================================
# ANALYTICS
# ============================================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview"""
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
    """Export all analytics data"""
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
            "features": ["RAG", "Streaming", "Adaptive Config", "Anti-Repetition", "Intelligent Classification"]
        }
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("="*70)
    logger.info("🚀 INCIDENT AI - GROQ API EDITION STARTING")
    logger.info("="*70)
    logger.info(f"🤖 Model: {MODEL_NAME}")
    logger.info(f"🔑 Groq API Key: {'✅ Set' if GROQ_API_KEY else '❌ Not set'}")
    logger.info(f"🔍 RAG: {'✅ Enabled' if rag else '❌ Disabled'}")
    logger.info(f"📊 Incident Count: {rag.count_incidents() if rag else 0}")
    logger.info(f"🧠 Intelligent Classification: ✅ Enabled")
    logger.info(f"🌐 Port: {os.getenv('PORT', '8000')}")
    logger.info("="*70)

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
    print(f"🧠 Intelligent Classification: ✅ No hardcoded greetings!")
    
    if not GROQ_API_KEY:
        print("\n⚠️  WARNING: GROQ_API_KEY not set!")
        print("Get your FREE key at: https://console.groq.com")
        print("Then set it: export GROQ_API_KEY='your-key-here'")
    
    print("\n📍 Main Endpoints:")
    print("  • POST /analyze                - Basic analysis")
    print("  • POST /analyze/stream         - Streaming with intent detection")
    print("  • POST /analyze/agentic-stream - Conversation-aware with context")
    print("  • POST /incidents/seed         - Add examples")
    print("  • GET  /api/analytics/overview - Analytics")
    
    print("\n✨ New Features:")
    print("  • 🎯 Dynamic message classification (no hardcoded lists!)")
    print("  • 🔍 Detailed incident match feedback with confidence scores")
    print("  • 📊 Similarity percentages and match quality indicators")
    print("  • 💬 Context-aware follow-up handling")
    
    print("\n" + "="*70 + "\n")
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")