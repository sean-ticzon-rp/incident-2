"""
RAG System - Using Hugging Face Inference API for embeddings
NO PyTorch needed - keeps Docker image small!
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import httpx
import uuid
import logging
import os
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face API configuration
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

class IncidentRAG:
    """
    RAG system for incident management
    Uses Hugging Face API for embeddings (no local model!)
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "incidents"
    ):
        """Initialize RAG system"""
        try:
            # Increase timeout for Railway's slower network
            self.client = QdrantClient(url=qdrant_url, timeout=60.0)
            self.collection_name = collection_name
            self.embedding_dim = 384
            
            if not HF_API_KEY:
                logger.warning("⚠️  HUGGINGFACE_API_KEY not set - embeddings will fail!")
            
            # Create collection if it doesn't exist
            self._init_collection()
            logger.info("✅ RAG system initialized (Hugging Face API)")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise
    
    def _init_collection(self):
        """Create vector collection for incidents"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection_name}'")
    
    async def _get_embedding_async(self, text: str) -> list:
        """Get embedding from Hugging Face API (async)"""
        if not HF_API_KEY:
            raise Exception("HUGGINGFACE_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text,
            "options": {
                "wait_for_model": True
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # HF API returns nested array, flatten it
                if isinstance(result, list) and len(result) > 0:
                    return result[0] if isinstance(result[0], list) else result
                return result
            elif response.status_code == 503:
                # Model is loading, wait and retry
                logger.info("Model loading, retrying...")
                await asyncio.sleep(2)
                return await self._get_embedding_async(text)
            else:
                raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
    
    def _get_embedding_sync(self, text: str) -> list:
        """Synchronous wrapper for getting embeddings"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._get_embedding_async(text))
    
    def add_incident(
        self,
        incident_id: str,
        title: str,
        description: str,
        service: str,
        root_cause: str = "",
        resolution: str = "",
        severity: str = "medium",
        tags: list = None
    ) -> str:
        """
        Add a past incident to the knowledge base
        """
        
        # Create searchable text
        text = f"""
        Incident: {title}
        Description: {description}
        Service: {service}
        Root Cause: {root_cause}
        Resolution: {resolution}
        """.strip()
        
        # Get embedding from Hugging Face API
        try:
            embedding = self._get_embedding_sync(text)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
        
        # Create point
        doc_id = str(uuid.uuid4())
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={
                "incident_id": incident_id,
                "title": title,
                "description": description,
                "service": service,
                "root_cause": root_cause,
                "resolution": resolution,
                "severity": severity,
                "tags": tags or [],
                "text": text
            }
        )
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        logger.info(f"Added incident: {incident_id}")
        return doc_id
    
    def search_similar(
        self,
        query: str,
        service: str = None,
        limit: int = 3
    ) -> list:
        """
        Search for similar past incidents
        """
        
        # Get embedding from Hugging Face API
        try:
            query_vector = self._get_embedding_sync(query)
        except Exception as e:
            logger.error(f"Failed to get embedding for search: {e}")
            return []
        
        # Build search parameters
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        
        # Add service filter if specified
        if service:
            search_params["query_filter"] = Filter(
                must=[
                    FieldCondition(
                        key="service",
                        match=MatchValue(value=service)
                    )
                ]
            )
        
        # Search
        results = self.client.search(**search_params)
        
        # Format results
        similar_incidents = []
        for hit in results:
            similar_incidents.append({
                "incident_id": hit.payload.get("incident_id"),
                "similarity_score": round(hit.score, 3),
                "title": hit.payload.get("title"),
                "description": hit.payload.get("description"),
                "service": hit.payload.get("service"),
                "root_cause": hit.payload.get("root_cause"),
                "resolution": hit.payload.get("resolution"),
                "severity": hit.payload.get("severity"),
                "tags": hit.payload.get("tags", [])
            })
        
        logger.info(f"Found {len(similar_incidents)} similar incidents")
        return similar_incidents
    
    def count_incidents(self) -> int:
        """Count total incidents in knowledge base"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0

# Seed with example data
def seed_example_data(rag: IncidentRAG):
    """Add some example incidents for testing"""
    
    examples = [
        {
            "incident_id": "INC-00234",
            "title": "High CPU usage on api-gateway",
            "description": "CPU spiked to 95% at 2:15pm. Users reporting slow response times and timeouts.",
            "service": "api-gateway",
            "root_cause": "Memory leak in authentication middleware after v2.3.4 deployment",
            "resolution": "Rolled back to v2.3.3, deployed fixed version v2.3.5 with proper cleanup in auth middleware",
            "severity": "high",
            "tags": ["cpu", "memory-leak", "production"]
        },
        {
            "incident_id": "INC-00567",
            "title": "Database connection pool exhausted",
            "description": "All database connections in use, users getting 504 Gateway Timeout errors",
            "service": "user-service",
            "root_cause": "Missing index on users table causing slow queries, connections piling up",
            "resolution": "Added index on users.email column, connection time dropped from 30s to 0.1s",
            "severity": "critical",
            "tags": ["database", "timeout", "production"]
        },
        {
            "incident_id": "INC-00891",
            "title": "Memory usage gradually increasing",
            "description": "Memory went from 2GB to 8GB over 6 hours, eventually causing OOM kills",
            "service": "payment-service",
            "root_cause": "Cache without eviction policy, objects accumulating in memory",
            "resolution": "Implemented LRU cache with size limit, added monitoring for cache size",
            "severity": "high",
            "tags": ["memory", "cache", "production"]
        },
        {
            "incident_id": "INC-01123",
            "title": "API returning 500 errors",
            "description": "50% of requests failing with Internal Server Error, error logs show NullPointerException",
            "service": "order-api",
            "root_cause": "Config change removed required environment variable DATABASE_URL",
            "resolution": "Restored environment variable, added validation to prevent startup without required configs",
            "severity": "critical",
            "tags": ["api", "config", "production"]
        },
        {
            "incident_id": "INC-01445",
            "title": "Slow API response times",
            "description": "Average response time increased from 200ms to 3000ms, no errors in logs",
            "service": "api-gateway",
            "root_cause": "Third-party API (payment provider) experiencing issues, our timeout was 5 seconds",
            "resolution": "Reduced timeout to 1 second, added circuit breaker pattern, implemented fallback",
            "severity": "medium",
            "tags": ["performance", "latency", "production"]
        }
    ]
    
    print("\n📥 Seeding example incidents...")
    for example in examples:
        try:
            rag.add_incident(**example)
        except Exception as e:
            print(f"Failed to add {example['incident_id']}: {e}")
    
    print(f"✅ Added {len(examples)} example incidents")
    print(f"📊 Total incidents in database: {rag.count_incidents()}")