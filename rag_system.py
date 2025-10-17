"""
RAG System - QUICK FIX VERSION
Removes service filtering to make similarity search work immediately
"""

import os
import uuid
import logging
import asyncio
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cohere API configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
EMBEDDING_MODEL = "embed-english-light-v3.0"
COHERE_API_URL = "https://api.cohere.ai/v1/embed"

class IncidentRAG:
    """RAG system for incident management"""

    def __init__(self, collection_name: str = "incidents"):
        """Initialize RAG system"""
        try:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60.0)
            self.collection_name = collection_name
            self.embedding_dim = 384

            if not COHERE_API_KEY:
                logger.warning("⚠️ COHERE_API_KEY not set - embeddings will fail!")

            self._init_collection()
            logger.info("✅ RAG system initialized (Cohere API - NO SERVICE FILTER)")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise

    def _init_collection(self):
        """Create vector collection - SIMPLE VERSION"""
        try:
            # Try to get the collection
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' exists ({collection_info.points_count} incidents)")
            
        except Exception as get_error:
            # Collection doesn't exist, create it
            logger.info(f"📦 Creating new collection '{self.collection_name}'...")
            
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created collection '{self.collection_name}'")
                
            except UnexpectedResponse as create_error:
                if "already exists" in str(create_error).lower():
                    logger.info(f"✅ Collection already exists (race condition)")
                else:
                    logger.error(f"Failed to create collection: {create_error}")
                    raise

    async def get_embedding(self, text: str) -> list:
        """Get embedding from Cohere API (async)"""
        if not COHERE_API_KEY:
            raise Exception("COHERE_API_KEY not set. Get free key at https://dashboard.cohere.com/api-keys")

        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": EMBEDDING_MODEL,
            "texts": [text],
            "input_type": "search_document"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(COHERE_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result["embeddings"][0]
                return embedding
            else:
                error_text = response.text
                logger.error(f"❌ Cohere API error: {response.status_code} - {error_text}")
                raise Exception(f"Cohere API error: {response.status_code} - {error_text}")

    async def get_query_embedding(self, text: str) -> list:
        """Get embedding for search queries"""
        if not COHERE_API_KEY:
            raise Exception("COHERE_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": EMBEDDING_MODEL,
            "texts": [text],
            "input_type": "search_query"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(COHERE_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result["embeddings"][0]
            else:
                error_text = response.text
                logger.error(f"❌ Cohere API error: {response.status_code} - {error_text}")
                raise Exception(f"Cohere API error: {response.status_code} - {error_text}")

    async def add_incident_async(self, incident_id: str, title: str, description: str,
                                 service: str, root_cause: str = "", resolution: str = "",
                                 severity: str = "medium", tags: list = None) -> str:
        """Add a past incident to the knowledge base"""
        text = f"Incident: {title}\nDescription: {description}\nService: {service}\nRoot Cause: {root_cause}\nResolution: {resolution}".strip()
        
        try:
            embedding = await self.get_embedding(text)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

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

        self.client.upsert(collection_name=self.collection_name, points=[point])
        logger.info(f"✅ Added incident: {incident_id}")
        return doc_id

    async def search_similar_async(self, query: str, service: str = None, limit: int = 3) -> list:
        """
        Search for similar past incidents
        FIXED: NO SERVICE FILTERING - searches all incidents regardless of service
        """
        try:
            query_vector = await self.get_query_embedding(query)
        except Exception as e:
            logger.error(f"Failed to get embedding for search: {e}")
            return []

        # CRITICAL FIX: Remove service filter completely
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True
            # NO FILTER - search everything!
        }

        logger.info(f"🔍 Searching ALL incidents (no service filter)")

        try:
            results = self.client.search(**search_params)
            logger.info(f"✅ Found {len(results)} similar incidents")
            
            return [
                {
                    "incident_id": hit.payload.get("incident_id"),
                    "similarity_score": round(hit.score, 3),
                    "title": hit.payload.get("title"),
                    "description": hit.payload.get("description"),
                    "service": hit.payload.get("service"),
                    "root_cause": hit.payload.get("root_cause"),
                    "resolution": hit.payload.get("resolution"),
                    "severity": hit.payload.get("severity"),
                    "tags": hit.payload.get("tags", [])
                }
                for hit in results
            ]
        except Exception as search_error:
            logger.error(f"Search failed: {search_error}")
            return []

    def count_incidents(self) -> int:
        """Count total incidents in knowledge base"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0

# Helper to seed example data
async def seed_example_data_async(rag: IncidentRAG):
    """Add some example incidents for testing"""
    examples = [
        {
            "incident_id": "INC-00234",
            "title": "High CPU usage on api-gateway",
            "description": "CPU spiked to 95% at 2:15pm causing slow response times",
            "service": "api-gateway",
            "root_cause": "Memory leak after v2.3.4 deployment",
            "resolution": "Rolled back to v2.3.3 and deployed v2.3.5 with proper connection pooling",
            "severity": "high",
            "tags": ["cpu", "memory-leak", "performance"]
        },
        {
            "incident_id": "INC-00567",
            "title": "Database connection pool exhausted",
            "description": "All connections in use, 504 errors across all services",
            "service": "database",
            "root_cause": "Missing index on users table caused slow queries that held connections",
            "resolution": "Added composite index on users(email, status) and increased pool size",
            "severity": "critical",
            "tags": ["database", "timeout", "performance"]
        },
        {
            "incident_id": "INC-00789",
            "title": "Memory leak in payment processing",
            "description": "Payment service memory usage growing steadily, eventually crashes",
            "service": "payment-service",
            "root_cause": "Webhook responses not properly closed, file handles accumulating",
            "resolution": "Fixed webhook client to properly close connections and added memory monitoring",
            "severity": "high",
            "tags": ["memory-leak", "payment", "crash"]
        }
    ]
    
    for ex in examples:
        await rag.add_incident_async(**ex)
    
    logger.info(f"✅ Seeded {len(examples)} example incidents")