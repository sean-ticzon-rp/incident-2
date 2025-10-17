"""
RAG System - Using Cohere API for embeddings
FREE tier: 1000 calls/month - Perfect for testing!
"""

import os
import uuid
import logging
import asyncio
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cohere API configuration - FREE tier available!
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
EMBEDDING_MODEL = "embed-english-light-v3.0"  # Fast, free, good quality
COHERE_API_URL = "https://api.cohere.ai/v1/embed"

class IncidentRAG:
    """
    RAG system for incident management
    Uses Cohere API for embeddings - FREE tier!
    """

    def __init__(self, collection_name: str = "incidents"):
        """Initialize RAG system"""
        try:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60.0)
            self.collection_name = collection_name
            self.embedding_dim = 384  # embed-english-light-v3.0 dimension

            if not COHERE_API_KEY:
                logger.warning("⚠️ COHERE_API_KEY not set - embeddings will fail!")

            self._init_collection()
            logger.info("✅ RAG system initialized (Cohere API - FREE)")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise

    def _init_collection(self):
        """Create vector collection for incidents if it doesn't exist"""
        try:
            # Try to get the collection
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' already exists ({collection_info.points_count} incidents)")
        except Exception as get_error:
            # Collection doesn't exist, try to create it
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created new collection '{self.collection_name}'")
            except UnexpectedResponse as create_error:
                if "already exists" in str(create_error).lower():
                    logger.info(f"✅ Collection '{self.collection_name}' already exists (race condition handled)")
                else:
                    logger.error(f"Failed to create collection: {create_error}")
                    raise
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
                raise

    async def get_embedding(self, text: str) -> list:
        """Get embedding from Cohere API (async) - FREE tier available!"""
        if not COHERE_API_KEY:
            raise Exception("COHERE_API_KEY not set. Get free key at https://dashboard.cohere.com/api-keys")

        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": EMBEDDING_MODEL,
            "texts": [text],  # Cohere accepts array of texts
            "input_type": "search_document"  # For storing documents
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(COHERE_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Cohere returns: {"embeddings": [[...]], "id": "...", "texts": [...]}
                embedding = result["embeddings"][0]
                logger.info(f"✅ Got embedding, dimension: {len(embedding)}")
                return embedding
            else:
                error_text = response.text
                logger.error(f"❌ Cohere API error: {response.status_code} - {error_text}")
                raise Exception(f"Cohere API error: {response.status_code} - {error_text}")

    async def get_query_embedding(self, text: str) -> list:
        """Get embedding for search queries (uses different input_type)"""
        if not COHERE_API_KEY:
            raise Exception("COHERE_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": EMBEDDING_MODEL,
            "texts": [text],
            "input_type": "search_query"  # For search queries
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
        """Add a past incident to the knowledge base (ASYNC VERSION)"""
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
        """Search for similar past incidents (ASYNC VERSION)"""
        try:
            query_vector = await self.get_query_embedding(query)  # Use search_query type
        except Exception as e:
            logger.error(f"Failed to get embedding for search: {e}")
            return []

        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True
        }

        if service:
            search_params["query_filter"] = Filter(
                must=[FieldCondition(key="service", match=MatchValue(value=service))]
            )

        results = self.client.search(**search_params)
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

    def count_incidents(self) -> int:
        """Count total incidents in knowledge base"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0

# Optional helper to seed example data
async def seed_example_data_async(rag: IncidentRAG):
    """Add some example incidents for testing (ASYNC VERSION)"""
    examples = [
        {
            "incident_id": "INC-00234",
            "title": "High CPU usage on api-gateway",
            "description": "CPU spiked to 95% at 2:15pm.",
            "service": "api-gateway",
            "root_cause": "Memory leak after v2.3.4 deployment",
            "resolution": "Rolled back and deployed v2.3.5",
            "severity": "high",
            "tags": ["cpu", "memory-leak"]
        },
        {
            "incident_id": "INC-00567",
            "title": "Database connection pool exhausted",
            "description": "All connections in use, 504 errors",
            "service": "user-service",
            "root_cause": "Missing index on users table",
            "resolution": "Added index",
            "severity": "critical",
            "tags": ["database", "timeout"]
        }
    ]
    for ex in examples:
        await rag.add_incident_async(**ex)
    logger.info(f"✅ Seeded {len(examples)} example incidents")