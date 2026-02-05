"""
FastAPI application for product embedding and semantic search.
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # Load .env file
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.embedder import ProductEmbedder


# Configuration from environment or defaults
QDRANT_URL = os.environ.get("QDRANT_URL", "https://qdrant-m4804ssokwsggcgkgws0wcoc.salestainable.nl:443")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "houthandel_products")

# Global embedder instance
_embedder: Optional[ProductEmbedder] = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")


class SearchResult(BaseModel):
    """Single search result."""
    name: str
    price: Optional[float] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    specs: Optional[dict] = None


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    count: int


class EmbedRequest(BaseModel):
    """Request to embed products from a directory."""
    input_dir: str = Field(default="output", description="Directory containing JSON files")


class EmbedResponse(BaseModel):
    """Response from embedding operation."""
    success: bool
    products_loaded: int
    message: str


class EmbedTextRequest(BaseModel):
    """Request to embed text."""
    text: str = Field(..., min_length=1, description="Text to embed")


class EmbedTextResponse(BaseModel):
    """Response with embedding vector."""
    text: str
    vector: List[float]
    dimensions: int


class AddDocumentsRequest(BaseModel):
    """Request to add documents to Qdrant with embeddings."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to embed and store")
    text_fields: List[str] = Field(
        default=["name", "title", "description", "content", "text"],
        description="Fields to concatenate for embedding text"
    )
    id_field: Optional[str] = Field(
        default=None,
        description="Field to use as document ID (auto-generated if not set)"
    )


class AddDocumentsResponse(BaseModel):
    """Response from adding documents."""
    success: bool
    documents_added: int
    ids: List[str]
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage embedder lifecycle."""
    global _embedder
    
    if QDRANT_API_KEY:
        try:
            _embedder = ProductEmbedder(
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME
            )
            print(f"✓ Connected to Qdrant at {QDRANT_URL}")
        except Exception as e:
            print(f"⚠ Failed to connect to Qdrant: {e}")
            _embedder = None
    else:
        print("⚠ QDRANT_API_KEY not set - embedder not initialized")
        _embedder = None
    
    yield
    
    _embedder = None
    print("✓ Embedder shut down")


app = FastAPI(
    title="Product Embedding API",
    description="Semantic search and embedding for product data using BGE-M3",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedder_ready": _embedder is not None,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION_NAME
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_products(request: SearchRequest) -> SearchResponse:
    """
    Semantic search for products.
    
    Uses BGE-M3 embeddings to find similar products based on the query text.
    """
    if not _embedder:
        raise HTTPException(status_code=503, detail="Embedder not initialized. Check QDRANT_API_KEY.")
    
    results = _embedder.search(request.query, request.limit)
    
    search_results = []
    for r in results:
        search_results.append(SearchResult(
            name=r.get("name", ""),
            price=r.get("price"),
            description=r.get("description", "")[:500] if r.get("description") else None,
            source_url=r.get("source_url"),
            category=r.get("category"),
            brand=r.get("brand"),
            images=r.get("images", [])[:3],
            specs=r.get("specs", {}).get("raw") if r.get("specs") else None
        ))
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        count=len(search_results)
    )


@app.get("/api/search")
async def search_products_get(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=100, description="Max results")
) -> SearchResponse:
    """
    Semantic search for products (GET endpoint).
    """
    request = SearchRequest(query=q, limit=limit)
    return await search_products(request)


@app.post("/api/embed", response_model=EmbedResponse)
async def embed_products(request: EmbedRequest) -> EmbedResponse:
    """
    Load and embed products from JSON files.
    
    Reads all JSON files from the specified directory and upserts
    embeddings to the Qdrant collection.
    """
    if not _embedder:
        raise HTTPException(status_code=503, detail="Embedder not initialized. Check QDRANT_API_KEY.")
    
    try:
        _embedder.setup_collection()
        products = _embedder.load_products_from_dir(request.input_dir)
        
        if not products:
            return EmbedResponse(
                success=False,
                products_loaded=0,
                message=f"No valid products found in {request.input_dir}"
            )
        
        _embedder.embed_and_upsert(products)
        
        return EmbedResponse(
            success=True,
            products_loaded=len(products),
            message=f"Successfully embedded {len(products)} products"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/embed-text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest) -> EmbedTextResponse:
    """
    Get embedding vector for a text query.
    
    Use this to get the vector, then query Qdrant directly with your own SDK.
    Returns a 1024-dimensional vector from BGE-M3.
    """
    if not _embedder:
        raise HTTPException(status_code=503, detail="Embedder not initialized. Check QDRANT_API_KEY.")
    
    vector = _embedder.model.encode(request.text).tolist()
    
    return EmbedTextResponse(
        text=request.text,
        vector=vector,
        dimensions=len(vector)
    )


@app.post("/api/documents", response_model=AddDocumentsResponse)
async def add_documents(request: AddDocumentsRequest) -> AddDocumentsResponse:
    """
    Add flexible documents to Qdrant with embeddings.
    
    Send any JSON documents - they will be embedded using specified text_fields
    and stored in Qdrant. All fields are preserved in the payload.
    
    Example:
    ```json
    {
        "documents": [
            {"name": "Product X", "description": "A great product", "price": 99.99, "custom_field": "anything"},
            {"title": "Article Y", "content": "Full text here...", "author": "John"}
        ],
        "text_fields": ["name", "title", "description", "content"],
        "id_field": "sku"
    }
    ```
    """
    if not _embedder:
        raise HTTPException(status_code=503, detail="Embedder not initialized. Check QDRANT_API_KEY.")
    
    if not request.documents:
        return AddDocumentsResponse(
            success=False,
            documents_added=0,
            ids=[],
            message="No documents provided"
        )
    
    try:
        import uuid
        from qdrant_client.models import PointStruct
        
        # Ensure collection exists
        _embedder.setup_collection()
        
        # Build embedding texts and IDs
        texts = []
        ids = []
        
        for doc in request.documents:
            # Build embedding text from specified fields
            text_parts = []
            for field in request.text_fields:
                if field in doc and doc[field]:
                    text_parts.append(str(doc[field]))
            
            embedding_text = " ".join(text_parts) if text_parts else str(doc)
            texts.append(embedding_text)
            
            # Get or generate ID
            if request.id_field and request.id_field in doc:
                doc_id = str(doc[request.id_field])
            elif "id" in doc:
                doc_id = str(doc["id"])
            else:
                # Generate UUID from content hash for deduplication
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, embedding_text[:500]))
            
            ids.append(doc_id)
        
        # Generate embeddings
        vectors = _embedder.model.encode(texts, show_progress_bar=False).tolist()
        
        # Create points
        points = []
        for doc_id, vector, doc, text in zip(ids, vectors, request.documents, texts):
            payload = dict(doc)
            payload["_embedding_text"] = text  # Store what was embedded
            points.append(PointStruct(
                id=doc_id,
                vector=vector,
                payload=payload
            ))
        
        # Upsert to Qdrant
        _embedder.client.upsert(
            collection_name=_embedder.collection_name,
            points=points
        )
        
        return AddDocumentsResponse(
            success=True,
            documents_added=len(points),
            ids=ids,
            message=f"Successfully added {len(points)} documents"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
