"""
FastAPI application for product embedding and semantic search.
"""

import os
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # Load .env file
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
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
    collection: Optional[str] = Field(default=None, description="Collection name (uses default if not set)")


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


class EmbedFilesResponse(BaseModel):
    """Response from multi-file upload embedding operation."""
    success: bool
    collection: str
    files_received: int
    files_processed: int
    products_loaded: int
    products_skipped: int
    files_errors: List[Dict[str, str]] = Field(default_factory=list)
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
    collection: Optional[str] = Field(
        default=None,
        description="Collection name (uses default if not set)"
    )


class AddDocumentsResponse(BaseModel):
    """Response from adding documents."""
    success: bool
    documents_added: int
    ids: List[str]
    message: str


class StreamDocumentsResponse(BaseModel):
    """Response from streaming large JSON document ingestion."""
    success: bool
    documents_seen: int
    documents_added: int
    documents_skipped: int
    batches_processed: int
    collection: str
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
    
    collection = request.collection or COLLECTION_NAME
    results = _embedder.search(request.query, request.limit, collection_name=collection)
    
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
    limit: int = Query(default=10, ge=1, le=100, description="Max results"),
    collection: Optional[str] = Query(default=None, description="Collection name")
) -> SearchResponse:
    """
    Semantic search for products (GET endpoint).
    """
    request = SearchRequest(query=q, limit=limit, collection=collection)
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


@app.post("/api/embed-files", response_model=EmbedFilesResponse)
async def embed_uploaded_files(
    collection: Optional[str] = Query(default=None, description="Collection name (uses default if not set)"),
    files: List[UploadFile] = File(...),
) -> EmbedFilesResponse:
    """
    Upload multiple JSON files, embed product records, and upsert to a target collection.

    - Expects multipart/form-data with one or more files under the `files` field.
    - Each JSON file must contain an array of product objects.
    - Invalid records are skipped and reported in counts.
    """
    if not _embedder:
        raise HTTPException(status_code=503, detail="Embedder not initialized. Check QDRANT_API_KEY.")

    target_collection = (collection or COLLECTION_NAME).strip() or COLLECTION_NAME
    files_errors: List[Dict[str, str]] = []
    all_products = []
    products_skipped = 0
    files_processed = 0

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for upload in files:
        filename = upload.filename or "unknown.json"

        if not filename.lower().endswith(".json"):
            files_errors.append({
                "file": filename,
                "error": "Unsupported file type; only .json files are allowed"
            })
            continue

        try:
            raw = await upload.read()
            if not raw:
                files_errors.append({"file": filename, "error": "File is empty"})
                continue

            try:
                data = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                files_errors.append({"file": filename, "error": "File must be UTF-8 encoded JSON"})
                continue
            except json.JSONDecodeError as e:
                files_errors.append({"file": filename, "error": f"Invalid JSON: {str(e)}"})
                continue

            parsed_products, skipped_count = _embedder.parse_products(data)
            products_skipped += skipped_count
            all_products.extend(parsed_products)
            files_processed += 1
        except Exception as e:
            files_errors.append({"file": filename, "error": str(e)})
        finally:
            await upload.close()

    if not all_products:
        return EmbedFilesResponse(
            success=False,
            collection=target_collection,
            files_received=len(files),
            files_processed=files_processed,
            products_loaded=0,
            products_skipped=products_skipped,
            files_errors=files_errors,
            message="No valid products found in uploaded files"
        )

    try:
        _embedder.setup_collection(collection_name=target_collection)
        _embedder.embed_and_upsert(all_products, collection_name=target_collection)

        return EmbedFilesResponse(
            success=True,
            collection=target_collection,
            files_received=len(files),
            files_processed=files_processed,
            products_loaded=len(all_products),
            products_skipped=products_skipped,
            files_errors=files_errors,
            message=f"Successfully embedded {len(all_products)} products into {target_collection}"
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
        collection = request.collection or COLLECTION_NAME
        _embedder.setup_collection(collection_name=collection)
        _embedder.client.upsert(
            collection_name=collection,
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


@app.post("/api/documents-stream", response_model=StreamDocumentsResponse)
async def add_documents_stream(
    file: UploadFile = File(...),
    embedding_field: str = Form(default="embedding_text", description="Field to embed first if present"),
    text_fields: Optional[str] = Form(default=None, description="Comma-separated or JSON array of fields"),
    payload_fields: Optional[str] = Form(default=None, description="Comma-separated or JSON array of payload fields to keep"),
    id_field: Optional[str] = Form(default=None, description="Field to use as document ID"),
    collection: Optional[str] = Form(default=None, description="Collection name (uses default if not set)"),
    batch_size: int = Query(default=200, ge=10, le=1000, description="Embedding/upsert batch size")
) -> StreamDocumentsResponse:
    """
    Stream a large JSON array file and ingest documents in batches.

    This endpoint is designed for very large files and avoids loading the entire
    JSON payload into memory. The uploaded file must contain a JSON array.
    """
    if not _embedder:
        raise HTTPException(status_code=503, detail="Embedder not initialized. Check QDRANT_API_KEY.")

    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")

    target_collection = (collection or COLLECTION_NAME).strip() or COLLECTION_NAME

    if text_fields:
        try:
            if text_fields.strip().startswith("["):
                parsed_text_fields = json.loads(text_fields)
                if not isinstance(parsed_text_fields, list) or not all(isinstance(f, str) for f in parsed_text_fields):
                    raise ValueError("text_fields JSON must be an array of strings")
                use_text_fields = parsed_text_fields
            else:
                use_text_fields = [field.strip() for field in text_fields.split(",") if field.strip()]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid text_fields: {str(e)}")
    else:
        use_text_fields = ["product_name", "omschrijving", "description", "name", "title", "content", "text"]

    if payload_fields:
        try:
            if payload_fields.strip().startswith("["):
                parsed_payload_fields = json.loads(payload_fields)
                if not isinstance(parsed_payload_fields, list) or not all(isinstance(f, str) for f in parsed_payload_fields):
                    raise ValueError("payload_fields JSON must be an array of strings")
                use_payload_fields = parsed_payload_fields
            else:
                use_payload_fields = [field.strip() for field in payload_fields.split(",") if field.strip()]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload_fields: {str(e)}")
    else:
        use_payload_fields = [
            "product_name",
            "price_excl",
            "price_incl",
            "sku_id",
            "stock_status",
            "image_url",
            "product_url",
            "ean",
            "description",
            "omschrijving",
            "specificaties",
            "brand",
            "category",
        ]

    try:
        import uuid
        import ijson
        from qdrant_client.models import PointStruct

        _embedder.setup_collection(collection_name=target_collection)

        documents_seen = 0
        documents_added = 0
        documents_skipped = 0
        batches_processed = 0
        batch_docs: List[Dict[str, Any]] = []
        batch_ids: List[str] = []
        batch_texts: List[str] = []

        def flush_batch() -> int:
            nonlocal batch_docs, batch_ids, batch_texts, batches_processed
            if not batch_docs:
                return 0

            vectors = _embedder.model.encode(batch_texts, show_progress_bar=False).tolist()
            points = []
            for doc_id, vector, doc, text in zip(batch_ids, vectors, batch_docs, batch_texts):
                payload = dict(doc)
                payload["_embedding_text"] = text
                points.append(PointStruct(id=doc_id, vector=vector, payload=payload))

            _embedder.client.upsert(collection_name=target_collection, points=points)
            added = len(points)
            batches_processed += 1

            batch_docs = []
            batch_ids = []
            batch_texts = []
            return added

        file.file.seek(0)
        for doc in ijson.items(file.file, "item"):
            documents_seen += 1

            if not isinstance(doc, dict):
                documents_skipped += 1
                continue

            embedding_text_value = doc.get(embedding_field)
            if embedding_text_value is not None and str(embedding_text_value).strip():
                embedding_text = str(embedding_text_value)
            else:
                text_parts = []
                for field in use_text_fields:
                    value = doc.get(field)
                    if value is not None and value != "":
                        text_parts.append(str(value))
                embedding_text = " ".join(text_parts) if text_parts else str(doc)

            if not embedding_text.strip():
                documents_skipped += 1
                continue

            if id_field and id_field in doc and doc[id_field] is not None:
                doc_id = str(doc[id_field])
            elif "sku_id" in doc and doc["sku_id"] is not None:
                doc_id = str(doc["sku_id"])
            elif "id" in doc and doc["id"] is not None:
                doc_id = str(doc["id"])
            else:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, embedding_text[:500]))

            payload = {key: doc[key] for key in use_payload_fields if key in doc}
            payload["_embedding_text"] = embedding_text
            payload["_source_file"] = file.filename
            batch_docs.append(payload)
            batch_ids.append(doc_id)
            batch_texts.append(embedding_text)

            if len(batch_docs) >= batch_size:
                documents_added += flush_batch()

        documents_added += flush_batch()

        success = documents_added > 0
        message = (
            f"Successfully streamed {documents_added} documents into {target_collection}"
            if success else
            "No valid documents were ingested"
        )

        return StreamDocumentsResponse(
            success=success,
            documents_seen=documents_seen,
            documents_added=documents_added,
            documents_skipped=documents_skipped,
            batches_processed=batches_processed,
            collection=target_collection,
            message=message,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
