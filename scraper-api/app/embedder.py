import json
import glob
import os
import uuid
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from app.models import Product

class ProductEmbedder:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str = "houthandel_products"):
        # For Qdrant Cloud, extract host and use grpc or REST API
        if "cloud.qdrant.io" in qdrant_url:
            # Extract host from URL - Qdrant Cloud uses HTTPS on port 443
            host = qdrant_url.replace("https://", "").replace("http://", "").split(":")[0].rstrip("/")
            self.client = QdrantClient(
                url=f"https://{host}",
                api_key=qdrant_api_key,
            )
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        self.collection_name = collection_name
        
        # Use local BAAI/bge-m3 model via sentence-transformers (supports Dutch)
        print("Loading BAAI/bge-m3 model locally (this may take a moment on first run)...")
        self.model = SentenceTransformer("BAAI/bge-m3")
        print("Initialized ProductEmbedder with local BAAI/bge-m3 model")
        
    def setup_collection(self, vector_size: int = 1024, collection_name: str = None):
        """Create collection if it doesn't exist. BAAI/bge-m3 uses 1024 dimensions."""
        collection = collection_name or self.collection_name
        if self.client.collection_exists(collection):
            # Check dimension compatibility
            try:
                info = self.client.get_collection(collection)
                current_size = info.config.params.vectors.size
                if current_size != vector_size:
                     print(f"Collection '{collection}' exists but has vector size {current_size} (expected {vector_size}).")
                     print("Recreating collection...")
                     self.client.delete_collection(collection)
                     self.client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                     )
                else:
                    print(f"Collection '{collection}' already exists with correct size.")
            except Exception as e:
                print(f"Warning checking collection: {e}")
        else:
            print(f"Creating collection '{collection}' with size {vector_size}...")
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def parse_products(self, data: Any) -> Tuple[List[Product], int]:
        """Parse a JSON array into validated Product objects.

        Returns a tuple of (valid_products, skipped_count).
        """
        if not isinstance(data, list):
            return [], 0

        products: List[Product] = []
        skipped = 0

        for item in data:
            if not isinstance(item, dict):
                skipped += 1
                continue

            if "url" in item and "source_url" not in item:
                item["source_url"] = item["url"]

            try:
                if "scrape_success" not in item:
                    item["scrape_success"] = True

                p = Product(**item)
                if p.scrape_success and p.name and p.price is not None:
                    products.append(p)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        return products, skipped

    def load_products_from_dir(self, directory: str) -> List[Product]:
        """Load all products from JSON files in directory."""
        products = []
        files = glob.glob(os.path.join(directory, "*.json"))
        print(f"Found {len(files)} JSON files in {directory}")
        
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    parsed_products, _ = self.parse_products(data)
                    products.extend(parsed_products)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        print(f"Loaded {len(products)} valid products total.")
        return products

    def embed_and_upsert(self, products: List[Product], batch_size: int = 50, collection_name: str = None):
        """Generate embeddings and upsert to Qdrant."""
        if not products:
            print("No products to process.")
            return

        collection = collection_name or self.collection_name

        # Prepare texts
        print("Preparing texts...")
        texts = []
        for p in products:
            if not p.embedding_text:
                p.compute_embedding_text()
            texts.append(p.embedding_text)

        # Process in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"Processing {len(texts)} products in {total_batches} batches...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_products = products[i:i + batch_size]
            
            try:
                # Use local BGE-M3 model for embedding via sentence-transformers
                vectors = self.model.encode(batch_texts, show_progress_bar=False)
                vectors_list = vectors.tolist()

                points = []
                for j, (product, vector) in enumerate(zip(batch_products, vectors_list)):
                    payload = product.model_dump(mode="json")
                    
                    try:
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, product.source_url))
                        payload["id"] = point_id
                    except Exception:
                        point_id = str(uuid.uuid4())

                    points.append(PointStruct(
                        id=point_id, 
                        vector=vector,
                        payload=payload
                    ))
                
                if points:
                    self.client.upsert(
                        collection_name=collection,
                        points=points
                    )
                
                print(f"Upserted batch {i//batch_size + 1}/{total_batches}")
                
            except Exception as e:
                print(f"Error processing batch {i} (skipping): {e}")
                time.sleep(2) # Backoff slightly on error
            
        print("Done!")

    def search(self, query: str, limit: int = 5, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search products semantically using local BGE-M3 model."""
        collection = collection_name or self.collection_name
        try:
            # Get query embedding using local model
            query_vector = self.model.encode(query).tolist()

            results = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit
            ).points
            
            return [hit.payload for hit in results]
        except Exception as e:
            print(f"Search failed: {e}")
            return []

