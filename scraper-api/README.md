# Product Embedding API

Semantic search and embedding for product data using BGE-M3 and Qdrant.

## Setup

```bash
cd scraper-api

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Embed products from JSON files to Qdrant
python -m app.cli embed --url https://your-qdrant.com:443 --key YOUR_API_KEY

# Search products
python -m app.cli search "douglas hout 2 meter" --url https://your-qdrant.com:443 --key YOUR_API_KEY
```

### API Server

```bash
# Set environment variables
export QDRANT_URL=https://your-qdrant.com:443
export QDRANT_API_KEY=your-key

# Start server
uvicorn app.main:app --reload
```

#### Endpoints

**Search products (GET):**
```bash
curl "http://localhost:8000/api/search?q=douglas%20hout&limit=5"
```

**Search products (POST):**
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "douglas hout", "limit": 10}'
```

**Embed products:**
```bash
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"input_dir": "output"}'
```

**Upload multiple JSON files and embed to `houthandel_products`:**
```bash
curl -X POST "http://localhost:8000/api/embed-files?collection=bouwmaat_products" \
  -H "accept: application/json" \
  -F "files=@output/douglas_products.json;type=application/json" \
  -F "files=@output/vuren_products.json;type=application/json"
```

The endpoint:
- accepts multipart file uploads (`files` field)
- expects each file to contain a JSON array of products
- skips invalid records and returns counts + per-file errors
- uploads to the `collection` query parameter (or defaults to `houthandel_products`)

**Stream a very large JSON file (constant-memory ingestion):**
```bash
curl -X POST "http://localhost:8000/api/documents-stream?batch_size=200" \
  -H "accept: application/json" \
  -F "file=@output/bouwmaat_big.json;type=application/json" \
  -F "collection=bouwmaat_products" \
  -F "id_field=sku_id" \
  -F "embedding_field=embedding_text"
```

Use `/api/documents-stream` for very large JSON arrays to avoid loading all
documents into memory at once. The endpoint parses incrementally and upserts
to Qdrant in batches.

For Bouwmaat-style records, the endpoint now:
- embeds `embedding_text` first (fallbacks to common text fields)
- uses `sku_id` as ID when available
- stores a useful payload subset by default (`product_name`, prices, sku, stock, image/url, ean, description, omschrijving, specificaties, etc.)

You can override payload fields explicitly:
```bash
-F 'payload_fields=["product_name","sku_id","product_url","price_excl","specificaties"]'
```

## Product Data Format

Products are stored with the following structure:

```json
{
  "id": "abc123def456",
  "source_url": "https://example.com/product",
  "name": "Product Name",
  "description": "Full product description...",
  "brand": "Brand Name",
  "category": "Hout > Douglas",
  "price": 149.99,
  "currency": "EUR",
  "specs": {
    "raw": {
      "Lengte": "2 meter",
      "Breedte": "150mm"
    }
  },
  "images": ["https://example.com/img1.jpg"]
}
```

## Configuration

Set environment variables (prefix `EMBED_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | - | Qdrant server URL |
| `QDRANT_API_KEY` | - | Qdrant API key |
| `COLLECTION_NAME` | houthandel_products | Qdrant collection name |

## Docker

```bash
# Build and run
docker compose up -d

# Or build manually
docker build -t product-embedding-api .
docker run -p 8000:8000 --env-file .env product-embedding-api
```

## Embedding Model

Uses **BAAI/bge-m3** locally via sentence-transformers:
- 1024 dimensions
- Multilingual support (Dutch included)
- ~2GB model download on first run
