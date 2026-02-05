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

## Embedding Model

Uses **BAAI/bge-m3** locally via sentence-transformers:
- 1024 dimensions
- Multilingual support (Dutch included)
- ~2GB model download on first run
