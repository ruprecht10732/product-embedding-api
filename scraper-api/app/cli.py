"""
CLI for product embedding and search.

Usage:
    python -m app.cli embed --url <qdrant-url> --key <api-key>
    python -m app.cli search "search query" --url <qdrant-url> --key <api-key>
"""

import argparse
import sys

try:
    from app.embedder import ProductEmbedder
except ImportError:
    ProductEmbedder = None


def main():
    parser = argparse.ArgumentParser(
        description="Product Embedding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed and upsert products to Qdrant")
    embed_parser.add_argument(
        "-i", "--input-dir",
        default="output",
        help="Directory containing JSON files to embed (default: output)"
    )
    embed_parser.add_argument(
        "--url",
        default="https://qdrant-m4804ssokwsggcgkgws0wcoc.salestainable.nl:443",
        help="Qdrant URL"
    )
    embed_parser.add_argument(
        "--key",
        default=None,
        help="Qdrant API Key"
    )
    embed_parser.add_argument(
        "--collection",
        default="houthandel_products",
        help="Qdrant collection name"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for products semantically")
    search_parser.add_argument(
        "query",
        help="Search query text"
    )
    search_parser.add_argument(
        "--url",
        default="https://qdrant-m4804ssokwsggcgkgws0wcoc.salestainable.nl:443",
        help="Qdrant URL"
    )
    search_parser.add_argument(
        "--key",
        default=None,
        help="Qdrant API Key"
    )
    search_parser.add_argument(
        "--collection",
        default="houthandel_products",
        help="Qdrant collection name"
    )
    search_parser.add_argument(
        "-n", "--limit",
        type=int,
        default=5,
        help="Number of results"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "embed":
        if ProductEmbedder is None:
            print("Error: Required packages not installed. Please run: pip install sentence-transformers qdrant-client")
            return 1
            
        print("Initializing embedder...")
        print(f"  Qdrant URL: {args.url}")
        print(f"  Collection: {args.collection}")
        
        embedder = ProductEmbedder(
            qdrant_url=args.url,
            qdrant_api_key=args.key,
            collection_name=args.collection
        )
        
        embedder.setup_collection()
        products = embedder.load_products_from_dir(args.input_dir)
        embedder.embed_and_upsert(products)
        return 0

    elif args.command == "search":
        if ProductEmbedder is None:
            print("Error: Required packages not installed.")
            return 1
            
        print(f"Searching for: '{args.query}'...")
        embedder = ProductEmbedder(
            qdrant_url=args.url,
            qdrant_api_key=args.key,
            collection_name=args.collection
        )
        
        results = embedder.search(args.query, args.limit)
        
        print(f"\nFound {len(results)} matches:\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res.get('name')}")
            print(f"   Price: €{res.get('price')}")
            print(f"   URL: {res.get('source_url')}")
            desc = res.get('description', '')
            print(f"   Desc: {desc[:100]}..." if desc else "   Desc: None")
            print()
            
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
