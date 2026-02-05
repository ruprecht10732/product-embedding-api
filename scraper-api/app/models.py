from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional


class ProductSpecs(BaseModel):
    """Product specifications as key-value pairs."""
    raw: dict[str, str] = Field(default_factory=dict, description="Raw specification data")


class Product(BaseModel):
    """Product data optimized for vector embedding."""
    
    # Core identification
    id: str = Field(..., description="Unique product identifier (generated from URL hash)")
    source_url: str = Field(..., description="Original URL the product was scraped from")
    
    # Product information
    name: str = Field(..., description="Product title/name")
    description: str = Field(default="", description="Full product description")
    brand: Optional[str] = Field(default=None, description="Brand/manufacturer name")
    category: Optional[str] = Field(default=None, description="Product category or breadcrumb")
    
    # Pricing
    price: Optional[float] = Field(default=None, description="Product price as number")
    currency: Optional[str] = Field(default=None, description="Currency code (USD, EUR, etc.)")
    price_raw: Optional[str] = Field(default=None, description="Original price string")
    
    # Details
    specs: ProductSpecs = Field(default_factory=ProductSpecs, description="Product specifications")
    images: list[str] = Field(default_factory=list, description="List of image URLs")
    
    # Reviews
    rating: Optional[float] = Field(default=None, description="Average rating (0-5)")
    review_count: Optional[int] = Field(default=None, description="Number of reviews")
    reviews_summary: Optional[str] = Field(default=None, description="Summary of customer reviews")
    
    # Metadata
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of data")
    scrape_success: bool = Field(default=True, description="Whether data is valid")
    scrape_errors: list[str] = Field(default_factory=list, description="Any errors")
    
    # Pre-computed for embedding
    embedding_text: str = Field(default="", description="Pre-formatted text for vector embedding")
    
    def compute_embedding_text(self) -> str:
        """Generate concatenated text optimized for embedding."""
        parts = []
        
        if self.name:
            parts.append(f"Product: {self.name}")
        
        if self.brand:
            parts.append(f"Brand: {self.brand}")
            
        if self.category:
            parts.append(f"Category: {self.category}")
            
        if self.description:
            # Truncate very long descriptions
            desc = self.description[:2000] if len(self.description) > 2000 else self.description
            parts.append(f"Description: {desc}")
            
        if self.specs.raw:
            specs_text = ", ".join(f"{k}: {v}" for k, v in self.specs.raw.items())
            parts.append(f"Specifications: {specs_text}")
            
        if self.price is not None:
            currency = self.currency or "EUR"
            parts.append(f"Price: {currency} {self.price}")
            
        if self.rating is not None:
            parts.append(f"Rating: {self.rating}/5")
            if self.review_count:
                parts.append(f"({self.review_count} reviews)")
                
        if self.reviews_summary:
            parts.append(f"Customer feedback: {self.reviews_summary}")
        
        self.embedding_text = "\n".join(parts)
        return self.embedding_text
