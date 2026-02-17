"""
GatorGPT Model Configuration

Pydantic-based configuration for model hyperparameters.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for GatorGPT model architecture."""
    
    vocab_size: int = Field(50257, description="Vocabulary size")
    d_model: int = Field(448, description="Model dimension")
    n_heads: int = Field(8, description="Number of attention heads")
    gqa_groups: int = Field(2, description="Number of GQA groups")
    max_len: int = Field(1024, description="Maximum sequence length")
    d_ff: int = Field(896, description="Feed-forward dimension")
    eps: float = Field(1e-5, description="Epsilon for normalization")
    dropout_p: float = Field(0.0, description="Dropout probability")
    n_blocks: int = Field(10, description="Number of transformer blocks")
    
    class Config:
        """Pydantic config."""
        frozen = True  # Make config immutable
    
    @classmethod
    def small(cls) -> "ModelConfig":
        """Small model configuration (20M parameters)."""
        return cls(
            d_model=256,
            n_heads=4,
            gqa_groups=2,
            d_ff=512,
            n_blocks=6,
        )
    
    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium model configuration (63M parameters) - Default."""
        return cls(
            d_model=448,
            n_heads=8,
            gqa_groups=2,
            d_ff=896,
            n_blocks=10,
        )
    
    @classmethod
    def large(cls) -> "ModelConfig":
        """Large model configuration (150M parameters)."""
        return cls(
            d_model=768,
            n_heads=12,
            gqa_groups=2,
            d_ff=1536,
            n_blocks=12,
        )
