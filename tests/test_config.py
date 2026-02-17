"""Tests for model configuration."""
import pytest
from gatorgpt.model.config import ModelConfig


class TestModelConfig:
    """Test ModelConfig."""
    
    def test_small_config(self):
        """Test small model configuration."""
        config = ModelConfig.small()
        
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_blocks == 6
        assert config.d_ff == 512
    
    def test_medium_config(self):
        """Test medium model configuration."""
        config = ModelConfig.medium()
        
        assert config.d_model == 448
        assert config.n_heads == 8
        assert config.n_blocks == 10
        assert config.d_ff == 896
    
    def test_large_config(self):
        """Test large model configuration."""
        config = ModelConfig.large()
        
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_blocks == 12
        assert config.d_ff == 1536
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            d_model=128,
            n_heads=4,
            n_blocks=4,
            d_ff=256,
            vocab_size=10000,
            max_len=256,
            dropout=0.2,
            n_kv_heads=2
        )
        
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_blocks == 4
        assert config.d_ff == 256
        assert config.vocab_size == 10000
        assert config.max_len == 256
        assert config.dropout == 0.2
        assert config.n_kv_heads == 2
    
    def test_head_dim_calculation(self):
        """Test head dimension calculation."""
        config = ModelConfig.small()
        
        assert config.head_dim == config.d_model // config.n_heads
    
    def test_kv_heads_default(self):
        """Test default KV heads calculation."""
        config = ModelConfig.small()
        
        # Default is n_heads // 2
        assert config.n_kv_heads == config.n_heads // 2
