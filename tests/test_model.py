"""Tests for model architecture components."""
import pytest
import torch
from gatorgpt.model.architecture import (
    RoPE,
    GroupedQueryAttention,
    SwiGLU,
    TransformerBlock,
    GatorGPT
)
from gatorgpt.model.config import ModelConfig


class TestRoPE:
    """Test Rotary Positional Encoding."""
    
    def test_rope_initialization(self):
        """Test RoPE initializes correctly."""
        rope = RoPE(dim=64, max_len=512)
        assert rope.dim == 64
        assert rope.max_len == 512
    
    def test_rope_forward(self, device):
        """Test RoPE forward pass."""
        batch_size, seq_len, d_model = 2, 10, 64
        rope = RoPE(dim=d_model, max_len=512).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output = rope(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should modify input


class TestGroupedQueryAttention:
    """Test Grouped Query Attention."""
    
    def test_gqa_initialization(self, model_config):
        """Test GQA initializes with correct dimensions."""
        gqa = GroupedQueryAttention(model_config)
        
        assert gqa.d_model == model_config.d_model
        assert gqa.n_heads == model_config.n_heads
        assert gqa.n_kv_heads == model_config.n_kv_heads
    
    def test_gqa_forward(self, model_config, device):
        """Test GQA forward pass."""
        batch_size, seq_len = 2, 10
        gqa = GroupedQueryAttention(model_config).to(device)
        x = torch.randn(batch_size, seq_len, model_config.d_model, device=device)
        
        output = gqa(x)
        
        assert output.shape == x.shape
    
    def test_gqa_with_mask(self, model_config, device):
        """Test GQA with attention mask."""
        batch_size, seq_len = 2, 10
        gqa = GroupedQueryAttention(model_config).to(device)
        x = torch.randn(batch_size, seq_len, model_config.d_model, device=device)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        output = gqa(x, mask=mask)
        
        assert output.shape == x.shape


class TestSwiGLU:
    """Test SwiGLU activation."""
    
    def test_swiglu_initialization(self, model_config):
        """Test SwiGLU initializes correctly."""
        swiglu = SwiGLU(model_config)
        
        assert swiglu.d_model == model_config.d_model
        assert swiglu.d_ff == model_config.d_ff
    
    def test_swiglu_forward(self, model_config, device):
        """Test SwiGLU forward pass."""
        batch_size, seq_len = 2, 10
        swiglu = SwiGLU(model_config).to(device)
        x = torch.randn(batch_size, seq_len, model_config.d_model, device=device)
        
        output = swiglu(x)
        
        assert output.shape == x.shape


class TestTransformerBlock:
    """Test Transformer Block."""
    
    def test_block_initialization(self, model_config):
        """Test TransformerBlock initializes correctly."""
        block = TransformerBlock(model_config)
        
        assert hasattr(block, 'attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'ln2')
    
    def test_block_forward(self, model_config, device):
        """Test TransformerBlock forward pass."""
        batch_size, seq_len = 2, 10
        block = TransformerBlock(model_config).to(device)
        x = torch.randn(batch_size, seq_len, model_config.d_model, device=device)
        
        output = block(x)
        
        assert output.shape == x.shape


class TestGatorGPT:
    """Test GatorGPT model."""
    
    def test_model_initialization(self, model_config):
        """Test GatorGPT initializes correctly."""
        model = GatorGPT(model_config)
        
        assert model.config == model_config
        assert len(model.blocks) == model_config.n_blocks
    
    def test_model_forward(self, model_config, device):
        """Test GatorGPT forward pass."""
        batch_size, seq_len = 2, 10
        model = GatorGPT(model_config).to(device)
        x = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        
        logits = model(x)
        
        assert logits.shape == (batch_size, seq_len, model_config.vocab_size)
    
    def test_model_count_parameters(self, model_config):
        """Test parameter counting."""
        model = GatorGPT(model_config)
        
        params = model.count_parameters()
        
        assert params > 0
        assert isinstance(params, int)
    
    def test_model_eval_mode(self, model_config, device):
        """Test model in eval mode."""
        model = GatorGPT(model_config).to(device)
        model.eval()
        
        batch_size, seq_len = 1, 5
        x = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x)
        
        # Should be deterministic in eval mode
        assert torch.allclose(logits1, logits2)
