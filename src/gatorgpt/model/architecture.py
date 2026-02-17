"""
GatorGPT Model Architecture

Core transformer components including:
- Rotary Positional Encoding (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU MLP
- Transformer Blocks
- Main GatorGPT Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional


class Rope(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    
    Applies rotary embeddings to incorporate positional information into the model.
    Each dimension pair is rotated by an angle proportional to its position.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(Rope, self).__init__()
        
        assert d_model % 2 == 0, "d_model must be even for proper RoPE implementation"
        self.d_model = d_model
        self.max_len = max_len
        
        # Position indices [0, 1, 2, ..., max_len-1]
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(1))
        
        # Frequency terms for rotation
        self.register_buffer(
            'div_term',
            torch.exp(
                torch.arange(0, d_model, 2) * 
                -(torch.log(torch.tensor(10000.0)) / d_model)
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get position indices for current sequence
        position_ids = self.position_ids[:seq_len]
        
        # Calculate rotation angles
        angles = position_ids * self.div_term
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Reshape input to separate even and odd dimensions
        x_pairs = x.view(batch_size, seq_len, d_model // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]
        
        # Apply 2D rotation
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
        
        rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated_x = rotated_pairs.view(batch_size, seq_len, d_model)
        
        return rotated_x


class GQA(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    A more efficient variant of Multi-Head Attention where key-value heads
    are grouped and shared across multiple query heads.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        gqa_groups: int = 2,
        max_len: int = 1024,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % gqa_groups == 0, "n_heads must be divisible by gqa_groups"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.gqa_groups = gqa_groups
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_heads // gqa_groups
        self.max_len = max_len
        
        # Linear projections (bias-free)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # RoPE for Q and K
        self.rope_q = Rope(d_model=n_heads * self.head_dim, max_len=max_len)
        self.rope_k = Rope(d_model=self.n_kv_heads * self.head_dim, max_len=max_len)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for grouped query attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask (currently unused)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # (B, T, H*D)
        k = self.k_proj(x)  # (B, T, H_kv*D)
        v = self.v_proj(x)  # (B, T, H_kv*D)
        
        # Apply RoPE
        q = self.rope_q(q)
        q = q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        k = self.rope_k(k)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        v = v.view(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        # Expand K and V to match number of query heads
        expand_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(expand_factor, dim=1)
        v = v.repeat_interleave(expand_factor, dim=1)
        
        # Scaled dot-product attention with flash attention
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=False
            )
        
        # Merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_heads * self.head_dim)
        
        # Output projection
        out = self.o_proj(out)
        
        return out


class MLP(nn.Module):
    """
    SwiGLU MLP for transformer block.
    
    Uses SwiGLU activation: up ⊗ swish(gate)
    """
    
    def __init__(self, d_model: int = 384, d_ff: int = 768):
        super().__init__()
        
        # Fused up + gate projection
        self.w1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        # Down projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        up, gate = self.w1(x).chunk(2, dim=-1)
        x = up * F.silu(gate)  # SwiGLU activation
        x = self.w2(x)
        return x


class Block(nn.Module):
    """
    Transformer Block with pre-normalization.
    
    Structure:
    - RMSNorm -> GQA -> Residual
    - RMSNorm -> MLP -> Residual
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        gqa_groups: int = 2,
        max_len: int = 1024,
        d_ff: int = 768,
        eps: float = 1e-5,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        
        self.rms1 = nn.modules.normalization.RMSNorm(d_model, eps)
        self.rms2 = nn.modules.normalization.RMSNorm(d_model, eps)
        
        self.attn = GQA(d_model, n_heads, gqa_groups, max_len)
        self.mlp = MLP(d_model, d_ff)
        
        self.drop_attn = nn.Dropout(dropout_p)
        self.drop_mlp = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Attention with pre-norm and residual
        x = x + self.drop_attn(self.attn(self.rms1(x)))
        # MLP with pre-norm and residual
        x = x + self.drop_mlp(self.mlp(self.rms2(x)))
        return x


class GatorGPT(nn.Module):
    """
    GatorGPT: A transformer-based language model.
    
    Architecture features:
    - Rotary Positional Encoding (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU activation in MLP
    - RMSNorm for normalization
    - Weight tying between embedding and output layers
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 8,
        gqa_groups: int = 2,
        max_len: int = 1024,
        d_ff: int = 768,
        eps: float = 1e-5,
        dropout_p: float = 0.0,
        n_blocks: int = 10,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                n_heads=n_heads,
                gqa_groups=gqa_groups,
                max_len=max_len,
                d_ff=d_ff,
                eps=eps,
                dropout_p=dropout_p,
            ) for _ in range(n_blocks)
        ])
        
        # Final normalization
        self.final_rms = nn.modules.normalization.RMSNorm(d_model, eps)
        
        # Output projection (unembedding)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.unembed.weight = self.embed.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        h = self.embed(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Final normalization
        h = self.final_rms(h)
        
        # Project to vocabulary
        logits = self.unembed(h)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
