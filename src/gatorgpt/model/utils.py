"""
Model utilities for loading, saving, and inspecting GatorGPT models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from torchinfo import summary

from gatorgpt.model.architecture import GatorGPT
from gatorgpt.model.config import ModelConfig


def create_model(config: ModelConfig, device: str = "cuda") -> GatorGPT:
    """
    Create a GatorGPT model from configuration.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Initialized GatorGPT model
    """
    model = GatorGPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        gqa_groups=config.gqa_groups,
        max_len=config.max_len,
        d_ff=config.d_ff,
        eps=config.eps,
        dropout_p=config.dropout_p,
        n_blocks=config.n_blocks,
    )
    
    model = model.to(device)
    return model


def load_checkpoint(
    checkpoint_path: str,
    config: ModelConfig,
    device: str = "cuda",
    compile_model: bool = False
) -> GatorGPT:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on
        compile_model: Whether to compile model with torch.compile
        
    Returns:
        Loaded model
    """
    model = create_model(config, device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if compile_model:
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    
    return model


def save_checkpoint(
    model: GatorGPT,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        metadata: Optional additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, save_path)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(
    model: GatorGPT,
    input_shape: tuple = (1, 128),
    device: str = "cuda"
) -> None:
    """
    Print detailed model summary.
    
    Args:
        model: Model to summarize
        input_shape: Input shape for summary (batch_size, seq_len)
        device: Device model is on
    """
    print("\n" + "="*80)
    print("GatorGPT Model Summary")
    print("="*80)
    
    # Model configuration
    print(f"\nConfiguration:")
    print(f"  Vocabulary Size: {model.vocab_size:,}")
    print(f"  Model Dimension: {model.d_model}")
    print(f"  Max Sequence Length: {model.max_len}")
    print(f"  Number of Blocks: {len(model.blocks)}")
    
    # Parameter counts
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    # Memory estimate (rough)
    memory_mb = (params['total'] * 4) / (1024 ** 2)  # Assuming float32
    print(f"\nEstimated Memory (FP32): {memory_mb:.2f} MB")
    
    print("\n" + "="*80)
    
    # Detailed layer summary
    try:
        summary(
            model,
            input_size=input_shape,
            dtypes=[torch.long],
            device=device,
            verbose=1
        )
    except Exception as e:
        print(f"Could not generate detailed summary: {e}")
