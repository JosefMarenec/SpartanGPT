"""
Training Configuration

Pydantic-based configuration for training hyperparameters.
"""

from typing import Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = Field(3, description="Number of training epochs")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.01, description="Learning rate")
    weight_decay: float = Field(0.01, description="Weight decay")
    
    # Data parameters
    max_length: int = Field(512, description="Maximum sequence length")
    stride: int = Field(256, description="Stride for sliding window")
    train_split: float = Field(0.95, description="Train/val split ratio")
    
    # Evaluation
    eval_freq: int = Field(5000, description="Evaluate every N steps")
    eval_iter: int = Field(1000, description="Number of eval iterations")
    
    # Early stopping
    patience: int = Field(3, description="Early stopping patience")
    
    # Logging
    sample_tokens: int = Field(1024, description="Tokens to generate for samples")
    progress_chunks: int = Field(20, description="Progress logging frequency")
    
    # Checkpointing
    checkpoint_dir: str = Field("checkpoints", description="Checkpoint directory")
    save_every: int = Field(10000, description="Save checkpoint every N steps")
    
    # WandB
    use_wandb: bool = Field(True, description="Use Weights & Biases")
    wandb_project: str = Field("GatorGPT", description="WandB project name")
    wandb_run_name: Optional[str] = Field(None, description="WandB run name")
    
    # Data loading
    num_workers: int = Field(4, description="DataLoader workers")
    prefetch_factor: int = Field(2, description="DataLoader prefetch factor")
    
    # Data source
    dataset_name: str = Field("roneneldan/TinyStories", description="HuggingFace dataset")
    training_cutoff: int = Field(100_000, description="Number of training examples")
    
    class Config:
        """Pydantic config."""
        frozen = False
