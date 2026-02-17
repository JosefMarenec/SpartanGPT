"""Dataset and DataLoader utilities for GatorGPT training."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class FastDataset(Dataset):
    """Fast dataset with pre-computed sliding windows."""
    
    def __init__(self, tokens: List[int], max_length: int = 256, stride: int = 128):
        self.tokens = np.array(tokens, dtype=np.int32)
        self.max_length = max_length
        self.starts = np.arange(0, len(tokens) - max_length, stride)
    
    def __len__(self) -> int:
        return len(self.starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.max_length
        input_ids = torch.from_numpy(self.tokens[start:end].copy()).long()
        target_ids = torch.from_numpy(self.tokens[start+1:end+1].copy()).long()
        return input_ids, target_ids

def create_fast_dataloader(
    tokens: List[int],
    batch_size: int = 8,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2
) -> DataLoader:
    """Create optimized DataLoader for training."""
    dataset = FastDataset(tokens, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
