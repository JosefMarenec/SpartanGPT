"""Tests for training dataset."""
import pytest
import torch
from gatorgpt.training.dataset import TinyStoriesDataset


class TestTinyStoriesDataset:
    """Test TinyStories dataset."""
    
    def test_dataset_initialization(self, tokenizer):
        """Test dataset initializes correctly."""
        dataset = TinyStoriesDataset(
            split="train",
            tokenizer=tokenizer,
            max_length=128,
            subset_size=100  # Small subset for testing
        )
        
        assert len(dataset) > 0
        assert dataset.max_length == 128
    
    def test_dataset_getitem(self, tokenizer):
        """Test dataset item retrieval."""
        dataset = TinyStoriesDataset(
            split="train",
            tokenizer=tokenizer,
            max_length=128,
            subset_size=10
        )
        
        item = dataset[0]
        
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    def test_dataset_length(self, tokenizer):
        """Test dataset length."""
        subset_size = 50
        dataset = TinyStoriesDataset(
            split="train",
            tokenizer=tokenizer,
            max_length=128,
            subset_size=subset_size
        )
        
        assert len(dataset) == subset_size
    
    def test_dataset_token_shapes(self, tokenizer):
        """Test token shapes match max_length."""
        max_length = 64
        dataset = TinyStoriesDataset(
            split="train",
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=10
        )
        
        item = dataset[0]
        
        assert item["input_ids"].shape[0] == max_length
        assert item["labels"].shape[0] == max_length
    
    @pytest.mark.parametrize("split", ["train", "validation"])
    def test_dataset_splits(self, tokenizer, split):
        """Test different dataset splits."""
        dataset = TinyStoriesDataset(
            split=split,
            tokenizer=tokenizer,
            max_length=128,
            subset_size=10
        )
        
        assert len(dataset) > 0
