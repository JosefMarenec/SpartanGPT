"""Test configuration for pytest."""
import pytest
import torch


@pytest.fixture
def device():
    """Fixture for device (CPU for tests)."""
    return "cpu"


@pytest.fixture
def model_config():
    """Fixture for a small model config for testing."""
    from gatorgpt.model.config import ModelConfig
    return ModelConfig.small()


@pytest.fixture
def tokenizer():
    """Fixture for tiktoken tokenizer."""
    import tiktoken
    return tiktoken.get_encoding("p50k_base")
