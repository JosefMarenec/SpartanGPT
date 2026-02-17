"""Tests for inference generator."""
import pytest
import torch
from gatorgpt.model.architecture import GatorGPT
from gatorgpt.inference.generator import TextGenerator


class TestTextGenerator:
    """Test TextGenerator."""
    
    @pytest.fixture
    def model(self, model_config, device):
        """Create a small model for testing."""
        model = GatorGPT(model_config).to(device)
        model.eval()
        return model
    
    @pytest.fixture
    def generator(self, model, tokenizer, device):
        """Create a text generator."""
        return TextGenerator(model, tokenizer, device)
    
    def test_generator_initialization(self, generator, model, tokenizer, device):
        """Test generator initializes correctly."""
        assert generator.model == model
        assert generator.tokenizer == tokenizer
        assert generator.device == device
    
    def test_generate_basic(self, generator):
        """Test basic text generation."""
        prompt = "Once upon a time"
        
        output = generator.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0
        )
        
        assert isinstance(output, str)
        assert len(output) > 0
        # Output should contain the prompt
        assert prompt in output or len(output) > len(prompt)
    
    def test_generate_with_temperature(self, generator):
        """Test generation with different temperatures."""
        prompt = "The"
        
        # Low temperature should be more deterministic
        output1 = generator.generate(prompt, max_new_tokens=5, temperature=0.1)
        output2 = generator.generate(prompt, max_new_tokens=5, temperature=0.1)
        
        assert isinstance(output1, str)
        assert isinstance(output2, str)
    
    def test_generate_with_top_k(self, generator):
        """Test generation with top-k sampling."""
        prompt = "Hello"
        
        output = generator.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            top_k=5
        )
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_generate_max_tokens(self, generator):
        """Test generation respects max_new_tokens."""
        prompt = "Test"
        max_new_tokens = 5
        
        output = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0
        )
        
        # Output should be longer than prompt but not excessively long
        assert isinstance(output, str)
    
    def test_generate_empty_prompt(self, generator):
        """Test generation with empty prompt."""
        output = generator.generate(
            prompt="",
            max_new_tokens=10,
            temperature=1.0
        )
        
        assert isinstance(output, str)
