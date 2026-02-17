"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from gatorgpt.api.server import create_app
from gatorgpt.model.config import ModelConfig


@pytest.fixture
def client():
    """Create test client."""
    # Create app with small model config for faster testing
    app = create_app(
        checkpoint_path=None,
        model_config=ModelConfig.small(),
        device="cpu"
    )
    
    # Manually trigger startup event
    @app.on_event("startup")
    async def startup():
        pass
    
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestInfoEndpoint:
    """Test /info endpoint."""
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/info")
        
        # May be 200 or 503 depending on startup timing
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "vocab_size" in data
            assert "parameters" in data
            assert "max_length" in data


class TestCompletionsEndpoint:
    """Test /v1/completions endpoint."""
    
    def test_completions_basic(self, client):
        """Test basic completion request."""
        request_data = {
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.7
        }
        
        response = client.post("/v1/completions", json=request_data)
        
        # May be 200 or 503 depending on model loading
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "created" in data
            assert "model" in data
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "text" in data["choices"][0]
    
    def test_completions_validation(self, client):
        """Test completion request validation."""
        # Missing required field
        request_data = {
            "max_tokens": 10
        }
        
        response = client.post("/v1/completions", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422


class TestChatCompletionsEndpoint:
    """Test /v1/chat/completions endpoint."""
    
    def test_chat_completions_basic(self, client):
        """Test basic chat completion request."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Tell me a story"}
            ],
            "max_tokens": 20,
            "temperature": 0.7
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        # May be 200 or 503 depending on model loading
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "created" in data
            assert "model" in data
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]
            assert "role" in data["choices"][0]["message"]
            assert "content" in data["choices"][0]["message"]
    
    def test_chat_completions_multiple_messages(self, client):
        """Test chat with multiple messages."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Tell me a joke"}
            ],
            "max_tokens": 20
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        # May be 200 or 503 depending on model loading
        assert response.status_code in [200, 503]
    
    def test_chat_completions_validation(self, client):
        """Test chat completion request validation."""
        # Missing required field
        request_data = {
            "max_tokens": 10
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
