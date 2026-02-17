"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional


class CompletionRequest(BaseModel):
    """Request model for text completion."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(50, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(5, description="Top-k sampling parameter")


class CompletionChoice(BaseModel):
    """Single completion choice."""
    text: str
    index: int
    finish_reason: str = "length"


class CompletionResponse(BaseModel):
    """Response model for text completion."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]


class ChatMessage(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[ChatMessage]
    max_tokens: int = Field(50, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(5, description="Top-k sampling parameter")


class ChatChoice(BaseModel):
    """Single chat completion choice."""
    message: ChatMessage
    index: int
    finish_reason: str = "length"


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    vocab_size: int
    parameters: int
    max_length: int
