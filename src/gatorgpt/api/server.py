"""FastAPI server for GatorGPT inference."""

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import tiktoken
import time
import uuid
from pathlib import Path

from gatorgpt.model.architecture import GatorGPT
from gatorgpt.model.config import ModelConfig
from gatorgpt.model.utils import load_checkpoint
from gatorgpt.inference.generator import TextGenerator
from gatorgpt.api.models import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ChatRequest,
    ChatResponse,
    ChatChoice,
    ChatMessage,
    HealthResponse,
    ModelInfo,
)


class GatorGPTServer:
    """GatorGPT API server."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generator = TextGenerator(model, tokenizer, device)
    
    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate text completion."""
        generated_text = self.generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model="GatorGPT",
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0
                )
            ]
        )
    
    def generate_chat(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion."""
        # Combine messages into prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"
        
        generated_text = self.generator.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        return ChatResponse(
            id=f"chat-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model="GatorGPT",
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=generated_text),
                    index=0
                )
            ]
        )


def create_app(
    checkpoint_path: str = None,
    model_config: ModelConfig = None,
    device: Optional[str] = None
) -> FastAPI:
    """Create FastAPI application."""
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    app = FastAPI(
        title="GatorGPT API",
        description="OpenAI-compatible API for GatorGPT",
        version="0.1.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load model on startup
    @app.on_event("startup")
    async def load_model():
        if model_config is None:
            config = ModelConfig.medium()
        else:
            config = model_config
        
        tokenizer = tiktoken.get_encoding("p50k_base")
        
        if checkpoint_path and Path(checkpoint_path).exists():
            model = load_checkpoint(checkpoint_path, config, device)
        else:
            # Create untrained model if no checkpoint
            from gatorgpt.model.utils import create_model
            model = create_model(config, device)
        
        model.eval()
        app.state.server = GatorGPTServer(model, tokenizer, device)
        app.state.model_config = config
    
    # Routes
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=hasattr(app.state, "server")
        )
    
    @app.get("/info", response_model=ModelInfo)
    async def model_info():
        """Model information endpoint."""
        if not hasattr(app.state, "server"):
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        config = app.state.model_config
        model = app.state.server.model
        
        return ModelInfo(
            model_name="GatorGPT",
            vocab_size=config.vocab_size,
            parameters=model.count_parameters(),
            max_length=config.max_len
        )
    
    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest):
        """Text completion endpoint (OpenAI-compatible)."""
        if not hasattr(app.state, "server"):
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return app.state.server.generate_completion(request)
    
    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def chat_completions(request: ChatRequest):
        """Chat completion endpoint (OpenAI-compatible)."""
        if not hasattr(app.state, "server"):
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return app.state.server.generate_chat(request)
    
    return app
