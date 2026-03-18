# SpartanGPT

GatorGPT is a lightweight 63M parameter transformer-based language model trained on the TinyStories dataset. It features modern architectural components including Grouped Query Attention (GQA), Rotary Positional Encoding (RoPE), and SwiGLU activation.

## Features

- **Modern Architecture**: GQA, RoPE, SwiGLU MLP, RMSNorm
- **Production-Ready API**: FastAPI server with OpenAI-compatible endpoints
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Modular Design**: Clean separation of concerns with dedicated modules for training, inference, and API
- **Type-Safe**: Full type hints and Pydantic validation
- **GPU Optimized**: Flash attention and BF16 support for A100 GPUs

## Project Structure

```
SpartanGPT/
├── src/gatorgpt/              # Main package
│   ├── model/                 # Model architecture and utilities
│   │   ├── architecture.py    # Rope, GQA, MLP, Block, GatorGPT
│   │   ├── config.py          # Model configuration
│   │   └── utils.py           # Model loading, saving
│   ├── training/              # Training infrastructure
│   │   ├── dataset.py         # Dataset and DataLoader
│   │   └── config.py          # Training configuration
│   ├── inference/             # Text generation
│   │   └── generator.py       # TextGenerator with sampling
│   └── api/                   # FastAPI server
│       ├── server.py          # API application
│       └── models.py          # Request/response models
├── scripts/                   # CLI utilities (coming soon)
├── config/                    # Configuration files
├── requirements.txt           # Dependencies
├── pyproject.toml             # Package configuration
├── Dockerfile                 # Production container
└── docker-compose.yml         # Docker orchestration
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/SpartanGPT.git
cd SpartanGPT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### With Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t gatorgpt .
docker run -d --gpus all -p 8000:8000 gatorgpt
```

## Quick Start

### Starting the API Server

```bash
# Using uvicorn directly
uvicorn gatorgpt.api.server:create_app --host 0.0.0.0 --port 8000 --factory

# With docker-compose
docker-compose up -d
```

### Using the API

**Text Completion:**

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_k": 5
    }
)

print(response.json()["choices"][0]["text"])
```

**Chat Completion:**

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Tell me a story about a robot"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Model Info:**

```bash
curl http://localhost:8000/info
```

## API Endpoints

- `GET /health` - Health check
- `GET /info` - Model information
- `POST /v1/completions` - Text completion (OpenAI-compatible)
- `POST /v1/chat/completions` - Chat completion (OpenAI-compatible)

## Model Architecture

### Configuration

| Parameter | Small | Medium (Default) | Large |
|-----------|-------|------------------|-------|
| d_model | 256 | 448 | 768 |
| n_heads | 4 | 8 | 12 |
| n_blocks | 6 | 10 | 12 |
| d_ff | 512 | 896 | 1536 |
| Parameters | ~20M | ~63M | ~150M |

### Features

- **Grouped Query Attention (GQA)**: 2 KV heads per query head for memory efficiency
- **Rotary Positional Encoding (RoPE)**: Superior positional awareness
- **SwiGLU**: Modern activation function for MLP layers
- **RMSNorm**: Efficient normalization
- **Weight Tying**: Shared embedding and output layers

## Training

Training infrastructure is available in `src/gatorgpt/training/`. To train a model:

```python
from gatorgpt.model.config import ModelConfig
from gatorgpt.model.utils import create_model
from gatorgpt.training import TrainingConfig

# Create model
config = ModelConfig.medium()
model = create_model(config, device="cuda")

# Training code here (see scripts/train.py)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format code
black src/

# Lint
ruff src/

# Run tests (if available)
pytest tests/
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
MODEL_PATH=checkpoints/model.pt
MODEL_SIZE=medium
DEVICE=cuda
API_HOST=0.0.0.0
API_PORT=8000
```

## Docker Deployment

The project includes production-ready Docker configuration:

```bash
# Build image
docker build -t gatorgpt .

# Run with GPU
docker run -d --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  gatorgpt

# Using docker-compose
docker-compose up -d
```

## Model Links

- 🤗 **HuggingFace**: [kunjcr2/GatorGPT2](https://huggingface.co/kunjcr2/GatorGPT2)
- 🐋 **Docker Hub**: [kunjcr2/gatorgpt](https://hub.docker.com/repository/docker/kunjcr2/gatorgpt)

## Performance

- **Training**: Optimized with torch.compile and flash attention
- **Inference**: BF16 support for A100 GPUs
- **Memory Efficient**: GQA reduces KV cache size
- **Fast DataLoaders**: Parallel tokenization and preprocessing

## License

Apache License 2.0

## Contributing

Contributions welcome! Please see the project structure and follow the existing code style.

## Acknowledgments

- Built on PyTorch 2.0+
- Trained on [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- Inspired by modern LLM architectures (LLaMA, GPT)
