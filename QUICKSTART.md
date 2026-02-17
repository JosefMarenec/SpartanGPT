# 🚀 GatorGPT Quick Start Guide

## Running the Server

### Step 1: Activate Virtual Environment

```bash
cd /Users/josefmarenec/Desktop/SpartanGPT
source venv/bin/activate
```

### Step 2: Start the Server

```bash
# Start on port 8000 (default)
uvicorn gatorgpt.api.server:create_app --host 0.0.0.0 --port 8000 --factory

# Or on port 8001
uvicorn gatorgpt.api.server:create_app --host 0.0.0.0 --port 8001 --factory
```

### Step 3: Access the API

Once running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/info

### Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

---

## Testing the API

### Option 1: Use Swagger UI (Easiest)

1. Open http://localhost:8000/docs in your browser
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters and click "Execute"

### Option 2: Use the Verification Script

```bash
# Make sure server is running first, then:
python scripts/verify_api.py
```

### Option 3: Use curl

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info

# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Option 4: Use Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": "Tell me a story",
        "max_tokens": 50,
        "temperature": 0.7
    }
)
print(response.json())
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/gatorgpt

# Run specific test file
pytest tests/test_model.py

# Verbose output
pytest -v
```

---

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Start server
uvicorn gatorgpt.api.server:create_app --host 0.0.0.0 --port 8000 --factory

# Run tests
pytest

# Run verification script
python scripts/verify_api.py

# Deactivate environment
deactivate
```

---

## Troubleshooting

**Server won't start:**
- Make sure virtual environment is activated: `source venv/bin/activate`
- Check if another process is using the port: try a different port number

**Import errors:**
- Run `pip install -e .` to install the package in editable mode
- Make sure all dependencies are installed: `pip install -r requirements.txt`

**Model not loaded:**
- This is okay for testing - the server will create an untrained model
- For real usage, you need to train the model or load a checkpoint

---

## Next Steps

1. **Train the Model**: The current model is untrained and will generate random output
2. **Load a Checkpoint**: If you have a trained checkpoint, specify it when starting the server
3. **Customize Configuration**: Edit `config/` files to adjust model settings
4. **Deploy with Docker**: Use `docker-compose up` for production deployment

For more details, see [README.md](README.md)
