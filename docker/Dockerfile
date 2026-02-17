# Multi-stage build for production inference
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install Python and essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package
RUN pip3 install -e .

# Non-root user for security
RUN useradd -m -u 1000 gatorgpt && chown -R gatorgpt:gatorgpt /app
USER gatorgpt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["uvicorn", "gatorgpt.api.server:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
