#!/bin/bash
# Start GatorGPT API server

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
DEVICE=${DEVICE:-cuda}

echo "Starting GatorGPT API Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Device: $DEVICE"

# Start server
uvicorn gatorgpt.api.server:create_app --host $HOST --port $PORT --factory
