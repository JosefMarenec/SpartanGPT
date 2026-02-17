#!/usr/bin/env python3
"""Simple script to verify GatorGPT API is working correctly."""
import requests
import sys
from typing import Dict, Any
import json


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {message}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ{Colors.END} {message}")


def print_header(message: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{message}{Colors.END}")


def test_health_check(base_url: str) -> bool:
    """Test /health endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "healthy":
            print_success(f"Health check passed: {data}")
            return True
        else:
            print_error(f"Health check failed: {data}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test /info endpoint."""
    try:
        response = requests.get(f"{base_url}/info", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        required_fields = ["model_name", "vocab_size", "parameters", "max_length"]
        if all(field in data for field in required_fields):
            print_success("Model info retrieved successfully:")
            print(f"  - Model: {data['model_name']}")
            print(f"  - Parameters: {data['parameters']:,}")
            print(f"  - Vocab Size: {data['vocab_size']:,}")
            print(f"  - Max Length: {data['max_length']}")
            return True
        else:
            print_error(f"Missing required fields in response: {data}")
            return False
    except Exception as e:
        print_error(f"Model info error: {e}")
        return False


def test_completion(base_url: str) -> bool:
    """Test /v1/completions endpoint."""
    try:
        payload = {
            "prompt": "Once upon a time",
            "max_tokens": 30,
            "temperature": 0.7,
            "top_k": 5
        }
        
        print_info(f"Sending completion request with prompt: '{payload['prompt']}'")
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            generated_text = data["choices"][0]["text"]
            print_success("Completion generated successfully:")
            print(f"  Prompt: {payload['prompt']}")
            print(f"  Generated: {generated_text[:100]}...")
            return True
        else:
            print_error(f"Invalid completion response: {data}")
            return False
    except Exception as e:
        print_error(f"Completion error: {e}")
        return False


def test_chat_completion(base_url: str) -> bool:
    """Test /v1/chat/completions endpoint."""
    try:
        payload = {
            "messages": [
                {"role": "user", "content": "Tell me a short story about a robot"}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "top_k": 5
        }
        
        print_info("Sending chat completion request...")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0]["message"]
            print_success("Chat completion generated successfully:")
            print(f"  User: {payload['messages'][0]['content']}")
            print(f"  Assistant: {message['content'][:100]}...")
            return True
        else:
            print_error(f"Invalid chat response: {data}")
            return False
    except Exception as e:
        print_error(f"Chat completion error: {e}")
        return False


def main():
    """Run all API verification tests."""
    # Default to localhost:8001 since that's what's running
    base_url = "http://localhost:8001"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print_header(f"🧪 GatorGPT API Verification")
    print_info(f"Testing API at: {base_url}")
    
    results = []
    
    # Test 1: Health Check
    print_header("1. Health Check")
    results.append(("Health Check", test_health_check(base_url)))
    
    # Test 2: Model Info
    print_header("2. Model Information")
    results.append(("Model Info", test_model_info(base_url)))
    
    # Test 3: Completion
    print_header("3. Text Completion")
    results.append(("Text Completion", test_completion(base_url)))
    
    # Test 4: Chat Completion
    print_header("4. Chat Completion")
    results.append(("Chat Completion", test_chat_completion(base_url)))
    
    # Summary
    print_header("📊 Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some tests failed{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
