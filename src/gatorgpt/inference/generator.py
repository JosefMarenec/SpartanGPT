"""Text generation utilities for GatorGPT."""

import torch
import tiktoken
from torch.amp import autocast
from typing import Optional


class TextGenerator:
    """Text generation with sampling strategies."""
    
    def __init__(self, model, tokenizer, device: str = "cuda", context_size: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_size = context_size
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.75,
        top_k: int = 5
    ) -> str:
        """Generate text from a prompt."""
        self.model.eval()
        
        # Encode prompt
        token_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        # Generate
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get context window
                idx_cond = token_ids[:, -self.context_size:]
                
                # Forward pass
                with autocast("cuda", torch.bfloat16):
                    logits = self.model(idx_cond)
                
                # Get next token logits
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(topk_vals, dim=-1)
                idx_next_local = torch.multinomial(probs, num_samples=1)
                idx_next = torch.gather(topk_idx, -1, idx_next_local)
                
                # Append to sequence
                token_ids = torch.cat((token_ids, idx_next), dim=1)
        
        # Decode
        generated_text = self.tokenizer.decode(token_ids.squeeze(0).tolist())
        return generated_text


def generate_text(model, prompt: str, tokenizer, max_new_tokens: int = 50, device: str = "cuda") -> str:
    """Simple text generation function."""
    generator = TextGenerator(model, tokenizer, device)
    return generator.generate(prompt, max_new_tokens)
