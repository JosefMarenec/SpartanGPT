"""
GatorGPT Model Architecture

This module contains the core transformer architecture components for GatorGPT.
"""

__version__ = "0.1.0"

from gatorgpt.model.architecture import GatorGPT, Rope, GQA, MLP, Block
from gatorgpt.model.config import ModelConfig

__all__ = [
    "GatorGPT",
    "Rope",
    "GQA",
    "MLP",
    "Block",
    "ModelConfig",
]
