# src/core/__init__.py

"""
Core Components
=============

Core functionality for text analysis:
- Language processing
- LLM integration
- Configuration management
"""

from .language_processing.factory import create_text_processor
from .llm.factory import create_llm

__all__ = [
    "create_text_processor",
    "create_llm",
]
