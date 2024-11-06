# src/core/language_processing/__init__.py

"""
Language Processing
=================

Language-specific text processing components:
- BaseTextProcessor: Abstract base class
- EnglishTextProcessor: English language support
- FinnishTextProcessor: Finnish language support
- TextProcessorFactory: Factory for creating processors
"""

from .base import BaseTextProcessor
from .english import EnglishTextProcessor
from .factory import TextProcessorFactory, create_text_processor
from .finnish import FinnishTextProcessor

__all__ = [
    "BaseTextProcessor",
    "EnglishTextProcessor",
    "FinnishTextProcessor",
    "create_text_processor",
    "TextProcessorFactory",
]
