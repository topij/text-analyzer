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
from .finnish import FinnishTextProcessor
from .factory import create_text_processor, TextProcessorFactory

__all__ = [
    'BaseTextProcessor',
    'EnglishTextProcessor',
    'FinnishTextProcessor',
    'create_text_processor',
    'TextProcessorFactory',
]