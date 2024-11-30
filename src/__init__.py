# src/__init__.py

"""
Semantic Text Analyzer
=====================

A comprehensive toolkit for semantic text analysis, including:
- Keyword extraction
- Theme analysis
- Category classification
- Multi-language support
"""

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.language_processing import (
    BaseTextProcessor,
    EnglishTextProcessor,
    FinnishTextProcessor,
    create_text_processor,
)
from src.schemas import (
    AnalysisParameters,
    CategoryAnalysisResult,
    KeywordAnalysisResult,
    KeywordInfo,
    ThemeAnalysisResult,
)

# Version information
__version__ = "1.0.0-alpha.1"

# Expose key components at package level
__all__ = [
    # Main analyzers
    "KeywordAnalyzer",
    "ThemeAnalyzer",
    "CategoryAnalyzer",
    # Schema types
    "KeywordAnalysisResult",
    "KeywordInfo",
    "ThemeAnalysisResult",
    "CategoryAnalysisResult",
    "AnalysisParameters",
    # Language processing
    "create_text_processor",
    "BaseTextProcessor",
    "EnglishTextProcessor",
    "FinnishTextProcessor",
    # Version
    "__version__",
]
