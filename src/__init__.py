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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .analyzers.category_analyzer import CategoryAnalyzer, CategoryOutput
from .analyzers.keyword_analyzer import KeywordAnalyzer, KeywordOutput
from .analyzers.theme_analyzer import ThemeAnalyzer, ThemeOutput
from .core.language_processing.factory import create_text_processor
from .core.llm.factory import create_llm

# Setup package logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


class SemanticAnalyzer:
    """Main interface for semantic text analysis."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the semantic analyzer.

        Args:
            config_path: Path to configuration directory
            llm_provider: LLM provider to use ("openai" or "anthropic")
            llm_model: Specific model to use
            language: Default language code
            **kwargs: Additional configuration parameters
        """
        self.config_path = config_path or Path("config")
        self.language = language

        # Initialize LLM
        self.llm = create_llm(llm_provider, llm_model, **kwargs)

        # Initialize analyzers
        self.keyword_analyzer = KeywordAnalyzer(self.llm, kwargs.get("keyword_config"))
        self.theme_analyzer = ThemeAnalyzer(self.llm, kwargs.get("theme_config"))
        self.category_analyzer = CategoryAnalyzer(
            categories=kwargs.get("categories", {}),
            llm=self.llm,
            config=kwargs.get("category_config"),
        )

        # Create text processor if language is specified
        self.text_processor = create_text_processor(language) if language else None

    async def analyze(self, text: str, analysis_types: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Analyze text using specified analyzers.

        Args:
            text: Input text to analyze
            analysis_types: List of analysis types to perform
                          ("keywords", "themes", "categories")
            **kwargs: Additional parameters for specific analyzers

        Returns:
            Dict containing results from each requested analyzer
        """
        import asyncio

        # Default to all analysis types if none specified
        analysis_types = analysis_types or ["keywords", "themes", "categories"]

        # Prepare text if processor is available
        processed_text = self.text_processor.process_text(text) if self.text_processor else text

        # Create tasks for requested analyses
        tasks = []
        if "keywords" in analysis_types:
            tasks.append(self.keyword_analyzer.analyze(processed_text, **kwargs.get("keyword_params", {})))
        if "themes" in analysis_types:
            tasks.append(self.theme_analyzer.analyze(processed_text, **kwargs.get("theme_params", {})))
        if "categories" in analysis_types:
            tasks.append(self.category_analyzer.analyze(processed_text, **kwargs.get("category_params", {})))

        # Run analyses concurrently
        results = await asyncio.gather(*tasks)

        # Combine results
        combined_results = {}
        for result, analysis_type in zip(results, analysis_types):
            combined_results[analysis_type] = result.dict()

        return combined_results

    async def analyze_batch(
        self, texts: List[str], analysis_types: Optional[List[str]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Analyze multiple texts concurrently.

        Args:
            texts: List of input texts
            analysis_types: Types of analysis to perform
            **kwargs: Additional parameters

        Returns:
            List of analysis results for each text
        """
        import asyncio

        tasks = [self.analyze(text, analysis_types, **kwargs) for text in texts]

        return await asyncio.gather(*tasks)


# Version information
__version__ = "1.0.0-alpha.1"


# Convenience functions
async def analyze_text(text: str, analysis_types: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Quick analysis of a single text.

    Args:
        text: Input text
        analysis_types: Types of analysis to perform
        **kwargs: Configuration parameters

    Returns:
        Analysis results
    """
    analyzer = SemanticAnalyzer(**kwargs)
    return await analyzer.analyze(text, analysis_types)


async def analyze_texts(texts: List[str], analysis_types: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
    """Quick analysis of multiple texts.

    Args:
        texts: Input texts
        analysis_types: Types of analysis to perform
        **kwargs: Configuration parameters

    Returns:
        List of analysis results
    """
    analyzer = SemanticAnalyzer(**kwargs)
    return await analyzer.analyze_batch(texts, analysis_types)
