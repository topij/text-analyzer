# src/semantic_analyzer/analyzer.py

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.config import AnalyzerConfig
from src.core.language_processing.factory import create_text_processor
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """Main interface for semantic text analysis."""

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        file_utils: Optional[FileUtils] = None,
        llm=None,
        language: Optional[str] = None,
        categories: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initialize semantic analyzer."""
        # Initialize core components
        self.file_utils = file_utils or FileUtils()
        self.config = config or AnalyzerConfig(self.file_utils)

        # Get analyzer configurations
        analysis_config = self.config.config.get("semantic_analyzer", {}).get("analysis", {})

        # Set up language processing
        self.language = language or self.config.config.get("semantic_analyzer", {}).get("default_language", "en")
        self.text_processor = create_text_processor(language=self.language, config=kwargs.get("processor_config"))

        # Initialize LLM if not provided
        if not llm:
            from src.core.llm.factory import create_llm

            model_config = self.config.config.get("semantic_analyzer", {}).get("models", {})
            llm = create_llm(**model_config)
        self.llm = llm

        # Initialize analyzers with their specific configurations
        self.keyword_analyzer = KeywordAnalyzer(
            llm=self.llm,
            config=analysis_config.get("keywords", {}),
            language_processor=self.text_processor,
        )

        self.theme_analyzer = ThemeAnalyzer(llm=self.llm, config=analysis_config.get("themes", {}))

        self.category_analyzer = CategoryAnalyzer(
            categories=categories or {},
            llm=self.llm,
            config=analysis_config.get("categories", {}),
        )

        logger.info(f"Initialized SemanticAnalyzer with language: {self.language}")

    def _validate_analysis_types(self, types: List[str]) -> None:
        """Validate analysis types."""
        valid_types = {"keywords", "themes", "categories"}
        invalid = [t for t in types if t not in valid_types]
        if invalid:
            raise ValueError(f"Invalid analysis types: {invalid}")

    async def analyze(self, text: str, analysis_types: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Analyze text using specified analyzers."""
        analysis_types = analysis_types or ["keywords", "themes", "categories"]

        try:
            # Validate analysis types
            self._validate_analysis_types(analysis_types)

            # Initialize results
            combined_results = {
                t: {
                    "error": None,
                    "confidence": 1.0,
                    t: [] if t != "categories" else {},
                    "language": self.language,
                }
                for t in analysis_types
            }

            # Validate input
            if not text or not isinstance(text, str):
                error = {"error": "Invalid input text", "confidence": 0.0}
                for t in analysis_types:
                    combined_results[t].update(error)
                return combined_results

            # Create analysis tasks
            tasks = []
            analyzers = {
                "keywords": self.keyword_analyzer,
                "themes": self.theme_analyzer,
                "categories": self.category_analyzer,
            }

            for analysis_type in analysis_types:
                analyzer = analyzers[analysis_type]
                params = kwargs.get(f"{analysis_type}_params", {})
                tasks.append(analyzer.analyze(text, **params))

            # Run analyses
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result, analysis_type in zip(results, analysis_types):
                if isinstance(result, Exception):
                    combined_results[analysis_type].update({"error": str(result), "confidence": 0.0})
                else:
                    combined_results[analysis_type] = result.model_dump()

            return combined_results

        except ValueError as e:
            raise  # Re-raise ValueError for test_error_handling
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            error_info = {"error": str(e), "confidence": 0.0}
            return {t: error_info for t in analysis_types}

    async def analyze_batch(
        self, texts: List[str], analysis_types: Optional[List[str]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Analyze multiple texts concurrently."""
        tasks = [self.analyze(text, analysis_types, **kwargs) for text in texts]

        return await asyncio.gather(*tasks)

    def save_results(self, results: Dict[str, Any], filename: str, output_type: str = "processed") -> str:
        """Save analysis results."""
        return self.file_utils.save_yaml(
            results,
            filename,
            output_type=output_type,
            include_timestamp=self.file_utils.config.get("include_timestamp", True),
        )
