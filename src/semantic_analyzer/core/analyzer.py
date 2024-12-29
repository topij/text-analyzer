"""Core semantic analysis functionality."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseChatModel

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.config import AnalyzerConfig
from src.core.language_processing import create_text_processor

from src.loaders.parameter_handler import ParameterHandler
from src.loaders.models import CategoryConfig

from src.schemas import (
    CategoryAnalysisResult,
    CategoryOutput,
    CompleteAnalysisResult,
    KeywordAnalysisResult,
    ThemeAnalysisResult,
    ThemeOutput,
)

from ..base import (
    BaseSemanticAnalyzer,
    ResultProcessingMixin,
    AnalyzerFactory,
    AnalysisError,
)

logger = logging.getLogger(__name__)


class CoreSemanticAnalyzer(BaseSemanticAnalyzer, ResultProcessingMixin):
    """Core semantic analysis functionality."""

    ANALYZER_MAPPING = {
        "keywords": {
            "attr_name": "keyword_analyzer",
            "class": KeywordAnalyzer,
            "singular": "keyword",
        },
        "themes": {
            "attr_name": "theme_analyzer",
            "class": ThemeAnalyzer,
            "singular": "theme",
        },
        "categories": {
            "attr_name": "category_analyzer",
            "class": CategoryAnalyzer,
            "singular": "category",
        },
    }

    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        llm: Optional[BaseChatModel] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        **kwargs,
    ):
        """Initialize core analyzer.
        
        Args:
            parameter_file: Path to parameter file
            file_utils: File utilities instance
            llm: Language model instance
            categories: Optional category configurations
            **kwargs: Additional configuration options
        """
        super().__init__(
            parameter_file=parameter_file,
            file_utils=file_utils,
            llm=llm,
            **kwargs
        )
        self._init_categories(categories)
        self._init_analyzers()

    def _init_categories(self, categories: Optional[Dict[str, CategoryConfig]]) -> None:
        """Initialize categories with validation."""
        self.categories = categories or {}

    def _init_analyzers(self) -> None:
        """Initialize individual analyzers."""
        for analysis_type, config in self.ANALYZER_MAPPING.items():
            analyzer = AnalyzerFactory.create_analyzer(
                analyzer_type=analysis_type,
                analyzer_class=config["class"],
                llm=self.llm,
                file_utils=self.file_utils,
                categories=self.categories if analysis_type == "categories" else None,
            )
            setattr(self, config["attr_name"], analyzer)

    async def _create_analysis_task(
        self,
        analysis_type: str,
        text: str,
        **kwargs,
    ) -> Awaitable[Any]:
        """Create analysis coroutine for specified type."""
        config = self.ANALYZER_MAPPING[analysis_type]
        analyzer = getattr(self, config["attr_name"])
        return analyzer.analyze(text, **kwargs)

    def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        language: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> CompleteAnalysisResult:
        """Run analysis pipeline.
        
        Args:
            text: Text to analyze
            analysis_types: Types of analysis to perform
            language: Language of text
            timeout: Analysis timeout in seconds
            **kwargs: Additional analysis parameters
            
        Returns:
            CompleteAnalysisResult with analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Set language if specified
            if language:
                self.set_language(language)

            # Validate analysis types
            types_to_run = self._validate_analysis_types(analysis_types)

            # Create tasks
            tasks = [
                self._create_analysis_task(
                    analysis_type=analysis_type,
                    text=text,
                    **kwargs
                )
                for analysis_type in types_to_run
            ]

            # Run analyses
            try:
                results = asyncio.run(
                    asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                )
            except asyncio.TimeoutError as e:
                raise AnalysisError(f"Analysis timed out after {timeout} seconds") from e

            # Process results
            processed_results = self._process_analysis_results(results, types_to_run)

            return CompleteAnalysisResult(
                text=text,
                language=language or "en",
                results=processed_results,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise AnalysisError(f"Analysis failed: {e}") from e

    def _create_type_error_result(self, analysis_type: str, error: str) -> Any:
        """Create error result for specific analysis type."""
        if analysis_type == "themes":
            return ThemeAnalysisResult(
                themes=[],
                error=error,
                timestamp=datetime.now(),
            )
        elif analysis_type == "categories":
            return CategoryAnalysisResult(
                categories=[],
                error=error,
                timestamp=datetime.now(),
            )
        elif analysis_type == "keywords":
            return KeywordAnalysisResult(
                keywords=[],
                error=error,
                timestamp=datetime.now(),
            )
        else:
            return {"error": error, "timestamp": datetime.now()}
