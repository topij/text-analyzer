# src/semantic_analyzer/core/analyzer.py

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
from ..analyzer import BaseAnalyzerConfig

logger = logging.getLogger(__name__)


class CoreSemanticAnalyzer(BaseSemanticAnalyzer, ResultProcessingMixin):
    """Core semantic analysis functionality."""

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
        self._init_analyzers(use_excel=False)

    def _init_categories(self, categories: Optional[Dict[str, CategoryConfig]]) -> None:
        """Initialize categories with validation."""
        self.categories = categories or {}

    def _init_analyzers(self, use_excel: bool = False) -> None:
        """Initialize individual analyzers."""
        analyzer_types = BaseAnalyzerConfig.ANALYZER_TYPES

        for analysis_type, config in analyzer_types.items():
            analyzer = AnalyzerFactory.create_analyzer(
                analyzer_type=analysis_type,
                analyzer_class=config["analyzer_class"],
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
        config = BaseAnalyzerConfig.ANALYZER_TYPES[analysis_type]
        analyzer = getattr(self, config["attr_name"])
        return analyzer.analyze(text, **kwargs)

    def _create_type_error_result(self, analysis_type: str, error: str) -> Any:
        """Create error result for specific analysis type."""
        error_results = {
            "keywords": KeywordAnalysisResult(
                error=error,
                language=self.parameters.general.language,
                keywords=[],
                compound_words=[],
                domain_keywords={},
                success=False
            ),
            "themes": ThemeAnalysisResult(
                error=error,
                language=self.parameters.general.language,
                themes=[],
                theme_hierarchy={},
                success=False
            ),
            "categories": CategoryAnalysisResult(
                error=error,
                language=self.parameters.general.language,
                matches=[],
                success=False
            ),
        }
        return error_results.get(analysis_type, {"error": error})

    async def analyze(
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
            start_time = datetime.now()

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

            # Run analyses with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise AnalysisError(f"Analysis timed out after {timeout} seconds")

            processed_results = self._process_analysis_results(results, types_to_run)
            processing_time = (datetime.now() - start_time).total_seconds()

            return CompleteAnalysisResult(
                keywords=processed_results.get("keywords", KeywordAnalysisResult(
                    language=self.parameters.general.language,
                    keywords=[],
                    compound_words=[],
                    domain_keywords={},
                    success=False
                )),
                themes=processed_results.get("themes", ThemeAnalysisResult(
                    language=self.parameters.general.language,
                    themes=[],
                    theme_hierarchy={},
                    success=False
                )),
                categories=processed_results.get("categories", CategoryAnalysisResult(
                    language=self.parameters.general.language,
                    matches=[],
                    success=False
                )),
                language=self.parameters.general.language,
                processing_time=processing_time,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "language": self.parameters.general.language,
                }
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise AnalysisError(f"Analysis failed: {str(e)}")

    def _validate_analysis_types(
        self, types: Optional[List[str]] = None
    ) -> List[str]:
        """Validate and normalize analysis types.

        Args:
            types: List of requested analysis types

        Returns:
            List of validated and normalized analysis type names

        Raises:
            ValueError: If any requested type is invalid
        """
        valid_types = set(BaseAnalyzerConfig.ANALYZER_TYPES.keys())
        
        if not types:
            return list(valid_types)

        # Normalize input types to plural form
        normalized_types = []
        for t in types:
            # Handle both singular and plural forms
            if t.endswith("y"):
                plural = t[:-1] + "ies"
            elif not t.endswith("s"):
                plural = t + "s"
            else:
                plural = t

            normalized_types.append(plural)

        # Check for invalid types
        invalid_types = set(normalized_types) - valid_types
        if invalid_types:
            valid_forms = []
            for valid, config in BaseAnalyzerConfig.ANALYZER_TYPES.items():
                singular = config["singular_name"]
                valid_forms.extend([valid, singular])

            raise ValueError(
                f"Invalid analysis types: {invalid_types}. "
                f"Valid types are: {', '.join(valid_forms)}"
            )

        return normalized_types
