import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Awaitable

import pandas as pd
from langchain_core.language_models.openai import OpenAIChatModel

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.config.manager import ConfigManager
from src.core.language_processing import create_text_processor
from src.loaders.parameter_handler import ParameterHandler
from src.schemas import (
    CategoryAnalysisResult,
    CompleteAnalysisResult,
    KeywordAnalysisResult,
    ThemeAnalysisResult,
)
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from src.utils.output_formatter import OutputDetail

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """Main interface for semantic text analysis."""

    VALID_TYPES = {"keywords", "themes", "categories"}

    def __init__(
        self,
        config_manager: ConfigManager,
        parameter_file: Optional[Union[str, Path]] = None,
        llm: Optional[OpenAIChatModel] = None,
        **kwargs,
    ):
        """Initialize analyzer with parameters and components."""
        self.config_manager = config_manager
        self.file_utils = self.config_manager.file_utils
        self.analyzer_config = self.config_manager.get_config()
        self.llm = llm or self._create_llm()

        self._base_parameter_file = parameter_file
        self.parameter_handler = ParameterHandler(parameter_file)
        self.parameters = self.parameter_handler.get_parameters()
        self._init_analyzers()
        logger.info("Semantic analyzer initialized with configuration manager")

    def _create_llm(self):
        """Create LLM based on configuration."""
        model_config = self.config_manager.get_model_config()
        return OpenAIChatModel(
            model=model_config.default_model,
            provider=model_config.default_provider,
            parameters=model_config.parameters,
        )

    def _init_analyzers(self):
        """Initialize analyzers with proper configurations."""
        global_config = self.config_manager.get_config()
        model_params = global_config.models.parameters
        language = global_config.languages.default_language

        self.keyword_analyzer = KeywordAnalyzer(
            llm=self.llm,
            config={
                "max_keywords": model_params.get("max_keywords", 10),
                "min_keyword_length": model_params.get("min_keyword_length", 3),
            },
            language_processor=create_text_processor(language=language),
        )

        self.theme_analyzer = ThemeAnalyzer(
            llm=self.llm,
            config={
                "max_themes": model_params.get("max_themes", 5),
            },
            language_processor=create_text_processor(language=language),
        )

        self.category_analyzer = CategoryAnalyzer(
            categories=self.parameters.categories,
            llm=self.llm,
            config={"min_confidence": self.parameters.general.min_confidence},
            language_processor=create_text_processor(language=language),
        )

    def set_language(self, language: Optional[str] = None) -> None:
        """Update analyzer configuration for a new language."""
        try:
            language = (
                language
                or self.config_manager.get_config().languages.default_language
            )
            self.config_manager.config.languages.default_language = language
            self._init_analyzers()
            logger.info(f"Language updated to {language}")
        except Exception as e:
            logger.error(f"Error setting language: {e}")
            raise

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> CompleteAnalysisResult:
        """Run the analysis pipeline."""
        start_time = datetime.now()

        try:
            types_to_run = self._validate_analysis_types(analysis_types)
            tasks = []

            for analysis_type in types_to_run:
                coro = await self._create_analysis_task(
                    analysis_type, text, **kwargs
                )
                if coro:
                    tasks.append(asyncio.create_task(coro))

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )

            processed_results = self._process_analysis_results(
                results, types_to_run
            )

            return CompleteAnalysisResult(
                keywords=processed_results.get(
                    "keywords", self._create_error_result_by_type("keywords")
                ),
                themes=processed_results.get(
                    "themes", self._create_error_result_by_type("themes")
                ),
                categories=processed_results.get(
                    "categories",
                    self._create_error_result_by_type("categories"),
                ),
                language=self.config_manager.get_config().languages.default_language,
                success=all(r.success for r in processed_results.values()),
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise

    def _validate_analysis_types(
        self, types: Optional[List[str]] = None
    ) -> List[str]:
        """Validate and return analysis types to run."""
        if not types:
            return list(self.VALID_TYPES)

        invalid_types = set(types) - self.VALID_TYPES
        if invalid_types:
            raise ValueError(f"Invalid analysis types: {invalid_types}")

        return types

    async def _create_analysis_task(
        self, analysis_type: str, text: str, **kwargs
    ) -> Optional[Awaitable]:
        """Create analysis coroutine for a specified type."""
        try:
            if analysis_type == "keywords":
                return self.keyword_analyzer.analyze(text, **kwargs)
            elif analysis_type == "themes":
                return self.theme_analyzer.analyze(text, **kwargs)
            elif analysis_type == "categories":
                return self.category_analyzer.analyze(text, **kwargs)
            return None
        except Exception as e:
            logger.error(f"Error creating {analysis_type} task: {e}")
            return None

    def _process_analysis_results(
        self, results: List[Any], types: List[str]
    ) -> Dict[str, Any]:
        """Process and map analysis results to correct types."""
        processed = {}

        for analysis_type, result in zip(types, results):
            if isinstance(result, Exception):
                processed[analysis_type] = self._create_error_result_by_type(
                    analysis_type
                )
            else:
                processed[analysis_type] = result

        return processed

    def _create_error_result_by_type(self, analysis_type: str) -> Any:
        """Create appropriate error result for each type."""
        error_message = f"Analysis failed for {analysis_type}"
        if analysis_type == "keywords":
            return KeywordAnalysisResult(success=False, error=error_message)
        elif analysis_type == "themes":
            return ThemeAnalysisResult(success=False, error=error_message)
        elif analysis_type == "categories":
            return CategoryAnalysisResult(success=False, error=error_message)
