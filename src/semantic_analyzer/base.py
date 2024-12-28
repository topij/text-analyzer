"""Base classes and mixins for semantic analysis."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

from langchain_core.language_models import BaseChatModel

from src.core.config import AnalyzerConfig
from src.config import ConfigManager
from src.loaders.parameter_handler import ParameterHandler
from src.utils.formatting_config import ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from src.utils.output_formatter import OutputDetail
from src.core.llm.factory import create_llm
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class ResultProcessingMixin:
    """Mixin for processing analysis results."""
    
    def _process_analysis_results(self, results: List[Any], types: List[str]) -> Dict:
        """Process and convert analysis results."""
        processed_results = {}
        for result, analysis_type in zip(results, types):
            if isinstance(result, Exception):
                processed_results[analysis_type] = self._create_type_error_result(
                    analysis_type, str(result)
                )
            else:
                processed_results[analysis_type] = result
        return processed_results
    
    def _create_type_error_result(self, analysis_type: str, error: str):
        """Create error result for specific analysis type."""
        raise NotImplementedError("Subclasses must implement this method")


class AnalyzerFactory:
    """Factory for creating different types of analyzers."""
    
    @staticmethod
    def create_analyzer(
        analyzer_type: str,
        analyzer_class: Type,
        llm: BaseChatModel,
        file_utils: FileUtils,
        **kwargs
    ) -> Any:
        """Create and return an analyzer instance."""
        return analyzer_class(
            llm=llm,
            file_utils=file_utils,
            **kwargs
        )


class BaseSemanticAnalyzer:
    """Base class for semantic analysis functionality."""
    
    VALID_TYPES = {"keywords", "themes", "categories"}
    
    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs,
    ):
        """Initialize base analyzer components."""
        self.file_utils = file_utils or FileUtils()
        self._init_parameters(parameter_file)
        self._init_config_and_llm(llm)
        self._init_formatter(kwargs.get("format_config"))
        
    def _init_parameters(self, parameter_file: Optional[Union[str, Path]]) -> None:
        """Initialize and validate parameters."""
        if parameter_file:
            param_path = (
                Path(parameter_file)
                if isinstance(parameter_file, Path)
                else self.file_utils.get_data_path("parameters") / parameter_file
            )
            if not param_path.exists():
                raise FileNotFoundError(f"Parameter file not found: {param_path}")

            self.parameter_handler = ParameterHandler(param_path)
            self.parameters = self.parameter_handler.get_parameters()
            logger.debug(f"Loaded parameters from {param_path}")
        else:
            logger.debug("Using default parameters")
            self.parameter_handler = ParameterHandler()
            self.parameters = self.parameter_handler.get_parameters()
    
    def _init_config_and_llm(self, llm: Optional[BaseChatModel]) -> None:
        """Initialize configuration and LLM."""
        self.config_manager = ConfigManager(file_utils=self.file_utils)
        self.analyzer_config = AnalyzerConfig(
            file_utils=self.file_utils,
            config_manager=self.config_manager
        )
        
        self.llm = llm or create_llm(
            config_manager=self.config_manager,
            provider=self.analyzer_config.config.get("models", {}).get("default_provider"),
            model=self.analyzer_config.config.get("models", {}).get("default_model"),
        )
    
    def _init_formatter(self, format_config: Optional[ExcelOutputConfig] = None) -> None:
        """Initialize the Excel formatter."""
        self.formatter = ExcelAnalysisFormatter(
            file_utils=self.file_utils,
            config=format_config or ExcelOutputConfig(detail_level=OutputDetail.SUMMARY),
        )
    
    def _validate_analysis_types(self, types: Optional[List[str]] = None) -> List[str]:
        """Validate and return analysis types to run."""
        if not types:
            return list(self.VALID_TYPES)
        
        invalid_types = set(types) - self.VALID_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid analysis types: {invalid_types}. "
                f"Must be one of: {list(self.VALID_TYPES)}"
            )
        return types
