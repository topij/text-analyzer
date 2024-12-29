"""Base classes and mixins for semantic analysis."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Protocol
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from src.core.config import AnalyzerConfig
from src.config import ConfigManager
from src.loaders.parameter_handler import ParameterHandler
from src.utils.formatting_config import ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from src.utils.output_formatter import OutputDetail
from src.core.llm.factory import create_llm
from src.core.managers import EnvironmentManager
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base class for analysis errors."""
    pass


class ConfigurationError(AnalysisError):
    """Raised when there's an error in configuration."""
    pass


class AnalysisTypeError(AnalysisError):
    """Raised when an invalid analysis type is requested."""
    pass


class ResultProcessor(Protocol):
    """Protocol for result processing functionality."""
    
    def process_results(self, results: List[Any], types: List[str]) -> Dict:
        """Process analysis results."""
        ...

    def create_error_result(self, analysis_type: str, error: str) -> Any:
        """Create error result for specific analysis type."""
        ...


class ResultProcessingMixin:
    """Mixin for processing analysis results."""
    
    def _process_analysis_results(self, results: List[Any], types: List[str]) -> Dict:
        """Process and convert analysis results.
        
        Args:
            results: List of raw analysis results
            types: List of analysis types corresponding to results
            
        Returns:
            Dict mapping analysis types to processed results
        
        Raises:
            AnalysisError: If processing fails
        """
        try:
            processed_results = {}
            for result, analysis_type in zip(results, types):
                if isinstance(result, Exception):
                    processed_results[analysis_type] = self._create_type_error_result(
                        analysis_type, str(result)
                    )
                else:
                    processed_results[analysis_type] = result
            return processed_results
        except Exception as e:
            raise AnalysisError(f"Failed to process results: {e}") from e
    
    @abstractmethod
    def _create_type_error_result(self, analysis_type: str, error: str) -> Any:
        """Create error result for specific analysis type."""
        pass


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
        """Create and return an analyzer instance.
        
        Args:
            analyzer_type: Type of analyzer to create
            analyzer_class: Class to instantiate
            llm: Language model instance
            file_utils: File utilities instance
            **kwargs: Additional configuration options
            
        Returns:
            Configured analyzer instance
            
        Raises:
            ConfigurationError: If analyzer creation fails
        """
        try:
            return analyzer_class(
                llm=llm,
                file_utils=file_utils,
                **kwargs
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create analyzer: {e}") from e


class BaseSemanticAnalyzer(ABC):
    """Base class for semantic analysis functionality."""
    
    VALID_TYPES = {"keywords", "themes", "categories"}
    
    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs,
    ):
        """Initialize base analyzer components.
        
        Args:
            parameter_file: Path to parameter file
            file_utils: FileUtils instance (required)
            llm: Language model instance
            **kwargs: Additional configuration options
            
        Raises:
            ConfigurationError: If initialization fails
            ValueError: If file_utils is None and EnvironmentManager not initialized
        """
        try:
            # Try to get components from EnvironmentManager first
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                self.file_utils = components["file_utils"]
                self.config_manager = components["config_manager"]
            except RuntimeError:
                if file_utils is None:
                    raise ValueError(
                        "FileUtils instance must be provided to BaseSemanticAnalyzer. "
                        "Use EnvironmentManager to get a shared FileUtils instance."
                    )
                self.file_utils = file_utils
                raise ValueError(
                    "ConfigManager must be provided to BaseSemanticAnalyzer. "
                    "Use EnvironmentManager to get a shared ConfigManager instance."
                )

            # Create analyzer config after we have file_utils and config_manager
            self.analyzer_config = AnalyzerConfig(
                file_utils=self.file_utils,
                config_manager=self.config_manager
            )

            self._init_parameters(parameter_file)
            self._init_llm(llm)
            self._init_formatter(kwargs.get("format_config"))
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize analyzer: {e}") from e
    
    def _init_parameters(self, parameter_file: Optional[Union[str, Path]]) -> None:
        """Initialize and validate parameters."""
        if parameter_file:
            param_path = (
                Path(parameter_file)
                if isinstance(parameter_file, Path)
                else self.file_utils.get_data_path("parameters") / parameter_file
            )
            if not param_path.exists():
                raise ConfigurationError(f"Parameter file not found: {param_path}")

            self.parameter_handler = ParameterHandler(
                file_path=param_path,
                file_utils=self.file_utils
            )
            self.parameters = self.parameter_handler.get_parameters()
            logger.debug(f"Loaded parameters from {param_path}")
        else:
            logger.debug("Using default parameters")
            self.parameter_handler = ParameterHandler(file_utils=self.file_utils)
            self.parameters = self.parameter_handler.get_parameters()
    
    def _init_llm(self, llm: Optional[BaseChatModel]) -> None:
        """Initialize LLM."""
        try:
            # Get config and ensure it's a dict
            config = self.analyzer_config.get_config()
            if not isinstance(config, dict):
                config = config.model_dump()

            # Get model settings
            model_config = config.get("models", {})
            provider = model_config.get("default_provider")
            model = model_config.get("default_model")

            logger.debug(f"Initializing LLM with provider={provider}, model={model}")
            
            self.llm = llm or create_llm(
                analyzer_config=self.analyzer_config,
                provider=provider,
                model=model,
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize LLM: {e}") from e
    
    def _init_formatter(self, format_config: Optional[ExcelOutputConfig] = None) -> None:
        """Initialize the Excel formatter."""
        try:
            self.formatter = ExcelAnalysisFormatter(
                file_utils=self.file_utils,
                config=format_config or ExcelOutputConfig(detail_level=OutputDetail.SUMMARY),
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize formatter: {e}") from e
    
    def _validate_analysis_types(self, types: Optional[List[str]] = None) -> List[str]:
        """Validate and return analysis types to run.
        
        Args:
            types: List of analysis types to validate
            
        Returns:
            List of validated analysis types
            
        Raises:
            AnalysisTypeError: If any type is invalid
        """
        if not types:
            return list(self.VALID_TYPES)
        
        invalid_types = set(types) - self.VALID_TYPES
        if invalid_types:
            raise AnalysisTypeError(
                f"Invalid analysis types: {invalid_types}. "
                f"Must be one of: {list(self.VALID_TYPES)}"
            )
        return types
    
    @abstractmethod
    def analyze(self, text: str, **kwargs) -> Any:
        """Analyze text content.
        
        Args:
            text: Text to analyze
            **kwargs: Additional analysis parameters
            
        Returns:
            Analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        pass
