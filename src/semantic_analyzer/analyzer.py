# src/semantic_analyzer/analyzer.py

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Awaitable, Type, TypeVar, Protocol

import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.config import AnalyzerConfig
from src.core.language_processing import create_text_processor
from src.core.managers import EnvironmentManager

from src.loaders.parameter_handler import (
    ParameterHandler,
)

from src.config import ConfigManager
from src.core.config import AnalyzerConfig
from src.config.models import GlobalConfig

from src.loaders.models import CategoryConfig

from src.schemas import (
    CategoryAnalysisResult,
    CategoryOutput,
    CompleteAnalysisResult,
    KeywordAnalysisResult,
    ThemeAnalysisResult,
    ThemeOutput,
)

from src.excel_analysis.base import ExcelAnalysisBase
from src.excel_analysis.parameters import AnalysisParameters
from src.analyzers.excel_support import (
    ExcelKeywordAnalyzer,
    ExcelThemeAnalyzer,
    ExcelCategoryAnalyzer,
)

from src.utils.formatting_config import ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from src.utils.output_formatter import OutputDetail

from src.schemas import CompleteAnalysisResult
from src.core.llm.factory import create_llm
from FileUtils import FileUtils
from .base import BaseSemanticAnalyzer, ResultProcessingMixin, AnalyzerFactory

logger = logging.getLogger(__name__)

from tqdm import tqdm  # For progress reporting

# Type variables for generic analyzer types
T = TypeVar('T')

class AnalyzerMapping(Protocol):
    """Protocol for analyzer mapping configuration."""
    analyzer_class: Type[T]
    attr_name: str
    singular_name: str = ''

class BaseAnalyzerConfig:
    """Base configuration for analyzer mappings."""
    
    ANALYZER_TYPES = {
        "keywords": {
            "analyzer_class": KeywordAnalyzer,
            "attr_name": "keyword_analyzer",
            "singular_name": "keyword",
        },
        "themes": {
            "analyzer_class": ThemeAnalyzer,
            "attr_name": "theme_analyzer",
            "singular_name": "theme",
        },
        "categories": {
            "analyzer_class": CategoryAnalyzer,
            "attr_name": "category_analyzer",
            "singular_name": "category",
        },
    }

    EXCEL_ANALYZER_TYPES = {
        "keywords": {
            "analyzer_class": ExcelKeywordAnalyzer,
            "attr_name": "keyword_analyzer",
            "singular_name": "keyword",
        },
        "themes": {
            "analyzer_class": ExcelThemeAnalyzer,
            "attr_name": "theme_analyzer",
            "singular_name": "theme",
        },
        "categories": {
            "analyzer_class": ExcelCategoryAnalyzer,
            "attr_name": "category_analyzer",
            "singular_name": "category",
        },
    }

class ExcelSemanticAnalyzer(BaseSemanticAnalyzer, ResultProcessingMixin):
    """Enhanced SemanticAnalyzer with Excel support."""

    def __init__(
        self,
        content_file: Union[str, Path],
        parameter_file: Union[str, Path],
        llm: Optional[BaseChatModel] = None,
        file_utils: Optional[FileUtils] = None,
        output_config: Optional[ExcelOutputConfig] = None,
        **kwargs,
    ):
        """Initialize analyzer with Excel support and formatting."""
        if file_utils is None:
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                file_utils = components["file_utils"]
            except RuntimeError:
                raise ValueError(
                    "FileUtils instance must be provided to ExcelSemanticAnalyzer. "
                    "Use EnvironmentManager to get a shared FileUtils instance."
                )

        super().__init__(
            parameter_file=parameter_file,
            file_utils=file_utils,
            llm=llm,
            format_config=output_config,
            **kwargs
        )
        
        self.content_file = content_file
        self.parameter_file = parameter_file
        self._init_analyzers(use_excel=True)
        logger.info("Excel Semantic Analyzer initialized with formatting support")

    def _init_analyzers(self, use_excel: bool = False) -> None:
        """Initialize individual analyzers with proper configuration."""
        language = self.parameters.general.language
        logger.info(f"Initializing analyzers for language: {language}")

        analyzer_types = (
            BaseAnalyzerConfig.EXCEL_ANALYZER_TYPES if use_excel 
            else BaseAnalyzerConfig.ANALYZER_TYPES
        )

        for analyzer_type, config in analyzer_types.items():
            logger.debug(f"Initializing {analyzer_type} analyzer...")
            
            analyzer_kwargs = {
                "llm": self.llm,
                "file_utils": self.file_utils,
            }

            if use_excel:
                analyzer_kwargs.update({
                    "content_file": self.content_file,
                    "parameter_file": self.parameter_file,
                    "content_column": getattr(self, 'content_column', 'content'),
                })
            else:
                analyzer_kwargs["config"] = {"language": language}

            analyzer = config["analyzer_class"](**analyzer_kwargs)
            setattr(self, config["attr_name"], analyzer)

        logger.info("Successfully initialized all analyzers")

    async def analyze_excel(
        self,
        content_file: Optional[Union[str, Path, pd.DataFrame]] = None,
        analysis_types: Optional[List[str]] = None,
        batch_size: int = 10,
        save_results: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        format_config: Optional[ExcelOutputConfig] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Analyze Excel content with enhanced error handling."""
        start_time = datetime.now()

        try:
            self._verify_analyzers()

            if not hasattr(self, "formatter"):
                self.formatter = ExcelAnalysisFormatter(
                    file_utils=self.file_utils,
                    config=format_config or ExcelOutputConfig(detail_level=OutputDetail.SUMMARY),
                )

            if content_file is not None:
                if isinstance(content_file, pd.DataFrame):
                    content_df = content_file
                else:
                    content_df = self.file_utils.load_single_file(
                        file_path=content_file, input_type="raw"
                    )
                kwargs["content"] = content_df

            types_to_run = self._validate_analysis_types(analysis_types)
            logger.debug(f"Running analysis types: {types_to_run}")

            analysis_tasks = []
            for analyzer_type in types_to_run:
                config = BaseAnalyzerConfig.EXCEL_ANALYZER_TYPES[analyzer_type]
                analyzer = getattr(self, config["attr_name"])
                
                analysis_kwargs = {k: v for k, v in kwargs.items() if k != 'show_progress'}
                task = analyzer.analyze_excel(batch_size=batch_size, **analysis_kwargs)
                analysis_tasks.append(task)

            results = await asyncio.gather(*analysis_tasks)
            formatted_df = self.formatter.format_output(results, types_to_run)

            formatted_df["analysis_timestamp"] = datetime.now()
            formatted_df["processing_time"] = (datetime.now() - start_time).total_seconds()
            formatted_df["language"] = self.parameters.general.language

            if save_results and output_file:
                logger.info(f"Saving results to {output_file}...")
                self.formatter.save_excel_results(
                    formatted_df, output_file, include_summary=True
                )

            return formatted_df

        except Exception as e:
            logger.error(f"Excel analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Excel analysis failed: {str(e)}")

    def _verify_analyzers(self) -> None:
        """Verify that all required analyzers are properly initialized."""
        logger.debug("Verifying analyzer initialization...")
        
        analyzer_types = BaseAnalyzerConfig.EXCEL_ANALYZER_TYPES
        if not any(hasattr(self, config["attr_name"]) for config in analyzer_types.values()):
            raise ValueError("No analyzers have been initialized")
        
        for analyzer_type, config in analyzer_types.items():
            attr_name = config["attr_name"]
            if not hasattr(self, attr_name):
                logger.warning(f"{analyzer_type} analyzer not initialized")
                continue
                
            analyzer = getattr(self, attr_name)
            if not analyzer:
                raise ValueError(f"{analyzer_type} analyzer is None")
                
            logger.debug(f"Verified {analyzer_type} analyzer")
            
        logger.debug("All available analyzers verified successfully")

    def _combine_results(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine results from different analyzers."""
        combined_df = self.content.copy()

        for analysis_type, df in results.items():
            result_columns = [col for col in df.columns if col != self.content_column]
            for col in result_columns:
                new_col = f"{analysis_type}_{col}" if not col.startswith(analysis_type) else col
                combined_df[new_col] = df[col]

        return combined_df

    def _save_results(self, results_df: pd.DataFrame, output_file: Union[str, Path]) -> Path:
        """Save analysis results to Excel."""
        try:
            saved_files, _ = self.file_utils.save_data_to_storage(
                data={"Analysis Results": results_df},
                output_type="processed",
                file_name=output_file,
                output_filetype="xlsx",
                include_timestamp=True,
            )

            saved_path = Path(next(iter(saved_files.values())))
            logger.info(f"Saved results to: {saved_path}")
            return saved_path

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def _validate_analysis_types(self, types: Optional[List[str]] = None) -> List[str]:
        """Validate and return analysis types to run."""
        valid_types = {"keywords", "themes", "categories"}

        if not types:
            return list(valid_types)

        invalid_types = set(types) - valid_types
        if invalid_types:
            raise ValueError(
                f"Invalid analysis types: {invalid_types}. "
                f"Must be one of: {list(valid_types)}"
            )

        return types

class SemanticAnalyzer(BaseSemanticAnalyzer, ResultProcessingMixin):
    """Main interface for semantic text analysis."""

    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        llm: Optional[BaseChatModel] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        environment_manager: Optional[EnvironmentManager] = None,
        config_manager: Optional[ConfigManager] = None,
        **kwargs,
    ):
        """Initialize analyzer with parameters and components."""
        try:
            if environment_manager:
                components = environment_manager.get_components()
                file_utils = components["file_utils"]
                config_manager = components["config_manager"]

            # Store the base parameter file path
            self._base_parameter_file = parameter_file

            super().__init__(
                parameter_file=parameter_file,
                file_utils=file_utils,
                llm=llm,
                config_manager=config_manager,
                **kwargs
            )
            self._init_categories(categories)
            self._init_analyzers(use_excel=False)

            logger.info("Semantic Analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Semantic Analyzer: {e}")
            raise

    def _init_categories(self, categories: Optional[Dict[str, CategoryConfig]]) -> None:
        """Initialize categories with validation."""
        self.categories = categories or self.parameters.categories or {}
        if not self.categories:
            logger.warning("No categories available from parameters or explicit input")

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        **kwargs
    ) -> CompleteAnalysisResult:
        """Run semantic analysis on text.

        Args:
            text: Text to analyze
            analysis_types: List of analysis types to run
            **kwargs: Additional analysis parameters. Special parameters:
                - language: Language to use for analysis. Will be removed from kwargs.
                - timeout: Timeout for analysis in seconds. Will be removed from kwargs.

        Returns:
            CompleteAnalysisResult with all analysis results
        """
        try:
            start_time = datetime.now()
            
            # Handle special parameters
            if 'language' in kwargs:
                self.set_language(kwargs.pop('language'))
            
            # Remove timeout parameter if present since individual analyzers don't use it
            if 'timeout' in kwargs:
                kwargs.pop('timeout')
            
            self._verify_analyzers()
            try:
                types_to_run = self._validate_analysis_types(analysis_types)
            except ValueError as e:
                # Return a failed result with the validation error
                return CompleteAnalysisResult(
                    keywords=KeywordAnalysisResult(
                        language=self.parameters.general.language,
                        keywords=[],
                        compound_words=[],
                        domain_keywords={},
                        success=False
                    ),
                    themes=ThemeAnalysisResult(
                        language=self.parameters.general.language,
                        themes=[],
                        theme_hierarchy={},
                        success=False
                    ),
                    categories=CategoryAnalysisResult(
                        language=self.parameters.general.language,
                        matches=[],
                        success=False
                    ),
                    language=self.parameters.general.language,
                    processing_time=0.0,
                    success=False,
                    error=str(e),
                    metadata={
                        "analysis_timestamp": datetime.now().isoformat(),
                        "language": self.parameters.general.language,
                    }
                )

            logger.debug(f"Running analysis types: {types_to_run}")

            analysis_tasks = []
            for analyzer_type in types_to_run:
                config = BaseAnalyzerConfig.ANALYZER_TYPES[analyzer_type]
                analyzer = getattr(self, config["attr_name"])
                task = analyzer.analyze(text, **kwargs)
                analysis_tasks.append(task)

            results = await asyncio.gather(*analysis_tasks)
            processed_results = self._process_analysis_results(results, types_to_run)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Determine overall success based on individual analyzer successes
            overall_success = all(
                result.success
                for analyzer_type in types_to_run
                if (result := processed_results.get(analyzer_type)) is not None
            )

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
                success=overall_success,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "language": self.parameters.general.language,
                }
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def _verify_analyzers(self) -> None:
        """Verify that all required analyzers are properly initialized."""
        logger.debug("Verifying analyzer initialization...")
        
        analyzer_types = BaseAnalyzerConfig.ANALYZER_TYPES
        if not any(hasattr(self, config["attr_name"]) for config in analyzer_types.values()):
            raise ValueError("No analyzers have been initialized")
        
        for analyzer_type, config in analyzer_types.items():
            attr_name = config["attr_name"]
            if not hasattr(self, attr_name):
                logger.warning(f"{analyzer_type} analyzer not initialized")
                continue
                
            analyzer = getattr(self, attr_name)
            if not analyzer:
                raise ValueError(f"{analyzer_type} analyzer is None")
                
            logger.debug(f"Verified {analyzer_type} analyzer")
            
        logger.debug("All available analyzers verified successfully")

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

    def _init_analyzers(self, use_excel: bool = False) -> None:
        """Initialize analyzers with proper configuration."""
        try:
            # Set up language configuration
            language = self.parameters.general.language
            language_config = self.config_manager.get_language_config()

            # Build processor config
            processor_config = {
                "min_word_length": self.parameters.general.min_keyword_length,
                "include_compounds": self.parameters.general.include_compounds,
                **language_config.languages.get(language, {}),
            }

            # Create language processor
            self._initialize_text_processor()

            # Get base config
            base_config = self.analyzer_config.get_analyzer_config("base")
            base_config.update(
                {
                    "language": language,
                    "min_confidence": self.parameters.general.min_confidence,
                    "focus_on": self.parameters.general.focus_on,
                }
            )

            logger.debug(f"Initializing analyzers with config: {base_config}")

            # Initialize analyzers
            self.keyword_analyzer = KeywordAnalyzer(
                llm=self.llm,
                config=base_config,
                language_processor=self.language_processor,
            )

            self.theme_analyzer = ThemeAnalyzer(
                llm=self.llm,
                config=base_config,
                language_processor=self.language_processor,
            )

            self.category_analyzer = CategoryAnalyzer(
                llm=self.llm,
                config=base_config,
                language_processor=self.language_processor,
                categories=self.categories,
            )

            # Verify initialization
            self.verify_configuration()

            logger.info(f"All analyzers initialized for language: {language}")

        except Exception as e:
            logger.error(f"Failed to initialize analyzers: {e}")
            raise

    def _initialize_text_processor(self) -> None:
        """Initialize text processor for language handling."""
        try:
            self.language_processor = create_text_processor(
                language=self.parameters.general.language,
                config={
                    "min_keyword_length": self.parameters.general.min_keyword_length,
                    "include_compounds": self.parameters.general.include_compounds,
                },
                file_utils=self.file_utils
            )
            logger.debug(f"Initialized text processor for language: {self.parameters.general.language}")
        except Exception as e:
            logger.error(f"Failed to initialize text processor: {e}")
            raise

    def verify_configuration(self) -> None:
        """Verify all components are properly configured."""
        try:
            logger.info("Verifying analyzer configuration:")

            # Check language processor
            if not self.language_processor:
                raise ValueError("Language processor not initialized")

            # Check analyzers
            for analysis_type, config in BaseAnalyzerConfig.ANALYZER_TYPES.items():
                attr_name = config["attr_name"]
                analyzer_class = config["analyzer_class"]

                if not hasattr(self, attr_name):
                    raise ValueError(f"Missing required analyzer: {attr_name}")

                analyzer = getattr(self, attr_name)
                if not isinstance(analyzer, analyzer_class):
                    raise ValueError(
                        f"Invalid analyzer type for {attr_name}: {type(analyzer)}"
                    )

            # Log configuration details
            logger.info(f"Language: {self.parameters.general.language}")
            logger.info(f"Categories loaded: {len(self.categories)}")
            for name, cat in self.categories.items():
                logger.info(
                    f"  - {name}: {len(cat.keywords)} keywords, threshold: {cat.threshold}"
                )
            logger.info(
                f"Language processor: {type(self.language_processor).__name__}"
            )

        except Exception as e:
            logger.error(f"Configuration verification failed: {e}")
            raise

    def set_language(self, language: str) -> None:
        """Update analyzer configuration for new language."""
        try:
            # Get base parameter file name
            base_name = "parameters"
            if self._base_parameter_file:
                base_name = Path(self._base_parameter_file).stem.split("_")[0]
            elif hasattr(self, "parameter_handler") and hasattr(self.parameter_handler, "parameter_file"):
                base_name = Path(self.parameter_handler.parameter_file).stem.split("_")[0]

            # Construct language-specific parameter file path
            param_path = (
                self.file_utils.get_data_path("parameters")
                / f"{base_name}_{language}.xlsx"
            )
            logger.debug(f"Looking for parameter file: {param_path}")

            # Update parameter handler with new file
            self.parameter_handler = ParameterHandler(param_path)
            self.parameters = self.parameter_handler.get_parameters()

            # Create new config dictionary based on current config
            new_config = self.analyzer_config.config.copy()
            new_config["language"] = language
            self.config = new_config

            # Create new language processor
            self._initialize_text_processor()

            # Reinitialize analyzers with new configuration
            self._init_analyzers()
            logger.info(f"Language switched to {language}")

        except Exception as e:
            logger.error(f"Error setting language to {language}: {e}")
            raise

    @classmethod
    def from_excel(
        cls,
        content_file: Union[str, Path],
        parameter_file: Union[str, Path],
        **kwargs,
    ) -> ExcelSemanticAnalyzer:
        """Create Excel-aware analyzer instance.

        Args:
            content_file: Path to content Excel file
            parameter_file: Path to parameter Excel file
            **kwargs: Additional configuration options

        Returns:
            ExcelSemanticAnalyzer instance
        """
        return ExcelSemanticAnalyzer(
            content_file=content_file, parameter_file=parameter_file, **kwargs
        )

    async def _create_analysis_task(
        self,
        analysis_type: str,
        text: str,
        **kwargs,
    ) -> Optional[Awaitable]:
        """Create analysis coroutine for specified type."""
        try:
            config = BaseAnalyzerConfig.ANALYZER_TYPES.get(analysis_type)
            if not config:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return None

            attr_name = config["attr_name"]
            analyzer = getattr(self, attr_name)
            logger.debug(f"Creating {analysis_type} analysis task")

            # Get type-specific parameters from kwargs
            param_key = f"{config['singular_name']}_params"
            type_params = kwargs.get(param_key, {})

            return analyzer.analyze(text, **type_params)

        except Exception as e:
            logger.error(f"Error creating {analysis_type} task: {e}")
            return None

    def _convert_theme_output(self, output: ThemeOutput) -> ThemeAnalysisResult:
        """Convert ThemeOutput to ThemeAnalysisResult."""
        return ThemeAnalysisResult(
            themes=output.themes,
            theme_hierarchy=output.theme_hierarchy,
            language=output.language,
            success=output.success,
            error=output.error,
        )

    def _convert_category_output(
        self, output: CategoryOutput
    ) -> CategoryAnalysisResult:
        """Convert CategoryOutput to CategoryAnalysisResult."""
        return CategoryAnalysisResult(
            matches=output.categories,  # Note: CategoryOutput.categories maps to CategoryAnalysisResult.matches
            language=output.language,
            success=output.success,
            error=output.error,
        )

    def save_results(
        self,
        results: CompleteAnalysisResult,
        output_file: str,
        output_type: str = "processed",
    ) -> Path:
        """Save analysis results to Excel file."""
        # Create formatter with default config
        formatter = ExcelAnalysisFormatter(
            file_utils=self.file_utils,
            config=ExcelOutputConfig(detail_level=OutputDetail.DETAILED),
        )

        # Format results
        formatted_results = formatter.format_output(
            {
                "keywords": results.keywords,
                "themes": results.themes,
                "categories": results.categories,
            },
            ["keywords", "themes", "categories"],
        )

        # Create DataFrame
        results_df = pd.DataFrame([formatted_results])

        # Add metadata columns
        results_df["language"] = results.language
        results_df["processing_time"] = f"{results.processing_time:.2f}s"
        results_df["success"] = str(results.success)
        if results.error:
            results_df["error"] = results.error

        # Save using formatter
        return formatter.save_excel_results(
            results_df, output_file, include_summary=True
        )

    async def analyze_excel_content(
        self, text: str, **kwargs
    ) -> Dict[str, Any]:
        """Analyze single text content from Excel."""
        results = {}

        for analysis_type in BaseAnalyzerConfig.ANALYZER_TYPES:
            try:
                # Get analyzer configuration
                config = BaseAnalyzerConfig.ANALYZER_TYPES[analysis_type]
                attr_name = config["attr_name"]

                if not hasattr(self, attr_name):
                    raise AttributeError(f"Missing analyzer: {attr_name}")

                analyzer = getattr(self, attr_name)
                result = await analyzer.analyze(text)

                # Convert CategoryOutput to CategoryAnalysisResult
                if analysis_type == "categories" and isinstance(
                    result, CategoryOutput
                ):
                    results[analysis_type] = CategoryAnalysisResult(
                        matches=result.categories,  # Map categories to matches
                        language=result.language,
                        success=result.success,
                        error=result.error,
                    )
                else:
                    results[analysis_type] = result

                logger.debug(f"Completed {analysis_type} analysis")

            except Exception as e:
                logger.error(f"Error in {analysis_type} analysis: {e}")
                results[analysis_type] = self._create_error_result_by_type(
                    analysis_type
                )

        return results

    async def analyze_excel(
        self,
        content_file: Union[str, Path, pd.DataFrame],
        analysis_types: Optional[List[str]] = None,
        content_column: str = "content",
        batch_size: int = 10,
        save_results: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        format_config: Optional[ExcelOutputConfig] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Analyze Excel content with proper formatting.

        Args:
            analysis_types: List of analysis types to run. Can be singular or plural form:
                          'keyword'/'keywords', 'theme'/'themes', 'category'/'categories'
        """
        start_time = datetime.now()

        try:
            # Load content if needed
            if isinstance(content_file, (str, Path)):
                content_df = self.file_utils.load_single_file(
                    file_path=content_file, input_type="raw"
                )
            else:
                content_df = content_file

            if content_column not in content_df.columns:
                raise ValueError(f"Column '{content_column}' not found")

            # Update formatter config if provided
            if format_config:
                self.formatter = ExcelAnalysisFormatter(
                    file_utils=self.file_utils, config=format_config
                )

            # Validate and normalize analysis types
            types_to_run = self._validate_analysis_types(analysis_types)
            logger.info(f"Running analysis types: {types_to_run}")

            # Run analyses
            results_df = await self._run_analyses(
                types_to_run=types_to_run,
                content_df=content_df,
                content_column=content_column,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs,
            )

            # Add metadata
            results_df["analysis_timestamp"] = datetime.now()
            results_df["processing_time"] = (
                datetime.now() - start_time
            ).total_seconds()
            results_df["language"] = self.parameters.general.language

            if save_results and output_file:
                self.formatter.save_excel_results(
                    results_df=results_df,
                    output_file=output_file,
                    include_summary=True,
                )

            return results_df

        except Exception as e:
            logger.error(f"Excel analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Excel analysis failed: {str(e)}")

    async def _run_analyses(
        self,
        types_to_run: List[str],
        content_df: pd.DataFrame,
        content_column: str,
        batch_size: int,
        show_progress: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run individual analyses on Excel content."""
        analysis_results = []

        # Create progress bar for rows
        rows_iter = (
            tqdm(
                content_df.iterrows(),
                total=len(content_df),
                desc="Processing rows",
            )
            if show_progress
            else content_df.iterrows()
        )

        for idx, row in rows_iter:
            text = str(row[content_column])
            if show_progress:
                tqdm.write(f"\nProcessing row {idx + 1}")

            try:
                # Analyze single text content
                result = await self.analyze_excel_content(text, **kwargs)

                # Format result for the row
                formatted = self.formatter.format_output(
                    results=result, analysis_types=types_to_run
                )

                # Add original content
                formatted[content_column] = text
                analysis_results.append(formatted)

                if show_progress:
                    tqdm.write("✓ Row completed")

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                analysis_results.append({content_column: text, "error": str(e)})

        return pd.DataFrame(analysis_results)

    async def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 3,
        timeout: float = 30.0,
        **kwargs,
    ) -> List[CompleteAnalysisResult]:
        """Process multiple texts with controlled concurrency.

        Args:
            texts: List of texts to analyze
            batch_size: Maximum number of concurrent analyses
            timeout: Timeout per batch in seconds
            **kwargs: Additional analysis parameters

        Returns:
            List[CompleteAnalysisResult]: Results for each text
        """
        results = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_tasks = [
                    self.analyze(text, timeout=timeout, **kwargs)
                    for text in batch
                ]

                # Run batch with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=timeout * len(batch),
                    )
                    results.extend(self._process_batch_results(batch_results))
                except asyncio.TimeoutError:
                    logger.error(f"Batch {i//batch_size + 1} timed out")
                    results.extend(
                        [
                            self._create_error_result(
                                "Batch processing timed out", datetime.now()
                            )
                            for _ in batch
                        ]
                    )

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return [
                self._create_error_result(str(e), datetime.now()) for _ in texts
            ]

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
        # Create a mapping of valid types (both singular and plural)
        valid_types = set()
        valid_forms = []
        for plural, config in BaseAnalyzerConfig.ANALYZER_TYPES.items():
            singular = config["singular_name"]
            valid_types.add(plural)
            valid_types.add(singular)
            valid_forms.extend([plural, singular])

        if not types:
            return list(BaseAnalyzerConfig.ANALYZER_TYPES.keys())

        # Check for invalid types
        invalid_types = set(types) - valid_types
        if invalid_types:
            raise ValueError(
                f"Invalid analysis types: {invalid_types}. "
                f"Valid types are: {', '.join(valid_forms)}"
            )

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

        return list(set(normalized_types))

    def _create_error_result_by_type(self, analysis_type: str) -> Any:
        """Create appropriate error result for each type."""
        if analysis_type == "keywords":
            return KeywordAnalysisResult(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language=self.parameters.general.language,
                success=False,
                error="Analysis failed",
            )
        elif analysis_type == "themes":
            return ThemeAnalysisResult(
                themes=[],
                theme_hierarchy={},
                language=self.parameters.general.language,
                success=False,
                error="Analysis failed",
            )
        elif analysis_type == "categories":
            return CategoryAnalysisResult(
                matches=[],
                language=self.parameters.general.language,
                success=False,
                error="Analysis failed",
            )

    def _process_analysis_results(
        self,
        results: List[Any],
        types: List[str],
    ) -> Dict[str, Any]:
        """Process and convert analysis results to correct types."""
        processed = {}

        for analysis_type, result in zip(types, results):
            logger.debug(f"Processing {analysis_type} result: {result}")

            if isinstance(result, Exception):
                logger.error(f"Error in {analysis_type} analysis: {result}")
                processed[analysis_type] = self._create_error_result_by_type(
                    analysis_type
                )
            else:
                if analysis_type == "keywords":
                    processed[analysis_type] = KeywordAnalysisResult(
                        keywords=result.keywords,
                        compound_words=result.compound_words,  # Now available
                        domain_keywords=result.domain_keywords,
                        language=result.language
                        or self.parameters.general.language,
                        success=result.success,
                        error=result.error,
                    )
                elif analysis_type == "themes":
                    processed[analysis_type] = ThemeAnalysisResult(
                        themes=result.themes,
                        theme_hierarchy=result.theme_hierarchy,
                        language=result.language
                        or self.parameters.general.language,
                        success=result.success,
                        error=result.error,
                    )
                    # elif analysis_type == "categories":
                    #     processed[analysis_type] = CategoryAnalysisResult(
                    #         matches=result.categories,  # Map categories to matches
                    #         language=result.language
                    #         or self.parameters.general.language,
                    #         success=result.success,
                    #         error=result.error,
                    #     )
                elif analysis_type == "categories":
                    logger.debug(
                        f"Processing category result type: {type(result)}"
                    )
                    logger.debug(f"Raw category result: {result}")

                    # Explicitly handle CategoryOutput
                    category_result = None
                    try:
                        if isinstance(result, CategoryOutput):
                            category_result = CategoryAnalysisResult(
                                matches=result.categories,  # Map categories to matches
                                language=result.language
                                or self.parameters.general.language,
                                success=result.success,
                                error=result.error,
                            )
                        else:
                            logger.warning(
                                f"Unexpected category result type: {type(result)}"
                            )
                            category_result = self._create_error_result_by_type(
                                "categories"
                            )

                        processed[analysis_type] = category_result
                        logger.debug(
                            f"Processed category result: {category_result}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error processing category result: {e}",
                            exc_info=True,
                        )
                        processed[analysis_type] = (
                            self._create_error_result_by_type("categories")
                        )

        return processed

    def _process_batch_results(
        self,
        results: List[Any],
    ) -> List[CompleteAnalysisResult]:
        """Process batch analysis results."""
        processed = []
        for result in results:
            if isinstance(result, Exception):
                processed.append(
                    self._create_error_result(str(result), datetime.now())
                )
            else:
                processed.append(result)
        return processed

    def _create_error_result(
        self,
        error: str,
        start_time: datetime,
    ) -> CompleteAnalysisResult:
        """Create error result with proper structure."""
        processing_time = (datetime.now() - start_time).total_seconds()

        return CompleteAnalysisResult(
            keywords=KeywordAnalysisResult(
                success=False,
                error=error,
                language=self.parameters.general.language,
            ),
            themes=ThemeAnalysisResult(
                success=False,
                error=error,
                language=self.parameters.general.language,
            ),
            categories=CategoryAnalysisResult(
                success=False,
                error=error,
                language=self.parameters.general.language,
            ),
            language=self.parameters.general.language,
            success=False,
            error=error,
            processing_time=processing_time,
        )

    def change_llm_provider(
        self, provider: str, model: Optional[str] = None
    ) -> None:
        """Change LLM provider and optionally model."""
        try:
            # Get provider config
            config = self.config_manager.get_config()
            if not isinstance(config, dict):
                config = config.model_dump()
            
            model_config = config.get("models", {})
            provider_config = model_config.get("providers", {}).get(provider)

            if not provider_config:
                logger.error(f"Provider {provider} not found in configuration")
                return

            # Get model to use
            if model and model in provider_config.get("available_models", {}):
                new_model = model
            else:
                # Use provider's default model if none specified
                new_model = next(
                    iter(provider_config.get("available_models", {}).keys())
                )

            logger.debug(
                f"Changing provider to: {provider}, model: {new_model}"
            )

            # Update configurations at all levels
            # 1. Update analyzer config
            self.analyzer_config.config["models"]["default_provider"] = provider
            self.analyzer_config.config["models"]["default_model"] = new_model

            # 2. Update config manager's config (handle both dict and Pydantic model cases)
            if hasattr(self.config_manager, "_config"):
                if hasattr(self.config_manager._config, "model_dump"):
                    config_dict = self.config_manager._config.model_dump()
                    config_dict["models"]["default_provider"] = provider
                    config_dict["models"]["default_model"] = new_model
                    self.config_manager._config = GlobalConfig(**config_dict)
                else:
                    if "models" not in self.config_manager._config:
                        self.config_manager._config["models"] = {}
                    self.config_manager._config["models"][
                        "default_provider"
                    ] = provider
                    self.config_manager._config["models"][
                        "default_model"
                    ] = new_model

            # Force reinitialize LLM with new configuration
            self.llm = create_llm(
                analyzer_config=self.analyzer_config,
                provider=provider,
                model=new_model,
            )

            # Reinitialize analyzers with new LLM
            self._init_analyzers()

            # Verify change
            current_provider = self.analyzer_config.config["models"][
                "default_provider"
            ]
            current_model = self.analyzer_config.config["models"][
                "default_model"
            ]
            logger.info(
                f"LLM provider changed to {current_provider} using model {current_model}"
            )

        except Exception as e:
            logger.error(f"Failed to change LLM provider: {e}", exc_info=True)
            raise

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers.

        Returns:
            List of provider names
        """
        try:
            if not hasattr(self, "config_manager"):
                raise ValueError("config_manager not initialized")

            config = self.config_manager.get_config()
            if not isinstance(config, dict):
                config = config.model_dump()
            
            model_config = config.get("models", {})
            provider_names = list(model_config.get("providers", {}).keys())
            logger.debug(f"Found providers: {provider_names}")
            return provider_names
        except Exception as e:
            logger.error(f"Failed to get available providers: {e}", exc_info=True)
            return []
