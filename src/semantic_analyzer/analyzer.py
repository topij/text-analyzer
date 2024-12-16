# src/semantic_analyzer/analyzer.py

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Awaitable

import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.config import AnalyzerConfig
from src.core.language_processing import create_text_processor

from src.loaders.parameter_handler import (
    ParameterHandler,
)

from src.config import ConfigManager
from src.core.config import AnalyzerConfig

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

logger = logging.getLogger(__name__)

from tqdm import tqdm  # For progress reporting


class ExcelSemanticAnalyzer(ExcelAnalysisBase):
    """Enhanced SemanticAnalyzer with Excel support."""

    ANALYZER_MAPPING = {
        "keywords": ("keyword_analyzer", ExcelKeywordAnalyzer),
        "themes": ("theme_analyzer", ExcelThemeAnalyzer),
        "categories": ("category_analyzer", ExcelCategoryAnalyzer),
    }

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
        # Initialize base components
        super().__init__(
            content_file=content_file,
            parameter_file=parameter_file,
            file_utils=file_utils,
            **kwargs,
        )

        # Create LLM if needed
        self.llm = llm or create_llm()

        # Initialize formatter with configuration
        self.formatter = ExcelAnalysisFormatter(
            file_utils=self.file_utils,
            config=output_config
            or ExcelOutputConfig(detail_level=OutputDetail.SUMMARY),
        )

        # Initialize analyzers
        self._init_analyzers()
        logger.info(
            "Excel Semantic Analyzer initialized with formatting support"
        )

    def _init_analyzers(self) -> None:
        """Initialize individual analyzers with proper configuration."""
        language = self.parameters.parameters.general.language
        logger.info(f"Initializing analyzers for language: {language}")

        for analyzer_type, (
            attr_name,
            analyzer_class,
        ) in self.ANALYZER_MAPPING.items():
            logger.debug(f"Initializing {analyzer_type} analyzer...")
            analyzer = analyzer_class(
                content_file=self.content,
                parameter_file=self.parameter_file,
                llm=self.llm,
                content_column=self.content_column,
                file_utils=self.file_utils,
            )
            setattr(self, attr_name, analyzer)

        logger.info(f"Successfully initialized all analyzers")

    async def analyze_excel(
        self,
        analysis_types: Optional[List[str]] = None,
        batch_size: int = 10,
        save_results: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        format_config: Optional[ExcelOutputConfig] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Analyze Excel content with enhanced formatting.

        Args:
            analysis_types: List of analysis types to perform
            batch_size: Size of processing batches
            save_results: Whether to save results to Excel
            output_file: Optional output file path
            show_progress: Whether to show progress bars
            format_config: Optional formatting configuration
            **kwargs: Additional analysis parameters

        Returns:
            DataFrame with formatted analysis results
        """
        start_time = datetime.now()

        try:
            # Validate analysis types
            types_to_run = self._validate_analysis_types(analysis_types)
            logger.info(f"Running analysis types: {types_to_run}")

            # Initialize results storage
            analysis_results = {}

            # Create progress bar if requested
            types_iter = (
                tqdm(types_to_run, desc="Analysis Progress")
                if show_progress
                else types_to_run
            )

            # Run individual analyses
            for analysis_type in types_iter:
                logger.info(f"Running {analysis_type} analysis...")
                attr_name = self.ANALYZER_MAPPING[analysis_type][0]
                analyzer = getattr(self, attr_name)

                # Process with progress reporting
                if show_progress:
                    tqdm.write(f"\nProcessing {analysis_type.capitalize()}...")

                # Run analysis
                result_df = await analyzer.analyze_excel(
                    batch_size=batch_size, **kwargs
                )
                analysis_results[analysis_type] = result_df

                if show_progress:
                    tqdm.write(f"✓ Completed {analysis_type} analysis")

            # Format results using enhanced formatter
            logger.info("Formatting results...")
            formatted_df = self.formatter.format_analysis_results(
                analysis_results=analysis_results,
                original_content=self.content,
                content_column=self.content_column,
            )

            # Add analysis metadata
            formatted_df["analysis_timestamp"] = datetime.now()
            formatted_df["processing_time"] = (
                datetime.now() - start_time
            ).total_seconds()
            formatted_df["language"] = (
                self.parameters.parameters.general.language
            )

            # Save if requested
            if save_results and output_file:
                logger.info("Saving formatted results...")
                saved_path = self.formatter.save_excel_results(
                    results_df=formatted_df,
                    output_file=output_file,
                    include_summary=True,
                )
                logger.info(f"Results saved to: {saved_path}")

            # Display summary if progress shown
            if show_progress:
                self.formatter.display_results_summary(formatted_df)

            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis completed in {total_time:.2f} seconds")

            return formatted_df

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise

    def _combine_results(
        self, results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Combine results from different analyzers.

        Args:
            results: Dict of DataFrames from each analyzer

        Returns:
            Combined DataFrame with all results
        """
        # Start with content DataFrame
        combined_df = self.content.copy()

        # Add results from each analyzer
        for analysis_type, df in results.items():
            # Get result columns (excluding content column)
            result_columns = [
                col for col in df.columns if col != self.content_column
            ]

            # Add columns with analyzer prefix
            for col in result_columns:
                new_col = (
                    f"{analysis_type}_{col}"
                    if not col.startswith(analysis_type)
                    else col
                )
                combined_df[new_col] = df[col]

        return combined_df

    def _save_results(
        self, results_df: pd.DataFrame, output_file: Union[str, Path]
    ) -> Path:
        """Save analysis results to Excel.

        Args:
            results_df: DataFrame with analysis results
            output_file: Output file path

        Returns:
            Path to saved file
        """
        try:
            # Save using FileUtils
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

    def _validate_analysis_types(
        self, types: Optional[List[str]] = None
    ) -> List[str]:
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


class SemanticAnalyzer:
    """Main interface for semantic text analysis."""

    VALID_TYPES = {"keywords", "themes", "categories"}

    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        llm: Optional[BaseChatModel] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        **kwargs,
    ):
        """Initialize analyzer with parameters and components."""
        self.file_utils = file_utils or FileUtils()

        # Resolve parameter file path
        if parameter_file:
            param_path = (
                Path(parameter_file)
                if isinstance(parameter_file, Path)
                else self.file_utils.get_data_path("parameters")
                / parameter_file
            )
            logger.debug(f"Using parameter file: {param_path}")
            if not param_path.exists():
                raise FileNotFoundError(
                    f"Parameter file not found: {param_path}"
                )
        else:
            logger.warning("No parameter file specified, will use defaults")
            param_path = None

        # Initialize configuration management
        self.config_manager = ConfigManager(file_utils=self.file_utils)
        self.analyzer_config = AnalyzerConfig(
            file_utils=self.file_utils, config_manager=self.config_manager
        )

        # Get model configuration
        model_config = self.config_manager.get_model_config()

        # Create LLM if not provided
        self.llm = llm or create_llm(
            config_manager=self.config_manager,
            provider=model_config.default_provider,
            model=model_config.default_model,
        )

        # Initialize with parameters
        self.parameter_handler = ParameterHandler(param_path)
        self.parameters = self.parameter_handler.get_parameters()

        logger.debug(f"Loaded parameter file: {param_path}")
        logger.debug(f"Parameter contents: {self.parameters}")

        # Store categories (from params or explicit)
        self.categories = categories or self.parameters.categories
        if not self.categories:
            logger.warning(
                "No categories available from parameters or explicit input"
            )

        # Initialize analyzers with proper configuration
        self._init_analyzers()

    def _init_analyzers(self) -> None:
        """Initialize analyzers with correct language and parameters."""
        language = self.parameters.general.language
        language_config = self.config_manager.get_language_config()

        logger.debug(f"Parameters loaded: {self.parameters}")  # Debug log
        logger.debug(
            f"Categories from parameters: {self.parameters.categories}"
        )

        # Build config dict for language processor
        processor_config = {
            "min_word_length": self.parameters.general.min_keyword_length,
            "include_compounds": self.parameters.general.include_compounds,
            **language_config.languages.get(language, {}),
        }

        # Create language processor
        self.language_processor = create_text_processor(
            language=language,
            config=processor_config,
        )

        # Get base config from analyzer config
        base_config = self.analyzer_config.get_analyzer_config("base")
        base_config.update(
            {
                "language": language,
                "min_confidence": self.parameters.general.min_confidence,
                "focus_on": self.parameters.general.focus_on,
            }
        )

        logger.debug(f"Initializing analyzers with config: {base_config}")

        # Initialize all analyzers with proper configuration
        self.keyword_analyzer = KeywordAnalyzer(
            llm=self.llm,
            config=base_config,
            language_processor=self.language_processor,
        )
        logger.debug("Keyword analyzer initialized")

        self.theme_analyzer = ThemeAnalyzer(
            llm=self.llm,
            config=base_config,
            language_processor=self.language_processor,
        )
        logger.debug("Theme analyzer initialized")

        # Use categories from parameters or explicitly provided oneses
        categories = self.categories or self.parameters.categories
        logger.debug(f"Using categories for analyzer: {categories}")

        self.category_analyzer = CategoryAnalyzer(
            llm=self.llm,
            config=base_config,
            language_processor=self.language_processor,
            categories=categories,
        )
        logger.debug(
            f"Category analyzer initialized with {len(categories)} categories"
        )

        # Log successful initialization
        logger.info(f"All analyzers initialized for language: {language}")

        # Add validation to ensure all analyzers are ready
        self.verify_analyzers()

    def verify_analyzers(self) -> None:
        """Verify all analyzers are properly initialized."""
        required_analyzers = {
            "keyword_analyzer": KeywordAnalyzer,
            "theme_analyzer": ThemeAnalyzer,
            "category_analyzer": CategoryAnalyzer,
        }

        for name, cls in required_analyzers.items():
            if not hasattr(self, name):
                raise ValueError(f"Missing required analyzer: {name}")

            analyzer = getattr(self, name)
            if not isinstance(analyzer, cls):
                raise ValueError(
                    f"Invalid analyzer type for {name}: {type(analyzer)}"
                )

            if not hasattr(analyzer, "analyze"):
                raise ValueError(
                    f"Analyzer {name} missing required 'analyze' method"
                )

            logger.debug(f"Verified {name} initialization")

    def verify_configuration(self) -> None:
        """Verify all components are properly configured."""
        logger.info("Verifying analyzer configuration:")
        logger.info(f"Language: {self.parameters.general.language}")
        logger.info(f"Categories loaded: {len(self.parameters.categories)}")
        for name, cat in self.parameters.categories.items():
            logger.info(
                f"  - {name}: {len(cat.keywords)} keywords, threshold: {cat.threshold}"
            )
        logger.info(
            f"Language processor: {type(self.language_processor).__name__}"
        )
        logger.info(f"Base config: {self.analyzer_config}")

    def set_language(self, language: str) -> None:
        """Update analyzer configuration for new language."""
        try:
            # Get base parameter file name
            base_name = "parameters"
            if self._base_parameter_file:
                base_name = Path(self._base_parameter_file).stem.split("_")[0]

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
            self.language_processor = create_text_processor(
                language=language, config=self.config
            )

            # Reinitialize analyzers with new configuration
            self._init_analyzers()
            logger.info(f"Language switched to {language}")

        except Exception as e:
            logger.error(f"Error setting language to {language}: {e}")
            raise

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        language: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> CompleteAnalysisResult:
        """Run analysis pipeline."""
        start_time = datetime.now()

        try:
            # Update language if specified
            if language:
                self.set_language(language)

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
                language=self.parameters.general.language,
                success=all(r.success for r in processed_results.values()),
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return self._create_error_result(str(e), start_time)

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
        analyzers = {
            "keywords": (self.keyword_analyzer, "keyword_params"),
            "themes": (self.theme_analyzer, "theme_params"),
            "categories": (self.category_analyzer, "category_params"),
        }

        try:
            if analysis_type not in analyzers:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return None

            analyzer, param_key = analyzers[analysis_type]
            logger.debug(f"Creating {analysis_type} analysis task")

            # Get type-specific parameters from kwargs
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

    async def analyze_excel(
        self,
        analysis_types: Optional[List[str]] = None,
        batch_size: int = 10,
        save_results: bool = True,
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        format_config: Optional[ExcelOutputConfig] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Analyze Excel content with enhanced formatting."""
        start_time = datetime.now()

        try:
            # Update formatter config if provided
            if format_config:
                self.formatter = ExcelAnalysisFormatter(
                    file_utils=self.file_utils, config=format_config
                )

            # Run analyses
            types_to_run = self._validate_analysis_types(analysis_types)
            analysis_results = await self._run_analyses(
                types_to_run, batch_size, show_progress, **kwargs
            )

            # Format results using formatter
            formatted_df = self.formatter.format_output(
                analysis_results, types_to_run
            )

            # Add metadata
            formatted_df["analysis_timestamp"] = datetime.now()
            formatted_df["processing_time"] = (
                datetime.now() - start_time
            ).total_seconds()
            formatted_df["language"] = (
                self.parameters.parameters.general.language
            )

            # Save results if requested
            if save_results and output_file:
                self.formatter.save_excel_results(
                    formatted_df, output_file, include_summary=True
                )

            return formatted_df

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise

    async def _run_analyses(
        self,
        types_to_run: List[str],
        batch_size: int,
        show_progress: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run individual analyses."""
        analysis_results = {}

        # Create progress bar if requested
        types_iter = (
            tqdm(types_to_run, desc="Analysis Progress")
            if show_progress
            else types_to_run
        )

        for analysis_type in types_iter:
            logger.info(f"Running {analysis_type} analysis...")
            attr_name = self.ANALYZER_MAPPING[analysis_type][0]
            analyzer = getattr(self, attr_name)

            if show_progress:
                tqdm.write(f"\nProcessing {analysis_type.capitalize()}...")

            # Run analysis
            result = await analyzer.analyze_excel(
                batch_size=batch_size, **kwargs
            )
            analysis_results[analysis_type] = result

            if show_progress:
                tqdm.write(f"✓ Completed {analysis_type} analysis")

        return analysis_results

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
        """Validate and return analysis types to run."""
        if not types:
            return list(self.VALID_TYPES)

        invalid_types = set(types) - self.VALID_TYPES
        if invalid_types:
            raise ValueError(f"Invalid analysis types: {invalid_types}")

        return types

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
            if isinstance(result, Exception):
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
                elif analysis_type == "categories":
                    processed[analysis_type] = CategoryAnalysisResult(
                        matches=result.categories,  # Map categories to matches
                        language=result.language
                        or self.parameters.general.language,
                        success=result.success,
                        error=result.error,
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

    def _create_type_error_result(self, analysis_type: str, error: str) -> Any:
        """Create error result for specific analysis type."""
        base_error = {
            "success": False,
            "error": error,
            "language": self.parameters.general.language,
            "keywords": [],
            "compound_words": [],
            "domain_keywords": {},
        }

        if analysis_type == "keywords":
            return KeywordAnalysisResult(**base_error)
        elif analysis_type == "themes":
            return ThemeAnalysisResult(
                themes=[], theme_hierarchy={}, **base_error
            )
        elif analysis_type == "categories":
            return CategoryAnalysisResult(matches=[], **base_error)

        return base_error

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        # Clean up resources
        if hasattr(self, "llm") and hasattr(self.llm, "aclose"):
            await self.llm.aclose()
