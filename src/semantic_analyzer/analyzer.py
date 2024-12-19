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
            # Verify analyzers before starting
            self._verify_analyzers()

            # Initialize formatter if needed
            if not hasattr(self, "formatter"):
                self.formatter = ExcelAnalysisFormatter(
                    file_utils=self.file_utils,
                    config=format_config
                    or ExcelOutputConfig(detail_level=OutputDetail.SUMMARY),
                )

            # Load content if provided
            if content_file is not None:
                if isinstance(content_file, pd.DataFrame):
                    content_df = content_file
                else:
                    content_df = self.file_utils.load_single_file(
                        file_path=content_file, input_type="raw"
                    )
                kwargs["content"] = content_df

            # Validate analysis types
            types_to_run = self._validate_analysis_types(analysis_types)
            logger.debug(f"Running analysis types: {types_to_run}")

            # Run analyses
            analysis_results = await self._run_analyses(
                types_to_run, batch_size, show_progress, **kwargs
            )

            # Format results
            formatted_df = self.formatter.format_output(
                analysis_results, types_to_run
            )

            # Add metadata
            formatted_df["analysis_timestamp"] = datetime.now()
            formatted_df["processing_time"] = (
                datetime.now() - start_time
            ).total_seconds()
            formatted_df["language"] = self.parameters.general.language

            # Save if requested
            if save_results and output_file:
                logger.info(f"Saving results to {output_file}...")
                self.formatter.save_excel_results(
                    formatted_df, output_file, include_summary=True
                )

            return formatted_df

        except Exception as e:
            logger.error(f"Excel analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Excel analysis failed: {str(e)}")

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

    # Define valid types consistently using plural form
    VALID_TYPES = {"keywords", "themes", "categories"}

    # Mapping between analysis types and their analyzers
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
        """Initialize analyzer with parameters and components."""
        try:
            # Initialize core components
            self.file_utils = file_utils or FileUtils()

            # Load and validate parameters
            self._init_parameters(parameter_file)

            # Initialize categories
            self._init_categories(categories)

            # Initialize config and LLM
            self._init_config_and_llm(llm)

            # Initialize analyzers
            self._init_analyzers()

            # Initialize formatter
            self.formatter = ExcelAnalysisFormatter(
                file_utils=self.file_utils,
                config=kwargs.get("format_config")
                or ExcelOutputConfig(detail_level=OutputDetail.SUMMARY),
            )

            logger.info("Semantic Analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Semantic Analyzer: {e}")
            raise

    def _init_parameters(
        self, parameter_file: Optional[Union[str, Path]]
    ) -> None:
        """Initialize and validate parameters."""
        if parameter_file:
            param_path = (
                Path(parameter_file)
                if isinstance(parameter_file, Path)
                else self.file_utils.get_data_path("parameters")
                / parameter_file
            )
            if not param_path.exists():
                raise FileNotFoundError(
                    f"Parameter file not found: {param_path}"
                )

            self.parameter_handler = ParameterHandler(param_path)
            self.parameters = self.parameter_handler.get_parameters()
            logger.debug(f"Loaded parameters from {param_path}")
        else:
            logger.debug("Using default parameters")
            self.parameter_handler = ParameterHandler()
            self.parameters = self.parameter_handler.get_parameters()

    def _init_categories(
        self, categories: Optional[Dict[str, CategoryConfig]]
    ) -> None:
        """Initialize categories with validation."""
        self.categories = categories or self.parameters.categories or {}
        if not self.categories:
            logger.warning(
                "No categories available from parameters or explicit input"
            )

    def _init_config_and_llm(self, llm: Optional[BaseChatModel]) -> None:
        """Initialize configuration and LLM."""
        self.config_manager = ConfigManager(file_utils=self.file_utils)
        self.analyzer_config = AnalyzerConfig(
            file_utils=self.file_utils, config_manager=self.config_manager
        )

        self.llm = llm or create_llm(
            config_manager=self.config_manager,
            provider=self.analyzer_config.config.get("models", {}).get(
                "default_provider"
            ),
            model=self.analyzer_config.config.get("models", {}).get(
                "default_model"
            ),
        )

    def _init_analyzers(self) -> None:
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
            self.language_processor = create_text_processor(
                language=language,
                config=processor_config,
            )

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

    def verify_configuration(self) -> None:
        """Verify all components are properly configured."""
        try:
            logger.info("Verifying analyzer configuration:")

            # Check language processor
            if not self.language_processor:
                raise ValueError("Language processor not initialized")

            # Check analyzers
            for analysis_type, config in self.ANALYZER_MAPPING.items():
                attr_name = config["attr_name"]
                expected_class = config["class"]

                if not hasattr(self, attr_name):
                    raise ValueError(f"Missing required analyzer: {attr_name}")

                analyzer = getattr(self, attr_name)
                if not isinstance(analyzer, expected_class):
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

        # Add debug logs for categories
        if "categories" in (analysis_types or []):
            logger.debug(
                f"Category analyzer categories: {self.category_analyzer.categories}"
            )

        try:
            types_to_run = self._validate_analysis_types(analysis_types)
            tasks = []

            for analysis_type in types_to_run:
                # Skip category analysis if no categories configured
                if (
                    analysis_type == "categories"
                    and not self.category_analyzer.categories
                ):
                    logger.warning(
                        "Skipping category analysis - no categories configured"
                    )
                    continue

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

            # Categories are already set in analyzer initialization
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

        for analysis_type in self.VALID_TYPES:
            try:
                # Get analyzer configuration
                analyzer_config = self.ANALYZER_MAPPING[analysis_type]
                analyzer_attr = analyzer_config["attr_name"]

                if not hasattr(self, analyzer_attr):
                    raise AttributeError(f"Missing analyzer: {analyzer_attr}")

                analyzer = getattr(self, analyzer_attr)
                result = await analyzer.analyze(text)

                # Convert CategoryOutput to CategoryAnalysisResult
                if analysis_type == "categories" and isinstance(
                    result, CategoryOutput
                ):
                    results[analysis_type] = CategoryAnalysisResult(
                        matches=result.categories,
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
        if not types:
            return list(self.VALID_TYPES)

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
        invalid_types = set(normalized_types) - self.VALID_TYPES
        if invalid_types:
            valid_forms = []
            for valid in self.VALID_TYPES:
                singular = self.ANALYZER_MAPPING[valid]["singular"]
                valid_forms.extend([valid, singular])

            raise ValueError(
                f"Invalid analysis types: {invalid_types}. "
                f"Valid types are: {', '.join(valid_forms)}"
            )

        return normalized_types

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
