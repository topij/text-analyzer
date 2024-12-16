# src/semantic_analyzer/analyzer.py

# import asyncio
# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Awaitable, Dict, List, Optional, Union

# import pandas as pd
# from langchain_core.language_models import BaseChatModel

# from src.analyzers.category_analyzer import CategoryAnalyzer
# from src.analyzers.keyword_analyzer import KeywordAnalyzer
# from src.analyzers.theme_analyzer import ThemeAnalyzer

# from src.core.config import AnalyzerConfig
# from src.core.language_processing import create_text_processor
# from src.core.llm.factory import create_llm


# from FileUtils import FileUtils

# logger = logging.getLogger(__name__)


# # src/semantic_analyzer/analyzer.py

# import asyncio
# import logging
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Awaitable, Dict, List, Optional, Union

# import pandas as pd
# from langchain_core.language_models import BaseChatModel


# from src.core.llm.factory import create_llm
# from src.loaders.parameter_handler import ParameterHandler
# from src.schemas import (
#     CategoryAnalysisResult,
#     CategoryOutput,
#     CompleteAnalysisResult,
#     KeywordAnalysisResult,
#     ThemeAnalysisResult,
#     ThemeOutput,
# )
# from FileUtils import FileUtils


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

from src.loaders.parameter_handler import (  # get_parameter_file_path,
    ParameterHandler,
)
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
        **kwargs,
    ):
        """Initialize analyzer with parameters and components."""
        self.file_utils = file_utils or FileUtils()
        self.analyzer_config = AnalyzerConfig(file_utils=self.file_utils)
        self.llm = llm or create_llm(config=self.analyzer_config)

        # Store base parameter file path
        self._base_parameter_file = parameter_file

        # Initialize with default parameters
        self.parameter_handler = ParameterHandler(parameter_file)
        self.parameters = self.parameter_handler.get_parameters()
        self._init_analyzers()
        logger.info("Semantic analyzer initialization complete")

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

    def _init_analyzers(self) -> None:
        """Initialize analyzers with correct language and parameters."""
        # Create language processor first
        language = self.parameters.general.language

        # Build config dict for language processor
        config = {
            "min_word_length": self.parameters.general.min_keyword_length,
            "include_compounds": self.parameters.general.include_compounds,
        }

        self.language_processor = create_text_processor(
            language=language,
            config=config,
        )
        logger.debug(f"Created language processor for {language}")

        # Base config for all analyzers
        base_config = {
            "language": language,
            "min_confidence": self.parameters.general.min_confidence,
            "focus_on": self.parameters.general.focus_on,
        }

        # Initialize analyzers with proper config
        self.keyword_analyzer = KeywordAnalyzer(
            llm=self.llm,
            config={
                **base_config,
                "max_keywords": self.parameters.general.max_keywords,
                "min_keyword_length": self.parameters.general.min_keyword_length,
                "include_compounds": self.parameters.general.include_compounds,
                "weights": self.parameters.analysis_settings.weights.model_dump(),
            },
            language_processor=self.language_processor,
        )
        logger.debug("Initialized keyword analyzer")

        self.theme_analyzer = ThemeAnalyzer(
            llm=self.llm,
            config={
                **base_config,
                "max_themes": self.parameters.general.max_themes,
            },
            language_processor=self.language_processor,
        )
        logger.debug("Initialized theme analyzer")

        # Initialize category analyzer with parameter categories
        self.category_analyzer = CategoryAnalyzer(
            categories=self.parameters.categories,  # Pass categories from parameters
            llm=self.llm,
            config=base_config,
            language_processor=self.language_processor,
        )
        logger.debug(
            f"Initialized category analyzer with {len(self.parameters.categories)} categories"
        )

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
        try:
            if analysis_type == "keywords":
                return self.keyword_analyzer.analyze(
                    text, **kwargs.get("keyword_params", {})
                )
            elif analysis_type == "themes":
                return self.theme_analyzer.analyze(
                    text, **kwargs.get("theme_params", {})
                )
            elif analysis_type == "categories":
                return self.category_analyzer.analyze(
                    text, **kwargs.get("category_params", {})
                )
            return None
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

    # def save_results(
    #     self,
    #     results: CompleteAnalysisResult,
    #     output_file: str,
    #     output_type: str = "processed",
    # ) -> Path:
    #     """Save analysis results to Excel file.

    #     Args:
    #         results: Analysis results to save
    #         output_file: Output file name
    #         output_type: Output directory type

    #     Returns:
    #         Path: Path to saved file
    #     """
    #     # Convert results to DataFrames
    #     dfs = {
    #         "Keywords": self._format_keyword_results(results.keywords),
    #         "Themes": self._format_theme_results(results.themes),
    #         "Categories": self._format_category_results(results.categories),
    #         "Summary": self._create_summary(results),
    #     }

    #     # Save using FileUtils
    #     saved_files, _ = self.file_utils.save_data_to_storage(
    #         data=dfs,
    #         output_filetype="xlsx",
    #         file_name=output_file,
    #         output_type=output_type,
    #     )

    #     # Return path to saved file
    #     return Path(next(iter(saved_files.values())))

    # def _format_keyword_results(
    #     self, results: KeywordAnalysisResult
    # ) -> pd.DataFrame:
    #     """Format keyword results for Excel."""
    #     rows = []
    #     for kw in results.keywords:
    #         rows.append(
    #             {
    #                 "Keyword": kw.keyword,
    #                 "Score": f"{kw.score:.2f}",
    #                 "Domain": kw.domain or "",
    #                 "Compound Parts": (
    #                     ", ".join(kw.compound_parts)
    #                     if kw.compound_parts
    #                     else ""
    #                 ),
    #             }
    #         )
    #     return pd.DataFrame(rows)

    # def _format_theme_results(
    #     self, results: ThemeAnalysisResult
    # ) -> pd.DataFrame:
    #     """Format theme results for Excel."""
    #     rows = []
    #     for theme in results.themes:
    #         rows.append(
    #             {
    #                 "Theme": theme.name,
    #                 "Description": theme.description,
    #                 "Confidence": f"{theme.confidence:.2f}",
    #                 "Keywords": ", ".join(theme.keywords),
    #                 "Parent Theme": theme.parent_theme or "",
    #             }
    #         )
    #     return pd.DataFrame(rows)

    # def _format_category_results(
    #     self, results: CategoryAnalysisResult
    # ) -> pd.DataFrame:
    #     """Format category results for Excel."""
    #     rows = []
    #     for cat in results.matches:
    #         rows.append(
    #             {
    #                 "Category": cat.name,
    #                 "Confidence": f"{cat.confidence:.2f}",
    #                 "Description": cat.description,
    #                 "Evidence": "\n".join(e.text for e in cat.evidence),
    #             }
    #         )
    #     return pd.DataFrame(rows)

    # def _create_summary(self, results: CompleteAnalysisResult) -> pd.DataFrame:
    #     """Create summary sheet for Excel output."""
    #     return pd.DataFrame(
    #         [
    #             {
    #                 "Language": results.language,
    #                 "Success": results.success,
    #                 "Error": results.error or "",
    #                 "Processing Time": f"{results.processing_time:.2f}s",
    #                 "Keywords Found": len(results.keywords.keywords),
    #                 "Themes Found": len(results.themes.themes),
    #                 "Categories Found": len(results.categories.matches),
    #             }
    #         ]
    #     )

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