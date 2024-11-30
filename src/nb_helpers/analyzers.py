# src/nb_helpers/analyzers.py

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.analyzers.base import AnalyzerOutput, BaseAnalyzer
from src.core.language_parameters import LanguageParameterManager
from src.core.language_processing import create_text_processor
from src.core.llm.factory import create_llm
from src.loaders.models import GeneralParameters, ParameterSet
from src.loaders.parameter_handler import ParameterHandler  # Updated import
from src.nb_helpers.base import DebugMixin
from src.nb_helpers.testers import CategoryTester, KeywordTester, ThemeTester
from src.nb_helpers.visualizers import (  # , format_confidence_bar,
    create_analysis_summary,
)
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


@dataclass
class AnalysisOptions:
    """Options for analysis output."""

    show_confidence: bool = True
    show_evidence: bool = True
    show_keywords: bool = True
    show_raw_data: bool = True
    debug_mode: bool = True
    language: Optional[str] = None
    parameter_file: Optional[str] = None


class TextAnalyzer(DebugMixin):
    def __init__(self, options: Optional[AnalysisOptions] = None):
        """Initialize analyzer with options."""
        self.options = options or AnalysisOptions()
        # Use ParameterHandler instead of LanguageParameterManager
        self.parameter_handler = ParameterHandler(
            file_path=self.options.parameter_file if self.options else None
        )
        logger.debug(f"Initialized TextAnalyzer with options: {self.options}")

    def display_analysis(self, text: str, analyzer_name: str) -> None:
        """Display analysis setup and results."""
        # Get language from parameter handler if available
        language = (
            self.options.language
            or self.parameter_handler.language
            or self._detect_language(text)
        )
        logger.debug(f"Starting {analyzer_name} analysis in {language}")
        logger.debug(f"Parameter file: {self.options.parameter_file}")

        print(f"\n{analyzer_name} Analysis")
        print("=" * 50)
        print(f"\nDetected Language: {language}")

        if self.options.parameter_file:
            print(f"Using parameters from: {self.options.parameter_file}")

        print("\nInput Text:")
        print("-" * 20)
        print(text.strip())

        print("\nAnalyzing...")
        print("-" * 20)

    def _detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            from langdetect import detect

            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"


async def analyze_keywords(
    text: str, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Analyze text for keywords with language support."""
    logger.debug("Starting keyword analysis")
    analyzer = TextAnalyzer(options)

    try:
        # Initialize language processor with correct language
        language_processor = create_text_processor(
            language=options.language if options else "en"
        )

        # Create tester with proper initialization
        tester = KeywordTester(
            config={
                "language": options.language if options else "en",
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            language_processor=language_processor,
        )

        logger.debug("Running keyword analysis")
        results = await tester.analyze(text)
        logger.debug("Keyword analysis completed")

        logger.debug("Formatting results")
        tester.format_results(results, detailed=True)
        if options and options.debug_mode:
            logger.debug("Displaying debug information")
            analyzer.display_debug_info(results)

        return results

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        if options and options.debug_mode:
            analyzer.display_debug_info({"error": str(e), "success": False})
        return {"error": str(e), "success": False}


async def analyze_themes(
    text: str, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Analyze text for themes with language support."""
    logger.debug("Starting theme analysis")
    analyzer = TextAnalyzer(options)
    analyzer.display_analysis(text, "Theme")

    # Get parameters from parameter handler
    params = analyzer.parameter_handler.get_parameters()

    # Update with language if specified
    if options and options.language:
        params.general.language = options.language

    logger.debug("Creating theme tester with parameters")
    tester = ThemeTester(config=params.model_dump())

    logger.debug("Running theme analysis")
    results = await tester.analyze(text)
    logger.debug("Theme analysis completed")

    logger.debug("Formatting results")
    tester.format_results(results, detailed=True)
    if options and options.debug_mode:
        logger.debug("Displaying debug information")
        analyzer.display_debug_info(results)

    return results


async def analyze_categories(
    text: str, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Analyze text for categories with language support."""
    logger.debug("Starting category analysis")
    analyzer = TextAnalyzer(options)
    analyzer.display_analysis(text, "Category")

    # Get parameters from parameter handler
    params = analyzer.parameter_handler.get_parameters()

    # Update with language if specified
    if options and options.language:
        params.general.language = options.language

    logger.debug("Creating category tester with parameters")
    tester = CategoryTester(
        config=params.model_dump(), categories=params.categories
    )

    logger.debug("Running category analysis")
    results = await tester.analyze(text)
    logger.debug("Category analysis completed")

    logger.debug("Formatting results")
    tester.format_results(results, detailed=True)
    if options and options.debug_mode:
        logger.debug("Displaying debug information")
        analyzer.display_debug_info(results)

    return results


async def analyze_text(
    text: str, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Run complete text analysis with language support."""
    logger.debug("Starting complete text analysis")
    results = {}

    analyzers = [
        ("keywords", analyze_keywords),
        ("themes", analyze_themes),
        ("categories", analyze_categories),
    ]

    for analyzer_type, analyze_func in analyzers:
        logger.debug(f"Running {analyzer_type} analysis")
        results[analyzer_type] = await analyze_func(text, options)
        logger.debug(f"{analyzer_type} analysis completed")
        print("\n" + "-" * 80 + "\n")

    logger.debug("Complete text analysis finished")
    return results


async def analyze_excel_content(
    input_file: Union[str, Path],
    output_file: str,
    content_column: str = "content",
    parameter_file: Optional[Union[str, Path]] = None,
    language_column: Optional[str] = None,
) -> None:
    """Process Excel file with language detection and parameters."""
    analyzer = TextAnalyzer()
    param_manager = LanguageParameterManager()

    # Load Excel file
    df = analyzer.file_utils.load_single_file(input_file)
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found")

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        text = str(row[content_column])
        print(f"\nProcessing text {idx + 1}/{total}")

        try:
            # Determine language
            language = None
            if language_column and language_column in df.columns:
                language = row[language_column]

            # Create analysis options
            options = AnalysisOptions(
                language=language,
                parameter_file=parameter_file,
                debug_mode=False,  # Disable debug mode for batch processing
            )

            # Run analysis
            analysis = await analyze_text(text, options)
            results.append(create_analysis_summary(analysis).iloc[0])
            print("✓ Analysis complete")

        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            results.append(
                pd.Series({"keywords": "", "categories": "", "themes": ""})
            )

    # Create output DataFrame
    output_df = pd.DataFrame(results)
    output_df.insert(0, content_column, df[content_column])
    if language_column:
        output_df.insert(
            1,
            "detected_language",
            [
                param_manager.detect_language(text)
                for text in df[content_column]
            ],
        )

    # Save results
    analyzer.file_utils.save_data_to_disk(
        data={"Analysis Results": output_df},
        output_filetype="xlsx",
        file_name=output_file,
    )

    print(f"\nResults saved to: {output_file}")
