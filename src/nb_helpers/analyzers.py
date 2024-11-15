# src/nb_helpers/debug.py -> src/nb_helpers/analyzers.py

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from src.nb_helpers.base import DebugMixin
from src.nb_helpers.testers import KeywordTester, ThemeTester, CategoryTester
from src.nb_helpers.visualizers import format_confidence_bar, create_analysis_summary

logger = logging.getLogger(__name__)


@dataclass
class AnalysisOptions:
    """Options for analysis output."""

    show_confidence: bool = True
    show_evidence: bool = True
    show_keywords: bool = True
    show_raw_data: bool = True
    debug_mode: bool = False


class TextAnalyzer(DebugMixin):
    """Text analysis with detailed output."""

    def __init__(self, options: Optional[AnalysisOptions] = None):
        self.options = options or AnalysisOptions()
        logger.debug("Initialized TextAnalyzer with options: %s", self.options)

    def display_analysis(self, text: str, analyzer_name: str, parameter_file: Optional[str] = None) -> None:
        """Display analysis setup and results."""
        logger.debug("Starting %s analysis", analyzer_name)
        logger.debug("Parameter file: %s", parameter_file if parameter_file else "None")

        print(f"\n{analyzer_name} Analysis")
        print("=" * 50)

        print("\nInput Text:")
        print("-" * 20)
        print(text.strip())

        print("\nAnalyzing...")
        print("-" * 20)


async def analyze_keywords(
    text: str, parameter_file: Optional[str] = None, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Analyze text for keywords with detailed output."""
    logger.debug("Starting keyword analysis")
    analyzer = TextAnalyzer(options)
    analyzer.display_analysis(text, "Keyword", parameter_file)

    logger.debug("Creating keyword tester with parameter file: %s", parameter_file)
    tester = KeywordTester(parameter_file=parameter_file)

    logger.debug("Running keyword analysis")
    results = await tester.analyze(text)
    logger.debug("Keyword analysis completed")

    logger.debug("Formatting results")
    tester.format_results(results, detailed=True)
    if options and options.debug_mode:
        logger.debug("Displaying debug information")
        analyzer.display_debug_info(results)

    return results


async def analyze_themes(
    text: str, parameter_file: Optional[str] = None, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Analyze text for themes with detailed output."""
    logger.debug("Starting theme analysis")
    analyzer = TextAnalyzer(options)
    analyzer.display_analysis(text, "Theme", parameter_file)

    logger.debug("Creating theme tester with parameter file: %s", parameter_file)
    tester = ThemeTester(parameter_file=parameter_file)

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
    text: str, parameter_file: Optional[str] = None, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Analyze text for categories with detailed output."""
    logger.debug("Starting category analysis")
    analyzer = TextAnalyzer(options)
    analyzer.display_analysis(text, "Category", parameter_file)

    logger.debug("Creating category tester with parameter file: %s", parameter_file)
    tester = CategoryTester(parameter_file=parameter_file)

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
    text: str, parameter_file: Optional[str] = None, options: Optional[AnalysisOptions] = None
) -> Dict[str, Any]:
    """Run complete text analysis with all analyzers."""
    logger.debug("Starting complete text analysis")
    results = {}

    analyzers = [("keywords", analyze_keywords), ("themes", analyze_themes), ("categories", analyze_categories)]

    for analyzer_type, analyze_func in analyzers:
        logger.debug("Running %s analysis", analyzer_type)
        results[analyzer_type] = await analyze_func(text, parameter_file, options)
        logger.debug("%s analysis completed", analyzer_type)
        print("\n" + "-" * 80 + "\n")

    logger.debug("Complete text analysis finished")
    return results
