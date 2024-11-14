# src/nb_helpers/debug.py
import logging
import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from src.nb_helpers.base import DebugMixin
from src.nb_helpers.testers import KeywordTester, ThemeTester, CategoryTester
from src.nb_helpers.visualizers import format_confidence_bar, create_analysis_summary
from src.loaders.parameter_adapter import ParameterAdapter
from src.loaders.models import CategoryConfig

@dataclass
class DebugOptions:
    show_confidence: bool = True
    show_evidence: bool = True
    show_keywords: bool = True
    show_raw_data: bool = True

class AnalysisDebugger(DebugMixin):
    def __init__(self, options: Optional[DebugOptions] = None):
        self.options = options or DebugOptions()
        self.parameter_adapter = None
    
    def setup_analysis(self, text: str, analyzer_name: str, parameter_file: Optional[str] = None) -> None:
        """Initialize debug analysis session."""
        print(f"\nDebug {analyzer_name} Analysis")
        print("=" * 50)
        
        print("\nInput Text:")
        print("-" * 20)
        print(text.strip())
        
        if parameter_file:
            self.parameter_adapter = ParameterAdapter(parameter_file)
            print("\nLoaded Parameters:")
            print("-" * 20)
            self._display_parameters()
        
        self.setup_debug_logging(f"src.analyzers.{analyzer_name.lower()}_analyzer")
        
    def _display_parameters(self) -> None:
        """Display loaded parameters if available."""
        if not self.parameter_adapter or not self.parameter_adapter.parameters:
            return
        
        params = self.parameter_adapter.parameters
        if hasattr(params, "general"):
            print("\nGeneral Parameters:")
            for k, v in params.general.dict().items():
                print(f"  {k}: {v}")

    def display_debug_info(self, results: Any) -> None:
        """Display detailed debug information."""
        print("\nDebug Information:")
        print("-" * 20)
        
        if self.options.show_confidence:
            self._display_confidence_stats(results)
            
        if self.options.show_raw_data:
            print("\nRaw Analysis Data:")
            if hasattr(results, "model_dump"):
                print(json.dumps(results.model_dump(), indent=2))
            else:
                print(json.dumps(results, indent=2))

    def _display_confidence_stats(self, results: Any) -> None:
        """Display confidence statistics if available."""
        confidence_scores = []
        
        if hasattr(results, "categories"):
            confidence_scores.extend(cat.confidence for cat in results.categories)
        elif hasattr(results, "keywords"):
            confidence_scores.extend(kw.score for kw in results.keywords)
        elif hasattr(results, "themes"):
            confidence_scores.extend(theme.confidence for theme in results.themes)
            
        if confidence_scores:
            avg_conf = sum(confidence_scores) / len(confidence_scores)
            print(f"\nConfidence Statistics:")
            print(f"  Average: {avg_conf:.2f}")
            print(f"  Max: {max(confidence_scores):.2f}")
            print(f"  Min: {min(confidence_scores):.2f}")

async def debug_analysis(
    text: str, 
    analyzer_type: str,
    tester_cls: type,
    parameter_file: Optional[str] = None,
    options: Optional[DebugOptions] = None
) -> Dict[str, Any]:
    """Generic debug analysis function."""
    debugger = AnalysisDebugger(options)
    debugger.setup_analysis(text, analyzer_type, parameter_file)
    
    print("\nRunning Analysis...")
    print("-" * 20)
    
    tester = tester_cls(parameter_file=parameter_file) if parameter_file else tester_cls()
    results = await tester.analyze(text)
    
    tester.format_results(results, detailed=True)
    debugger.display_debug_info(results)
    
    return results

async def debug_keyword_analysis(
    text: str, 
    parameter_file: Optional[str] = None, 
    options: Optional[DebugOptions] = None
) -> Dict[str, Any]:
    """Debug keyword analysis with detailed output."""
    return await debug_analysis(text, "Keyword", KeywordTester, parameter_file, options)

async def debug_theme_analysis(
    text: str, 
    parameter_file: Optional[str] = None,
    options: Optional[DebugOptions] = None
) -> Dict[str, Any]:
    """Debug theme analysis with detailed output."""
    return await debug_analysis(text, "Theme", ThemeTester, parameter_file, options)

async def debug_category_analysis(
    text: str, 
    parameter_file: Optional[str] = None,
    options: Optional[DebugOptions] = None
) -> Dict[str, Any]:
    """Debug category analysis with detailed output."""
    return await debug_analysis(text, "Category", CategoryTester, parameter_file, options)

async def debug_full_pipeline(
    text: str,
    parameter_file: Optional[str] = None,
    options: Optional[DebugOptions] = None
) -> Dict[str, Any]:
    """Run debug analysis for all analyzers."""
    results = {}
    for analyzer_type, tester_cls in [
        ("Keyword", KeywordTester),
        ("Theme", ThemeTester),
        ("Category", CategoryTester)
    ]:
        results[analyzer_type.lower()] = await debug_analysis(
            text, analyzer_type, tester_cls, parameter_file, options
        )
        print("\n" + "-" * 80 + "\n")
    return results