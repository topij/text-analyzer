# # src/nb_helpers/debug.py

# src/nb_helpers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from src.core.llm.factory import create_llm
from src.utils.FileUtils.file_utils import FileUtils

@dataclass
class AnalysisResult:
    """Base class for analysis results."""
    success: bool = True
    error: Optional[str] = None
    language: str = "unknown"
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None

class DisplayMixin:
    """Base display functionality for all result types."""
    def display_confidence_bar(self, value: float, width: int = 20) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)
    
    def format_results(self, results: Dict[str, Any], detailed: bool = True) -> None:
        if isinstance(results, dict) and results.get("error"):
            print(f"Error: {results['error']}")
            return
            
        self._display_specific_results(results, detailed)
        
        if detailed and isinstance(results, dict):
            self._display_metadata(results)

    @abstractmethod
    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        """Implement specific display logic in derived classes."""
        pass
    
    def _display_metadata(self, results: Dict[str, Any]) -> None:
        if metadata := results.get("metadata"):
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

class AnalysisTester(ABC):
    """Base class for all analysis testers."""
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        file_utils: Optional[FileUtils] = None
    ):
        self.llm = llm or create_llm()
        self.config = config or {}
        self.file_utils = file_utils or FileUtils()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Implement specific analysis in derived classes."""
        pass
    
    def _validate_input(self, text: str) -> Optional[str]:
        """Validate input text."""
        if not text:
            return "Empty input text"
        if not isinstance(text, str):
            return f"Invalid input type: {type(text)}, expected str"
        if len(text.strip()) == 0:
            return "Input text contains only whitespace"
        return None
        
    def _handle_error(self, error: str) -> Dict[str, Any]:
        """Create error response."""
        self.logger.error(f"Analysis failed: {error}")
        return AnalysisResult(
            success=False,
            error=str(error)
        ).__dict__

class DebugMixin:
    """Common debug functionality."""
    def setup_debug_logging(self, logger_name: str) -> None:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Configure debug handler if not present
        if not any(h.level == logging.DEBUG for h in logger.handlers):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
    
    def display_debug_info(self, results: Any) -> None:
        """Display debug information."""
        import json
        
        print("\nDebug Information:")
        print("-" * 20)
        
        if hasattr(results, "model_dump"):
            print(json.dumps(results.model_dump(), indent=2))
        elif isinstance(results, dict):
            print(json.dumps(results, indent=2))
        else:
            print(f"Results type: {type(results)}")
            print(str(results))

class LoaderMixin:
    """Common data loading functionality."""
    def __init__(self):
        self.file_utils = FileUtils()
    
    def _load_test_data(self, file_pattern: str = "test_content_{lang}.xlsx") -> Dict[str, str]:
        """Load test data from files."""
        try:
            texts = {}
            for lang in ["en", "fi"]:
                df = self.file_utils.load_single_file(
                    file_pattern.format(lang=lang),
                    input_type="raw"
                )
                if df is not None:
                    for _, row in df.iterrows():
                        key = f"{lang}_{row['type']}"
                        texts[key] = row['content']
            return texts
        except Exception as e:
            self.logger.warning(f"Could not load test data: {e}")
            return {}

    def _create_default_data(self) -> Dict[str, str]:
        """Create default test data."""
        return {
            "en_test": "Default test content for English.",
            "fi_test": "Oletustestisisältö suomeksi."
        }


# import logging
# import json
# from typing import Any, Dict, Optional, List
# from dataclasses import dataclass

# from src.nb_helpers.base import DebugMixin
# from src.nb_helpers.testers import KeywordTester, ThemeTester, CategoryTester
# from src.nb_helpers.visualizers import format_confidence_bar, create_analysis_summary
# from src.loaders.parameter_adapter import ParameterAdapter
# from src.loaders.models import CategoryConfig

# @dataclass
# class DebugOptions:
#     show_confidence: bool = True
#     show_evidence: bool = True
#     show_keywords: bool = True
#     show_raw_data: bool = True

# class AnalysisDebugger(DebugMixin):
#     def __init__(self, options: Optional[DebugOptions] = None):
#         self.options = options or DebugOptions()
#         self.parameter_adapter = None
    
#     def setup_analysis(self, text: str, analyzer_name: str, parameter_file: Optional[str] = None) -> None:
#         """Initialize debug analysis session."""
#         print(f"\nDebug {analyzer_name} Analysis")
#         print("=" * 50)
        
#         print("\nInput Text:")
#         print("-" * 20)
#         print(text.strip())
        
#         if parameter_file:
#             self.parameter_adapter = ParameterAdapter(parameter_file)
#             print("\nLoaded Parameters:")
#             print("-" * 20)
#             self._display_parameters()
        
#         self.setup_debug_logging(f"src.analyzers.{analyzer_name.lower()}_analyzer")
        
#     def _display_parameters(self) -> None:
#         """Display loaded parameters if available."""
#         if not self.parameter_adapter or not self.parameter_adapter.parameters:
#             return
        
#         params = self.parameter_adapter.parameters
#         if hasattr(params, "general"):
#             print("\nGeneral Parameters:")
#             for k, v in params.general.dict().items():
#                 print(f"  {k}: {v}")

#     def display_debug_info(self, results: Any) -> None:
#         """Display detailed debug information."""
#         print("\nDebug Information:")
#         print("-" * 20)
        
#         if self.options.show_confidence:
#             self._display_confidence_stats(results)
            
#         if self.options.show_raw_data:
#             print("\nRaw Analysis Data:")
#             if hasattr(results, "model_dump"):
#                 print(json.dumps(results.model_dump(), indent=2))
#             else:
#                 print(json.dumps(results, indent=2))

#     def _display_confidence_stats(self, results: Any) -> None:
#         """Display confidence statistics if available."""
#         confidence_scores = []
        
#         if hasattr(results, "categories"):
#             confidence_scores.extend(cat.confidence for cat in results.categories)
#         elif hasattr(results, "keywords"):
#             confidence_scores.extend(kw.score for kw in results.keywords)
#         elif hasattr(results, "themes"):
#             confidence_scores.extend(theme.confidence for theme in results.themes)
            
#         if confidence_scores:
#             avg_conf = sum(confidence_scores) / len(confidence_scores)
#             print(f"\nConfidence Statistics:")
#             print(f"  Average: {avg_conf:.2f}")
#             print(f"  Max: {max(confidence_scores):.2f}")
#             print(f"  Min: {min(confidence_scores):.2f}")

# async def debug_analysis(
#     text: str, 
#     analyzer_type: str,
#     tester_cls: type,
#     parameter_file: Optional[str] = None,
#     options: Optional[DebugOptions] = None
# ) -> Dict[str, Any]:
#     """Generic debug analysis function."""
#     debugger = AnalysisDebugger(options)
#     debugger.setup_analysis(text, analyzer_type, parameter_file)
    
#     print("\nRunning Analysis...")
#     print("-" * 20)
    
#     tester = tester_cls(parameter_file=parameter_file) if parameter_file else tester_cls()
#     results = await tester.analyze(text)
    
#     tester.format_results(results, detailed=True)
#     debugger.display_debug_info(results)
    
#     return results

# async def debug_keyword_analysis(
#     text: str, 
#     parameter_file: Optional[str] = None, 
#     options: Optional[DebugOptions] = None
# ) -> Dict[str, Any]:
#     """Debug keyword analysis with detailed output."""
#     return await debug_analysis(text, "Keyword", KeywordTester, parameter_file, options)

# async def debug_theme_analysis(
#     text: str, 
#     parameter_file: Optional[str] = None,
#     options: Optional[DebugOptions] = None
# ) -> Dict[str, Any]:
#     """Debug theme analysis with detailed output."""
#     return await debug_analysis(text, "Theme", ThemeTester, parameter_file, options)

# async def debug_category_analysis(
#     text: str, 
#     parameter_file: Optional[str] = None,
#     options: Optional[DebugOptions] = None
# ) -> Dict[str, Any]:
#     """Debug category analysis with detailed output."""
#     return await debug_analysis(text, "Category", CategoryTester, parameter_file, options)

# async def debug_full_pipeline(
#     text: str,
#     parameter_file: Optional[str] = None,
#     options: Optional[DebugOptions] = None
# ) -> Dict[str, Any]:
#     """Run debug analysis for all analyzers."""
#     results = {}
#     for analyzer_type, tester_cls in [
#         ("Keyword", KeywordTester),
#         ("Theme", ThemeTester),
#         ("Category", CategoryTester)
#     ]:
#         results[analyzer_type.lower()] = await debug_analysis(
#             text, analyzer_type, tester_cls, parameter_file, options
#         )
#         print("\n" + "-" * 80 + "\n")
#     return results