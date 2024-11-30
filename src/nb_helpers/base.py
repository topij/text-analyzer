# src/nb_helpers/base.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    def format_results(
        self, results: Dict[str, Any], detailed: bool = True
    ) -> None:
        if isinstance(results, dict) and results.get("error"):
            print(f"Error: {results['error']}")
            return

        self._display_specific_results(results, detailed)

        if detailed and isinstance(results, dict):
            self._display_metadata(results)

    @abstractmethod
    def _display_specific_results(
        self, results: Dict[str, Any], detailed: bool
    ) -> None:
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
        file_utils: Optional[FileUtils] = None,
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
        return AnalysisResult(success=False, error=str(error)).__dict__


class DebugMixin:
    """Common debug functionality."""

    def setup_debug_logging(self, logger_name: str) -> None:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Configure debug handler if not present
        if not any(h.level == logging.DEBUG for h in logger.handlers):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    def _load_test_data(
        self, file_pattern: str = "test_content_{lang}.xlsx"
    ) -> Dict[str, str]:
        """Load test data from files."""
        try:
            texts = {}
            for lang in ["en", "fi"]:
                df = self.file_utils.load_single_file(
                    file_pattern.format(lang=lang), input_type="raw"
                )
                if df is not None:
                    for _, row in df.iterrows():
                        key = f"{lang}_{row['type']}"
                        texts[key] = row["content"]
            return texts
        except Exception as e:
            self.logger.warning(f"Could not load test data: {e}")
            return {}

    def _create_default_data(self) -> Dict[str, str]:
        """Create default test data."""
        return {
            "en_test": "Default test content for English.",
            "fi_test": "Oletustestisisältö suomeksi.",
        }
