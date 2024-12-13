# src/utils/output_formatter.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.utils.formatting_config import BaseOutputConfig, OutputDetail

logger = logging.getLogger(__name__)


class BaseFormatter(ABC):
    """Abstract base class for all formatters."""

    def __init__(self, config: Optional[BaseOutputConfig] = None):
        """Initialize formatter with config."""
        self.config = config or BaseOutputConfig()

    @abstractmethod
    def format_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Any:
        """Format analysis results."""
        pass

    def _format_confidence(self, value: float) -> str:
        """Format confidence score."""
        return f" ({value:.2f})" if self.config.include_confidence else ""


class ResultFormatter(BaseFormatter):
    """Base formatter for individual analysis types."""

    def format_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Dict[str, str]:
        """Format analysis results according to type."""
        formatted = {}
        for analysis_type in analysis_types:
            if analysis_type in results:
                formatted[analysis_type] = self._format_type(
                    results[analysis_type], analysis_type
                )
        return formatted

    @abstractmethod
    def _format_type(self, result: Any, analysis_type: str) -> str:
        """Format specific analysis type result."""
        pass
