# src/utils/output_format.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import pandas as pd


class OutputDetail(str, Enum):
    """Output detail level."""

    MINIMAL = "minimal"  # Basic output - just results
    SUMMARY = "summary"  # Results with key metadata and evidence
    DETAILED = "detailed"  # Full output with all metadata
    DEBUG = "debug"  # Everything including internal states


@dataclass
class OutputConfig:
    """Base configuration for output formatting."""

    detail_level: OutputDetail = OutputDetail.SUMMARY
    show_scores: bool = True
    max_items: Optional[int] = None
    include_metadata: bool = False
    separator: str = ", "


class ColumnConfig(BaseModel):
    """Configuration for column formatting."""

    name: str
    format_template: str
    included_fields: List[str]
    confidence_threshold: Optional[float] = None
    max_items: Optional[int] = None


class BaseFormatter(ABC):
    """Abstract base formatter defining interface."""

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()

    @abstractmethod
    def format_results(self, results: Dict[str, Any]) -> Any:
        """Format analysis results according to config."""
        pass

    def _format_item_with_score(self, item: Any, score: float) -> str:
        """Format single item with optional score."""
        if not self.config.show_scores:
            return str(item)
        return f"{item} ({score:.2f})"

    def _truncate_items(self, items: List[Any]) -> List[Any]:
        """Truncate items list based on max_items config."""
        if self.config.max_items:
            return items[: self.config.max_items]
        return items


class ConsoleFormatter(BaseFormatter):
    """Formatter for console output."""

    def format_results(self, results: Dict[str, Any]) -> str:
        if self.config.detail_level == OutputDetail.MINIMAL:
            return self._format_minimal(results)
        elif self.config.detail_level == OutputDetail.SUMMARY:
            return self._format_summary(results)
        return self._format_detailed(results)

    def _format_minimal(self, results: Dict[str, Any]) -> str:
        """Format minimal results for console."""
        lines = []
        for analysis_type, result in results.items():
            items = self._get_items(result)
            formatted = [
                self._format_item_with_score(item.name, item.score)
                for item in self._truncate_items(items)
            ]
            lines.append(
                f"{analysis_type}: {self.config.separator.join(formatted)}"
            )
        return "\n".join(lines)


class ExcelFormatter(BaseFormatter):
    """Formatter for Excel output."""

    def format_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        data = {}

        # Format each analysis type
        for analysis_type, result in results.items():
            column_name = f"{analysis_type}_results"
            items = self._get_items(result)

            if self.config.detail_level == OutputDetail.MINIMAL:
                data[column_name] = self._format_minimal_column(items)
            else:
                data.update(self._format_detailed_columns(items, analysis_type))

        return pd.DataFrame(data)

    def _format_minimal_column(self, items: List[Any]) -> List[str]:
        """Format minimal column data."""
        formatted = [
            self._format_item_with_score(item.name, item.score)
            for item in self._truncate_items(items)
        ]
        return [self.config.separator.join(formatted)]


class JSONFormatter(BaseFormatter):
    """Formatter for JSON output."""

    def format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.detail_level == OutputDetail.MINIMAL:
            return self._format_minimal_json(results)
        return self._format_detailed_json(results)
