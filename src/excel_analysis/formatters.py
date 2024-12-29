# src/excel_analysis/formatters.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from FileUtils import FileUtils, OutputFileType
from src.core.managers import EnvironmentManager
from src.utils.formatting_config import ExcelOutputConfig, OutputDetail
from src.utils.output_formatter import BaseFormatter

logger = logging.getLogger(__name__)


class ExcelAnalysisFormatter(BaseFormatter):
    """Formatter for Excel output."""

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config: Optional[ExcelOutputConfig] = None,
    ):
        """Initialize formatter with FileUtils and config.
        
        Args:
            file_utils: FileUtils instance (required)
            config: Optional output configuration
            
        Raises:
            ValueError: If file_utils is None and EnvironmentManager not initialized
        """
        super().__init__(config or ExcelOutputConfig())
        
        # Try to get FileUtils from EnvironmentManager first
        if file_utils is None:
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                self.file_utils = components["file_utils"]
            except RuntimeError:
                raise ValueError(
                    "FileUtils instance must be provided to ExcelAnalysisFormatter. "
                    "Use EnvironmentManager to get a shared FileUtils instance."
                )
        else:
            self.file_utils = file_utils

    def format_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Dict[str, str]:
        """Format analysis results for Excel output."""
        formatted = {}

        for analysis_type in analysis_types:
            if analysis_type not in results:
                continue

            result = results[analysis_type]
            if self.config.detail_level == OutputDetail.MINIMAL:
                formatted[analysis_type] = self._format_minimal(
                    result, analysis_type
                )
            else:
                formatted[analysis_type] = self._format_detailed(
                    result, analysis_type
                )

        return formatted

    def _format_minimal(self, result: Any, analysis_type: str) -> str:
        """Format minimal output for Excel."""
        if analysis_type == "keywords":
            items = [
                f"{kw.keyword}{self._format_confidence(kw.score)}"
                for kw in result.keywords
            ]
        elif analysis_type == "themes":
            items = [
                f"{theme.name}{self._format_confidence(theme.confidence)}"
                for theme in result.themes
            ]
        elif analysis_type == "categories":
            items = [
                f"{cat.name}{self._format_confidence(cat.confidence)}"
                for cat in result.matches
            ]
        else:
            return ""

        return ", ".join(items)

    def _format_detailed(self, result: Any, analysis_type: str) -> str:
        """Format detailed output for Excel."""
        items = []

        if analysis_type == "keywords":
            for kw in result.keywords:
                item = f"{kw.keyword}{self._format_confidence(kw.score)}"
                if hasattr(kw, "domain") and kw.domain:
                    item += f" [{kw.domain}]"
                items.append(item)

        elif analysis_type == "themes":
            for theme in result.themes:
                item = (
                    f"{theme.name}{self._format_confidence(theme.confidence)}"
                )
                if hasattr(theme, "description"):
                    item += f": {theme.description}"
                items.append(item)

        elif analysis_type == "categories":
            for cat in result.matches:
                item = f"{cat.name}{self._format_confidence(cat.confidence)}"
                if hasattr(cat, "description"):
                    item += f": {cat.description}"
                items.append(item)

        return "; ".join(items)

    def save_excel_results(
        self,
        results_df: pd.DataFrame,
        output_file: Union[str, Path],
        include_summary: bool = True,
    ) -> Path:
        """Save formatted results to Excel."""
        sheets = {"Analysis Results": results_df}

        if include_summary and self.config.detail_level >= OutputDetail.SUMMARY:
            sheets["Summary"] = self._create_summary_sheet(results_df)

        saved_files, _ = self.file_utils.save_data_to_storage(
            data=sheets,
            output_type="processed",
            file_name=str(output_file),
            output_filetype=OutputFileType.XLSX,
            include_timestamp=True,
        )

        return Path(next(iter(saved_files.values())))

    def _create_summary_sheet(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics sheet."""
        summary_data = {
            "Metric": [
                "Total Records",
                "Records with Keywords",
                "Records with Themes",
                "Records with Categories",
            ],
            "Value": [
                len(results_df),
                results_df["keywords"].notna().sum(),
                results_df["themes"].notna().sum(),
                results_df["categories"].notna().sum(),
            ],
        }

        return pd.DataFrame(summary_data)
