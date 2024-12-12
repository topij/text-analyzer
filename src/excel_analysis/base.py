# src/excel_analysis/base.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.excel_analysis.parameters import AnalysisParameters
from FileUtils import FileUtils, OutputFileType

logger = logging.getLogger(__name__)


class ExcelAnalysisBase:
    """Base class for Excel-based analysis."""

    def __init__(
        self,
        content_file: Union[str, Path, pd.DataFrame],
        parameter_file: Union[str, Path],
        content_column: str = "content",
        file_utils: Optional[FileUtils] = None,
        **kwargs,
    ):
        """Initialize Excel analysis.

        Args:
            content_file: Path to content Excel file or DataFrame
            parameter_file: Path to parameter Excel file
            content_column: Name of content column
            file_utils: Optional FileUtils instance
            **kwargs: Additional configuration options
        """
        self.file_utils = file_utils or FileUtils()
        self.content_column = content_column
        self.config = kwargs

        # Load content based on input type
        if isinstance(content_file, pd.DataFrame):
            self.content = content_file
        else:
            self.content_file = Path(content_file)
            self.content = self._load_content()

        # Load and validate parameter file
        self.parameter_file = Path(parameter_file)
        param_path = (
            self.file_utils.get_data_path("parameters") / self.parameter_file
        )
        if not param_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_path}")

        self.parameters = AnalysisParameters(param_path, self.file_utils)

    def _load_content(self) -> pd.DataFrame:
        """Load content from Excel file using FileUtils.

        Returns:
            DataFrame containing content

        Raises:
            ValueError: If content file or column is invalid
        """
        try:
            # Get full path to content file
            content_path = (
                self.file_utils.get_data_path("raw") / self.content_file
            )
            if not content_path.exists():
                raise FileNotFoundError(
                    f"Content file not found: {content_path}"
                )

            # Load file using FileUtils
            loaded_data = self.file_utils.load_single_file(
                file_path=content_path,
                sheet_name=0,  # Load first sheet by default
            )

            if not isinstance(loaded_data, pd.DataFrame):
                raise ValueError(
                    f"Invalid content file format: {type(loaded_data)}"
                )

            if self.content_column not in loaded_data.columns:
                raise ValueError(
                    f"Content column '{self.content_column}' not found. "
                    f"Available columns: {list(loaded_data.columns)}"
                )

            # Ensure content column has string values
            loaded_data[self.content_column] = (
                loaded_data[self.content_column].fillna("").astype(str)
            )

            logger.info(
                f"Successfully loaded content file with {len(loaded_data)} rows"
            )
            return loaded_data

        except Exception as e:
            logger.error(f"Error loading content file {self.content_file}: {e}")
            raise ValueError(f"Failed to load content file: {str(e)}")

    def _get_texts(self) -> List[str]:
        """Get texts from content DataFrame.

        Returns:
            List of texts to analyze
        """
        if self.content is None or self.content.empty:
            return []

        return self.content[self.content_column].tolist()

    def _save_results(
        self, results_df: pd.DataFrame, output_file: Union[str, Path]
    ) -> Path:
        """Save analysis results to Excel.

        Args:
            results_df: DataFrame with analysis results
            output_file: Output file name (without extension)

        Returns:
            Path to saved file
        """
        try:
            # Ensure file name doesn't include extension
            output_name = Path(output_file).stem

            # Save using FileUtils
            saved_files, _ = self.file_utils.save_data_to_storage(
                data={"Analysis Results": results_df},
                file_name=output_name,
                output_type="processed",
                output_filetype=OutputFileType.XLSX,
                include_timestamp=True,
            )

            if not saved_files:
                raise ValueError("No files were saved")

            saved_path = Path(next(iter(saved_files.values())))
            logger.info(f"Saved results to: {saved_path}")
            return saved_path

        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise
