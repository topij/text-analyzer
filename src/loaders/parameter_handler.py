# src/loaders/parameter_handler.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List, Set
import pandas as pd
from pydantic import BaseModel, Field

from .models import GeneralParameters, CategoryConfig, PredefinedKeyword, AnalysisSettings, ParameterSet
from .parameter_config import ParameterConfigurations, ParameterSheets
from .parameter_validation import ParameterValidation
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class ParameterHandler:
    """Unified parameter handling with validation."""

    REQUIRED_PARAMETERS = {
        "general.max_keywords": {"type": int, "min": 1, "max": 20},
        "general.focus_on": {"type": str},
        "general.column_name_to_analyze": {"type": str},
    }

    def __init__(self, file_path: Optional[Union[str, Path]] = None, file_utils: Optional[FileUtils] = None):
        """Initialize parameter handler."""
        self.file_utils = file_utils or FileUtils()

        # Convert to absolute path and resolve any relative parts
        # Convert file path using helper
        if file_path:
            self.file_path = get_parameter_file_path(file_path, self.file_utils)
            logger.debug(f"Using parameter file: {self.file_path}")
        else:
            self.file_path = None

        self.config = ParameterConfigurations()
        self.validator = ParameterValidation()
        self.parameters = None
        self.language = self._detect_language()

    def _detect_language(self) -> str:
        """Detect parameter file language."""
        if not self.file_path:
            return "en"
        return ParameterConfigurations.detect_language(str(self.file_path))

    def get_parameters(self) -> ParameterSet:
        """Get loaded and validated parameters."""
        if self.parameters is None:
            self._load_and_validate_parameters()
        return self.parameters

    def _load_and_validate_parameters(self) -> None:
        """Load and validate parameters."""
        try:
            if not self.file_path or not self.file_path.exists():
                logger.debug(f"File path does not exist: {self.file_path}")
                self.parameters = ParameterSet(**self.config.get_default_config())
                return

            logger.debug(f"Loading Excel file from: {self.file_path}")
            # Load Excel sheets
            sheets = self.file_utils.load_excel_sheets(self.file_path)
            logger.debug(f"Found sheets: {list(sheets.keys())}")

            # Get expected sheet name
            general_sheet_name = ParameterSheets.get_sheet_name("GENERAL", self.language)
            logger.debug(f"Looking for general parameters in sheet: {general_sheet_name}")

            # Get general sheet
            general_sheet = sheets.get(general_sheet_name)
            if general_sheet is not None:
                logger.debug(f"General sheet columns: {general_sheet.columns.tolist()}")
                logger.debug(f"General sheet content:\n{general_sheet}")
            else:
                logger.warning(f"Sheet {general_sheet_name} not found")

            # Parse parameters
            general_params = self._parse_general_parameters(general_sheet)
            logger.debug(f"Parsed general parameters: {general_params}")

            # Parse other sheets
            config = {
                "general": general_params,
                "categories": self._parse_categories(sheets),
                "predefined_keywords": self._parse_keywords(sheets),
                "excluded_keywords": self._parse_excluded_keywords(sheets),
                "analysis_settings": self._parse_settings(sheets),
            }

            # Create parameter set
            self.parameters = ParameterSet(**config)
            logger.debug(f"Created parameter set: {self.parameters.model_dump()}")

        except Exception as e:
            logger.error(f"Error loading parameters: {e}", exc_info=True)
            self.parameters = ParameterSet(**self.config.get_default_config())
            raise

    def _ensure_required_parameters(self, params: Dict[str, Any]) -> None:
        """Ensure all required parameters are present with valid values."""
        defaults = {"focus_on": "general content analysis", "column_name_to_analyze": "text", "max_keywords": 10}

        # Add any missing required parameters from defaults
        for param, default in defaults.items():
            if param not in params or params[param] is None:
                params[param] = default
                logger.debug(f"Using default value for {param}: {default}")

    def _parse_general_parameters(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Parse general parameters sheet."""
        params = self.config.get_default_config()["general"].copy()
        logger.debug(f"Starting with default general parameters: {params}")

        if df is not None and not df.empty:
            param_col = "parameter"
            value_col = "value"

            if param_col in df.columns and value_col in df.columns:
                logger.debug("Found parameter and value columns")
                logger.debug(f"Full dataframe:\n{df}")

                for _, row in df.iterrows():
                    if pd.notna(row[param_col]) and pd.notna(row[value_col]):
                        param_name = str(row[param_col]).strip()
                        value = self._convert_value(row[value_col])
                        logger.debug(f"Setting {param_name} = {value}")
                        params[param_name] = value
            else:
                logger.warning(f"Required columns not found. Available columns: {df.columns.tolist()}")

        logger.debug(f"Final general parameters: {params}")
        return params

    def _parse_categories(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, CategoryConfig]:
        """Parse categories sheet."""
        categories = {}
        sheet_name = ParameterSheets.get_sheet_name("CATEGORIES", self.language)
        df = sheets.get(sheet_name)

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if "category" in df.columns and pd.notna(row["category"]):
                    cat_name = str(row["category"]).strip()
                    categories[cat_name] = CategoryConfig(
                        description=row.get("description", ""),
                        keywords=self._split_list(row.get("keywords", "")),
                        threshold=float(row.get("threshold", 0.5)),
                        parent=row.get("parent") if pd.notna(row.get("parent")) else None,
                    )
        return categories

    def _parse_keywords(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, PredefinedKeyword]:
        """Parse predefined keywords sheet."""
        keywords = {}
        sheet_name = ParameterSheets.get_sheet_name("KEYWORDS", self.language)
        df = sheets.get(sheet_name)

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if "keyword" in df.columns and pd.notna(row["keyword"]):
                    keyword = str(row["keyword"]).strip()
                    keywords[keyword] = PredefinedKeyword(
                        importance=float(row.get("importance", 1.0)),
                        domain=row.get("domain") if pd.notna(row.get("domain")) else None,
                    )
        return keywords

    def _parse_excluded_keywords(self, sheets: Dict[str, pd.DataFrame]) -> Set[str]:
        """Parse excluded keywords sheet."""
        excluded = set()
        sheet_name = ParameterSheets.get_sheet_name("EXCLUDED", self.language)
        df = sheets.get(sheet_name)

        if df is not None and not df.empty:
            if "keyword" in df.columns:
                excluded = {str(row["keyword"]).strip() for _, row in df.iterrows() if pd.notna(row["keyword"])}
        return excluded

    def _parse_settings(self, sheets: Dict[str, pd.DataFrame]) -> AnalysisSettings:
        """Parse analysis settings sheet."""
        default_settings = self.config.get_default_config()["analysis_settings"]

        sheet_name = ParameterSheets.get_sheet_name("SETTINGS", self.language)
        df = sheets.get(sheet_name)

        if df is not None and not df.empty:
            # Parse settings here if needed
            pass

        return AnalysisSettings(**default_settings)

    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert parameter values to appropriate types."""
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower == "true":
                return True
            if value_lower == "false":
                return False
            try:
                return float(value) if "." in value else int(value)
            except ValueError:
                return value
        return value

    @staticmethod
    def _split_list(value: Any) -> List[str]:
        """Split comma-separated string into list."""
        if pd.isna(value):
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validate current parameters."""
        if self.parameters is None:
            self._load_and_validate_parameters()
        return self.validator.validate_parameters(self.parameters.model_dump())

    def update_parameters(self, **kwargs) -> ParameterSet:
        """Update parameters with new values."""
        if self.parameters is None:
            self._load_and_validate_parameters()

        config = self.parameters.model_dump()
        for section, values in kwargs.items():
            if section in config:
                if isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values

        self.parameters = ParameterSet(**config)
        return self.parameters


def verify_parameter_file(file_path: Union[str, Path]) -> None:
    """Verify parameter file existence and content."""
    path = Path(file_path).resolve()
    print(f"\nParameter File Verification:")
    print(f"Absolute path: {path}")
    print(f"File exists: {path.exists()}")

    if path.exists():
        # Load and check content
        import pandas as pd

        try:
            xlsx = pd.ExcelFile(path)
            print("\nFound sheets:")
            for sheet in xlsx.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet)
                print(f"\n{sheet}:")
                print(df.head())
        except Exception as e:
            print(f"Error reading file: {e}")


def get_parameter_file_path(file_name: Union[str, Path], file_utils: Optional[FileUtils] = None) -> Path:
    """Get the full path for a parameter file.

    Args:
        file_name: Name or path of parameter file
        file_utils: Optional FileUtils instance

    Returns:
        Path: Full path to parameter file location
    """
    fu = file_utils or FileUtils()
    # Get the parameters directory
    param_dir = fu.get_data_path("parameters")
    return param_dir / Path(file_name).name
