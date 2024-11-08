# src/loaders/parameter_adapter.py

from typing import Optional, Union, Dict, Any, Set, List
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import logging
from .models import (
    GeneralParameters,
    CategoryConfig,
    PredefinedKeyword,
    AnalysisSettings,
    ParameterSet
)
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)

class ParameterValidator:
    """Validator for analysis parameters."""

    def validate(self, params: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
        """Validate parameters and return validation results.

        Args:
            params: Parameters to validate

        Returns:
            tuple containing:
            - bool: Whether validation passed
            - List[str]: Warning messages
            - List[str]: Error messages
        """
        warnings = []
        errors = []

        try:
            # Validate general parameters
            general = self.validate_parameters(params)

            # Check for conflicting keywords
            if hasattr(params, "predefined_keywords") and hasattr(params, "excluded_keywords"):
                conflicts = set(params.predefined_keywords) & set(params.excluded_keywords)
                if conflicts:
                    warnings.append(f"Keywords appear in both predefined and excluded: {conflicts}")

            # Check confidence thresholds
            if hasattr(general, "min_confidence"):
                if general.min_confidence < 0.1:
                    warnings.append("Very low minimum confidence threshold")
                elif general.min_confidence > 0.9:
                    warnings.append("Very high minimum confidence threshold")

            # Validate categories
            if hasattr(params, "categories"):
                for cat_name, cat in params.categories.items():
                    if cat.threshold < 0.1:
                        warnings.append(f"Very low threshold for category {cat_name}")

            return len(errors) == 0, warnings, errors

        except ValueError as e:
            errors.append(str(e))
            return False, warnings, errors

        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
            return False, warnings, errors

    def validate_parameters(self, params: Dict[str, Any]) -> GeneralParameters:
        """Validate and convert parameters."""
        if "general" in params:
            return GeneralParameters(**params["general"])
        return GeneralParameters(**params)

    def validate_runtime_params(
        self, base_params: GeneralParameters, runtime_params: Dict[str, Any]
    ) -> GeneralParameters:
        """Validate runtime parameter updates."""
        params = base_params.model_dump()
        params.update(runtime_params)
        return GeneralParameters(**params)


class ParameterEntry(BaseModel):
    """Entry for predefined keywords."""

    keyword: str
    domain: Optional[str] = None
    importance: float = 1.0
    compound_parts: List[str] = []

class ParameterAdapter:
    """Adapter for loading and converting analysis parameters."""

    DEFAULT_CONFIG = {
        "general": {
            "max_keywords": 10,
            "min_keyword_length": 3,
            "language": "en",
            "focus_on": None,
            "include_compounds": True,
            "max_themes": 3,
            "min_confidence": 0.3,
            "column_name_to_analyze": "text"
        },
        "categories": {},
        "predefined_keywords": {},
        "excluded_keywords": set(),
        "analysis_settings": {
            "theme_analysis": {
                "enabled": True,
                "min_confidence": 0.5
            },
            "weights": {
                "statistical": 0.4,
                "llm": 0.6
            }
        }
    }

    SHEET_NAMES = {
        "general": "General Parameters",
        "categories": "Categories",
        "keywords": "Predefined Keywords",
        "excluded": "Excluded Keywords",
        "settings": "Analysis Settings",
    }

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """Initialize adapter with parameter file path."""
        self.file_utils = FileUtils()
        self.file_path = Path(file_path) if file_path else None
        self.parameters = None
        self.load_and_convert()
        logger.debug(f"Initialized ParameterAdapter with file: {file_path}")

    def get_parameters(self) -> ParameterSet:
        """Get loaded parameters."""
        return self.parameters

    def _load_excel(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Load all sheets from Excel file using FileUtils."""
        try:
            sheets = {}
            excel_data = self.file_utils.load_excel_sheets(file_path)
            
            # Map sheet names
            sheet_mapping = {
                "General Parameters": "general",
                "Categories": "categories",
                "Predefined Keywords": "keywords",
                "Excluded Keywords": "excluded",
                "Analysis Settings": "settings"
            }
            
            # Convert loaded sheets to expected format
            for sheet_name, internal_name in sheet_mapping.items():
                if sheet_name in excel_data:
                    sheets[internal_name] = excel_data[sheet_name]
            
            return sheets
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise

    def _parse_general_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse general parameters sheet."""
        if df.empty:
            return {}

        params = self.DEFAULT_CONFIG["general"].copy()

        # Get max_keywords - Should be 8 by default
        params["max_keywords"] = 8

        # Detect Finnish from filename
        if self.file_path and "parametrit" in str(self.file_path).lower():
            params["language"] = "fi"
            params["column_name_to_analyze"] = "keskustelu"

        if "parameter" in df.columns and "value" in df.columns:
            for _, row in df.iterrows():
                param = row["parameter"]
                value = row["value"]
                if pd.notna(value):
                    if param == "language":
                        params["language"] = str(value).lower()
                    elif param == "column_name_to_analyze":
                        params["column_name_to_analyze"] = str(value)
                    elif param == "focus_on":
                        params["focus_on"] = str(value)

        return params

    def _parse_categories(self, df: pd.DataFrame) -> Dict[str, CategoryConfig]:
        """Parse categories sheet."""
        categories = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if pd.notna(row.get("category")):
                    categories[row["category"]] = CategoryConfig(
                        description=row.get("description", ""),
                        keywords=str(row.get("keywords", "")).split(",") if pd.notna(row.get("keywords")) else [],
                        threshold=float(row.get("threshold", 0.5)),
                        parent=row.get("parent") if pd.notna(row.get("parent")) else None,
                    )
        return categories

    def _parse_predefined_keywords(self, df: pd.DataFrame) -> Dict[str, PredefinedKeyword]:
        """Parse predefined keywords sheet."""
        if df.empty:
            return {}

        keywords = {}
        for _, row in df.iterrows():
            if pd.notna(row.get("keyword")):
                keyword = row["keyword"]
                keywords[keyword] = PredefinedKeyword(
                    importance=float(row.get("importance", 1.0)),
                    domain=row.get("domain") if pd.notna(row.get("domain")) else None,
                    compound_parts=(
                        str(row.get("compound_parts", "")).split(",") if pd.notna(row.get("compound_parts")) else []
                    ),
                )

        return keywords

    def _parse_excluded_keywords(self, df: pd.DataFrame) -> Set[str]:
        """Parse excluded keywords sheet."""
        excluded = set()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if pd.notna(row.get("keyword")):
                    excluded.add(str(row["keyword"]).strip())
        return excluded

    def _parse_analysis_settings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse analysis settings sheet."""
        if df.empty:
            return self.DEFAULT_CONFIG["analysis_settings"]

        settings = {
            "theme_analysis": {"enabled": True, "min_confidence": 0.5},
            "weights": {"statistical": 0.4, "llm": 0.6},
        }

        for _, row in df.iterrows():
            if pd.notna(row.get("setting")):
                value = row["value"]
                parts = row["setting"].split(".")
                current = settings
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value

        return settings

    def load_and_convert(self) -> ParameterSet:
        """Load and convert parameters from file."""
        try:
            if not self.file_path or not self.file_path.exists():
                logger.debug("Using default parameters")
                self.parameters = ParameterSet(**self.DEFAULT_CONFIG)
                return self.parameters

            # Use FileUtils to load the Excel file
            sheets = self._load_excel(self.file_path)

            # Parse sheets
            general_params = self._parse_general_parameters(sheets.get("general", pd.DataFrame()))
            categories = self._parse_categories(sheets.get("categories", pd.DataFrame()))
            predefined_keywords = self._parse_predefined_keywords(sheets.get("keywords", pd.DataFrame()))
            excluded_keywords = self._parse_excluded_keywords(sheets.get("excluded", pd.DataFrame()))
            analysis_settings = self._parse_analysis_settings(sheets.get("settings", pd.DataFrame()))

            # Merge with defaults
            config = {
                "general": {**self.DEFAULT_CONFIG["general"], **general_params},
                "categories": categories,
                "predefined_keywords": predefined_keywords,
                "excluded_keywords": excluded_keywords,
                "analysis_settings": analysis_settings
            }

            self.parameters = ParameterSet(**config)
            return self.parameters
            
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.parameters = ParameterSet(**self.DEFAULT_CONFIG)
            return self.parameters

    def _convert_general_params(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert old parameter format to new."""
        params = {}
        param_mapping = {
            "max_kws": "max_keywords",
            "column": "column_name_to_analyze",
            "language": "language",
            "min_length": "min_keyword_length",
            "excluded": "excluded_keywords",
        }

        for _, row in df.iterrows():
            key = row["parameter"]
            value = row["value"]

            # Map old parameter names to new ones
            key = param_mapping.get(key, key)
            params[key] = value

        # Handle Finnish detection
        if any(df["parameter"].str.contains("max_kws")):
            if df.loc[df["parameter"] == "max_kws", "value"].iloc[0] == 8:
                params["max_keywords"] = 8

        # Handle Finnish path detection
        if self.file_path and "parametrit" in self.file_path.stem.lower():
            params["language"] = "fi"
            params["column_name_to_analyze"] = "keskustelu"

        return params

    def update_parameters(self, **kwargs) -> ParameterSet:
        """Update parameters with new values."""
        config = self.parameters.model_dump()

        for section, values in kwargs.items():
            if section in config:
                if isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values

        self.parameters = ParameterSet(**config)
        return self.parameters
