# src/loaders/parameter_handler.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List, Set
import pandas as pd
from pydantic import ValidationError, BaseModel, Field

from .models import (
    GeneralParameters,
    DomainContext,
    CategoryConfig,
    PredefinedKeyword,
    AnalysisSettings,
    ParameterSet,
)
from src.loaders.parameter_config import (
    ParameterConfigurations,
    ParameterSheets,
)
from src.loaders.parameter_validation import ParameterValidation
from .parameter_config import ParameterSheets, ParameterConfigurations
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class ParameterHandler:
    """Unified parameter handling with validation."""

    REQUIRED_PARAMETERS = {
        "general.max_keywords": {"type": int, "min": 1, "max": 20},
        "general.focus_on": {"type": str},
        "general.column_name_to_analyze": {"type": str},
    }

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
    ):
        """Initialize parameter handler."""
        self.file_utils = file_utils or FileUtils()
        self.config = ParameterConfigurations()  # Initialize config
        self.validator = ParameterValidation()  # Initialize validator
        self.file_path = (
            get_parameter_file_path(file_path, self.file_utils)
            if file_path
            else None
        )
        self.language = self._detect_language()
        self._load_and_validate_parameters()

    def _detect_language(self) -> str:
        """Detect parameter file language."""
        if not self.file_path:
            return "en"
        return ParameterConfigurations.detect_language(str(self.file_path))

    def get_parameters(self) -> ParameterSet:
        """Get loaded and validated parameters."""
        if not hasattr(self, "parameters"):
            self._load_and_validate_parameters()
        return self.parameters

    def _load_and_validate_parameters(self) -> None:
        """Load and validate parameters from file."""
        try:
            if not self.file_path or not self.file_path.exists():
                logger.debug("No parameter file specified, using defaults")
                default_config = self.config.DEFAULT_CONFIG.copy()
                default_config["general"]["language"] = self.language
                self.parameters = ParameterSet(**default_config)
                return

            sheets = self.file_utils.load_excel_sheets(self.file_path)
            logger.debug(f"Found sheets: {list(sheets.keys())}")

            # Create config with properly mapped parameters
            config = {
                "general": self._parse_general_parameters(
                    sheets.get(
                        ParameterSheets.get_sheet_name("general", self.language)
                    )
                ),
                "categories": self._parse_categories(
                    sheets.get(
                        ParameterSheets.get_sheet_name(
                            "categories", self.language
                        )
                    )
                ),
                "predefined_keywords": self._parse_keywords(
                    sheets.get(
                        ParameterSheets.get_sheet_name(
                            "keywords", self.language
                        )
                    )
                ),
                "excluded_keywords": self._parse_excluded_keywords(
                    sheets.get(
                        ParameterSheets.get_sheet_name(
                            "excluded", self.language
                        )
                    )
                ),
                "analysis_settings": self._parse_settings(
                    sheets.get(
                        ParameterSheets.get_sheet_name(
                            "settings", self.language
                        )
                    )
                ),
                "domain_context": self._parse_domains(
                    sheets.get(
                        ParameterSheets.get_sheet_name("domains", self.language)
                    )
                ),
            }

            # Ensure general section exists with language
            if "general" not in config:
                config["general"] = {}
            config["general"]["language"] = self.language

            # Create and validate ParameterSet
            self.parameters = ParameterSet(**config)

        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            logger.debug("Using default configuration")
            default_config = self.config.DEFAULT_CONFIG.copy()
            default_config["general"]["language"] = self.language
            self.parameters = ParameterSet(**default_config)

    def _parse_general_parameters(
        self, df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Parse general parameters using existing mappings from ParameterSheets."""
        params = {}

        if df is None or df.empty:
            return params

        try:
            # Get column names and parameter mappings for the current language
            column_names = ParameterSheets.get_column_names(
                "general", self.language
            )
            param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
                "parameters"
            ][self.language]

            param_col = column_names["parameter"]
            value_col = column_names["value"]

            if param_col in df.columns and value_col in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[param_col]) and pd.notna(row[value_col]):
                        param_name = str(row[param_col]).strip()

                        # Map parameter name to internal name using existing mappings
                        internal_name = param_mappings.get(param_name)
                        if internal_name is None:
                            logger.warning(
                                f"Unknown parameter name: {param_name}"
                            )
                            continue

                        value = self._convert_value(row[value_col])
                        params[internal_name] = value
                        logger.debug(
                            f"Mapped '{param_name}' to '{internal_name}': {value}"
                        )

            # Ensure language is set
            if "language" not in params:
                params["language"] = self.language

            return params

        except Exception as e:
            logger.error(f"Error parsing general parameters: {e}")
            return {
                "language": self.language
            }  # At minimum, preserve language setting

    def _parse_keywords(
        self, df: Optional[pd.DataFrame]
    ) -> Dict[str, PredefinedKeyword]:
        """Parse keywords sheet."""
        if df is None or df.empty:
            return {}

        keywords = {}
        try:
            column_names = ParameterSheets.get_column_names(
                "keywords", self.language
            )
            kw_col = column_names["keyword"]

            if kw_col not in df.columns:
                logger.warning(
                    f"Keyword column '{kw_col}' not found. Available columns: {df.columns.tolist()}"
                )
                return keywords

            for _, row in df.iterrows():
                if pd.notna(row[kw_col]):
                    keyword = str(row[kw_col]).strip()
                    importance_col = column_names.get("importance")
                    domain_col = column_names.get("domain")

                    importance = 1.0
                    if importance_col and importance_col in df.columns:
                        importance = float(row.get(importance_col, 1.0))

                    domain = None
                    if domain_col and domain_col in df.columns:
                        domain = (
                            row.get(domain_col)
                            if pd.notna(row.get(domain_col))
                            else None
                        )

                    keywords[keyword] = PredefinedKeyword(
                        importance=importance, domain=domain
                    )
                    logger.debug(
                        f"Added keyword: {keyword} (importance: {importance}, domain: {domain})"
                    )

        except Exception as e:
            logger.error(f"Error parsing keywords: {str(e)}")

        return keywords

    def _parse_excluded_keywords(self, df: Optional[pd.DataFrame]) -> Set[str]:
        """Parse excluded keywords sheet."""
        excluded = set()
        if df is None or df.empty:
            return excluded

        try:
            column_names = ParameterSheets.get_column_names(
                "excluded", self.language
            )
            kw_col = column_names["keyword"]

            if kw_col not in df.columns:
                logger.warning(
                    f"Keyword column '{kw_col}' not found. Available columns: {df.columns.tolist()}"
                )
                return excluded

            for _, row in df.iterrows():
                if pd.notna(row[kw_col]):
                    keyword = str(row[kw_col]).strip()
                    excluded.add(keyword)
                    logger.debug(f"Added excluded keyword: {keyword}")

        except Exception as e:
            logger.error(f"Error parsing excluded keywords: {str(e)}")

        return excluded

    def _parse_categories(
        self, df: Optional[pd.DataFrame]
    ) -> Dict[str, CategoryConfig]:
        """Parse categories sheet."""
        categories = {}
        if df is None or df.empty:
            return categories

        try:
            column_names = ParameterSheets.get_column_names(
                "categories", self.language
            )
            cat_col = column_names["category"]

            if cat_col not in df.columns:
                logger.warning(
                    f"Category column '{cat_col}' not found. Available columns: {df.columns.tolist()}"
                )
                return categories

            for _, row in df.iterrows():
                if pd.notna(row[cat_col]):
                    cat_name = str(row[cat_col]).strip()

                    description = ""
                    if (
                        "description" in column_names
                        and column_names["description"] in df.columns
                    ):
                        description = row.get(column_names["description"], "")

                    keywords = []
                    if (
                        "keywords" in column_names
                        and column_names["keywords"] in df.columns
                    ):
                        keywords = self._split_list(
                            row.get(column_names["keywords"], "")
                        )

                    threshold = 0.5
                    if (
                        "threshold" in column_names
                        and column_names["threshold"] in df.columns
                    ):
                        threshold = float(
                            row.get(column_names["threshold"], 0.5)
                        )

                    parent = None
                    if (
                        "parent" in column_names
                        and column_names["parent"] in df.columns
                    ):
                        parent = (
                            row.get(column_names["parent"])
                            if pd.notna(row.get(column_names["parent"]))
                            else None
                        )

                    categories[cat_name] = CategoryConfig(
                        description=description,
                        keywords=keywords,
                        threshold=threshold,
                        parent=parent,
                    )
                    logger.debug(
                        f"Added category: {cat_name} with {len(keywords)} keywords"
                    )

        except Exception as e:
            logger.error(f"Error parsing categories: {str(e)}")

        return categories

    def _parse_domains(
        self, df: Optional[pd.DataFrame]
    ) -> Dict[str, DomainContext]:
        """Parse domains sheet."""
        domains = {}
        if df is None or df.empty:
            return domains

        try:
            column_names = ParameterSheets.get_column_names(
                "domains", self.language
            )
            name_col = column_names["name"]

            if name_col not in df.columns:
                logger.warning(
                    f"Name column '{name_col}' not found. Available columns: {df.columns.tolist()}"
                )
                return domains

            for _, row in df.iterrows():
                if pd.notna(row[name_col]):
                    domain_name = str(row[name_col]).strip()

                    # Ensure all required fields are present
                    domain = DomainContext(
                        name=domain_name,
                        description=str(
                            row.get(column_names.get("description", ""), "")
                        ),
                        key_terms=self._split_list(
                            row.get(column_names.get("key_terms", ""), "")
                        ),
                        context=str(
                            row.get(column_names.get("context", ""), "")
                        ),
                        stopwords=self._split_list(
                            row.get(column_names.get("stopwords", ""), "")
                        ),
                    )

                    domains[domain_name] = domain
                    logger.debug(
                        f"Added domain: {domain_name} with {len(domain.key_terms)} key terms"
                    )

        except Exception as e:
            logger.error(f"Error parsing domains: {str(e)}")
            logger.debug(
                f"Row data: {row.to_dict() if 'row' in locals() else 'N/A'}"
            )

        return domains

    def _parse_settings(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Parse settings sheet with default config fallback."""
        # Get default settings from config
        settings = self.config.DEFAULT_CONFIG.get(
            "analysis_settings", {}
        ).copy()

        if df is None or df.empty:
            return settings

        try:
            column_names = ParameterSheets.get_column_names(
                "settings", self.language
            )
            setting_col = column_names["setting"]
            value_col = column_names["value"]

            if setting_col in df.columns and value_col in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[setting_col]) and pd.notna(row[value_col]):
                        setting_name = str(row[setting_col]).strip()
                        # Get mapping for the setting name
                        internal_name = ParameterSheets.PARAMETER_MAPPING[
                            "settings"
                        ]["parameters"][self.language].get(setting_name)

                        if internal_name:
                            value = self._convert_value(row[value_col])

                            # Handle nested settings
                            if "." in internal_name:
                                section, param = internal_name.split(".", 1)
                                if section not in settings:
                                    settings[section] = {}
                                settings[section][param] = value
                            else:
                                settings[internal_name] = value

                            logger.debug(
                                f"Set setting '{internal_name}' to: {value}"
                            )

        except Exception as e:
            logger.error(f"Error parsing settings: {e}")

        return settings

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
        if not hasattr(self, "parameters"):
            self._load_and_validate_parameters()
        return self.validator.validate_parameters(self.parameters.model_dump())

    def update_parameters(self, **kwargs) -> ParameterSet:
        """Update parameters with new values."""
        if not hasattr(self, "parameters"):
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


def get_parameter_file_path(
    file_name: Union[str, Path], file_utils: Optional[FileUtils] = None
) -> Path:
    """Get the full path for a parameter file."""
    fu = file_utils or FileUtils()
    param_dir = fu.get_data_path("parameters")
    return param_dir / Path(file_name).name


def verify_parameter_file(file_path: Union[str, Path]) -> None:
    """Verify parameter file existence and content."""
    path = Path(file_path).resolve()
    print(f"\nParameter File Verification:")
    print(f"Absolute path: {path}")
    print(f"File exists: {path.exists()}")

    if path.exists():
        try:
            xlsx = pd.ExcelFile(path)
            print("\nFound sheets:")
            for sheet in xlsx.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet)
                print(f"\n{sheet}:")
                print(df.head())
        except Exception as e:
            print(f"Error reading file: {e}")
