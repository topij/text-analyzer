# src/loaders/parameter_handler.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from src.config.manager import ConfigManager
from src.core.config import AnalyzerConfig
from src.core.managers import EnvironmentManager

from src.loaders.parameter_config import (
    ParameterConfigurations,
    ParameterSheets,
)
from src.loaders.parameter_validation import ParameterValidation
from FileUtils import FileUtils

from .models import (
    AnalysisSettings,
    CategoryConfig,
    DomainContext,
    GeneralParameters,
    ParameterSet,
    PredefinedKeyword,
)
from .parameter_config import ParameterConfigurations, ParameterSheets

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
        """Initialize parameter handler.
        
        Args:
            file_path: Optional path to parameter file
            file_utils: FileUtils instance (required)
            
        Raises:
            ValueError: If file_utils is None and EnvironmentManager not initialized
        """
        # Try to get components from EnvironmentManager first
        if file_utils is None:
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                self.file_utils = components["file_utils"]
                self.analyzer_config = AnalyzerConfig(
                    file_utils=self.file_utils,
                    config_manager=components["config_manager"]
                )
            except RuntimeError:
                raise ValueError(
                    "FileUtils instance must be provided to ParameterHandler. "
                    "Use EnvironmentManager to get a shared FileUtils instance."
                )
        else:
            self.file_utils = file_utils
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                self.analyzer_config = AnalyzerConfig(
                    file_utils=self.file_utils,
                    config_manager=components["config_manager"]
                )
            except RuntimeError:
                raise ValueError(
                    "EnvironmentManager must be initialized before creating ParameterHandler. "
                    "Initialize EnvironmentManager first."
                )

        self.config = ParameterConfigurations()
        self.validator = ParameterValidation()

        # Handle file path
        if isinstance(file_path, (str, Path)):
            self.file_path = Path(file_path)
            if not self.file_path.exists():
                raise FileNotFoundError(
                    f"Parameter file not found: {self.file_path}"
                )
            logger.debug(f"Using parameter file: {self.file_path}")
        else:
            self.file_path = None
            logger.debug("No parameter file specified, using defaults")

        self.language = self._detect_language()
        self._load_and_validate_parameters()

    def _detect_language(self) -> str:
        """Detect parameter file language."""
        if not self.file_path:
            return "en"
        return ParameterConfigurations.detect_language(str(self.file_path))

    def get_parameters(self) -> ParameterSet:
        """Get loaded and validated parameters with debug logging."""
        base_params = self.analyzer_config.config
        if not hasattr(self, "parameters"):
            self._load_and_validate_parameters()

        logger.debug(f"Loaded parameters: {self.parameters.model_dump()}")
        return self.parameters

    def _create_parameter_set(self, config: Dict[str, Any]) -> ParameterSet:
        """Create ParameterSet with validation."""
        try:
            # Ensure general parameters
            if "general" not in config:
                config["general"] = {}

            # Ensure language is set
            config["general"]["language"] = self.language

            # Create parameter set
            params = ParameterSet(**config)

            # Validate parameters
            is_valid, warnings, errors = self.validate_parameters(params)
            if not is_valid:
                logger.error(f"Parameter validation failed: {errors}")

            logger.debug(f"Created parameter set: {params.model_dump()}")
            return params

        except Exception as e:
            logger.error(f"Error creating parameter set: {e}")
            # Return minimal valid parameter set
            return ParameterSet(general={"language": self.language})

    def _validate_mandatory_fields(
        self, df: pd.DataFrame, column_names: Dict[str, str]
    ) -> None:
        """Validate presence of mandatory parameters with stricter checking."""
        param_col = column_names["parameter"]

        # Map Excel parameter names to internal names
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ][self.language]
        mandatory_fields = {
            "column_name_to_analyze",  # These are the internal names
            "focus_on",
            "max_keywords",
        }

        # Create reverse mapping from Excel names to internal names
        excel_to_internal = {
            excel: internal for excel, internal in param_mappings.items()
        }

        # Get actual parameters from DataFrame
        found_params = set()
        for _, row in df.iterrows():
            excel_param = row[param_col]
            if excel_param in excel_to_internal:
                internal_name = excel_to_internal[excel_param]
                found_params.add(internal_name)

        # Find missing mandatory fields
        missing = mandatory_fields - found_params
        if missing:
            # Get Excel names for missing fields
            missing_excel = []
            for internal in missing:
                excel_names = [
                    excel
                    for excel, int_name in param_mappings.items()
                    if int_name == internal
                ]
                missing_excel.extend(excel_names)

            raise ValueError(
                f"Missing mandatory parameters: {', '.join(missing_excel)}"
            )

    def _load_and_validate_parameters(self) -> None:
        """Load and validate parameters with mandatory field checking."""
        try:
            if not self.file_path or not self.file_path.exists():
                logger.debug("No parameter file specified, using defaults")
                default_config = self.config.DEFAULT_CONFIG.copy()
                default_config["general"]["language"] = self.language
                self.parameters = ParameterSet(**default_config)
                return

            # Load all sheets from Excel
            sheets = self.file_utils.load_excel_sheets(self.file_path)
            config = {}

            # Load General Parameters
            general_sheet = ParameterSheets.get_sheet_name(
                "general", self.language
            )
            if general_sheet in sheets:
                # First validate mandatory fields
                self._validate_mandatory_fields(
                    sheets[general_sheet],
                    ParameterSheets.get_column_names("general", self.language),
                )
                # Then parse parameters
                config["general"] = self._parse_general_parameters(
                    sheets[general_sheet]
                )
            else:
                logger.warning(f"Required sheet '{general_sheet}' not found")
                raise ValueError(f"Required sheet '{general_sheet}' not found")
            # Load Categories
            categories_sheet = ParameterSheets.get_sheet_name(
                "categories", self.language
            )
            if categories_sheet in sheets:
                config["categories"] = self._parse_categories(
                    sheets[categories_sheet]
                )
            else:
                logger.debug(
                    "No categories sheet found, using empty categories"
                )
                config["categories"] = {}

            # Load Predefined Keywords
            keywords_sheet = ParameterSheets.get_sheet_name(
                "keywords", self.language
            )
            if keywords_sheet in sheets:
                config["predefined_keywords"] = self._parse_keywords(
                    sheets[keywords_sheet]
                )
            else:
                logger.debug("No predefined keywords sheet found")
                config["predefined_keywords"] = {}

            # Load Excluded Keywords
            excluded_sheet = ParameterSheets.get_sheet_name(
                "excluded", self.language
            )
            if excluded_sheet in sheets:
                config["excluded_keywords"] = self._parse_excluded_keywords(
                    sheets[excluded_sheet]
                )
            else:
                logger.debug("No excluded keywords sheet found")
                config["excluded_keywords"] = set()

            # Load Domain Context
            domains_sheet = ParameterSheets.get_sheet_name(
                "domains", self.language
            )
            if domains_sheet in sheets:
                config["domain_context"] = self._parse_domains(
                    sheets[domains_sheet]
                )
            else:
                logger.debug("No domain context sheet found")
                config["domain_context"] = {}

            # Load Analysis Settings
            settings_sheet = ParameterSheets.get_sheet_name(
                "settings", self.language
            )
            if settings_sheet in sheets:
                config["analysis_settings"] = self._parse_settings(
                    sheets[settings_sheet]
                )
            else:
                logger.debug("No settings sheet found, using defaults")
                config["analysis_settings"] = self.config.DEFAULT_CONFIG[
                    "analysis_settings"
                ]

            # Don't override explicit values with defaults for general params
            if "language" not in config["general"]:
                config["general"]["language"] = self.language
            if "min_keyword_length" not in config["general"]:
                config["general"]["min_keyword_length"] = 3
            if "include_compounds" not in config["general"]:
                config["general"]["include_compounds"] = True

            # Create ParameterSet with complete config
            self.parameters = ParameterSet(**config)
            logger.debug(f"Loaded parameters from {self.file_path}")

            # Validate loaded parameters using validator instance
            is_valid, warnings, errors = self.validator.validate_parameters(
                self.parameters.model_dump()
            )
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {errors}")
            if warnings:
                for warning in warnings:
                    logger.warning(warning)

        # except Exception as e:
        #     logger.error(f"Error loading parameters: {e}")
        #     raise ValueError(str(e))
        except ValueError as e:
            # Re-raise ValueError directly to maintain the correct exception type
            raise
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            raise ValueError(str(e))

    def _parse_general_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse general parameters with validation."""
        params = {}
        column_names = ParameterSheets.get_column_names(
            "general", self.language
        )
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ][self.language]
        param_col = column_names["parameter"]
        value_col = column_names["value"]

        # This validation now happens before we get here
        excel_mappings = {
            excel: internal for excel, internal in param_mappings.items()
        }

        for _, row in df.iterrows():
            excel_name = row[param_col]
            if excel_name in excel_mappings:
                internal_name = excel_mappings[excel_name]
                params[internal_name] = self._convert_value(row[value_col])

        return params

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
            logger.warning("No categories found in parameter file")
            return categories

        try:
            column_names = ParameterSheets.get_column_names(
                "categories", self.language
            )
            cat_col = column_names["category"]
            logger.debug(
                f"Parsing categories with columns: {column_names}"
            )  # Debug columns

            if cat_col not in df.columns:
                logger.warning(
                    f"Category column '{cat_col}' not found. Available columns: {df.columns.tolist()}"
                )
                return categories

            for _, row in df.iterrows():
                if pd.notna(row[cat_col]):
                    cat_name = str(row[cat_col]).strip()
                    config = CategoryConfig(
                        description=str(
                            row.get(column_names["description"], "")
                        ),
                        keywords=self._split_list(
                            row.get(column_names["keywords"], "")
                        ),
                        threshold=float(
                            row.get(column_names["threshold"], 0.5)
                        ),
                        parent=(
                            row.get(column_names["parent"])
                            if pd.notna(row.get(column_names["parent"]))
                            else None
                        ),
                    )
                    categories[cat_name] = config
                    logger.debug(
                        f"Loaded category: {cat_name} with config: {config}"
                    )

            return categories

        except Exception as e:
            logger.error(f"Error parsing categories: {str(e)}", exc_info=True)
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

    def _convert_value(self, value: Any) -> Any:
        """Enhanced value conversion with type preservation."""
        if pd.isna(value):
            return None

        if isinstance(value, str):
            # Handle boolean strings
            value_lower = value.lower()
            if value_lower == "true":
                return True
            if value_lower == "false":
                return False

            # Try number conversion
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                # Keep string values as-is
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
