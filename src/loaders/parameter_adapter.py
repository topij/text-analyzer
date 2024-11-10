# src/loaders/parameter_adapter.py

from typing import Optional, Union, Dict, Any, Set, List
from pathlib import Path
import pandas as pd
import logging
from pydantic import BaseModel

from .models import (
    GeneralParameters,
    CategoryConfig,
    PredefinedKeyword,
    AnalysisSettings,
    ParameterSet,
    DomainContext
)
from .parameter_config import ParameterConfigurations, ParameterSheets
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)

class ParameterValidator:
    """Validator for analysis parameters."""

    def validate(self, params: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
        """Validate parameters and return validation results."""
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

        except Exception as e:
            errors.append(str(e))
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

class ParameterAdapter:
    """Adapter for loading and converting analysis parameters."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """Initialize adapter with parameter file path."""
        self.file_utils = FileUtils()
        self.file_path = Path(file_path) if file_path else None
        self.parameters = None
        self.language = self._detect_file_language()
        self.config = ParameterConfigurations()
        self.load_and_convert()
        logger.debug(f"Initialized ParameterAdapter with file: {file_path}, language: {self.language}")

    def _detect_file_language(self) -> str:
        """Detect language from file path."""
        if not self.file_path:
            return "en"
        return ParameterConfigurations.detect_language(str(self.file_path))

    def _load_excel(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Load sheets with language-specific names."""
        try:
            excel_data = self.file_utils.load_excel_sheets(file_path)
            sheets = {}
            
            # Map sheets using language-specific names
            sheet_types = ["GENERAL", "CATEGORIES", "KEYWORDS", "EXCLUDED", "SETTINGS", "DOMAINS", "PROMPTS"]
            
            for sheet_type in sheet_types:
                sheet_name = ParameterSheets.get_sheet_name(sheet_type, self.language)
                if sheet_name in excel_data:
                    sheets[sheet_type.lower()] = excel_data[sheet_name]
                elif self.language != "en":
                    # Try English name as fallback
                    en_name = ParameterSheets.get_sheet_name(sheet_type, "en")
                    if en_name in excel_data:
                        sheets[sheet_type.lower()] = excel_data[en_name]
            
            return sheets
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise

    def _parse_general_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse general parameters with language support."""
        if df.empty:
            return {}

        params = ParameterConfigurations.get_default_config()["general"].copy()
        
        # Get column names for current language
        param_col = ParameterConfigurations.get_column_name("general", "parameter", self.language)
        value_col = ParameterConfigurations.get_column_name("general", "value", self.language)
        
        if param_col in df.columns and value_col in df.columns:
            for _, row in df.iterrows():
                param = row[param_col]
                value = row[value_col]
                if pd.notna(value):
                    # Get internal parameter name
                    internal_name = next(
                        (k for k, v in ParameterConfigurations.get_column_names("general", self.language).items() 
                         if v == param),
                        param
                    )
                    params[internal_name] = value

        return params

    def _parse_categories(self, df: pd.DataFrame) -> Dict[str, CategoryConfig]:
        """Parse categories sheet with language support."""
        categories = {}
        if df is not None and not df.empty:
            # Get column names for current language
            col_names = ParameterConfigurations.get_column_names("categories", self.language)
            
            for _, row in df.iterrows():
                category_col = col_names.get("category", "category")
                if pd.notna(row.get(category_col)):
                    categories[row[category_col]] = CategoryConfig(
                        description=row.get(col_names.get("description", "description"), ""),
                        keywords=(
                            str(row.get(col_names.get("keywords", "keywords"), "")).split(",")
                            if pd.notna(row.get(col_names.get("keywords", "keywords")))
                            else []
                        ),
                        threshold=float(row.get(col_names.get("threshold", "threshold"), 0.5)),
                        parent=(
                            row.get(col_names.get("parent", "parent"))
                            if pd.notna(row.get(col_names.get("parent", "parent")))
                            else None
                        ),
                    )
        return categories

    def _parse_predefined_keywords(self, df: pd.DataFrame) -> Dict[str, PredefinedKeyword]:
        """Parse predefined keywords sheet with language support."""
        if df.empty:
            return {}

        keywords = {}
        col_names = ParameterConfigurations.get_column_names("keywords", self.language)
        
        for _, row in df.iterrows():
            keyword_col = col_names.get("keyword", "keyword")
            if pd.notna(row.get(keyword_col)):
                keyword = row[keyword_col]
                keywords[keyword] = PredefinedKeyword(
                    importance=float(row.get(col_names.get("importance", "importance"), 1.0)),
                    domain=row.get(col_names.get("domain", "domain"))
                    if pd.notna(row.get(col_names.get("domain", "domain")))
                    else None,
                    compound_parts=(
                        str(row.get(col_names.get("compound_parts", "compound_parts"), "")).split(",")
                        if pd.notna(row.get(col_names.get("compound_parts", "compound_parts")))
                        else []
                    ),
                )
        return keywords

    def _parse_domain_context(self, df: pd.DataFrame) -> Dict[str, DomainContext]:
        """Parse domain context sheet with language support."""
        domains = {}
        if df is not None and not df.empty:
            col_names = ParameterConfigurations.get_column_names("domains", self.language)
            
            for _, row in df.iterrows():
                name_col = col_names.get("name", "name")
                if pd.notna(row.get(name_col)):
                    domains[row[name_col]] = DomainContext(
                        name=row[name_col],
                        description=row.get(col_names.get("description", "description"), ""),
                        key_terms=(
                            [term.strip() for term in 
                             str(row.get(col_names.get("key_terms", "key_terms"), "")).split(",")]
                            if pd.notna(row.get(col_names.get("key_terms", "key_terms")))
                            else []
                        ),
                        context=row.get(col_names.get("context", "context"), ""),
                        stopwords=(
                            [word.strip() for word in 
                             str(row.get(col_names.get("stopwords", "stopwords"), "")).split(",")]
                            if pd.notna(row.get(col_names.get("stopwords", "stopwords")))
                            else []
                        )
                    )
        return domains

    def load_and_convert(self) -> ParameterSet:
        """Load and convert parameters from file."""
        try:
            if not self.file_path or not self.file_path.exists():
                logger.debug("Using default parameters")
                self.parameters = ParameterSet(**ParameterConfigurations.get_default_config())
                return self.parameters

            # Load and parse sheets
            sheets = self._load_excel(self.file_path)
            
            # Parse all sheets using language-specific column names
            config = {
                "general": {
                    **ParameterConfigurations.get_default_config()["general"],
                    **self._parse_general_parameters(sheets.get("general", pd.DataFrame()))
                },
                "categories": self._parse_categories(sheets.get("categories", pd.DataFrame())),
                "predefined_keywords": self._parse_predefined_keywords(sheets.get("keywords", pd.DataFrame())),
                "excluded_keywords": self._parse_excluded_keywords(sheets.get("excluded", pd.DataFrame())),
                "analysis_settings": self._parse_analysis_settings(sheets.get("settings", pd.DataFrame())),
                "domain_context": self._parse_domain_context(sheets.get("domains", pd.DataFrame()))
            }

            self.parameters = ParameterSet(**config)
            return self.parameters
            
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.parameters = ParameterSet(**ParameterConfigurations.get_default_config())
            return self.parameters

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