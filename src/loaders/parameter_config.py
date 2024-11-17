# src/loaders/parameter_config.py

from enum import Enum
from typing import Dict, Any


class ParameterSheets(str, Enum):
    """Excel sheet names with multilingual support."""

    # English sheet names
    GENERAL_EN = "General Parameters"
    KEYWORDS_EN = "Predefined Keywords"
    EXCLUDED_EN = "Excluded Keywords"
    CATEGORIES_EN = "Categories"
    PROMPTS_EN = "Custom Prompts"
    DOMAINS_EN = "Domain Context"
    SETTINGS_EN = "Analysis Settings"

    # Finnish sheet names
    GENERAL_FI = "yleiset säännöt"
    KEYWORDS_FI = "haettavat avainsanat"
    EXCLUDED_FI = "älä käytä"
    CATEGORIES_FI = "kategoriat"
    PROMPTS_FI = "kehotteet"
    DOMAINS_FI = "aihepiirin lisätietot"
    SETTINGS_FI = "lisäasetukset"

    @classmethod
    def get_sheet_name(cls, sheet_type: str, language: str = "en") -> str:
        """Get sheet name for specified language."""
        try:
            return getattr(cls, f"{sheet_type}_{language.upper()}").value
        except AttributeError:
            # Fallback to English if language-specific name not found
            return getattr(cls, f"{sheet_type}_EN").value


class ParameterConfigurations:
    """Configuration for parameter handling."""

    # Default configuration
    DEFAULT_CONFIG = {
        "general": {
            "max_keywords": 10,
            "min_keyword_length": 3,
            "language": "en",
            "focus_on": "general content analysis",  # Added default
            "include_compounds": True,
            "max_themes": 3,
            "min_confidence": 0.3,
            "column_name_to_analyze": "text",  # Added default
        },
        "categories": {},
        "predefined_keywords": {},
        "excluded_keywords": set(),
        "analysis_settings": {
            "theme_analysis": {"enabled": True, "min_confidence": 0.5},
            "weights": {"statistical": 0.4, "llm": 0.6},
        },
        "domain_context": {},
    }

    # Parameter name mappings
    PARAM_NAMES = {
        "general": {
            "en": {
                "parameter": "parameter",
                "value": "value",
                "description": "description",
                "max_keywords": "max_keywords",
                "language": "language",
                "focus_on": "focus_on",
                "min_keyword_length": "min_keyword_length",
                "include_compounds": "include_compounds",
                "column_name": "column_name_to_analyze",
            },
            "fi": {
                "parameter": "parametri",
                "value": "arvo",
                "description": "kuvaus",
                "max_keywords": "maksimi_avainsanat",
                "language": "kieli",
                "focus_on": "keskity_aihepiiriin",
                "min_keyword_length": "min_sanan_pituus",
                "include_compounds": "sisällytä_yhdyssanat",
                "column_name": "analysoitava_sarake",
            },
        },
        "categories": {
            "en": {
                "category": "category",
                "description": "description",
                "keywords": "keywords",
                "threshold": "threshold",
                "parent": "parent",
            },
            "fi": {
                "category": "kategoria",
                "description": "kuvaus",
                "keywords": "avainsanat",
                "threshold": "kynnysarvo",
                "parent": "yläkategoria",
            },
        },
        "domains": {
            "en": {
                "name": "name",
                "description": "description",
                "key_terms": "key_terms",
                "context": "context",
                "stopwords": "stopwords",
            },
            "fi": {
                "name": "nimi",
                "description": "kuvaus",
                "key_terms": "keskeiset_termit",
                "context": "konteksti",
                "stopwords": "ohitettavat_sanat",
            },
        },
        "keywords": {
            "en": {
                "keyword": "keyword",
                "importance": "importance",
                "domain": "domain",
                "compound_parts": "compound_parts",
            },
            "fi": {
                "keyword": "avainsana",
                "importance": "tärkeys",
                "domain": "aihepiiri",
                "compound_parts": "yhdyssanan_osat",
            },
        },
    }

    @classmethod
    def get_column_names(cls, sheet_type: str, language: str = "en") -> Dict[str, str]:
        """Get column names for specified sheet type and language."""
        return cls.PARAM_NAMES.get(sheet_type, {}).get(language, {})

    @classmethod
    def get_column_name(cls, sheet_type: str, column: str, language: str = "en") -> str:
        """Get specific column name for language."""
        columns = cls.get_column_names(sheet_type, language)
        return columns.get(column, column)  # Fallback to original name

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get copy of default configuration."""
        return cls.DEFAULT_CONFIG.copy()

    @classmethod
    def detect_language(cls, file_path: str) -> str:
        """Detect language from file name."""
        file_path = file_path.lower()
        if any(fi_indicator in file_path for fi_indicator in ["parametrit", "suomi", "finnish"]):
            return "fi"
        return "en"
