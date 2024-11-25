# src/loaders/parameter_config.py

from enum import Enum
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ParameterSheets:
    """Excel parameter configuration with language support."""

    SHEET_MAPPING = {
        "general": {"en": "General Parameters", "fi": "yleiset säännöt"},
        "keywords": {"en": "Predefined Keywords", "fi": "haettavat avainsanat"},
        "excluded": {"en": "Excluded Keywords", "fi": "älä käytä"},
        "categories": {"en": "Categories", "fi": "kategoriat"},
        "domains": {"en": "Domain Context", "fi": "aihepiirin lisätietot"},
        "settings": {"en": "Analysis Settings", "fi": "lisäasetukset"},
    }
    # Parameter name mapping to internal names
    PARAMETER_MAPPING = {
        "general": {
            "columns": {
                "en": {
                    "parameter": "parameter",
                    "value": "value",
                    "description": "description",
                },
                "fi": {
                    "parameter": "parametri",
                    "value": "arvo",
                    "description": "kuvaus",
                },
            },
            "parameters": {
                "en": {
                    "max_keywords": "max_keywords",
                    "language": "language",
                    "focus_on": "focus_on",
                    "min_keyword_length": "min_keyword_length",
                    "include_compounds": "include_compounds",
                    "column_name_to_analyze": "column_name_to_analyze",
                    "max_themes": "max_themes",
                    "min_confidence": "min_confidence",
                },
                "fi": {
                    "maksimi_avainsanat": "max_keywords",
                    "kieli": "language",
                    "keskity_aihepiiriin": "focus_on",
                    "min_sanan_pituus": "min_keyword_length",
                    "sisällytä_yhdyssanat": "include_compounds",
                    "analysoitava_sarake": "column_name_to_analyze",
                    "maksimi_teemat": "max_themes",
                    "min_luottamus": "min_confidence",
                },
            },
        },
        "keywords": {
            "columns": {
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
            "parameters": {
                "en": {
                    "keyword": "keyword",
                    "importance": "importance",
                    "domain": "domain",
                    "compound_parts": "compound_parts",
                },
                "fi": {
                    "avainsana": "keyword",
                    "tärkeys": "importance",
                    "aihepiiri": "domain",
                    "yhdyssanan_osat": "compound_parts",
                },
            },
        },
        "excluded": {
            "columns": {
                "en": {"keyword": "keyword", "reason": "reason"},
                "fi": {"keyword": "avainsana", "reason": "syy"},
            },
            "parameters": {
                "en": {"keyword": "keyword", "reason": "reason"},
                "fi": {"avainsana": "keyword", "syy": "reason"},
            },
        },
        "categories": {
            "columns": {
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
            "parameters": {
                "en": {
                    "category": "category",
                    "description": "description",
                    "keywords": "keywords",
                    "threshold": "threshold",
                    "parent": "parent",
                },
                "fi": {
                    "kategoria": "category",
                    "kuvaus": "description",
                    "avainsanat": "keywords",
                    "kynnysarvo": "threshold",
                    "yläkategoria": "parent",
                },
            },
        },
        "domains": {
            "columns": {
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
            "parameters": {
                "en": {
                    "name": "name",
                    "description": "description",
                    "key_terms": "key_terms",
                    "context": "context",
                    "stopwords": "stopwords",
                },
                "fi": {
                    "nimi": "name",
                    "kuvaus": "description",
                    "keskeiset_termit": "key_terms",
                    "konteksti": "context",
                    "ohitettavat_sanat": "stopwords",
                },
            },
        },
        "settings": {
            "columns": {
                "en": {
                    "setting": "setting",
                    "value": "value",
                    "description": "description",
                },
                "fi": {
                    "setting": "asetus",
                    "value": "arvo",
                    "description": "kuvaus",
                },
            },
            "parameters": {
                "en": {
                    "theme_analysis.min_confidence": "theme_analysis.min_confidence",
                    "weights.statistical": "weights.statistical",
                    "weights.llm": "weights.llm",
                },
                "fi": {
                    "teema_analyysi.min_luottamus": "theme_analysis.min_confidence",
                    "painot.tilastollinen": "weights.statistical",
                    "painot.llm": "weights.llm",
                },
            },
        },
    }

    @classmethod
    def get_sheet_name(cls, internal_name: str, language: str = "en") -> str:
        """Get the official sheet name for a given language."""
        lang = language.lower()[:2]
        if lang not in ["en", "fi"]:
            lang = "en"

        try:
            sheet_name = cls.SHEET_MAPPING[internal_name][lang]
            logger.debug(
                f"Sheet name for {internal_name} ({lang}): {sheet_name}"
            )
            return sheet_name
        except KeyError:
            logger.warning(
                f"Sheet name not found for {internal_name}, using English"
            )
            return cls.SHEET_MAPPING[internal_name]["en"]

    @classmethod
    def validate_sheet_name(
        cls, sheet_name: str, internal_name: str, language: str = "en"
    ) -> bool:
        """Validate that a sheet name exactly matches the expected name."""
        expected_name = cls.get_sheet_name(internal_name, language)
        return sheet_name == expected_name  # Strict equality check

    @classmethod
    def get_column_names(
        cls, sheet_type: str, language: str = "en"
    ) -> Dict[str, str]:
        """Get column names for a specific sheet type and language."""
        lang = language.lower()[:2]
        if lang not in ["en", "fi"]:
            lang = "en"

        try:
            return cls.PARAMETER_MAPPING[sheet_type]["columns"][lang]
        except KeyError:
            logger.warning(
                f"Column names not found for {sheet_type}, using English"
            )
            return cls.PARAMETER_MAPPING[sheet_type]["columns"]["en"]

    @classmethod
    def get_parameter_mapping(
        cls, sheet_type: str, language: str = "en"
    ) -> Dict[str, str]:
        """Get parameter mapping for a specific sheet type and language."""
        lang = language.lower()[:2]
        if lang not in ["en", "fi"]:
            lang = "en"

        try:
            return cls.PARAMETER_MAPPING[sheet_type]["parameters"][lang]
        except KeyError:
            logger.warning(
                f"Parameter mapping not found for {sheet_type}, using English"
            )
            return cls.PARAMETER_MAPPING[sheet_type]["parameters"]["en"]


class ParameterConfigurations:
    """Default configurations and utilities."""

    DEFAULT_CONFIG = {
        "general": {
            "max_keywords": 10,
            "min_keyword_length": 3,
            "language": "en",
            "focus_on": "general content analysis",
            "include_compounds": True,
            "max_themes": 3,
            "min_confidence": 0.3,
            "column_name_to_analyze": "text",
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

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get copy of default configuration."""
        return cls.DEFAULT_CONFIG.copy()

    @classmethod
    def detect_language(cls, file_path: str) -> str:
        """Detect language from file name."""
        file_path = file_path.lower()
        if any(
            fi_indicator in file_path
            for fi_indicator in ["fi", "fin", "finnish", "suomi"]
        ):
            return "fi"
        return "en"
