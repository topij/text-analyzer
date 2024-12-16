# tests/helpers/config.py

from typing import Dict, Any, Optional


def create_test_config(
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create test configuration with optional overrides."""
    base_config = {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "models": {
            "default_provider": "openai",
            "default_model": "gpt-4o-mini",
            "parameters": {"temperature": 0.0, "max_tokens": 1000},
        },
        "languages": {
            "default_language": "en",
            "languages": {
                "en": {
                    "min_word_length": 3,
                    "excluded_patterns": [r"^\d+$", r"^[^a-zA-Z0-9]+$"],
                },
                "fi": {
                    "min_word_length": 3,
                    "excluded_patterns": [r"^\d+$", r"^[^a-zA-ZäöåÄÖÅ0-9]+$"],
                    "voikko_path": None,
                },
            },
        },
        "features": {"use_caching": True, "batch_processing": True},
        "analysis": {
            "keywords": {
                "max_keywords": 10,
                "min_confidence": 0.3,
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            "themes": {"max_themes": 3, "min_confidence": 0.3},
            "categories": {"min_confidence": 0.3},
        },
    }

    if config_override:
        _deep_merge(base_config, config_override)

    return base_config


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
    """Deep merge two dictionaries."""
    for key, value in dict2.items():
        if (
            isinstance(value, dict)
            and key in dict1
            and isinstance(dict1[key], dict)
        ):
            _deep_merge(dict1[key], value)
        else:
            dict1[key] = value
