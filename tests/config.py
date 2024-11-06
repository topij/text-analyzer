# tests/config.py

"""Test configuration and setup."""


def get_test_config() -> dict:
    """Get test configuration."""
    return {
        "semantic_analyzer": {
            "default_language": "en",
            "content_column": "content",
            "analysis": {
                "keywords": {
                    "max_keywords": 5,
                    "min_keyword_length": 3,
                    "include_compounds": True,
                },
                "themes": {
                    "max_themes": 3,
                    "min_confidence": 0.5,
                    "include_hierarchy": True,
                },
                "categories": {
                    "max_categories": 3,
                    "min_confidence": 0.3,
                    "require_evidence": True,
                },
            },
            "models": {
                "default_provider": "openai",
                "default_model": "gpt-4o-mini",
                "parameters": {"temperature": 0.0, "max_tokens": 1000},
            },
        }
    }
