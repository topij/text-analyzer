# tests/conftest.py

import pytest
import logging
import os
from pathlib import Path
from typing import Dict, Any

from src.core.config import AnalyzerConfig
from src.config.manager import ConfigManager
from FileUtils import FileUtils


TEST_DIRECTORY_STRUCTURE = {
    "data": ["raw", "processed", "config", "parameters"],
    "logs": [],
    "reports": [],
    "models": [],
}

# Test environment variables
TEST_ENV_VARS = {
    "OPENAI_API_KEY": "sk-test-key-123",
    "ANTHROPIC_API_KEY": "anthro-test-key-123",
    "AZURE_OPENAI_API_KEY": "azure-test-key-123",
    "AZURE_OPENAI_ENDPOINT": "https://test-endpoint.azure.com",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
}


def pytest_configure(config):
    """Configure pytest with environment setup."""
    # Set up test environment variables
    for key, value in TEST_ENV_VARS.items():
        os.environ[key] = value

    # Configure test markers
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(scope="session")
def file_utils(tmp_path_factory) -> FileUtils:
    """Create FileUtils instance for testing."""
    tmp_path = tmp_path_factory.mktemp("test_data")
    return FileUtils(
        project_root=tmp_path,
        directory_structure=TEST_DIRECTORY_STRUCTURE,
        create_directories=True,
    )


@pytest.fixture(scope="session")
def test_config_manager(
    tmp_path_factory, file_utils: FileUtils
) -> ConfigManager:
    """Create ConfigManager with test configuration."""
    # Create test config directory
    tmp_path = tmp_path_factory.mktemp("test_config")
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    # Create test config file
    config_file = config_dir / "config.yaml"
    config_content = """
logging:
    level: DEBUG
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
models:
    default_provider: openai
    default_model: gpt-4o-mini
    parameters:
        temperature: 0.0
        max_tokens: 1000
languages:
    default_language: en
    languages:
        en:
            min_word_length: 3
        fi:
            min_word_length: 3
            voikko_path: null
features:
    use_caching: true
    batch_processing: true
analysis:
    keywords:
        max_keywords: 10
        min_confidence: 0.3
        weights:
            statistical: 0.4
            llm: 0.6
    themes:
        max_themes: 3
        min_confidence: 0.3
    categories:
        min_confidence: 0.3
    """
    config_file.write_text(config_content)

    # Create .env file in test config directory
    env_file = config_dir / ".env"
    env_content = "\n".join(
        f"{key}={value}" for key, value in TEST_ENV_VARS.items()
    )
    env_file.write_text(env_content)

    # Initialize config manager with test setup
    config_manager = ConfigManager(
        file_utils=file_utils,
        config_dir=str(config_dir),
        project_root=tmp_path,
        custom_directory_structure=TEST_DIRECTORY_STRUCTURE,
    )

    return config_manager


@pytest.fixture(scope="session")
def test_analyzer_config(test_config_manager: ConfigManager) -> AnalyzerConfig:
    """Create AnalyzerConfig for testing."""
    return AnalyzerConfig(config_manager=test_config_manager)


@pytest.fixture
def test_parameters() -> Dict[str, Any]:
    """Provide test parameter configurations."""
    return {
        "general": {
            "max_keywords": 8,
            "min_keyword_length": 3,
            "language": "en",
            "focus_on": "technical content",
            "include_compounds": True,
            "max_themes": 3,
            "min_confidence": 0.3,
            "column_name_to_analyze": "content",
        },
        "categories": {
            "technical": {
                "description": "Technical content",
                "keywords": ["software", "api", "data"],
                "threshold": 0.6,
            },
            "business": {
                "description": "Business content",
                "keywords": ["revenue", "growth", "market"],
                "threshold": 0.6,
            },
        },
        "analysis_settings": {
            "theme_analysis": {"enabled": True, "min_confidence": 0.5},
            "weights": {"statistical": 0.4, "llm": 0.6},
        },
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_env():
    """Automatically set up test environment variables for each test."""
    # Store original environment variables
    original_env = {key: os.environ.get(key) for key in TEST_ENV_VARS}

    # Set test environment variables
    for key, value in TEST_ENV_VARS.items():
        os.environ[key] = value

    yield

    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
