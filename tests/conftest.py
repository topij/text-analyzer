# tests/conftest.py

import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Generator, Any
from dotenv import load_dotenv
import logging

import pytest

from src.config.manager import ConfigManager, LoggingConfig
from src.core.config import AnalyzerConfig
from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel
from FileUtils import FileUtils

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pytest.PytestDeprecationWarning)

logger = logging.getLogger(__name__)


# Load environment variables at the start of test session
def pytest_sessionstart(session):
    """Load environment variables before test session starts."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
        logger.info(f"Loaded environment from {env_path}")


@pytest.fixture(scope="session")
def project_path() -> Path:
    """Get project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def file_utils(project_path: Path) -> FileUtils:
    """Create FileUtils instance."""
    return FileUtils(project_root=project_path)


@pytest.fixture(scope="session")
def config_manager(file_utils: FileUtils) -> ConfigManager:
    """Create ConfigManager instance for testing."""
    return ConfigManager(file_utils=file_utils, config_dir="config")


@pytest.fixture(scope="session")
def analyzer_config(config_manager: ConfigManager) -> AnalyzerConfig:
    """Create AnalyzerConfig instance for testing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment")

    config = AnalyzerConfig(
        file_utils=config_manager.file_utils, config_manager=config_manager
    )

    # Override test-specific settings
    config.config["models"]["default_model"] = "gpt-4o-mini"
    return config


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Basic test configuration."""
    return {
        "language": "en",
        "min_confidence": 0.3,
        "focus_on": "test content",
        "max_keywords": 10,
        "min_keyword_length": 3,
        "include_compounds": True,
        "max_themes": 3,
    }


@pytest.fixture(scope="session")
def mock_llm(analyzer_config: AnalyzerConfig) -> BaseChatModel:
    """Create mock LLM instance."""
    return create_llm(config=analyzer_config)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_path() -> Path:
    """Get project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def file_utils(project_path: Path) -> FileUtils:
    """Create FileUtils instance."""
    return FileUtils(project_root=project_path)


@pytest.fixture(scope="session")
def test_content() -> Dict[str, Dict[str, str]]:
    """Provide test content."""
    return {
        "en": {
            "technical": """Machine learning models are trained using large datasets.
                        Neural network architecture includes multiple layers.
                        Data preprocessing and feature engineering are crucial.""",
            "business": """Q3 financial results show 15% revenue growth.
                        Customer acquisition costs decreased while retention improved.
                        Market expansion strategy focuses on emerging sectors.""",
        },
    }
