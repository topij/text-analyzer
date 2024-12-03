# tests/conftest.py

import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Generator

from src.core.config import AnalyzerConfig
from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel

import pytest

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pytest.PytestDeprecationWarning)

from FileUtils import FileUtils


@pytest.fixture(scope="session")
def analyzer_config(file_utils: FileUtils) -> AnalyzerConfig:
    """Create AnalyzerConfig instance for testing."""
    return AnalyzerConfig(file_utils=file_utils)


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
