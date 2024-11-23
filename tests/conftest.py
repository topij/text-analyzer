# tests/conftest.py

import os
import sys
from pathlib import Path
import pytest
import warnings
import asyncio
from typing import Dict, Generator

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pytest.PytestDeprecationWarning)

from src.utils.FileUtils.file_utils import FileUtils


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
            "technical": """Machine learning models are trained using large datasets to recognize patterns.
                        Neural network architecture includes multiple layers for feature extraction.
                        Data preprocessing and feature engineering are crucial steps.""",
            "business": """Q3 financial results show 15% revenue growth and improved profit margins.
                        Customer acquisition costs decreased while retention rates increased.
                        Market expansion strategy focuses on emerging technology sectors.""",
        },
        "fi": {
            "technical": """Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja.
                        Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen.
                        Datan esikäsittely ja piirteiden suunnittelu ovat keskeisiä vaiheita.""",
            "business": """Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet.
                        Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani.
                        Markkinalaajennusstrategia keskittyy nouseviin teknologiasektoreihin.""",
        },
    }
