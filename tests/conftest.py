# tests/conftest.py

import pytest
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import yaml

from src.core.config import AnalyzerConfig
from src.config.manager import ConfigManager
from src.core.managers import EnvironmentManager, EnvironmentConfig
from FileUtils import FileUtils

# Configure logger
logger = logging.getLogger(__name__)

TEST_DIRECTORY_STRUCTURE = {
    "data": ["raw", "processed", "config/stop_words", "parameters"],
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
    "APP_LOGGING_LEVEL": "DEBUG",
    "APP_MODELS_DEFAULT_PROVIDER": "openai",
    "APP_MODELS_DEFAULT_MODEL": "gpt-4o-mini",
    "VOIKKO_LIBRARY_PATH": "/opt/homebrew/lib/libvoikko.dylib",
}


def pytest_configure(config):
    """Configure pytest with environment setup."""
    # Set up test environment variables
    for key, value in TEST_ENV_VARS.items():
        os.environ[key] = value

    # Configure test markers
    config.addinivalue_line("markers", "asyncio: mark test as async")


def copy_stopwords_files(test_root: Path) -> None:
    """Copy stopwords files from repository to test directory."""
    src_dir = Path(__file__).parent.parent / "data" / "config" / "stop_words"
    dest_dir = test_root / "data" / "config" / "stop_words"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying stopwords files from {src_dir} to {dest_dir}")
    
    for file in ["en.txt", "fi.txt"]:
        src_file = src_dir / file
        if src_file.exists():
            logger.info(f"Found source file: {src_file}")
            shutil.copy2(src_file, dest_dir / file)
            logger.info(f"Copied {file} to {dest_dir / file}")
        else:
            # Try absolute path from workspace root
            workspace_src = Path.cwd() / "data" / "config" / "stop_words" / file
            if workspace_src.exists():
                logger.info(f"Found source file at workspace path: {workspace_src}")
                shutil.copy2(workspace_src, dest_dir / file)
                logger.info(f"Copied {file} to {dest_dir / file}")
            else:
                logger.warning(f"Stopwords file not found: {file} (tried {src_file} and {workspace_src})")


@pytest.fixture
def test_root() -> Generator[Path, None, None]:
    """Create a temporary test directory with required structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure
        for parent, children in TEST_DIRECTORY_STRUCTURE.items():
            parent_dir = temp_path / parent
            parent_dir.mkdir(exist_ok=True)
            for child in children:
                child_path = parent_dir / child
                if "/" in child:
                    child_path.mkdir(parents=True, exist_ok=True)
                else:
                    child_path.mkdir(exist_ok=True)
        
        # Copy stopwords files
        copy_stopwords_files(temp_path)
        
        yield temp_path


@pytest.fixture(scope="session")
def file_utils(tmp_path_factory) -> FileUtils:
    """Create FileUtils instance for testing."""
    tmp_path = tmp_path_factory.mktemp("test_data")
    
    # Initialize FileUtils
    utils = FileUtils(
        project_root=tmp_path,
        directory_structure=TEST_DIRECTORY_STRUCTURE,
        create_directories=True,
    )
    
    # Copy stopwords files to the test data directory
    copy_stopwords_files(tmp_path)
    
    return utils


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
    providers:
        openai:
            available_models:
                gpt-4o-mini:
                    description: "Fast and cost-effective for simpler tasks"
                    max_tokens: 4096
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


@pytest.fixture
def test_environment_manager(file_utils: FileUtils, test_config_manager: ConfigManager) -> Generator[EnvironmentManager, None, None]:
    """Create EnvironmentManager instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create necessary subdirectories
        config_dir = temp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy stopwords files
        copy_stopwords_files(temp_path)
        
        # Create config content
        config_content = """
environment: production
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  disable_existing_loggers: false

models:
  default_provider: mock
  default_model: mock-model
  parameters:
    temperature: 0.7
    max_tokens: 1000
  providers:
    mock:
      api_key: mock-key
      models:
        - mock-model
        - mock-model-large

languages:
  default_language: en
  languages:
    en:
      name: English
      code: en
    fi:
      name: Finnish
      code: fi
      voikko_path: "/opt/homebrew/lib/libvoikko.dylib"
      voikko_dict_path: "/opt/homebrew/lib/voikko"

features:
  use_cache: true
  batch_processing: true

analysis:
  keywords:
    min_confidence: 0.3
    max_keywords: 10
    min_keyword_length: 3
    include_compounds: true
    language: en
    focus_on: general content analysis
  themes:
    min_confidence: 0.3
    max_themes: 5
    language: en
    focus_on: general content analysis
  categories:
    min_confidence: 0.3
    language: en
    focus_on: general content analysis
"""
        # Create config.yaml
        base_config = config_dir / "config.yaml"
        base_config.write_text(config_content)
        
        # Initialize environment manager with test components
        config = EnvironmentConfig(
            project_root=temp_path,
            custom_directory_structure=TEST_DIRECTORY_STRUCTURE,
            config_dir=str(config_dir),
        )
        
        # Reset singleton state
        EnvironmentManager._instance = None
        EnvironmentManager._initialized = False
        
        # Create environment manager instance
        environment = EnvironmentManager(config=config)
        
        # Set up test components
        environment._components = {
            "file_utils": file_utils,
            "config_manager": test_config_manager,
        }
        
        yield environment
        
        # Cleanup
        environment._initialized = False
        environment._instance = None


@pytest.fixture(scope="session")
def test_analyzer_config(test_environment_manager: EnvironmentManager) -> AnalyzerConfig:
    """Create AnalyzerConfig for testing."""
    components = test_environment_manager.get_components()
    return AnalyzerConfig(
        file_utils=components["file_utils"],
        config_manager=components["config_manager"]
    )


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


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
