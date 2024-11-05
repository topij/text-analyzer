# tests/conftest.py

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env files
def pytest_configure(config):
    """Configure test environment."""
    # Priority order:
    # 1. .env.test (test-specific variables)
    # 2. .env (default variables)
    
    env_files = [
        Path(os.getcwd()) / '.env.test',
        Path(os.getcwd()) / '.env'
    ]
    
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded environment from {env_file}")

    # Ensure required environment variables are set
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in .env or .env.test file"
        )
    
    # Set test-specific environment variables
    os.environ['SEMANTIC_ANALYZER_ENV'] = 'test'
    os.environ['SEMANTIC_ANALYZER_CONFIG'] = str(Path(__file__).parent / 'config')

@pytest.fixture(scope='session')
def env_vars():
    """Fixture to provide environment variables to tests."""
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'VOIKKO_PATH': os.getenv('VOIKKO_PATH'),
        'SEMANTIC_ANALYZER_CONFIG': os.getenv('SEMANTIC_ANALYZER_CONFIG')
    }