#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path
import textwrap

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from FileUtils import FileUtils
from scripts.test_data_generator import TestDataGenerator
from src.config.manager import ConfigManager
from src.core.managers.environment_manager import EnvironmentManager, EnvironmentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env_setup(project_root: Path) -> bool:
    """Check if .env file exists and contains required variables."""
    env_file = project_root / ".env"
    env_template = project_root / "_archive" / ".env.test"
    
    if not env_file.exists():
        logger.warning("\n" + textwrap.dedent("""
            ⚠️  No .env file found! This is required for full functionality.
            
            Please create a .env file in the project root with the required environment variables.
            You can use the template at _archive/.env.test as a starting point:
            
            1. Copy _archive/.env.test to .env in the project root
            2. Fill in the required variables, especially:
               - OPENAI_API_KEY (required for test data generation)
               - ENV (development or production)
               - APP_LOGGING_LEVEL (INFO recommended for development)
            
            Example minimal .env file:
            ```
            ENV=development
            OPENAI_API_KEY=your_api_key_here
            APP_LOGGING_LEVEL=INFO
            ```
        """).strip())
        return False
        
    # Check for critical environment variables
    required_vars = ["OPENAI_API_KEY", "ENV"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning("\n" + textwrap.dedent(f"""
            ⚠️  Missing required environment variables: {', '.join(missing_vars)}
            
            Please ensure these variables are set in your .env file.
            Check the template at _archive/.env.test for reference.
        """).strip())
        return False
        
    return True

def main():
    """Initialize project structure and generate test data."""
    try:
        logger.info("Initializing project structure...")
        
        # Create FileUtils instance with project root
        project_root = Path(__file__).resolve().parent.parent
        file_utils = FileUtils(project_root=project_root)
        
        # Check environment setup first
        env_ok = check_env_setup(project_root)
        if not env_ok:
            logger.warning("\nContinuing setup with limited functionality. Some features may not work properly.")
        
        # Initialize directory structure
        for dir_type in ["raw", "processed", "parameters", "config"]:
            path = file_utils.get_data_path(dir_type)
            logger.info(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment with ConfigManager
        config_manager = ConfigManager(file_utils=file_utils)
        config_manager.init_environment()
        config_manager.init_paths()
        config_manager.init_file_utils()
        config_manager.load_configurations()

        # Initialize environment manager with shared FileUtils
        env_config = EnvironmentConfig(
            env_type="development",
            project_root=project_root,
            log_level="INFO"
        )
        environment = EnvironmentManager(env_config)
        
        # Set up shared components
        environment.file_utils = file_utils
        environment.config_manager = config_manager
        
        # Make sure the environment is properly initialized
        EnvironmentManager._instance = environment
        
        # Generate test data
        logger.info("Generating test data...")
        generator = TestDataGenerator(file_utils=file_utils)
        files = generator.generate_all(force=True)
        
        logger.info("Generated files:")
        for file_type, paths in files.items():
            for path in paths:
                logger.info(f"- {file_type}: {path}")
        
        logger.info("Project setup completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during project setup: {e}")
        if "OPENAI_API_KEY" in str(e):
            logger.error("\n" + textwrap.dedent("""
                ❌ Setup failed due to missing OpenAI API key.
                
                To fix this:
                1. Get an API key from https://platform.openai.com/api-keys
                2. Add it to your .env file as OPENAI_API_KEY=your_key_here
                3. Run the setup script again
            """).strip())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 