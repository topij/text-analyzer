#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from FileUtils import FileUtils
from scripts.test_data_generator import TestDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize project structure and generate test data."""
    try:
        logger.info("Initializing project structure...")
        file_utils = FileUtils()
        
        # Initialize directory structure
        for dir_type in ["raw", "processed", "parameters", "config"]:
            path = file_utils.get_data_path(dir_type)
            logger.info(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        
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
        return 1

if __name__ == "__main__":
    sys.exit(main()) 