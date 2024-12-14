# src/nb_helpers/nb_environment.py

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from src.core.config_management import ConfigManager, LoggingConfig
from src.core.config import AnalyzerConfig
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


def setup_notebook_environment(
    log_level: Optional[str] = "INFO",
    config_dir: str = "config",
    custom_directory_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Set up environment for notebook usage.

    Args:
        log_level: Optional logging level override
        config_dir: Name of config directory (default: "config")
        custom_directory_structure: Optional custom directory structure

    Returns:
        Dict containing initialized components
    """
    try:
        # Get project root correctly
        project_root = Path().resolve().parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        # Initialize FileUtils with proper root and logging
        file_utils = FileUtils(
            project_root=project_root,
            log_level=log_level,
            directory_structure=custom_directory_structure,
        )

        # Data directory should be under project root
        data_dir = project_root / "data"
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        # Initialize ConfigManager with custom structure if provided
        config_manager = ConfigManager(
            file_utils=file_utils,
            config_dir=config_dir,
            custom_directory_structure=custom_directory_structure,
            project_root=project_root,
        )

        # Set notebook-specific logging
        config_manager.config.logging = LoggingConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file_path="notebook.log",  # Will be created under data/logs
        )

        # Initialize analyzer config
        analyzer_config = AnalyzerConfig(
            file_utils=file_utils, config_manager=config_manager
        )

        # Validate directory structure
        if not config_manager.validate_directory_structure():
            logger.warning("Directory structure validation failed")

        # Create helpful message about environment setup
        logger.info(f"Notebook environment set up with root: {project_root}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Config directory: {config_manager.config_dir}")
        logger.info(f"Log level: {log_level}")

        return {
            "file_utils": file_utils,
            "config_manager": config_manager,
            "analyzer_config": analyzer_config,
            "project_root": project_root,
            "data_dir": data_dir,
        }

    except Exception as e:
        logger.error(f"Failed to set up notebook environment: {e}")
        raise


def verify_notebook_environment() -> bool:
    """Verify notebook environment setup with enhanced validation.

    Returns:
        bool: True if environment is properly configured
    """
    status = {
        "project_structure": False,
        "configuration": False,
        "environment_vars": False,
        "directory_structure": False,
    }

    try:
        # Initialize basic components
        file_utils = FileUtils()
        config_manager = ConfigManager(file_utils=file_utils)

        # Check project structure
        config_dir = file_utils.get_data_path("config")
        required_files = [
            "config.yaml",
            "models.yaml",
            "language_processing.yaml",
        ]
        missing_files = [
            f for f in required_files if not (config_dir / f).exists()
        ]

        if missing_files:
            logger.warning(f"Missing configuration files: {missing_files}")
        else:
            status["project_structure"] = True
            logger.info("Project structure verified")

        # Validate directory structure
        status["directory_structure"] = (
            config_manager.validate_directory_structure()
        )

        # Check configuration loading
        try:
            config = config_manager.get_config()
            status["configuration"] = True
            logger.info("Configuration loading verified")
        except Exception as e:
            logger.warning(f"Configuration loading failed: {e}")

        # Check environment variables
        required_vars = (
            config_manager.get_config()
            .get("models", {})
            .get("providers", {})
            .get(
                config_manager.get_config()
                .get("models", {})
                .get("default_provider", "openai"),
                {},
            )
            .get("required_env_vars", [])
        )

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
        else:
            status["environment_vars"] = True
            logger.info("Environment variables verified")

        # Display status
        print("\nNotebook Environment Status:")
        print("=" * 50)
        for check, passed in status.items():
            status_icon = "✓" if passed else "✗"
            print(f"{status_icon} {check.replace('_', ' ').title()}")
        print("=" * 50)

        return all(status.values())

    except Exception as e:
        logger.error(f"Environment verification failed: {e}")
        return False


def display_configuration(
    config_manager: Optional[ConfigManager] = None,
) -> None:
    """Display current configuration settings.

    Args:
        config_manager: Optional ConfigManager instance (will create new one if not provided)
    """
    try:
        config_manager = config_manager or ConfigManager(FileUtils())
        config = config_manager.get_config()

        print("\nCurrent Configuration:")
        print("=" * 50)

        # Display provider settings
        print("\nProvider Settings:")
        print(f"Default Provider: {config.models.default_provider}")
        print(f"Default Model: {config.models.default_model}")

        # Display language settings
        print("\nLanguage Settings:")
        print(f"Default Language: {config.languages.default_language}")

        # Display directory structure
        print("\nDirectory Structure:")
        directory_structure = (
            config_manager.get_file_utils().get_directory_structure()
        )
        for parent_dir, subdirs in directory_structure.items():
            print(f"\n{parent_dir}/")
            if isinstance(subdirs, dict):
                for subdir in subdirs:
                    print(f"  └── {subdir}/")
            elif isinstance(subdirs, list):
                for subdir in subdirs:
                    print(f"  └── {subdir}/")

        # Display feature flags
        print("\nEnabled Features:")
        for feature, enabled in config.features.items():
            status = "✓" if enabled else "✗"
            print(f"{status} {feature.replace('_', ' ').title()}")

        print("=" * 50)

    except Exception as e:
        logger.error(f"Error displaying configuration: {e}")
