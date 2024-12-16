# src/nb_helpers/nb_environment.py

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from FileUtils import FileUtils
from src.config.manager import ConfigManager
from src.config.models import (
    GlobalConfig,
    LoggingConfig,
    ModelConfig,
    LanguageConfig,
)

logger = logging.getLogger(__name__)


def setup_notebook_environment(
    log_level: Optional[str] = "INFO",
    config_dir: str = "config",
    custom_directory_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Set up environment for notebook usage."""
    try:
        # Get project root
        project_root = Path().resolve().parent.absolute()

        # Initialize FileUtils once
        directory_structure = (
            custom_directory_structure
            or ConfigManager.DEFAULT_DIRECTORY_STRUCTURE
        )

        file_utils = FileUtils(
            project_root=project_root,
            log_level=log_level,
            directory_structure=directory_structure,
        )

        # Pass the existing FileUtils instance to ConfigManager
        config_manager = ConfigManager(
            file_utils=file_utils,  # Pass the existing instance
            config_dir=config_dir,
            project_root=project_root,
            custom_directory_structure=directory_structure,
        )

        # Store initialized components
        initialized_components = {
            "file_utils": file_utils,
            "config_manager": config_manager,
            "project_root": project_root,
            "data_dir": project_root / "data",
        }

        logger.info(f"Notebook environment set up with root: {project_root}")
        return initialized_components

    except Exception as e:
        logger.error(f"Failed to set up notebook environment: {e}")
        raise


def verify_notebook_environment(
    initialized_components: Optional[Dict[str, Any]] = None
) -> bool:
    """Verify notebook environment setup."""
    status = {
        "project_structure": False,
        "configuration": False,
        "environment_vars": False,
        "directory_structure": False,
    }

    try:
        # Check if we have initialized components
        if not initialized_components:
            logger.warning(
                "No initialized components provided - creating new ones"
            )
            # Create new instance only if absolutely necessary
            project_root = Path().resolve().parent.absolute()
            file_utils = FileUtils(project_root=project_root)
            config_manager = ConfigManager(file_utils=file_utils)
        else:
            # Use provided components
            config_manager = initialized_components.get("config_manager")
            file_utils = initialized_components.get("file_utils")

            if not config_manager or not file_utils:
                raise ValueError(
                    "Missing required components in initialized_components"
                )

            logger.debug("Using existing initialized components")

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

        if not missing_files:
            status["project_structure"] = True
            logger.info("Project structure verified")

        # Check environment variables
        required_vars = {
            "openai": ["OPENAI_API_KEY"],
            "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
            "anthropic": ["ANTHROPIC_API_KEY"],
        }

        # Get current provider from config
        provider = config_manager.get_model_config().default_provider
        required_provider_vars = required_vars.get(provider, [])

        missing_vars = [
            var for var in required_provider_vars if not os.getenv(var)
        ]
        if not missing_vars:
            status["environment_vars"] = True
        else:
            logger.warning(
                f"Missing environment variables for {provider}: {missing_vars}"
            )

        # Check configuration
        try:
            config = config_manager.get_config()
            status["configuration"] = True
            logger.info("Configuration loading verified")

            # Log current settings
            model_config = config_manager.get_model_config()
            logger.info(f"Current provider: {model_config.default_provider}")
            logger.info(f"Current model: {model_config.default_model}")
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")

        # Check directory structure
        status["directory_structure"] = (
            config_manager.validate_directory_structure()
        )

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


def _format_directory_structure(
    directory_structure: Dict[str, Any], indent: int = 0
) -> List[str]:
    """Format directory structure into a list of lines with proper indentation."""
    lines = []
    prefix = "  " * indent

    for parent_dir, subdirs in sorted(directory_structure.items()):
        # Add parent directory with a trailing slash
        lines.append(f"{prefix}{parent_dir}/")

        # Handle subdirectories
        if isinstance(subdirs, list) and subdirs:
            for i, subdir in enumerate(sorted(subdirs)):
                is_last = i == len(subdirs) - 1
                # Use different symbols for last item vs others
                symbol = "└──" if is_last else "├──"
                lines.append(f"{prefix} {symbol} {subdir}/")
        elif isinstance(subdirs, dict):
            # Handle nested directory structure
            sublines = _format_directory_structure(subdirs, indent + 1)
            lines.extend(sublines)

    return lines


def display_configuration(
    config_manager: Optional[ConfigManager] = None,
    initialized_components: Optional[Dict[str, Any]] = None,
) -> None:
    """Display current configuration settings."""
    try:
        # First try to get config_manager from initialized components
        if not config_manager and initialized_components:
            config_manager = initialized_components.get("config_manager")

        if not config_manager:
            raise ValueError(
                "No config manager provided or found in initialized components"
            )

        config = config_manager.get_config()
        # Get FileUtils from config_manager instead of creating new one
        file_utils = config_manager.get_file_utils()

        print("\nCurrent Configuration:")
        print("=" * 50)

        # Display provider settings
        model_config = config_manager.get_model_config()
        print("\nProvider Settings:")
        print(f"Default Provider: {model_config.default_provider}")
        print(f"Default Model: {model_config.default_model}")

        # Display language settings
        language_config = config_manager.get_language_config()
        print("\nLanguage Settings:")
        print(f"Default Language: {language_config.default_language}")

        # Display directory structure
        print("\nDirectory Structure:")
        directory_structure = file_utils.get_directory_structure()
        structure_lines = _format_directory_structure(directory_structure)
        print("\n".join(structure_lines))

        # Display feature flags
        print("\nEnabled Features:")
        for feature, enabled in sorted(config.features.items()):
            status = "✓" if enabled else "✗"
            print(f"{status} {feature.replace('_', ' ').title()}")

        print("=" * 50)

    except Exception as e:
        logger.error(f"Error displaying configuration: {e}")
