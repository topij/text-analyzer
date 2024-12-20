# src/nb_helpers/environment.py

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

from FileUtils import FileUtils
from src.config.manager import ConfigManager
from src.core.config import AnalyzerConfig
from src.semantic_analyzer.analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


class EnvironmentType(str, Enum):
    """Environment types for analysis."""

    LOCAL = "local"
    AZURE = "azure"


class AnalysisEnvironment:
    """Environment manager supporting both local and Azure environments."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        env_type: Optional[str] = None,
        project_root: Optional[Path] = None,
        log_level: str = "INFO",
    ):
        """Initialize environment manager."""
        try:
            # Only initialize once
            if hasattr(self, "_initialized"):
                return

            # Set up basic logging first
            self._configure_logging(log_level)

            # Determine environment type
            self.env_type = self._determine_env_type(env_type)
            logger.info(f"Running in {self.env_type} environment")

            # Set up project root based on environment
            self.project_root = self._setup_project_root(project_root)
            logger.info(f"Project root: {self.project_root}")

            # Load environment variables
            self._load_environment()

            # Initialize components
            self._init_components()

            self._initialized = True
            logger.info("Analysis environment initialized successfully")

        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize environment: {e}")

    def _configure_logging(self, level: str) -> None:
        """Configure logging with proper handlers."""
        try:
            # Convert string level to numeric
            numeric_level = getattr(logging, level.upper(), logging.INFO)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            # Remove any existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Add console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            handler.setLevel(numeric_level)
            root_logger.addHandler(handler)

            logger.debug("Logging configured successfully")

        except Exception as e:
            print(
                f"Error configuring logging: {e}"
            )  # Use print as logger might not be ready
            raise

    def _determine_env_type(self, env_type: Optional[str]) -> EnvironmentType:
        """Determine the environment type."""
        if env_type:
            try:
                return EnvironmentType(env_type.lower())
            except ValueError:
                logger.warning(
                    f"Invalid environment type: {env_type}, defaulting to LOCAL"
                )
                return EnvironmentType.LOCAL

        # Auto-detect environment
        return (
            EnvironmentType.AZURE
            if os.getenv("AZUREML_RUN_ID")
            else EnvironmentType.LOCAL
        )

    def _setup_project_root(self, project_root: Optional[Path]) -> Path:
        """Set up project root path."""
        if project_root:
            return Path(project_root)

        if self.env_type == EnvironmentType.AZURE:
            return Path(os.getenv("AZUREML_RUN_ROOT", "/"))

        # For local environment
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent

    def _load_environment(self) -> None:
        """Load environment variables."""
        if self.env_type == EnvironmentType.LOCAL:
            env_paths = [
                self.project_root / ".env",
                self.project_root / ".env.local",
                Path.home() / ".env",
            ]

            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from: {env_path}")
                    break
        else:
            logger.info("Using Azure ML environment variables")

    def _init_components(self) -> None:
        """Initialize components."""
        try:
            # Create FileUtils configuration
            file_utils_config = {
                "project_root": self.project_root,
                "create_directories": True,
            }

            if self.env_type == EnvironmentType.AZURE:
                file_utils_config.update(
                    {
                        "use_azure_storage": True,
                        "azure_connection_string": os.getenv(
                            "AZURE_STORAGE_CONNECTION_STRING"
                        ),
                        "container_name": os.getenv(
                            "AZURE_STORAGE_CONTAINER", "analysis-data"
                        ),
                    }
                )

            # Initialize components
            self.file_utils = FileUtils(**file_utils_config)
            self.config_manager = ConfigManager(file_utils=self.file_utils)
            self.analyzer_config = AnalyzerConfig(
                file_utils=self.file_utils, config_manager=self.config_manager
            )

            logger.debug("Components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def get_components(self) -> Dict[str, Any]:
        """Get initialized components."""
        if not hasattr(self, "_initialized"):
            raise RuntimeError("Environment not properly initialized")

        return {
            "file_utils": self.file_utils,
            "config_manager": self.config_manager,
            "analyzer_config": self.analyzer_config,
            "project_root": self.project_root,
            "env_type": self.env_type,
        }

    def verify_setup(self) -> Dict[str, bool]:
        """Verify environment setup."""
        try:
            checks = {
                "Environment": self._verify_environment(),
                "API Access": self._verify_api_access(),
                "Components": self._verify_components(),
            }

            self._log_verification_results(checks)
            return checks

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"error": False}

    def _verify_environment(self) -> Dict[str, bool]:
        """Verify environment-specific requirements."""
        return {
            "Project root exists": self.project_root.exists(),
            "Data directories": self._verify_data_directories(),
            "Environment variables": self._verify_env_vars(),
        }

    def _verify_data_directories(self) -> bool:
        """Verify data directories exist."""
        required_dirs = ["raw", "processed", "config", "parameters"]
        data_path = self.project_root / "data"
        return all((data_path / d).exists() for d in required_dirs)

    def _verify_env_vars(self) -> bool:
        """Verify environment variables."""
        required_vars = ["OPENAI_API_KEY"]
        if self.env_type == EnvironmentType.AZURE:
            required_vars.extend(
                ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
            )

        return all(bool(os.getenv(var)) for var in required_vars)

    def _verify_api_access(self) -> Dict[str, bool]:
        """Verify API access."""
        return {
            "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
            "Azure OpenAI": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        }

    def _verify_components(self) -> Dict[str, bool]:
        """Verify components."""
        return {
            "FileUtils": hasattr(self, "file_utils"),
            "ConfigManager": hasattr(self, "config_manager"),
            "AnalyzerConfig": hasattr(self, "analyzer_config"),
        }

    def _log_verification_results(
        self, checks: Dict[str, Dict[str, bool]]
    ) -> None:
        """Log verification results."""
        for category, results in checks.items():
            logger.info(f"\n{category} checks:")
            for check, status in results.items():
                status_symbol = "✓" if status else "✗"
                logger.info(f"{status_symbol} {check}")


def setup_analysis_environment(
    env_type: Optional[str] = None,
    log_level: str = "WARNING",
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Setup analysis environment and return components."""
    try:
        # Convert project_root to Path if provided as string
        if isinstance(project_root, str):
            project_root = Path(project_root)

        # If project_root is not provided, determine it from current file
        if project_root is None:
            current_file = Path().resolve()
            # If we're in notebooks directory, go up one level
            if current_file.name == "notebooks":
                project_root = current_file.parent
            else:
                # Go up until we find the project root (where data dir exists)
                current = current_file
                while current.parent != current:  # Stop at root
                    if (current / "data").exists():
                        project_root = current
                        break
                    current = current.parent

        if not project_root or not (project_root / "data").exists():
            raise ValueError(f"Invalid project root: {project_root}")

        logger.info(f"Using project root: {project_root}")

        env = AnalysisEnvironment(
            env_type=env_type, project_root=project_root, log_level=log_level
        )

        # Verify setup
        if not env.verify_setup():
            logger.warning("Environment setup verification failed")

        return env.get_components()

    except Exception as e:
        logger.error(f"Failed to set up analysis environment: {e}")
        raise


def get_llm_info(analyzer: SemanticAnalyzer, detailed: bool = False) -> str:
    """Get information about the active LLM configuration."""
    try:
        # Get current config and LLM info
        config = analyzer.analyzer_config.config["models"]
        provider = config["default_provider"]
        model = config["default_model"]

        info = f"Active LLM: {provider} ({model})"

        if detailed:
            params = config.get("parameters", {})
            info += "\nParameters:"
            for key, value in params.items():
                # Format key name for display
                display_key = key.replace("_", " ").title()
                info += f"\n- {display_key}: {value}"

            # Add provider-specific info
            provider_config = config.get("providers", {}).get(provider, {})
            if provider_config:
                if "api_type" in provider_config:
                    info += f"\nAPI Type: {provider_config['api_type']}"
                if "api_version" in provider_config:
                    info += f"\nAPI Version: {provider_config['api_version']}"

        return info

    except Exception as e:
        logger.error(f"Error getting LLM info: {e}")
        return "Unable to get LLM information"


def get_available_providers(
    analyzer: SemanticAnalyzer,
) -> Dict[str, Dict[str, Any]]:
    """Get available LLM providers and their configurations."""
    config = analyzer.analyzer_config.config
    return config.get("models", {}).get("providers", {})


def change_llm_provider(
    analyzer: SemanticAnalyzer, provider: str, model: Optional[str] = None
) -> None:
    """Change LLM provider and optionally model.

    This is a convenience function that delegates to SemanticAnalyzer's implementation.
    """
    try:
        # Delegate to SemanticAnalyzer's implementation
        analyzer.change_llm_provider(provider, model)

    except Exception as e:
        logger.error(f"Failed to change LLM provider: {e}")
        raise
