# src/core/config/manager.py
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from FileUtils import FileUtils
from .models import (
    GlobalConfig,
    ModelConfig,
    LanguageConfig,
    LoggingConfig,
)  # Add needed imports

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager with FileUtils integration."""

    DEFAULT_DIRECTORY_STRUCTURE = {
        "data": ["raw", "interim", "processed", "config", "parameters", "logs"],
        "notebooks": [],
        "docs": [],
        "scripts": [],
        "src": [],
        "reports": [],
        "models": [],
    }

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_dir: str = "config",
        project_root: Optional[Path] = None,
        custom_directory_structure: Optional[Dict[str, Any]] = None,
    ):
        """Initialize configuration manager."""
        self._init_environment()
        self._init_paths(project_root, config_dir)

        # Store or create FileUtils
        if file_utils:
            self.file_utils = file_utils
            logger.debug("Using provided FileUtils instance")
        else:
            self._init_file_utils(custom_directory_structure)
            logger.debug("Created new FileUtils instance")

        # Load and validate configurations
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load configurations from all sources with correct precedence."""
        try:
            config_dir = self.file_utils.get_data_path("config")
            logger.debug(f"Loading configurations from: {config_dir}")

            # Start with base config
            base_config = self._load_yaml("config.yaml", required=True)
            self._config = base_config

            # Check environment
            env = os.getenv("ENV", "development").lower()

            # Load development config if in development
            if env == "development":
                dev_config = self._load_yaml("config.dev.yaml", required=False)
                if dev_config:
                    self._deep_merge(self._config, dev_config)

            # Set up logging with merged config
            self._setup_logging()

            # Validate and create final config object
            self.config = GlobalConfig(**self._config)
            logger.debug("Configuration loaded and validated successfully")

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise

    def get_config(self) -> GlobalConfig:
        """Get the complete configuration.

        Returns:
            GlobalConfig: The validated configuration object
        """
        if not hasattr(self, "config"):
            raise RuntimeError("Configuration not properly initialized")
        return self.config

    def _init_environment(self) -> None:
        """Initialize environment variables."""
        for env_file in [".env", ".env.local"]:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded environment from {env_path}")

        self.initial_log_level = os.getenv("LOG_LEVEL", "INFO")

    def _init_paths(
        self, project_root: Optional[Path], config_dir: str
    ) -> None:
        """Initialize project paths."""
        self.project_root = (
            Path(project_root) if project_root else Path().resolve()
        )
        self.data_dir = self.project_root / "data"
        self.config_dir = self.data_dir / config_dir
        self.logs_dir = self.data_dir / "logs"

    def _init_file_utils(
        self,
        custom_directory_structure: Optional[Dict[str, Any]],
    ) -> None:
        """Initialize FileUtils instance only if not provided."""
        fileutils_config = {
            "directory_structure": (
                custom_directory_structure
                if custom_directory_structure
                else self.DEFAULT_DIRECTORY_STRUCTURE
            )
        }

        self.file_utils = self._create_file_utils(
            config_override=fileutils_config,
            log_level=self.initial_log_level,
            create_directories=True,
        )

    def _create_file_utils(
        self,
        config_path: Path,
        config_override: Dict[str, Any],
        log_level: str,
        create_directories: bool = False,
    ) -> FileUtils:
        """Create new FileUtils instance."""
        return FileUtils(
            project_root=self.project_root,
            config_file=config_path if config_path.exists() else None,
            config_override=config_override,
            log_level=log_level,
            create_directories=create_directories,
        )

    def _update_file_utils_logging(self, new_log_level: str) -> None:
        """Update FileUtils with new log level."""
        if new_log_level != self.initial_log_level:
            self.file_utils = self._create_file_utils(
                self.config_dir / "fileutils_config.yaml",
                {"directory_structure": self.get_directory_structure()},
                new_log_level,
                create_directories=False,
            )
            logger.debug(f"Updated FileUtils log level to {new_log_level}")

    def get_directory_structure(self) -> Dict[str, Any]:
        """Get current directory structure from FileUtils."""
        return self.file_utils.get_directory_structure()

    def validate_directory_structure(self) -> bool:
        """Validate that required directories exist."""
        structure = self.get_directory_structure()

        for parent_dir, subdirs in structure.items():
            parent_path = self.project_root / parent_dir
            if not parent_path.exists():
                logger.warning(f"Missing directory: {parent_path}")
                return False

            if isinstance(subdirs, list):
                for subdir in subdirs:
                    if not (parent_path / subdir).exists():
                        logger.warning(
                            f"Missing subdirectory: {parent_path / subdir}"
                        )
                        return False

        return True

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        required_dirs = [self.data_dir, self.config_dir, self.logs_dir]
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def get_file_utils(self) -> FileUtils:
        """Get the configured FileUtils instance."""
        return self.file_utils

    def _load_yaml(
        self, filename: str, required: bool = False
    ) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            filename: Name of the YAML file
            required: Whether the file is required

        Returns:
            Dict containing configuration or empty dict if file not found
        """
        try:
            config_path = self.config_dir / filename
            if not config_path.exists():
                if required:
                    raise FileNotFoundError(
                        f"Required config file not found: {config_path}"
                    )
                logger.debug(f"Optional config file not found: {filename}")
                return {}

            return self.file_utils.load_yaml(config_path) or {}

        except Exception as e:
            if required:
                logger.error(f"Could not load required config {filename}: {e}")
                raise
            logger.debug(f"Could not load optional config {filename}: {e}")
            return {}

    def _load_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        prefix = "APP_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert APP_LOGGING_LEVEL to config["logging"]["level"]
                parts = key[len(prefix) :].lower().split("_")
                current = config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value

        return config

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        for key, value in dict2.items():
            if (
                isinstance(value, dict)
                and key in dict1
                and isinstance(dict1[key], dict)
            ):
                dict1[key] = self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value
        return dict1

    def _setup_logging(self) -> None:
        """Setup logging with proper paths."""
        try:
            # Get logging config with proper precedence
            log_config = self._config.get("logging", {})

            # Get log level with proper default
            log_level = log_config.get("level", "INFO").upper()

            # Only configure if root logger isn't already configured at this level
            root_logger = logging.getLogger()
            current_level = root_logger.getEffectiveLevel()
            target_level = getattr(logging, log_level)

            if current_level != target_level:
                # Remove existing handlers
                root_logger.handlers.clear()

                # Configure root logger
                root_logger.setLevel(target_level)

                # Create and configure handler
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    fmt=log_config.get(
                        "format",
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    ),
                    datefmt=log_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
                )
                handler.setFormatter(formatter)
                handler.setLevel(target_level)
                root_logger.addHandler(handler)

                # Configure FileUtils logger
                fileutils_logger = logging.getLogger("FileUtils")
                fileutils_logger.setLevel(target_level)

                logger.debug(f"Logging reconfigured to level: {log_level}")
            else:
                logger.debug(
                    f"Logging already configured at level: {log_level}"
                )

        except Exception as e:
            # On error, ensure basic logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,
            )
            logger.error(f"Error setting up logging: {e}")

    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration."""
        return self.config.models

    def get_language_config(self) -> LanguageConfig:
        """Get language-specific configuration."""
        return self.config.languages

    def get_provider_config(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get provider-specific configuration with credentials.

        Args:
            provider: Optional provider name (defaults to configured default)
            model: Optional model name to override default

        Returns:
            Dict[str, Any]: Provider configuration with credentials
        """
        model_config = self.get_model_config()
        provider = provider or model_config.default_provider
        model = model or model_config.default_model

        # Get base configuration
        provider_config = model_config.providers.get(provider, {}).copy()
        if not provider_config:
            raise ValueError(f"Unsupported provider: {provider}")

        # Add model parameters
        provider_config.update(model_config.parameters)

        # Add model name
        provider_config["model"] = model

        # Add provider-specific model config if available
        if (
            "available_models" in provider_config
            and model in provider_config["available_models"]
        ):
            provider_config.update(provider_config["available_models"][model])

        # Add credentials
        self._add_provider_credentials(provider_config, provider)

        return provider_config

    def _add_provider_credentials(
        self, config: Dict[str, Any], provider: str
    ) -> None:
        """Add provider-specific credentials to config."""
        provider_vars = {
            "azure": {
                "api_key": "AZURE_OPENAI_API_KEY",
                "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
                "azure_deployment": "AZURE_OPENAI_DEPLOYMENT_NAME",
            },
            "openai": {
                "api_key": "OPENAI_API_KEY",
            },
            "anthropic": {
                "api_key": "ANTHROPIC_API_KEY",
            },
        }

        if provider in provider_vars:
            for config_key, env_var in provider_vars[provider].items():
                value = os.getenv(env_var)
                if value:
                    config[config_key] = value

    def _ensure_essential_directories(self) -> None:
        """Ensure only essential directories exist."""
        essential_dirs = [self.config_dir, self.logs_dir]
        for directory in essential_dirs:
            directory.mkdir(parents=True, exist_ok=True)
