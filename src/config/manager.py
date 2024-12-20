# src/config/manager.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from FileUtils import FileUtils
from .models import GlobalConfig, ModelConfig, LanguageConfig

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
        self.file_utils = file_utils or FileUtils()
        # Use the provided project_root or get it from file_utils
        self.project_root = (
            Path(project_root) if project_root else self.file_utils.project_root
        )
        self.config_dir = config_dir

        # Store initialization variables for potential reinit
        self._custom_directory_structure = custom_directory_structure

        # Initialize all components
        try:
            self.init_environment()
            self.init_paths()
            self.init_file_utils()
            self.load_configurations()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self._ensure_minimal_config()

    def init_environment(self) -> None:
        """Initialize environment variables."""
        for env_file in [".env", ".env.local"]:
            env_path = self.project_root / env_file
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded environment from {env_path}")

    def init_paths(self) -> None:
        """Initialize project paths."""
        # Use project_root/data instead of creating new data dir
        self.data_dir = self.project_root / "data"
        self.config_path = self.data_dir / self.config_dir
        self.logs_dir = self.data_dir / "logs"

        # Create essential directories
        for path in [self.config_path, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")

    def init_file_utils(self) -> None:
        """Initialize or configure FileUtils."""
        if not self.file_utils:
            directory_structure = (
                self._custom_directory_structure
                or self.DEFAULT_DIRECTORY_STRUCTURE
            )
            self.file_utils = FileUtils(
                project_root=self.project_root,
                directory_structure=directory_structure,
            )
            logger.debug("Created new FileUtils instance")
        else:
            logger.debug("Using provided FileUtils instance")

    # src/config/manager.py

    def load_configurations(self) -> None:
        """Load configurations from all sources."""
        try:
            # Load base config from config.yaml
            env = (
                os.getenv("ENV")
                or os.getenv("ENVIRONMENT", "development").lower()
            )

            # Load base config
            config_file = self.config_path / "config.yaml"
            if config_file.exists():
                base_config = self.file_utils.load_yaml(config_file)
                logger.debug(f"Loaded base config from {config_file}")
            else:
                logger.warning(f"Base config file not found: {config_file}")
                base_config = {}

            # Start with default config
            self._config = {
                "environment": "development",
                "models": {
                    "default_provider": "openai",
                    "default_model": "gpt-4o-mini",
                    "providers": {
                        "openai": {
                            "model": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": 1000,
                        },
                        "anthropic": {
                            "model": "claude-3-sonnet-20240229",
                            "temperature": 0.0,
                            "max_tokens": 1000,
                        },
                    },
                    "parameters": {
                        "temperature": 0.0,
                        "max_tokens": 1000,
                        "top_p": 1.0,
                    },
                },
            }

            # Merge base config
            if base_config:
                self._config.update(base_config)
                logger.debug("Merged base configuration")

            # Load environment-specific config only if in development
            if env == "development":
                dev_config_file = self.config_path / "config.dev.yaml"
                if dev_config_file.exists():
                    dev_config = self.file_utils.load_yaml(dev_config_file)
                    if dev_config:
                        self._deep_merge(self._config, dev_config)
                        logger.debug(
                            f"Merged dev config from {dev_config_file}"
                        )

            # Create and validate config object
            self.config = GlobalConfig(**self._config)
            logger.debug(
                f"Configuration loaded and validated: {self.config.model_dump()}"
            )

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            self._ensure_minimal_config()

    def _ensure_minimal_config(self) -> None:
        """Ensure minimal working configuration."""
        logger.info("Creating minimal configuration")
        self.config = GlobalConfig(
            models=ModelConfig(default_model="gpt-4o-mini"),
            logging={"level": "INFO"},
            features={"use_caching": True},
        )

    def load_yaml(
        self, filename: str, required: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Load YAML configuration file."""
        try:
            config_file = self.config_path / filename
            if not config_file.exists():
                if required:
                    raise FileNotFoundError(
                        f"Required config file not found: {config_file}"
                    )
                return None

            return self.file_utils.load_yaml(config_file)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            if required:
                raise
            return None

    def get_config(self) -> GlobalConfig:
        """Get complete configuration."""
        if not hasattr(self, "config"):
            self._ensure_minimal_config()
        return self.config

    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration."""
        return self.get_config().models

    def get_language_config(self) -> LanguageConfig:
        """Get language-specific configuration."""
        return self.get_config().languages

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.get_config().get("analysis", {}).get(analyzer_type, {})

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in dict2.items():
            if (
                isinstance(value, dict)
                and key in dict1
                and isinstance(dict1[key], dict)
            ):
                self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value
