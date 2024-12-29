# src/config/manager.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from FileUtils import FileUtils
from src.core.config_base import BaseConfigManager
from .models import GlobalConfig, ModelConfig, LanguageConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass


class ConfigManager(BaseConfigManager):
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
        super().__init__(file_utils=file_utils, config_dir=config_dir, project_root=project_root)
        
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
        # Use project_root/data/config for configuration files
        self.data_dir = self.project_root / "data"
        self.config_path = self.data_dir / self.config_dir
        self.logs_dir = self.data_dir / "logs"

        # Create essential directories
        for path in [self.config_path, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")

        logger.debug(f"Using config path: {self.config_path}")

    def init_file_utils(self) -> None:
        """Initialize or configure FileUtils."""
        if not self.file_utils:
            logger.error("FileUtils instance must be provided to ConfigManager. Use EnvironmentManager to get a shared instance.")
            raise ValueError("FileUtils instance must be provided to ConfigManager")
        logger.debug(f"Using FileUtils with project root: {self.file_utils.project_root}")

    def load_configurations(self) -> None:
        """Load configurations from all sources."""
        try:
            # Get environment from env vars or default to development
            env = (
                os.getenv("ENV")
                or os.getenv("ENVIRONMENT", "development").lower()
            )
            logger.debug(f"Loading configurations for environment: {env}")

            # Look for config in data/config first
            data_config = self.project_root / "data" / "config" / "config.yaml"
            logger.debug(f"Looking for data config at: {data_config}")
            
            if data_config.exists():
                config_file = data_config
            else:
                # Fall back to config dir in project root
                config_file = self.config_path / "config.yaml"
            
            logger.debug(f"Using config file: {config_file}")
            
            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                raise FileNotFoundError(f"Config file not found: {config_file}")

            logger.debug(f"Loading config from: {config_file}")
            base_config = self.file_utils.load_yaml(str(config_file))
            if not base_config:
                logger.error(f"Failed to load config from: {config_file}")
                raise ValueError(f"Failed to load config from: {config_file}")
            logger.debug(f"Loaded base config: {base_config}")
            
            # If this is production config, use it directly
            if base_config.get("environment") == "production":
                logger.debug("Found production config, using it directly")
                self._config = GlobalConfig(**base_config)
                logger.debug(f"Created GlobalConfig: {self._config.model_dump()}")
                return

            # For non-production, try to load environment-specific config
            env_config_file = self.config_path / f"config.{env}.yaml"
            if env_config_file.exists():
                logger.debug(f"Loading env config from: {env_config_file}")
                env_config = self.file_utils.load_yaml(str(env_config_file))
                logger.debug(f"Loaded env config: {env_config}")
            else:
                env_config = {}
                logger.warning(f"Environment config not found: {env_config_file}")

            # Merge configurations
            config = {**base_config, **env_config}
            logger.debug(f"Merged config: {config}")

            # Add environment if not specified
            if "environment" not in config:
                config["environment"] = env

            # Create GlobalConfig
            self._config = GlobalConfig(**config)
            logger.debug(f"Created GlobalConfig: {self._config.model_dump()}")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to load configurations: {e}") from e

    def _ensure_minimal_config(self) -> None:
        """Ensure minimal configuration exists."""
        minimal_config = {
            "environment": "development",
            "models": {
                "default_provider": "openai",
                "default_model": "gpt-4o-mini",
                "parameters": {
                    "temperature": 0.0,
                    "max_tokens": 1000,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                },
                "providers": {
                    "openai": {
                        "available_models": {
                            "gpt-4o-mini": {
                                "description": "Fast and cost-effective for simpler tasks",
                                "max_tokens": 4096
                            }
                        }
                    }
                }
            }
        }
        logger.debug(f"Using minimal config: {minimal_config}")
        self._config = GlobalConfig(**minimal_config)

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

            return self.file_utils.load_yaml(str(config_file))
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            if required:
                raise
            return None

    def get_config(self) -> GlobalConfig:
        """Get complete configuration."""
        if not hasattr(self, "_config"):
            raise ConfigurationError("Configuration not loaded. Call load_configurations() first.")
        return self._config

    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration."""
        return self.get_config().models

    def get_language_config(self) -> LanguageConfig:
        """Get language-specific configuration."""
        return self.get_config().languages

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.get_config().get("analysis", {}).get(analyzer_type, {})

    def get_provider_config(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        try:
            # Get config, ensuring it's loaded
            config = self.get_config()
            model_config = config.models.model_dump()  # Convert to dict first

            # Get provider, defaulting to config value
            provider = provider or model_config.get("default_provider", "openai")
            model = model or model_config.get("default_model", "gpt-4")

            # Get base configuration for provider
            provider_config = {}
            if "providers" in model_config and provider in model_config["providers"]:
                provider_config = model_config["providers"][provider].copy()

            # If empty, use defaults
            if not provider_config:
                provider_config = {
                    "model": model,
                    "temperature": 0.0,
                    "max_tokens": 1000,
                }

            # Add model name
            provider_config["model"] = model

            return provider_config

        except Exception as e:
            logger.error(f"Failed to get provider config: {e}")
            # Return minimal config
            return {
                "model": model or "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1000,
            }

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
