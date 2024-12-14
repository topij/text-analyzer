# src/core/config_management.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

import yaml
from pydantic import BaseModel, Field, field_validator

from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class LoggingConfig(BaseModel):
    """Unified logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
    file_path: Optional[str] = None
    disable_existing_loggers: bool = False

    @field_validator("level")
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}")
        return v


class ModelConfig(BaseModel):
    """LLM model configuration."""

    default_provider: str = Field(default="azure")
    default_model: str = Field(default="gpt-4o-mini")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LanguageConfig(BaseModel):
    """Language processing configuration."""

    default_language: str = Field(default="en")
    languages: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class GlobalConfig(BaseModel):
    """Global application configuration."""

    environment: str = Field(default="development")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    languages: LanguageConfig = Field(default_factory=LanguageConfig)
    features: Dict[str, bool] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dictionary-like access."""
        try:
            value = getattr(self, key, None)
            if value is None:
                return default
            if isinstance(value, BaseModel):
                return value.model_dump()
            return value
        except Exception:
            return default

    def model_dump(self) -> Dict[str, Any]:
        """Override model_dump to convert nested models to dicts."""
        result = super().model_dump()
        for key, value in result.items():
            if isinstance(value, BaseModel):
                result[key] = value.model_dump()
        return result


class ConfigManager:
    """Unified configuration manager."""

    def __init__(
        self, file_utils: Optional[FileUtils] = None, config_dir: str = "config"
    ):
        """Initialize configuration manager.

        Args:
            file_utils: Optional FileUtils instance
            config_dir: Directory name for config files (default: "config")
        """
        # Load environment variables first
        for env_file in [".env", ".env.local"]:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded environment from {env_path}")

        self.file_utils = file_utils or FileUtils()
        self.config_dir = config_dir
        self._config: Dict[str, Any] = {}
        self._load_configurations()
        self._setup_logging()

    def _load_configurations(self) -> None:
        """Load configurations from all sources with correct precedence."""
        try:
            config_dir = self.file_utils.get_data_path(self.config_dir)
            logger.debug(f"Loading configurations from: {config_dir}")

            # Load configurations in order (lowest to highest priority)
            base_config = self._load_yaml("config.yaml", required=True)
            dev_config = self._load_yaml("config.dev.yaml", required=False)
            model_config = self._load_yaml("models.yaml", required=False)

            # Start with base config
            self._config = base_config

            # Merge additional configs
            if model_config:
                self._deep_merge(self._config, model_config)
            if dev_config:
                self._deep_merge(self._config, dev_config)

            # Override with environment variables
            env_config = self._load_env_vars()
            if env_config:
                self._deep_merge(self._config, env_config)

            # Validate final configuration
            self.config = GlobalConfig(**self._config)
            logger.debug("Configuration loaded and validated successfully")

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise

    def _load_yaml(
        self, filename: str, required: bool = False
    ) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            # Update to use config directory
            config_path = (
                self.file_utils.get_data_path(self.config_dir) / filename
            )
            return self.file_utils.load_yaml(config_path) or {}
        except FileNotFoundError:
            if required:
                raise
            return {}
        except Exception as e:
            logger.warning(f"Could not load {filename}: {e}")
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
        """Setup logging with configuration."""
        log_config = self.config.logging

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_config.level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Create and configure handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=log_config.format, datefmt=log_config.date_format
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Add file handler if specified
        if log_config.file_path:
            try:
                # Use FileUtils to create log directory
                log_dir = Path(log_config.file_path).parent
                self.file_utils.create_directory(log_dir)

                file_handler = logging.FileHandler(log_config.file_path)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)

            except Exception as e:
                logger.warning(f"Could not set up file logging: {e}")
                # Continue without file logging

    def get_config(self) -> GlobalConfig:
        """Get the complete configuration."""
        return self.config

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
