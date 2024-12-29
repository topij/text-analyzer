"""Configuration for semantic analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.manager import ConfigManager
from src.config.models import ModelConfig
from src.core.managers import EnvironmentManager
from src.core.config_base import BaseConfigManager

from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class AnalyzerConfig:
    """Configuration for semantic analysis."""

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        """Initialize analyzer configuration.
        
        Args:
            file_utils: FileUtils instance (required)
            config_manager: Optional ConfigManager instance
            
        Raises:
            ValueError: If file_utils is None or config_manager is None
        """
        # Try to get FileUtils from EnvironmentManager first
        if file_utils is None:
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                self.file_utils = components["file_utils"]
            except RuntimeError:
                raise ValueError(
                    "FileUtils instance must be provided to AnalyzerConfig. "
                    "Use EnvironmentManager to get a shared FileUtils instance."
                )
        else:
            self.file_utils = file_utils
        
        # Get ConfigManager from EnvironmentManager if not provided
        if config_manager is None:
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                self._config_manager = components["config_manager"]
            except RuntimeError:
                raise ValueError(
                    "ConfigManager must be provided to AnalyzerConfig. "
                    "Use EnvironmentManager to get a shared ConfigManager instance."
                )
        else:
            self._config_manager = config_manager

        # Get config as dict
        self.config = self._config_manager.get_config()
        if not isinstance(self.config, dict):
            self.config = self.config.model_dump()
        logger.debug(f"Using config from manager: {self.config}")
        self._validate_required_vars()

    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return self.config

    def _validate_required_vars(self) -> None:
        """Validate required environment variables."""
        provider = self.config.get("models", {}).get(
            "default_provider", "openai"
        )
        logger.debug(f"Validating required vars for provider: {provider}")
        required_vars = {
            "azure": [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT_NAME",
            ],
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
        }.get(provider, [])

        missing = [var for var in required_vars if not self._get_env_var(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables for {provider}: {', '.join(missing)}"
            )

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config_manager.get_model_config()

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self._config_manager.get_analyzer_config(analyzer_type)

    def get_features(self) -> Dict[str, bool]:
        """Get feature flags."""
        return self._config_manager.get_features()

    @property
    def default_language(self) -> str:
        """Get default language."""
        return self._config_manager.get_default_language()

    @property
    def content_column(self) -> str:
        """Get content column name."""
        return self._config_manager.get_content_column()

    def get_provider_config(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        return self._config_manager.get_provider_config(provider, model)

    def _add_provider_credentials(
        self, config: Dict[str, Any], provider: str
    ) -> None:
        """Add provider-specific credentials to config."""
        creds = {
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
        }.get(provider, {})

        for config_key, env_var in creds.items():
            if value := self._get_env_var(env_var):
                config[config_key] = value

    def _get_env_var(self, var_name: str) -> Optional[str]:
        """Get environment variable safely."""
        import os

        return os.getenv(var_name)
