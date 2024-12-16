# src/core/config.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.manager import ConfigManager
from src.config.models import ModelConfig

from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class AnalyzerConfig:
    """Configuration handler that integrates with ConfigManager."""

    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize configuration with unified config management.

        Args:
            file_utils: Optional FileUtils instance
        """
        self.file_utils = file_utils or FileUtils()

        # Initialize unified configuration
        self._config_manager = ConfigManager(file_utils=self.file_utils)

        # Get configuration
        self.config = self._config_manager.get_config().model_dump()

        # Validate required variables
        self._validate_required_vars()

    def _validate_required_vars(self) -> None:
        """Validate required environment variables based on provider."""
        provider = self.config["models"]["default_provider"]
        required_vars = (
            self._config_manager.get_model_config()
            .providers[provider]
            .get("required_env_vars", [])
        )

        missing = [var for var in required_vars if not self._get_env_var(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables for {provider}: {', '.join(missing)}"
            )

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
        model_config = self._config_manager.get_model_config()
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
                value = self._get_env_var(env_var)
                if value:
                    config[config_key] = value

    def _get_env_var(self, var_name: str) -> Optional[str]:
        """Get environment variable with proper error handling."""
        try:
            return self._config_manager._load_env_vars().get(var_name)
        except Exception as e:
            logger.debug(f"Error getting environment variable {var_name}: {e}")
            return None

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get("models", {})

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.config.get("analysis", {}).get(
            analyzer_type,
            self.config.get("analysis", {}).get(analyzer_type, {}),
        )

    def get_features(self) -> Dict[str, bool]:
        """Get feature flags."""
        return self.config.get("features", {})

    @property
    def default_language(self) -> str:
        """Get default language."""
        return self.config.get("default_language", "en")

    @property
    def content_column(self) -> str:
        """Get content column name."""
        return self.config.get("content_column", "content")
