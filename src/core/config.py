# src/core/config.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.manager import ConfigManager, ModelConfig
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class AnalyzerConfig:
    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        self.file_utils = file_utils or FileUtils()
        self._config_manager = config_manager or ConfigManager(
            file_utils=self.file_utils
        )
        # Get config as dict
        self.config = self._config_manager.get_config().model_dump()
        self._validate_required_vars()

    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return self.config

    def _validate_required_vars(self) -> None:
        """Validate required environment variables."""
        provider = self.config.get("models", {}).get(
            "default_provider", "openai"
        )
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
        return self.config.get(
            "models", {}
        )  # Return models section from config

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.config.get("analysis", {}).get(
            analyzer_type, {}  # Return empty dict if not found
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

    def get_provider_config(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        model_config = self.get_model_config()
        provider = provider or model_config.get("default_provider", "openai")
        model = model or model_config.get(
            "default_model", "gpt-4o-mini"
        )  # Update default model

        # Get base configuration
        provider_config = (
            model_config.get("providers", {}).get(provider, {}).copy()
        )
        if not provider_config:
            raise ValueError(f"Unsupported provider: {provider}")

        # Add model parameters
        provider_config.update(model_config.get("parameters", {}))

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
