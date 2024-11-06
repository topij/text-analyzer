# src/core/llm/factory.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMConfig:
    """LLM configuration management."""

    DEFAULT_CONFIG = {
        "default_provider": "openai",
        "default_model": "gpt-4o-mini",  # Updated default model
        "providers": {
            "openai": {"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 1000},
            "anthropic": {
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.0,
                "max_tokens": 1000,
            },
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config path."""
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and config_path.exists():
            try:
                from src.core.utils.file_utils import FileUtils

                return FileUtils().load_yaml(config_path)
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        return self.DEFAULT_CONFIG.copy()

    def get_model_config(self, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for specific provider and model."""
        provider = provider or self.config["default_provider"]
        if provider not in self.config["providers"]:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_config = self.config["providers"][provider].copy()
        if model:
            provider_config["model"] = model

        return provider_config


class LLMFactory:
    """Factory for creating LLM instances with updated configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config path."""
        self.config = LLMConfig(config_path)

    def create(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseChatModel:
        """Create an LLM instance.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Specific model to use
            **kwargs: Additional configuration parameters

        Returns:
            BaseChatModel: Configured LLM instance

        Raises:
            ValueError: For unsupported providers
        """
        # Get base configuration
        config = self.config.get_model_config(provider, model)

        # Override with any provided kwargs
        config.update(kwargs)

        provider = provider or self.config.config["default_provider"]

        # Create appropriate LLM instance
        if provider == "openai":
            return ChatOpenAI(**config)
        elif provider == "anthropic":
            return ChatAnthropic(**config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


def create_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Convenience function to create LLM instance."""
    factory = LLMFactory()
    return factory.create(provider, model, **kwargs)
