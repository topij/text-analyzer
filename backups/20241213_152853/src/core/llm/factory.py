# src/core/llm/factory.py
"""LLM factory for creating and configuring language models."""

import logging
from pathlib import Path

from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.core.config import AnalyzerConfig

logger = logging.getLogger(__name__)


class LLMConfig:
    """LLM configuration management."""

    DEFAULT_CONFIG = {
        "default_provider": "openai",
        "default_model": "gpt-4o-mini",  # Updated default model
        "providers": {
            "openai": {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 1000,
            },  # Updated default model
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
                from FileUtils import FileUtils

                user_config = FileUtils().load_yaml(config_path)
                return self._merge_config(self.DEFAULT_CONFIG, user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        return self.DEFAULT_CONFIG.copy()

    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults."""
        result = default.copy()

        for key, value in user.items():
            if (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def get_model_config(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration for specific provider and model."""
        provider = provider or self.config["default_provider"]
        if provider not in self.config["providers"]:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_config = self.config["providers"][provider].copy()
        if model:
            provider_config["model"] = model

        return provider_config


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[AnalyzerConfig] = None,
    **kwargs,
) -> BaseChatModel:
    """Create an LLM instance with specified configuration."""
    try:
        # Create or use provided config
        config = config or AnalyzerConfig()

        # Get provider-specific configuration
        provider = provider or config.config["models"]["default_provider"]
        provider_config = config.get_provider_config(provider, model)

        # Override with kwargs
        provider_config.update(kwargs)

        # Create appropriate LLM instance
        if provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=provider_config["azure_endpoint"],
                azure_deployment=provider_config["azure_deployment"],
                api_key=provider_config["api_key"],
                api_version=provider_config.get(
                    "api_version", "2024-02-15-preview"
                ),
                temperature=provider_config.get("temperature", 0),
                max_tokens=provider_config.get("max_tokens", 1000),
            )
        elif provider == "openai":
            return ChatOpenAI(
                api_key=provider_config["api_key"],
                model=provider_config["model"],
                temperature=provider_config.get("temperature", 0),
                max_tokens=provider_config.get("max_tokens", 1000),
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                api_key=provider_config["api_key"],
                model=provider_config["model"],
                temperature=provider_config.get("temperature", 0),
                max_tokens=provider_config.get("max_tokens", 1000),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        logger.error(f"Error creating LLM instance: {e}")
        raise
