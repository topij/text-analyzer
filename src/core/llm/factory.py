# src/core/llm/factory.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# from src.config.manager import ConfigManager
from src.core.config import ConfigManager
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


# def create_llm(
#     provider: Optional[str] = None,
#     model: Optional[str] = None,
#     config: Optional[AnalyzerConfig] = None,
#     config_manager: Optional[ConfigManager] = None,
#     **kwargs,
# ) -> BaseChatModel:
#     """Create an LLM instance with specified configuration."""
#     logger.info(f"Creating LLM instance: {provider} {model}")
#     try:
#         # Get provider configuration
#         if config:
#             # Use AnalyzerConfig
#             provider = provider or config.config["models"]["default_provider"]
#             model = model or config.config["models"]["default_model"]
#             provider_config = config.get_provider_config(provider, model)
#         elif config_manager:
#             # Use model config from ConfigManager
#             model_config = config_manager.get_model_config()
#             provider = provider or model_config.default_provider
#             model = model or model_config.default_model
#             # Create analyzer config with config_manager
#             analyzer_config = AnalyzerConfig(config_manager=config_manager)
#             provider_config = analyzer_config.get_provider_config(
#                 provider, model
#             )
#         else:
#             # Create minimal config
#             file_utils = FileUtils()
#             config_manager = ConfigManager(file_utils=file_utils)
#             analyzer_config = AnalyzerConfig(config_manager=config_manager)
#             model_config = config_manager.get_model_config()
#             provider = provider or model_config.default_provider
#             model = model or model_config.default_model
#             provider_config = analyzer_config.get_provider_config(
#                 provider, model
#             )

#         # Ensure we have a provider
#         if not provider:
#             raise ValueError(
#                 "No LLM provider specified or found in configuration"
#             )

#         # Override with kwargs
#         provider_config.update(kwargs)

#         # In create_llm function
#         logger.debug(f"Creating LLM with provider: {provider}, model: {model}")
#         logger.debug(f"Provider config: {provider_config}")

#         # Create appropriate LLM instance
#         if provider == "azure":
#             return AzureChatOpenAI(
#                 azure_endpoint=provider_config["azure_endpoint"],
#                 azure_deployment=provider_config["azure_deployment"],
#                 api_key=provider_config["api_key"],
#                 api_version=provider_config.get(
#                     "api_version", "2024-02-15-preview"
#                 ),
#                 temperature=provider_config.get("temperature", 0),
#                 max_tokens=provider_config.get("max_tokens", 1000),
#             )


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[AnalyzerConfig] = None,
    config_manager: Optional[ConfigManager] = None,
    **kwargs,
) -> BaseChatModel:
    """Create an LLM instance with specified configuration."""
    logger.info(f"Creating LLM instance: provider={provider}, model={model}")
    try:
        # Get provider configuration
        if config:
            # Use AnalyzerConfig
            provider = provider or config.config["models"]["default_provider"]
            model = model or config.config["models"]["default_model"]
            provider_config = config.get_provider_config(provider, model)
            logger.debug(
                f"Using AnalyzerConfig - provider: {provider}, model: {model}"
            )
        elif config_manager:
            # Use model config from ConfigManager
            model_config = config_manager.get_model_config()
            provider = provider or model_config.default_provider
            model = model or model_config.default_model
            analyzer_config = AnalyzerConfig(config_manager=config_manager)
            provider_config = analyzer_config.get_provider_config(
                provider, model
            )
            logger.debug(
                f"Using ConfigManager - provider: {provider}, model: {model}"
            )

        logger.debug(f"Final provider config: {provider_config}")
        logger.debug(f"Creating LLM with provider: {provider}, model: {model}")

        # Create appropriate LLM instance
        # if provider == "azure":
        #     return AzureChatOpenAI(
        #         azure_endpoint=provider_config["azure_endpoint"],
        #         azure_deployment=provider_config["azure_deployment"],
        #         api_key=provider_config["api_key"],
        #         api_version=provider_config.get(
        #             "api_version", "2024-02-15-preview"
        #         ),
        #         temperature=provider_config.get("temperature", 0),
        #         max_tokens=provider_config.get("max_tokens", 1000),
        #     )
        # elif provider == "openai":
        #     return ChatOpenAI(
        #         api_key=provider_config["api_key"],
        #         model=provider_config["model"],
        #         temperature=provider_config.get("temperature", 0),
        #         max_tokens=provider_config.get("max_tokens", 1000),
        #     )
        # elif provider == "anthropic":
        #     return ChatAnthropic(
        #         api_key=provider_config["api_key"],
        #         model=provider_config["model"],
        #         temperature=provider_config.get("temperature", 0),
        #         max_tokens=provider_config.get("max_tokens", 1000),
        #     )
        # # else:
        # #     raise ValueError(f"Unsupported provider: {provider}")

        # Add credentials to provider config
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment variables"
                )
            provider_config["api_key"] = api_key

            return ChatAnthropic(
                api_key=provider_config["api_key"],
                model=model,
                temperature=provider_config.get("temperature", 0),
                max_tokens=provider_config.get("max_tokens", 1000),
            )
        elif provider == "azure":
            # Add Azure credentials
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not (api_key and endpoint):
                raise ValueError(
                    "Azure OpenAI credentials not found in environment variables"
                )
            provider_config["api_key"] = api_key
            provider_config["azure_endpoint"] = endpoint

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
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables"
                )
            provider_config["api_key"] = api_key

            return ChatOpenAI(
                api_key=provider_config["api_key"],
                model=model,
                temperature=provider_config.get("temperature", 0),
                max_tokens=provider_config.get("max_tokens", 1000),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        logger.error(f"Failed to create LLM instance: {e}")
        raise
