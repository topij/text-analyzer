# src/core/llm/factory.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.core.config import AnalyzerConfig

logger = logging.getLogger(__name__)


def create_llm(
    analyzer_config: AnalyzerConfig,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseChatModel:
    """Create LLM instance based on configuration.

    Args:
        analyzer_config: Analyzer configuration
        provider: Optional provider override
        model: Optional model override

    Returns:
        Configured LLM instance
    """
    try:
        config = analyzer_config.get_config()
        if not isinstance(config, dict):
            config = config.model_dump()
        
        model_config = config.get("models", {})
        logger.debug(f"Model config: {model_config}")

        # Get provider and model from config if not overridden
        provider = provider or model_config.get("default_provider")
        model = model or model_config.get("default_model")

        logger.debug(f"Using provider={provider}, model={model}")

        if not provider:
            raise ValueError("No LLM provider specified")

        provider_config = model_config.get("providers", {}).get(provider)
        logger.debug(f"Provider config for {provider}: {provider_config}")
        if not provider_config:
            raise ValueError(f"Provider {provider} not found in configuration")

        # Get model settings
        model_settings = provider_config.get("available_models", {}).get(model)
        logger.debug(f"Model settings for {model}: {model_settings}")
        if not model_settings:
            # List available models to help debugging
            available = list(provider_config.get("available_models", {}).keys())
            logger.debug(f"Available models for {provider}: {available}")
            raise ValueError(
                f"Model {model} not found for provider {provider}. "
                f"Available models: {available}"
            )

        # Get global and provider-specific parameters
        parameters = {
            **model_config.get("parameters", {}),  # Global parameters
            **(model_settings or {}),  # Model-specific parameters
        }

        logger.debug(f"Using parameters: {parameters}")

        # Create provider-specific LLM
        if provider == "azure":
            return create_azure_llm(model_settings, parameters)
        elif provider == "openai":
            return create_openai_llm(model, parameters)
        elif provider == "anthropic":
            return create_anthropic_llm(model, parameters)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        logger.error(f"Failed to create LLM: {e}", exc_info=True)
        raise


def create_azure_llm(model_settings, parameters):
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not (api_key and endpoint):
        raise ValueError(
            "Azure OpenAI credentials not found in environment variables"
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=model_settings.get("deployment_name"),
        api_key=api_key,
        api_version=parameters.get("api_version", "2024-02-15-preview"),
        temperature=parameters.get("temperature", 0),
        max_tokens=parameters.get("max_tokens", 1000),
    )


def create_openai_llm(model, parameters):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables"
        )

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=parameters.get("temperature", 0),
        max_tokens=parameters.get("max_tokens", 1000),
    )


def create_anthropic_llm(model, parameters):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables"
        )

    return ChatAnthropic(
        api_key=api_key,
        model=model,
        temperature=parameters.get("temperature", 0),
        max_tokens=parameters.get("max_tokens", 1000),
    )
