# src/core/config.py

"""Configuration handler for semantic text analyzer."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class AnalyzerConfig:
    """Configuration handler that uses FileUtils and environment variables."""

    DEFAULT_CONFIG = {
        "default_language": "en",
        "content_column": "content",
        "analysis": {
            "keywords": {
                "max_keywords": 5,
                "min_keyword_length": 3,
                "include_compounds": True,
            },
            "themes": {
                "max_themes": 3,
                "min_confidence": 0.5,
                "include_hierarchy": True,
            },
            "categories": {
                "max_categories": 3,
                "min_confidence": 0.3,
                "require_evidence": True,
            },
        },
        "models": {
            "default_provider": "openai",  # Changed default to azure
            "default_model": "gpt-4o-mini",
            "parameters": {
                "temperature": 0.0,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "providers": {
                "azure": {
                    "api_version": "2024-02-15-preview",
                    "api_type": "azure",
                },
                "openai": {
                    "api_type": "open_ai",
                },
                "anthropic": {
                    "api_type": "anthropic",
                },
            },
        },
        "features": {
            "use_caching": True,
            "use_async": True,
            "use_batching": True,
            "enable_finnish_support": True,
        },
    }

    REQUIRED_ENV_VARS = {
        "azure": [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
        ],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }

    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize configuration handler.

        Args:
            file_utils: Optional FileUtils instance
        """
        # Load environment variables
        self._load_env_vars()

        # Initialize FileUtils
        self.file_utils = file_utils or FileUtils()

        # Load and merge configurations
        self.config = self._load_config()

        # Validate required variables
        self._validate_required_vars()

        # Only set up logging if not already configured
        if not logging.getLogger().handlers:
            self._setup_logging()

    def _load_env_vars(self) -> None:
        """Load environment variables from .env files."""
        for env_file in [".env", ".env.local"]:
            if Path(env_file).exists():
                load_dotenv(env_file)
                logger.debug(f"Loaded environment from {env_file}")

    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configurations from different sources."""
        config = self.DEFAULT_CONFIG.copy()

        try:
            # Load from FileUtils
            file_config = self.file_utils.config.get("semantic_analyzer", {})
            config = self._deep_merge(config, file_config)

            # Override with environment variables if present
            env_config = self._load_env_config()
            if env_config:
                config = self._deep_merge(config, env_config)

        except Exception as e:
            logger.warning(f"Error loading configuration: {e}. Using defaults.")

        return config

    def _validate_required_vars(self) -> None:
        """Validate required environment variables."""
        provider = self.config["models"]["default_provider"]
        required_vars = self.REQUIRED_ENV_VARS.get(provider, [])

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables for {provider}: {', '.join(missing)}"
            )

    # In src/core/config.py, update the get_provider_config method:

    def get_provider_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get provider-specific configuration with credentials.

        Args:
            provider: Optional provider name (defaults to configured default)
            model: Optional model name to override default

        Returns:
            Dict[str, Any]: Provider configuration with credentials
        """
        provider = provider or self.config["models"]["default_provider"]

        if provider not in self.REQUIRED_ENV_VARS:
            raise ValueError(f"Unsupported provider: {provider}")

        # Get base configuration for provider
        base_config = self.config["models"]["providers"][provider].copy()

        # Add default model parameters
        base_config.update(self.config["models"]["parameters"])

        # Set model name
        model = model or self.config["models"]["default_model"]
        base_config["model"] = model

        # Add provider-specific model config if available
        if "available_models" in self.config["models"]["providers"][provider]:
            if (
                model
                in self.config["models"]["providers"][provider][
                    "available_models"
                ]
            ):
                model_config = self.config["models"]["providers"][provider][
                    "available_models"
                ][model]
                base_config.update(model_config)

        # Add credentials based on provider
        if provider == "azure":
            base_config.update(
                {
                    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                    "azure_deployment": os.getenv(
                        "AZURE_OPENAI_DEPLOYMENT_NAME"
                    ),
                }
            )
        elif provider == "openai":
            base_config.update(
                {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            )
        elif provider == "anthropic":
            base_config.update(
                {
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                }
            )

        return base_config

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _load_env_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from environment variables."""
        config = {}

        # Model configuration from environment
        if model := os.getenv("SEMANTIC_ANALYZER_MODEL"):
            config["models"] = {"default_model": model}

        # Feature flags from environment
        for feature in ["use_caching", "use_async", "use_batching"]:
            if value := os.getenv(f"SEMANTIC_ANALYZER_{feature.upper()}"):
                if "features" not in config:
                    config["features"] = {}
                config["features"][feature] = value.lower() == "true"

        return config if config else None

    def _validate_required_vars(self) -> None:
        """Validate required environment variables."""
        required_vars = ["OPENAI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    def _setup_logging(self) -> None:
        """Set up logging only if not already configured."""
        root_logger = logging.getLogger()

        # If handlers exist and level is set, respect existing configuration
        if root_logger.handlers and root_logger.level != logging.NOTSET:
            return

        # Get logging config from loaded configuration
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        format_str = log_config.get("format", "%(levelname)s: %(message)s")

        # Configure handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(level)

    # Public interface methods remain the same
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get("models", self.DEFAULT_CONFIG["models"])

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.config.get("analysis", {}).get(
            analyzer_type,
            self.DEFAULT_CONFIG["analysis"].get(analyzer_type, {}),
        )

    def get_features(self) -> Dict[str, bool]:
        """Get feature flags."""
        return self.config.get("features", self.DEFAULT_CONFIG["features"])

    @property
    def default_language(self) -> str:
        """Get default language."""
        return self.config.get(
            "default_language", self.DEFAULT_CONFIG["default_language"]
        )

    @property
    def content_column(self) -> str:
        """Get content column name."""
        return self.config.get(
            "content_column", self.DEFAULT_CONFIG["content_column"]
        )

    def save_results(
        self,
        data: Dict[str, Any],
        filename: str,
        output_type: str = "processed",
    ) -> Path:
        """Save results using FileUtils."""
        return self.file_utils.save_yaml(
            data=data,
            file_path=filename,
            output_type=output_type,
            include_timestamp=self.file_utils.config.get(
                "include_timestamp", True
            ),
        )
