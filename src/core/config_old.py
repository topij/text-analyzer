# src/core/config.py


import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class AnalyzerConfig:
    """Configuration handler that uses FileUtils and environment variables."""

    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize configuration handler.



        Args:

            file_utils: Optional FileUtils instance

        """

        # Load environment variables

        self._load_env_vars()

        # Initialize FileUtils

        self.file_utils = file_utils or FileUtils()

        # Validate required variables

        self._validate_required_vars()

        # Get semantic analyzer config

        self.config = self.file_utils.config.get("semantic_analyzer", {})

        # Initialize logging using FileUtils configuration

        self._setup_logging()

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""

        for env_file in [".env", ".env.local"]:

            if Path(env_file).exists():

                load_dotenv(env_file)

                logger.debug(f"Loaded environment from {env_file}")

    def _validate_required_vars(self) -> None:
        """Validate required environment variables."""

        required_vars = ["OPENAI_API_KEY"]

        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:

            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _setup_logging(self) -> None:
        """Set up logging using FileUtils configuration."""

        # FileUtils already handles logging setup

        pass

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""

        model_config = self.config.get("models", {})

        return {
            "provider": model_config.get("default_provider", "openai"),
            "model": model_config.get("default_model", "gpt-4o-mini"),
            "parameters": model_config.get("parameters", {}),
        }

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type.



        Args:

            analyzer_type: Type of analyzer ("keywords", "themes", "categories")



        Returns:

            Configuration dictionary

        """

        return self.config.get("analysis", {}).get(analyzer_type, {})

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

    @property
    def environment(self) -> str:
        """Get current environment."""

        return os.getenv("ENVIRONMENT", "development")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""

        return self.environment == "production"

    def get_data_path(self, data_type: str = "raw") -> Path:
        """Get data directory path using FileUtils."""

        return self.file_utils.get_data_path(data_type)

    def save_results(self, data: Dict[str, Any], filename: str, output_type: str = "processed") -> Path:
        """Save results using FileUtils.



        Args:

            data: Data to save

            filename: Output filename

            output_type: Output directory type



        Returns:

            Path to saved file

        """

        return self.file_utils.save_yaml(
            data=data,
            file_path=filename,
            output_type=output_type,
            include_timestamp=self.file_utils.config.get("include_timestamp", True),
        )

    def load_configuration(self, filename: str) -> Dict[str, Any]:
        """Load configuration file using FileUtils.



        Args:

            filename: Configuration filename



        Returns:

            Configuration dictionary

        """

        return self.file_utils.load_yaml(filename, input_type="configurations")


# Global config instance

analyzer_config = AnalyzerConfig()


# Example usage:

"""

from src.core.config import analyzer_config



# Get model configuration

model_config = analyzer_config.get_model_config()



# Get analyzer-specific configuration

keyword_config = analyzer_config.get_analyzer_config('keywords')



# Get data path

data_path = analyzer_config.get_data_path('processed')



# Save results

results_path = analyzer_config.save_results(

    results_data,

    'analysis_results.yaml'

)
"""
