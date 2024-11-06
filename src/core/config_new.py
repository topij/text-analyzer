# src/core/config.py

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
    file_path: Optional[str] = None
    disable_existing_loggers: bool = False


class ModelConfig(BaseModel):
    """LLM model configuration."""

    default_provider: str = Field(default="openai")
    default_model: str = Field(default="gpt-4o-mini")
    parameters: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0, "max_tokens": 1000, "top_p": 1.0})


class AnalyzerConfig(BaseModel):
    """Analysis configuration."""

    default_language: str = Field(default="en")
    content_column: str = Field(default="content")

    analysis: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "keywords": {
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_compounds": True,
            },
            "themes": {"max_themes": 3, "min_confidence": 0.5},
            "categories": {"max_categories": 3, "min_confidence": 0.3},
        }
    )

    models: ModelConfig = Field(default_factory=ModelConfig)

    features: Dict[str, bool] = Field(
        default_factory=lambda: {
            "use_caching": True,
            "use_async": True,
            "use_batching": True,
            "enable_finnish_support": True,
        }
    )


class ProjectConfig(BaseModel):
    """Complete project configuration."""

    model_config = ConfigDict(extra="allow")

    # Project settings
    project_name: str = Field(default="semantic-text-analyzer")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")

    # Core configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    analyzer: AnalyzerConfig = Field(default_factory=AnalyzerConfig)

    # Data paths
    data_paths: Dict[str, str] = Field(
        default_factory=lambda: {
            "raw": "data/raw",
            "processed": "data/processed",
            "configurations": "data/configurations",
            "models": "models",
            "reports": "reports",
        }
    )

    # File handling
    file_handling: Dict[str, Any] = Field(
        default_factory=lambda: {
            "csv_delimiter": ";",
            "encoding": "utf-8",
            "include_timestamp": True,
            "output_format": "yaml",
        }
    )


class ConfigurationManager:
    """Centralized configuration management."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager."""
        # Find project root
        self.project_root = self._find_project_root()

        # Load environment and config
        self._load_environment()
        self._validate_environment()
        self.config = self._load_config(config_path)

        # Set up logging
        self._setup_logging()
        self.logger = self.get_logger(__name__)

        self.logger.info(f"Configuration initialized for {self.config.project_name}")

    def _find_project_root(self) -> Path:
        """Find project root directory."""
        current_dir = Path.cwd()
        root_indicators = ["config.yaml", "pyproject.toml", ".git", "setup.py"]

        while current_dir != current_dir.parent:
            if any((current_dir / indicator).exists() for indicator in root_indicators):
                return current_dir
            current_dir = current_dir.parent

        return Path.cwd()

    def _load_environment(self) -> None:
        """Load environment variables."""
        env_files = [".env", ".env.local"]
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                load_dotenv(env_path)
                logging.debug(f"Loaded environment from {env_file}")

    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        required_vars = ["OPENAI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _load_config(self, config_path: Optional[Path] = None) -> ProjectConfig:
        """Load configuration from file or use defaults."""
        if config_path is None:
            config_path = self.project_root / "config.yaml"

        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
                return ProjectConfig(**config_dict)
            else:
                logging.warning(f"No config file found at {config_path}, using defaults")
                return ProjectConfig()

        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            return ProjectConfig()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging_config = {
            "version": 1,
            "disable_existing_loggers": self.config.logging.disable_existing_loggers,
            "formatters": {
                "standard": {
                    "format": self.config.logging.format,
                    "datefmt": self.config.logging.date_format,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": self.config.logging.level,
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console"],
                    "level": self.config.logging.level,
                    "propagate": True,
                }
            },
        }

        # Add file handler if configured
        if self.config.logging.file_path:
            log_file = self.project_root / self.config.logging.file_path
            log_file.parent.mkdir(parents=True, exist_ok=True)

            logging_config["handlers"]["file"] = {
                "class": "logging.FileHandler",
                "filename": str(log_file),
                "formatter": "standard",
                "level": self.config.logging.level,
            }
            logging_config["loggers"][""]["handlers"].append("file")

        logging.config.dictConfig(logging_config)

    def get_path(self, path_type: str) -> Path:
        """Get configured path."""
        if path_type not in self.config.data_paths:
            raise ValueError(f"Unknown path type: {path_type}")

        path = self.project_root / self.config.data_paths[path_type]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)

    # Analyzer-specific methods
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "provider": self.config.analyzer.models.default_provider,
            "model": self.config.analyzer.models.default_model,
            "parameters": self.config.analyzer.models.parameters,
        }

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.config.analyzer.analysis.get(analyzer_type, {})

    def get_features(self) -> Dict[str, bool]:
        """Get feature flags."""
        return self.config.analyzer.features

    @property
    def default_language(self) -> str:
        """Get default language."""
        return self.config.analyzer.default_language

    @property
    def content_column(self) -> str:
        """Get content column name."""
        return self.config.analyzer.content_column

    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.config.environment

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def save_yaml(self, data: Dict[str, Any], filename: str, output_type: str = "processed") -> Path:
        """Save data to YAML file."""
        output_path = self.get_path(output_type) / filename

        with open(output_path, "w", encoding=self.config.file_handling["encoding"]) as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        return output_path

    def load_yaml(self, filename: str, input_type: str = "configurations") -> Dict[str, Any]:
        """Load YAML file."""
        file_path = self.get_path(input_type) / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding=self.config.file_handling["encoding"]) as f:
            return yaml.safe_load(f)


# Create global instance
config_manager = ConfigurationManager()
logger = config_manager.get_logger(__name__)
