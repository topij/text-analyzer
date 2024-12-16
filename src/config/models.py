# src/config/models.py

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """Unified logging configuration."""

    level: str = Field(default="DEBUG")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
    file_path: Optional[str] = None
    disable_existing_loggers: bool = False

    @field_validator("level")
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}")
        return v


class ModelConfig(BaseModel):
    """LLM model configuration."""

    default_provider: str = Field(default="openai")
    default_model: str = Field(default="gpt-4o-mini")  # Updated default model
    parameters: Dict[str, Any] = Field(default_factory=dict)
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LanguageConfig(BaseModel):
    """Language processing configuration."""

    default_language: str = Field(default="en")
    languages: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class GlobalConfig(BaseModel):
    """Global application configuration."""

    environment: str = Field(default="development")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    languages: LanguageConfig = Field(default_factory=LanguageConfig)
    features: Dict[str, bool] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dictionary-like access."""
        try:
            value = getattr(self, key, None)
            if value is None:
                return default
            if isinstance(value, BaseModel):
                return value.model_dump()
            return value
        except Exception:
            return default

    def model_dump(self) -> Dict[str, Any]:
        """Override model_dump to convert nested models to dicts."""
        result = super().model_dump()
        for key, value in result.items():
            if isinstance(value, BaseModel):
                result[key] = value.model_dump()
        return result
