# src/utils/formatting_config.py

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class OutputDetail(str, Enum):
    """Output detail level."""

    MINIMAL = "minimal"  # Basic output - just results
    SUMMARY = "summary"  # Results with key metadata and evidence
    DETAILED = "detailed"  # Full output with all metadata
    DEBUG = "debug"  # Everything including internal states


class BaseColumnFormat(BaseModel):
    """Configuration for column formatting."""

    column_name: str
    format_template: str
    included_fields: List[str]
    confidence_threshold: Optional[float] = None
    max_items: Optional[int] = None


class BaseOutputConfig(BaseModel):
    """Base configuration for output formatting."""

    detail_level: OutputDetail = OutputDetail.SUMMARY
    batch_size: int = 10
    include_confidence: bool = True
    max_length: Optional[int] = None


class ExcelOutputConfig(BaseOutputConfig):
    """Configuration for Excel output formatting."""

    keywords_format: BaseColumnFormat = Field(
        default_factory=lambda: BaseColumnFormat(
            column_name="keywords",
            format_template="{keyword} ({confidence})",
            included_fields=["keyword", "confidence"],
            confidence_threshold=0.3,
            max_items=5,
        )
    )
    themes_format: BaseColumnFormat = Field(
        default_factory=lambda: BaseColumnFormat(
            column_name="themes",
            format_template="{name}: {description}",
            included_fields=["name", "description"],
            confidence_threshold=0.5,
            max_items=3,
        )
    )
    categories_format: BaseColumnFormat = Field(
        default_factory=lambda: BaseColumnFormat(
            column_name="categories",
            format_template="{name} ({confidence})",
            included_fields=["name", "confidence"],
            confidence_threshold=0.3,
            max_items=3,
        )
    )
