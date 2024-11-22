# src/loaders/models.py

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict


class DomainContext(BaseModel):
    """Domain context configuration."""

    name: str
    description: str
    key_terms: List[str] = Field(default_factory=list)
    context: str = ""
    stopwords: List[str] = Field(default_factory=list)


class GeneralParameters(BaseModel):
    """Base parameters shared across analysis types."""

    max_keywords: int = Field(default=10, le=20, ge=1)
    min_keyword_length: int = Field(default=3, le=10, ge=2)
    language: str = Field(default="en")
    focus_on: Optional[str] = None
    include_compounds: bool = Field(default=True)
    max_themes: int = Field(default=3, le=10, ge=1)
    min_confidence: float = Field(default=0.3, le=1.0, ge=0.0)
    column_name_to_analyze: str = Field(default="text")

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in ["en", "fi"]:
            raise ValueError("Language must be 'en' or 'fi'")
        return v.lower()


class CategoryConfig(BaseModel):
    """Configuration for a category."""

    description: str = Field(default="")
    keywords: List[str] = Field(default_factory=list)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    parent: Optional[str] = None  # For hierarchical categories

    @field_validator("keywords")
    def split_keywords(cls, v):
        """Handle comma-separated keywords from Excel."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v


class PredefinedKeyword(BaseModel):
    """Configuration for a predefined keyword."""

    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    domain: Optional[str] = None
    compound_parts: List[str] = Field(default_factory=list)


class ThemeAnalysisSettings(BaseModel):
    """Theme analysis settings."""

    enabled: bool = Field(default=True)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalysisWeights(BaseModel):
    """Weights for different analysis components."""

    statistical: float = Field(default=0.4, ge=0.0, le=1.0)
    llm: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "AnalysisWeights":
        """Validate that weights sum to 1.0."""
        if abs(self.statistical + self.llm - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        return self


class AnalysisSettings(BaseModel):
    """Complete analysis settings."""

    theme_analysis: ThemeAnalysisSettings = Field(
        default_factory=ThemeAnalysisSettings
    )
    weights: AnalysisWeights = Field(default_factory=AnalysisWeights)


# class ParameterSet(BaseModel):
#     """Complete parameter set with all sections."""

#     general: GeneralParameters = Field(default_factory=GeneralParameters)
#     categories: Dict[str, CategoryConfig] = Field(default_factory=dict)
#     predefined_keywords: Dict[str, PredefinedKeyword] = Field(default_factory=dict)
#     excluded_keywords: Set[str] = Field(default_factory=set)
#     analysis_settings: AnalysisSettings = Field(default_factory=AnalysisSettings)
#     domain_context: Dict[str, DomainContext] = Field(default_factory=dict)

#     model_config = ConfigDict(validate_assignment=True, extra="allow")


class ParameterSet(BaseModel):
    """Complete parameter set with all sections."""

    general: GeneralParameters = Field(default_factory=GeneralParameters)
    categories: Dict[str, CategoryConfig] = Field(default_factory=dict)
    predefined_keywords: Dict[str, PredefinedKeyword] = Field(
        default_factory=dict
    )
    excluded_keywords: Set[str] = Field(default_factory=set)
    analysis_settings: AnalysisSettings = Field(
        default_factory=AnalysisSettings
    )
    domain_context: Dict[str, DomainContext] = Field(default_factory=dict)

    def print(self, indent: int = 2) -> None:
        """Print parameters in a readable format."""
        import json

        # Convert to dict and handle non-serializable types
        data = self.model_dump()
        # Convert set to list for JSON serialization
        data["excluded_keywords"] = list(data["excluded_keywords"])

        print(json.dumps(data, indent=indent, default=str))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper type conversion."""
        data = self.model_dump()
        # Convert set to list
        data["excluded_keywords"] = list(data["excluded_keywords"])
        return data

    def __str__(self) -> str:
        """String representation of parameters."""
        return (
            f"ParameterSet(\n"
            f"  general={self.general},\n"
            f"  categories={len(self.categories)} items,\n"
            f"  predefined_keywords={len(self.predefined_keywords)} items,\n"
            f"  excluded_keywords={len(self.excluded_keywords)} items,\n"
            f"  analysis_settings={self.analysis_settings},\n"
            f"  domain_context={len(self.domain_context)} items\n"
            f")"
        )
