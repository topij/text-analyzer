# src/models/parameter_models.py

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParameterSheets(str, Enum):
    """Excel sheet names for parameter loading."""
    GENERAL = "General Parameters"
    KEYWORDS = "Predefined Keywords"
    EXCLUDED = "Excluded Keywords"
    CATEGORIES = "Categories"
    PROMPTS = "Custom Prompts"
    DOMAINS = "Domain Context"

class GeneralParameters(BaseModel):
    """General extraction parameters."""

    model_config = ConfigDict(extra="allow")

    max_keywords: int = Field(default=8, ge=1, le=20, description="Maximum number of keywords to extract")
    max_themes: int = Field(default=3, ge=1, le=10, description="Maximum number of themes to identify")
    language: str = Field(default="en", description="Default language code (e.g., 'en', 'fi')")
    focus_on: str = Field(default="general topics", description="Analysis focus area")
    min_keyword_length: int = Field(default=3, ge=2, description="Minimum keyword length")
    include_compounds: bool = Field(default=True, description="Include compound words")

    @field_validator("language")
    def validate_language(cls, v: str) -> str:
        """Validate language code."""
        supported = {"en", "fi"}
        v = v.lower()
        if v not in supported:
            raise ValueError(f"Language '{v}' not supported. Use: {supported}")
        return v


class KeywordEntry(BaseModel):
    """Single keyword entry with metadata."""

    keyword: str = Field(..., min_length=1)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    domain: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class CategoryEntry(BaseModel):
    """Category definition with metadata."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)
    keywords: List[str] = Field(default_factory=list)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PromptTemplate(BaseModel):
    """Custom prompt template definition."""

    name: str = Field(..., min_length=1)
    version: str = Field(default="1.0.0")
    system_prompt: str = Field(..., min_length=10)
    user_prompt: str = Field(..., min_length=10)
    language: str = Field(default="en")
    notes: Optional[str] = Field(default=None)


class ExtractionParameters(BaseModel):
    """Complete parameter set for extraction."""

    general: GeneralParameters
    predefined_keywords: List[KeywordEntry] = Field(default_factory=list)
    excluded_keywords: List[str] = Field(default_factory=list)
    categories: Dict[str, CategoryEntry] = Field(default_factory=dict)
    custom_prompts: Dict[str, PromptTemplate] = Field(default_factory=dict)

    @field_validator("excluded_keywords")
    def clean_excluded_keywords(cls, v: List[str]) -> List[str]:
        """Clean and validate excluded keywords."""
        return [k.strip().lower() for k in v if k.strip()]

    def get_keywords_for_domain(self, domain: Optional[str] = None) -> List[str]:
        """Get predefined keywords filtered by domain."""
        if domain:
            return [entry.keyword for entry in self.predefined_keywords if entry.domain == domain]
        return [entry.keyword for entry in self.predefined_keywords]


class ValidationError(Exception):
    """Custom validation error with details."""

    def __init__(self, message: str, details: Dict[str, Any]):
        self.message = message
        self.details = details
        super().__init__(message)

class DomainContext(BaseModel):
    """Domain-specific context for analysis."""
    name: str
    description: str
    key_terms: List[str]
    context: str
    stopwords: Optional[List[str]] = None
    
    class Config:
        frozen = True

class AnalysisContext(BaseModel):
    """Analysis context from parameters."""
    domains: Dict[str, DomainContext] = Field(default_factory=dict)
    enhancement_prompt: Optional[str] = None
    
    class Config:
        frozen = True

# Example parameter Excel sheet "Domain Context":
"""
name        | description                  | key_terms                      | context                            | stopwords
business    | Business content analysis    | revenue,growth,profit,market   | Focus on business performance...   | new,current,various
technical   | Technical content analysis   | system,software,data,api       | Focus on technical aspects...      | using,basic,simple
"""
