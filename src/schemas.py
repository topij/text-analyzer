# src/schemas.py

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, confloat


class KeywordInfo(BaseModel):
    """Information about an extracted keyword."""

    keyword: str = Field(..., description="The extracted keyword or phrase")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    domain: Optional[str] = Field(None, description="Domain/category")
    compound_parts: Optional[List[str]] = Field(
        None, description="Parts if compound word"
    )


class KeywordAnalysisResult(BaseModel):
    """Complete result of keyword analysis."""

    keywords: List[KeywordInfo] = Field(
        default_factory=list,
        description="List of extracted keywords with scores and metadata",
    )
    compound_words: List[str] = Field(
        default_factory=list, description="List of identified compound words"
    )
    domain_keywords: Dict[str, List[str]] = Field(
        default_factory=dict, description="Keywords grouped by domain"
    )
    language: str = Field(
        ..., description="Detected language of the analyzed text"
    )
    success: bool = Field(
        default=True, description="Whether the analysis was successful"
    )
    error: Optional[str] = Field(
        None, description="Error message if analysis failed"
    )


class Evidence(BaseModel):
    """Evidence supporting a categorization."""

    text: str
    relevance: float = Field(ge=0.0, le=1.0)


class CategoryMatch(BaseModel):
    """Information about a category match."""

    name: str = Field(..., description="Category name")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score"
    )
    description: str = Field(default="", description="Category description")
    evidence: List[Evidence] = Field(
        default_factory=list, description="Supporting evidence"
    )
    themes: List[str] = Field(
        default_factory=list, description="Related themes"
    )


class CategoryOutput(BaseModel):
    """Complete result of category analysis."""

    categories: List[CategoryMatch] = Field(default_factory=list)
    language: str = Field(default="unknown")
    success: bool = Field(default=True)
    error: Optional[str] = None


class CategoryAnalysisResult(BaseModel):
    """Complete result of category analysis."""

    matches: List[CategoryMatch] = Field(
        default_factory=list,
        description="List of category matches with confidence scores",
    )
    language: str = Field(
        ..., description="Detected language of the analyzed text"
    )
    success: bool = Field(
        default=True, description="Whether the analysis was successful"
    )
    error: Optional[str] = Field(
        None, description="Error message if analysis failed"
    )


class ThemeInfo(BaseModel):
    """Information about an identified theme."""

    name: str = Field(..., description="Name of the identified theme")
    description: str = Field(
        ..., description="Detailed description of the theme"
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ..., description="Confidence score for this theme"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Keywords associated with this theme"
    )
    parent_theme: Optional[str] = Field(
        None, description="Parent theme if hierarchical"
    )


class ThemeAnalysisResult(BaseModel):
    """Complete result of theme analysis."""

    themes: List[ThemeInfo] = Field(
        default_factory=list,
        description="List of identified themes with metadata",
    )
    theme_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Hierarchical relationships between themes",
    )
    language: str = Field(
        ..., description="Detected language of the analyzed text"
    )
    success: bool = Field(
        default=True, description="Whether the analysis was successful"
    )
    error: Optional[str] = Field(
        None, description="Error message if analysis failed"
    )


class AnalysisParameters(BaseModel):
    """Essential parameters for text analysis."""

    language: str = Field(
        default="en", description="Language code (e.g., 'en', 'fi')"
    )
    max_keywords: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of keywords to extract",
    )
    categories: Dict[str, Dict[str, Any]] = Field(
        ...,  # This is required
        description="Category definitions with names, descriptions, and keywords",
    )
    min_confidence: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results",
    )
    focus_on: Optional[str] = Field(
        default=None, description="Optional focus area for analysis"
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "allow"  # Allow additional fields for flexibility


class CompleteAnalysisResult(BaseModel):
    """Combined results from all analysis types."""

    keywords: KeywordAnalysisResult
    themes: ThemeAnalysisResult
    categories: CategoryAnalysisResult
    language: str = Field(
        ..., description="Detected language of the analyzed text"
    )
    success: bool = Field(
        default=True, description="Whether the complete analysis was successful"
    )
    error: Optional[str] = Field(
        None, description="Error message if analysis failed"
    )
    processing_time: float = Field(
        ..., description="Total processing time in seconds"
    )
