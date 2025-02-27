# src/schemas.py

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, confloat

from src.analyzers.base import AnalyzerOutput


class KeywordInfo(BaseModel):
    """Information about an extracted keyword."""

    keyword: str = Field(..., description="The extracted keyword or phrase")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    domain: Optional[str] = Field(None, description="Domain/category")
    compound_parts: Optional[List[str]] = Field(
        None, description="Parts if compound word"
    )
    frequency: Optional[int] = Field(
        default=0,
        description="Number of occurrences in text"
    )
    is_compound: bool = Field(
        default=False,
        description="Whether this is a compound word/phrase"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the keyword"
    )


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""

    keywords: List[KeywordInfo]
    compound_words: List[str] = Field(default_factory=list)  # Add this field
    domain_keywords: Dict[str, List[str]]


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

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = Field(default="")
    evidence: List[Evidence] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)


class CategoryOutput(AnalyzerOutput):
    """Output model for category analysis."""

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


# class CategoryOutput(AnalyzerOutput):
#     """Output model for category analysis."""

#     categories: List[CategoryMatch] = Field(default_factory=list)
#     language: str = Field(default="unknown")
#     success: bool = Field(default=True)
#     error: Optional[str] = None
#     domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)

#     class Config:
#         arbitrary_types_allowed = True


# class CategoryAnalysisResult(BaseModel):
#     """Complete result of category analysis."""

#     matches: List[CategoryMatch] = Field(
#         default_factory=list,
#         description="List of category matches with confidence scores",
#     )
#     language: str = Field(
#         ..., description="Detected language of the analyzed text"
#     )
#     success: bool = Field(
#         default=True, description="Whether the analysis was successful"
#     )
#     error: Optional[str] = Field(
#         None, description="Error message if analysis failed"
#     )


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


class ThemeOutput(BaseModel):
    """Complete output for theme analysis."""

    themes: List[ThemeInfo] = Field(default_factory=list)
    theme_hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    language: str = Field(default="unknown")
    success: bool = Field(default=True)
    error: Optional[str] = None


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


class ThemeContext(BaseModel):
    """Context information from theme analysis to be used in other analyses."""

    main_themes: List[str] = Field(
        default_factory=list,
        description="List of main themes identified in the text"
    )
    theme_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Hierarchical relationships between themes"
    )
    theme_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Descriptions for each identified theme"
    )
    theme_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for each theme"
    )
    theme_keywords: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Keywords associated with each theme"
    )

    @classmethod
    def from_theme_result(cls, result: ThemeAnalysisResult) -> "ThemeContext":
        """Create ThemeContext from ThemeAnalysisResult."""
        if not result or not result.success:
            return cls()

        main_themes = [theme.name for theme in result.themes if not theme.parent_theme]
        theme_descriptions = {theme.name: theme.description for theme in result.themes}
        theme_confidence = {theme.name: theme.confidence for theme in result.themes}
        theme_keywords = {theme.name: theme.keywords for theme in result.themes if theme.keywords}

        return cls(
            main_themes=main_themes,
            theme_hierarchy=result.theme_hierarchy,
            theme_descriptions=theme_descriptions,
            theme_confidence=theme_confidence,
            theme_keywords=theme_keywords
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
