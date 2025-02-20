# Lite Semantic Analyzer

The Lite Semantic Analyzer is a lightweight version of the main semantic analyzer that performs all analyses (keywords, themes, and categories) in a single LLM call. This can significantly improve processing speed when analyzing large volumes of text.

## Features

- Single LLM call for all analysis types using structured output
- Theme-first analysis workflow with hierarchical theme context
- TF-IDF based keyword extraction with theme-enhanced scoring
- Compound word detection for both English and Finnish
- Technical term recognition and domain-aware scoring
- Caching support for TF-IDF results
- Language detection and validation

## Usage

```python
from src.analyzers.lite_analyzer import LiteSemanticAnalyzer
from src.core.llm.factory import create_llm

# Create LLM instance
llm = create_llm()

# Initialize analyzer
analyzer = LiteSemanticAnalyzer(
    llm=llm,
    language="en",  # or "fi" for Finnish
    tfidf_weight=0.5,  # Weight given to TF-IDF results (0.0-1.0)
    parameter_file="parameters.xlsx",  # Optional configuration file
    custom_stop_words={"custom", "stop", "words"}  # Optional
)

# Run analysis
result = await analyzer.analyze(
    text="Your text here",
    analysis_types=["keywords", "themes", "categories"]  # Optional - defaults to all types
)

# Process results
if result.success:
    # Theme results (processed first)
    print("\nThemes:")
    for theme in result.themes.themes:
        print(f"• {theme.name} ({theme.confidence:.2f})")
        print(f"  Description: {theme.description}")
    
    # Theme hierarchy
    print("\nTheme Hierarchy:")
    for main_theme, sub_themes in result.themes.theme_hierarchy.items():
        print(f"• {main_theme}")
        for sub in sub_themes:
            print(f"  - {sub}")
    
    # Theme-enhanced keywords
    print("\nKeywords:")
    for kw in result.keywords.keywords:
        print(f"• {kw.keyword} (score: {kw.score:.2f})")
        if kw.metadata.get("theme_relevance"):
            print(f"  Theme relevance: {kw.metadata['theme_relevance']:.2f}")
        if kw.is_compound:
            print(f"  Compound parts: {', '.join(kw.compound_parts)}")
        
    # Theme-guided categories
    print("\nCategories:")
    for cat in result.categories.matches:
        print(f"• {cat.name} ({cat.confidence:.2f})")
        if cat.themes:
            print(f"  Related themes: {', '.join(cat.themes)}")
        if cat.evidence:
            print("  Evidence:")
            for ev in cat.evidence:
                print(f"    - {ev.text} (relevance: {ev.relevance:.2f})")
```

## Configuration

The lite analyzer supports the following configuration options:

- `language`: Language of the text to analyze ("en" or "fi")
- `tfidf_weight`: Weight given to TF-IDF results (0.0-1.0)
- `custom_stop_words`: Optional set of additional stop words
- `cache_size`: Maximum number of cached TF-IDF results (default: 1000)
- `parameter_file`: Optional Excel parameter file for additional configuration
- `available_categories`: Optional set of valid categories to choose from
- `domain_context`: Optional domain-specific context for enhanced scoring

## Technical Details

### Theme-First Analysis Workflow

The lite analyzer implements a theme-first approach where themes provide context for other analyses:

1. Theme Analysis:
   - Identifies main themes and sub-themes
   - Creates hierarchical theme structure
   - Assigns confidence scores based on theme position and specificity
   - Considers theme relationships and context

2. Theme-Enhanced Keyword Analysis:
   - Uses identified themes to guide keyword extraction
   - Adjusts keyword scores based on theme relevance (up to 30% boost)
   - Considers TF-IDF statistical significance
   - Incorporates domain context and predefined keywords
   - Handles compound words and technical terms

3. Theme-Guided Category Analysis:
   - Matches categories based on thematic alignment
   - Calculates semantic similarity between categories and themes
   - Adjusts category scores based on theme relevance (up to 25% boost)
   - Uses theme-enhanced keywords as evidence
   - Considers theme hierarchy in scoring

### Keyword Extraction Process

The analyzer combines multiple approaches for keyword extraction:

1. Statistical Analysis:
   - TF-IDF for initial keyword identification
   - Frequency analysis and caching
   - Dynamic keyword limits based on text length

2. Theme-Based Enhancement:
   - Theme relevance scoring
   - Hierarchical theme context
   - Theme-keyword associations

3. Additional Scoring Factors:
   - Technical term patterns
   - Domain-specific terms
   - Compound word detection
   - Proper noun recognition
   - Length and frequency adjustments
   - Predefined keyword boosts

### Category Analysis

Categories are matched using a comprehensive approach:

1. Theme-Based Scoring:
   - Theme overlap assessment
   - Semantic similarity calculation
   - Theme hierarchy consideration

2. Evidence-Based Scoring:
   - Keyword presence and relevance
   - Theme-enhanced evidence strength
   - Compound word recognition

3. Combined Scoring:
   - Base confidence (0.7)
   - Theme bonus (up to 0.25)
   - Keyword bonus (up to 0.1)
   - Evidence-based adjustments

## Performance Considerations

- Single LLM call with structured output for efficiency
- TF-IDF caching for improved performance
- Dynamic keyword limits based on text length
- Optimized scoring calculations
- Efficient theme context handling

## Limitations

- Less granular control over individual analysis types
- Fixed scoring patterns for technical terms and compounds
- Limited to English and Finnish languages
- No support for custom domain-specific analyzers
- Cache size limited to 1000 entries by default 