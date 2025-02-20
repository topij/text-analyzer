# Lite Semantic Analyzer

The Lite Semantic Analyzer is a lightweight version of the main semantic analyzer that performs all analyses (keywords, themes, and categories) in a single LLM call. This can significantly improve processing speed when analyzing large volumes of text.

## Features

- Single LLM call for all analysis types (keywords, themes, categories)
- TF-IDF based keyword extraction with LLM refinement
- Compound word detection for both English and Finnish
- Technical term recognition and scoring
- Domain-aware keyword scoring
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
)

# Run analysis
result = await analyzer.analyze(
    text="Your text here",
    analysis_types=["keywords", "themes", "categories"]  # Optional - defaults to all types
)

# Process results
if result.success:
    print("\nKeywords:")
    for kw in result.keywords.keywords:
        print(f"• {kw.keyword} (score: {kw.score:.2f})")
    
    print("\nThemes:")
    for theme in result.themes.themes:
        print(f"• {theme.name} ({theme.confidence:.2f})")
        
    print("\nCategories:")
    for cat in result.categories.matches:
        print(f"• {cat.name} ({cat.confidence:.2f})")
```

## Configuration

The lite analyzer supports the following configuration options:

- `language`: Language of the text to analyze ("en" or "fi")
- `tfidf_weight`: Weight given to TF-IDF results (0.0-1.0)
- `custom_stop_words`: Optional set of additional stop words
- `cache_size`: Maximum number of cached TF-IDF results (default: 1000)
- `parameter_file`: Optional Excel parameter file for additional configuration
- `available_categories`: Optional set of valid categories to choose from

## Technical Details

### Analysis Workflow

The lite analyzer performs all analyses in a single LLM call, but still maintains the benefits of theme-based context:

1. Theme identification is performed first within the LLM call
2. Identified themes are used to enhance keyword and category analysis
3. Results are processed to ensure consistency across all analysis types

### Keyword Extraction

The lite analyzer combines multiple approaches for keyword extraction:

1. TF-IDF is used to identify potential keywords based on statistical significance
2. Keywords are filtered and scored based on:
   - Theme relevance and alignment
   - Technical term patterns
   - Domain-specific terms
   - Compound word patterns
   - Proper noun detection
   - Length and frequency adjustments

### Theme Analysis

Themes are identified by the LLM and organized into a hierarchical structure. Theme confidence is calculated based on:

- Position in theme hierarchy (main themes vs sub-themes)
- Specificity of the theme (multi-word themes get higher confidence)
- Overlap with extracted keywords
- Theme relationships and context

### Category Analysis

Categories are matched and scored using a comprehensive approach:

- Theme-based scoring:
  - Theme overlap assessment
  - Semantic similarity calculation
  - Theme hierarchy consideration
- Keyword-based scoring:
  - Keyword presence and relevance
  - Evidence strength from matching keywords
- Combined scoring:
  - Base confidence (0.7)
  - Theme bonus (up to 0.2)
  - Keyword bonus (up to 0.1)
  - Evidence-based adjustments

## Performance Considerations

- The lite analyzer is optimized for speed by using a single LLM call
- TF-IDF results are cached to improve performance on similar texts
- Processing time is typically faster than the full analyzer, especially for batch processing
- Trade-off: May provide slightly less detailed analysis compared to the full analyzer

## Limitations

- Less granular control over individual analysis types
- Fixed scoring patterns for technical terms and compounds
- Limited to English and Finnish languages
- No support for custom domain-specific analyzers 