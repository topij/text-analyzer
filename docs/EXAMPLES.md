# Usage Examples

Comprehensive examples for using the Semantic Text Analyzer.

## Usage Examples

## Basic Analysis

### Simple Text Analysis
```python
from src.semantic_analyzer import SemanticAnalyzer
from pathlib import Path

async def analyze_text():
    # Initialize with parameters
    analyzer = SemanticAnalyzer(
        parameter_file=Path("parameters") / "parameters_en.xlsx"
    )
    
    # Analyze with specific types
    result = await analyzer.analyze(
        text="""Machine learning models are trained using large datasets 
                to recognize patterns. The neural network architecture 
                includes multiple layers for feature extraction.""",
        analysis_types=["keywords", "themes", "categories"],
        timeout=60.0  # Optional timeout in seconds
    )
    
    # Handle results
    if result.success:
        # Keywords
        if result.keywords.success:
            print("\nKeywords:")
            for kw in result.keywords.keywords:
                print(f"• {kw.keyword} (score: {kw.score:.2f})")
                if kw.domain:
                    print(f"  Domain: {kw.domain}")
        
        # Themes
        if result.themes.success:
            print("\nThemes:")
            for theme in result.themes.themes:
                print(f"• {theme.name} ({theme.confidence:.2f})")
                print(f"  Description: {theme.description}")
                if theme.parent_theme:
                    print(f"  Parent: {theme.parent_theme}")
        
        # Categories
        if result.categories.success:
            print("\nCategories:")
            for cat in result.categories.matches:
                print(f"• {cat.name} ({cat.confidence:.2f})")
                if cat.evidence:
                    print("  Evidence:")
                    for ev in cat.evidence:
                        print(f"    - {ev.text}")
    else:
        print(f"Analysis failed: {result.error}")
```

### Finnish Text Analysis
```python
from src.semantic_analyzer import SemanticAnalyzer
from pathlib import Path

async def analyze_finnish_text():
    # Initialize with Finnish parameters
    analyzer = SemanticAnalyzer(
        parameter_file=Path("parameters") / "parameters_fi.xlsx",
        config={
            "language": "fi",
            "features": {
                "preserve_compounds": True
            }
        }
    )
    
    text = """Koneoppimismalleja koulutetaan suurilla datajoukolla 
              tunnistamaan kaavoja. Neuroverkon arkkitehtuuri 
              sisältää useita kerroksia."""
    
    result = await analyzer.analyze(
        text=text,
        analysis_types=["keywords", "themes"]
    )
    
    if result.success:
        # Show compound words
        print("\nCompound Words:")
        for kw in result.keywords.keywords:
            if kw.compound_parts:
                print(f"• {kw.keyword}")
                print(f"  Parts: {' + '.join(kw.compound_parts)}")
                print(f"  Score: {kw.score:.2f}")
```

### Excel Processing
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.formatting_config import ExcelOutputConfig, OutputDetail
from pathlib import Path

async def process_excel():
    # Create Excel analyzer
    analyzer = SemanticAnalyzer.from_excel(
        content_file=Path("data") / "input.xlsx",
        parameter_file=Path("parameters") / "parameters_en.xlsx"
    )
    
    # Configure output formatting
    format_config = ExcelOutputConfig(
        detail_level=OutputDetail.DETAILED,
        include_confidence=True,
        keywords_format={
            "column_name": "keywords",
            "format_template": "{keyword} ({confidence})",
            "confidence_threshold": 0.3,
            "max_items": 5
        }
    )
    
    try:
        # Process with progress tracking
        results_df = await analyzer.analyze_excel(
            analysis_types=["keywords", "themes", "categories"],
            batch_size=10,
            save_results=True,
            output_file="results.xlsx",
            show_progress=True,
            format_config=format_config
        )
        
        print(f"Processed {len(results_df)} documents")
        
    except Exception as e:
        print(f"Excel processing failed: {str(e)}")
```

### Custom Categories
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.schemas import CategoryConfig

# Define categories
categories = {
    "technical": CategoryConfig(
        description="Technical content",
        keywords=["software", "api", "data"],
        threshold=0.6
    ),
    "business": CategoryConfig(
        description="Business content",
        keywords=["revenue", "growth", "market"],
        threshold=0.6
    ),
    "educational": CategoryConfig(
        description="Educational content",
        keywords=["learning", "teaching", "training"],
        threshold=0.5,
        parent="technical"  # Example of hierarchical categories
    )
}

# Initialize analyzer with categories
analyzer = SemanticAnalyzer(
    parameter_file="parameters.xlsx",
    categories=categories
)

# Analyze with categories
result = await analyzer.analyze(
    text="Your text here",
    analysis_types=["categories"]
)

# Process category results
if result.categories.success:
    for cat in result.categories.matches:
        print(f"\nCategory: {cat.name}")
        print(f"Confidence: {cat.confidence:.2f}")
        print(f"Description: {cat.description}")
        if cat.evidence:
            print("Evidence:")
            for ev in cat.evidence:
                print(f"- {ev.text} (relevance: {ev.relevance:.2f})")
```


### Custom Configuration
```python
config = {
    "analysis": {
        "keywords": {
            "max_keywords": 5,
            "min_keyword_length": 3,
            "include_compounds": True,
            "weights": {
                "statistical": 0.4,
                "llm": 0.6
            }
        },
        "themes": {
            "max_themes": 3,
            "min_confidence": 0.5
        },
        "categories": {
            "min_confidence": 0.3
        }
    }
}

analyzer = SemanticAnalyzer(config=config)
```

### Error Handling
```python
async def robust_analysis(text: str):
    try:
        result = await analyzer.analyze(text)
        
        if not result.success:
            print(f"Analysis failed: {result.error}")
            return
            
        if result.keywords.error:
            print(f"Keyword analysis error: {result.keywords.error}")
            
        if result.themes.error:
            print(f"Theme analysis error: {result.themes.error}")
            
        if result.categories.error:
            print(f"Category analysis error: {result.categories.error}")
            
        return result
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
```

For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).