# Usage Examples

Comprehensive examples for using the Semantic Text Analyzer.

## Basic Analysis

### Simple Text Analysis
```python
from src.semantic_analyzer import SemanticAnalyzer

async def analyze_text():
    analyzer = SemanticAnalyzer()
    
    text = """Machine learning models are trained using large datasets 
              to recognize patterns. The neural network architecture 
              includes multiple layers for feature extraction."""
              
    result = await analyzer.analyze(text)
    
    # Access results
    print("\nKeywords:")
    for kw in result.keywords.keywords:
        print(f"• {kw.keyword} (score: {kw.score:.2f})")
        if kw.domain:
            print(f"  Domain: {kw.domain}")
            
    print("\nThemes:")
    for theme in result.themes.themes:
        print(f"\n• {theme.name}")
        print(f"  Confidence: {theme.confidence:.2f}")
        print(f"  Description: {theme.description}")
        
    print("\nCategories:")
    for cat in result.categories.matches:
        print(f"\n• {cat.name}")
        print(f"  Confidence: {cat.confidence:.2f}")
        print(f"  Description: {cat.description}")
```

### Finnish Text Analysis
```python
async def analyze_finnish_text():
    analyzer = SemanticAnalyzer(parameter_file="parameters_fi.xlsx")
    
    text = """Koneoppimismalleja koulutetaan suurilla datajoukolla 
              tunnistamaan kaavoja. Neuroverkon arkkitehtuuri 
              sisältää useita kerroksia."""
              
    result = await analyzer.analyze(text)
    
    # Print compound words
    print("\nCompound Words:")
    for kw in result.keywords.keywords:
        if kw.compound_parts:
            print(f"• {kw.keyword}")
            print(f"  Parts: {' + '.join(kw.compound_parts)}")
```

## Advanced Usage

### Custom Categories
```python
from src.schemas import CategoryConfig

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
    )
}

analyzer = SemanticAnalyzer(categories=categories)
```

### Batch Processing
```python
async def process_batch():
    analyzer = SemanticAnalyzer()
    
    texts = [
        "First document to analyze",
        "Second document to analyze",
        "Third document to analyze"
    ]
    
    results = await analyzer.analyze_batch(
        texts=texts,
        batch_size=2,
        timeout=30.0
    )
```

### Excel Processing
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.formatting_config import ExcelOutputConfig

# Create Excel analyzer
analyzer = SemanticAnalyzer.from_excel(
    content_file="input.xlsx",
    parameter_file="parameters.xlsx"
)

# Optional: Configure output formatting
format_config = ExcelOutputConfig()

# Process with progress tracking
results_df = await analyzer.analyze_excel(
    analysis_types=["keywords", "themes", "categories"],
    batch_size=10,
    save_results=True,
    output_file="results.xlsx",
    show_progress=True,
    format_config=format_config  # Optional formatting configuration
)
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