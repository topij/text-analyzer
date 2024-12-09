# Usage Examples

Comprehensive examples for using the Semantic Text Analyzer.

## Basic Usage

### Simple Text Analysis

```python
from semantic_analyzer import SemanticAnalyzer

async def analyze_text():
    # Initialize analyzer
    analyzer = SemanticAnalyzer()
    
    # Analyze text
    text = """Machine learning models are trained using large datasets 
              to recognize patterns. The neural network architecture 
              includes multiple layers for feature extraction."""
              
    result = await analyzer.analyze(text)
    
    # Print results
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

# Run analysis
await analyze_text()
```

### Finnish Text Analysis

```python
async def analyze_finnish_text():
    analyzer = SemanticAnalyzer()
    
    text = """Koneoppimismalleja koulutetaan suurilla datajoukolla 
              tunnistamaan kaavoja. Neuroverkon arkkitehtuuri 
              sisältää useita kerroksia."""
              
    result = await analyzer.analyze(text)
    
    # Print compound words
    print("\nCompound Words:")
    for word in result.keywords.compound_words:
        parts = analyzer.language_processor.get_compound_parts(word)
        print(f"• {word}")
        if parts:
            print(f"  Parts: {' + '.join(parts)}")

await analyze_finnish_text()
```

## Advanced Usage

### Custom Categories

```python
from src.schemas import CategoryConfig

# Define custom categories
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

analyzer = SemanticAnalyzer(
    config={"categories": categories}
)
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
    
    # Process in batches
    results = await analyzer.analyze_batch(
        texts=texts,
        batch_size=2,
        timeout=30.0
    )
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print(f"Keywords: {len(result.keywords.keywords)}")
        print(f"Themes: {len(result.themes.themes)}")
        print(f"Categories: {len(result.categories.matches)}")

await process_batch()
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
            "min_confidence": 0.5,
            "include_hierarchy": True
        },
        "categories": {
            "max_categories": 2,
            "min_confidence": 0.3,
            "require_evidence": True
        }
    },
    "features": {
        "use_caching": True,
        "use_async": True
    }
}

analyzer = SemanticAnalyzer(config=config)
```

### Azure Integration

```python
from FileUtils import FileUtils

# Initialize with Azure storage
file_utils = FileUtils.create_azure_utils(
    connection_string="your-connection-string"
)

# Create analyzer with Azure configuration
analyzer = SemanticAnalyzer(
    file_utils=file_utils,
    config={
        "models": {
            "default_provider": "azure",
            "default_model": "gpt-4o-mini"
        }
    }
)

# Analyze and save results
result = await analyzer.analyze("Your text here")
output_path = analyzer.save_results(
    results=result,
    output_file="analysis_results",
    output_type="processed"
)
```

### Custom Language Processing

```python
from src.core.language_processing import create_text_processor

# Configure language processor
language_processor = create_text_processor(
    language="fi",
    config={
        "min_word_length": 3,
        "excluded_patterns": [
            r"^\d+$",
            r"^[^a-zA-ZäöåÄÖÅ0-9]+$"
        ],
        "preserve_compounds": True
    }
)

# Create analyzer with custom processor
analyzer = SemanticAnalyzer(
    language_processor=language_processor
)
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

<!-- ### Performance Optimization

```python
import asyncio
from memory_profiler import profile

@profile
async def optimized_batch_processing(texts: List[str]):
    # Process in smaller batches
    batch_size = 2
    timeout = 30.0
    max_retries = 3
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Process batch with retries
        for attempt in range(max_retries):
            try:
                batch_results = await asyncio.wait_for(
                    analyzer.analyze_batch(batch),
                    timeout=timeout
                )
                results.extend(batch_results)
                break
            except asyncio.TimeoutError:
                print(f"Batch {i//batch_size} timed out, attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    results.extend([None] * len(batch))
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                results.extend([None] * len(batch))
                break
                
    return results
``` -->

For more examples and detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md).