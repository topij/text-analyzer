# Getting Started with Semantic Text Analyzer

This guide will help you get up and running with the Semantic Text Analyzer. We'll cover basic setup, common usage patterns, and best practices.

## Basic Setup

1. First, ensure you have the analyzer installed:
```bash
pip install semantic-text-analyzer
```

2. Set up your environment variables:
```bash
# For OpenAI
export OPENAI_API_KEY='your-key-here'

# For Azure OpenAI
export AZURE_OPENAI_API_KEY='your-key-here'
export AZURE_OPENAI_ENDPOINT='your-endpoint'
export AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment'

# For Anthropic (optional)
export ANTHROPIC_API_KEY='your-key-here'
```

3. Create a basic configuration file (config.yaml):
```yaml
models:
  default_provider: "azure"
  default_model: "gpt-4o-mini"
  parameters:
    temperature: 0.0
    max_tokens: 1000

features:
  use_caching: true
  use_async: true
  enable_finnish_support: true
```

## Basic Usage

### Simple Analysis

```python
from semantic_analyzer import SemanticAnalyzer

# Initialize analyzer
analyzer = SemanticAnalyzer()

# Analyze text
text = """Machine learning models are trained using large datasets 
          to recognize patterns. The neural network architecture 
          includes multiple layers for feature extraction."""

result = await analyzer.analyze(text)

# Access results
print("\nKeywords:")
for kw in result.keywords.keywords:
    print(f"- {kw.keyword} (score: {kw.score:.2f})")

print("\nThemes:")
for theme in result.themes.themes:
    print(f"- {theme.name}: {theme.description}")

print("\nCategories:")
for cat in result.categories.matches:
    print(f"- {cat.name} (confidence: {cat.confidence:.2f})")
```

### Finnish Language Support

The analyzer automatically detects language and applies appropriate processing:

```python
# Finnish text analysis
text_fi = """Koneoppimismalleja koulutetaan suurilla datajoukolla 
             tunnistamaan kaavoja. Neuroverkon arkkitehtuuri 
             sisältää useita kerroksia."""

result_fi = await analyzer.analyze(text_fi)
```

### Batch Processing

For analyzing multiple texts efficiently:

```python
texts = [
    "First document to analyze",
    "Second document to analyze",
    "Third document to analyze"
]

results = await analyzer.analyze_batch(
    texts=texts,
    batch_size=3,
    timeout=30.0
)
```

## Advanced Usage

### Custom Configuration

```python
analyzer = SemanticAnalyzer(
    parameter_file="parameters.xlsx",
    config={
        "max_keywords": 5,
        "min_confidence": 0.3,
        "analysis": {
            "keywords": {
                "include_compounds": True,
                "min_keyword_length": 3
            },
            "themes": {
                "max_themes": 3,
                "min_confidence": 0.5
            }
        }
    }
)
```

### Saving Results

```python
# Save analysis results
output_path = analyzer.save_results(
    results=result,
    output_file="analysis_results",
    output_type="processed"
)
```

### Working with Categories

```python
# Define custom categories
categories = {
    "technical": {
        "description": "Technical content",
        "keywords": ["software", "api", "data"],
        "threshold": 0.6
    },
    "business": {
        "description": "Business content",
        "keywords": ["revenue", "growth", "market"],
        "threshold": 0.6
    }
}

# Create analyzer with custom categories
analyzer = SemanticAnalyzer(
    config={"categories": categories}
)
```

## Best Practices

1. **Language Processing:**
   - Let the analyzer auto-detect language when possible
   - For Finnish text, ensure Voikko is properly installed
   - Consider text preprocessing for better results

2. **Performance:**
   - Use batch processing for multiple texts
   - Enable caching for repeated analyses
   - Configure appropriate timeouts for your use case

3. **Configuration:**
   - Use YAML files for static configuration
   - Use environment variables for credentials
   - Override specific settings at runtime as needed

4. **Error Handling:**
   - Always check the `success` field in results
   - Log and handle potential errors in results
   - Set appropriate confidence thresholds

## Next Steps

- Review the [Configuration Guide](CONFIGURATION_GUIDE.md) for detailed settings
- Check the [API Reference](API_REFERENCE.md) for complete functionality
- See the [Azure Guide](AZURE_GUIDE.md) for cloud deployment