# Configuration Guide

This guide covers the configuration options for the Semantic Text Analyzer.

## Configuration Methods

The analyzer supports multiple configuration methods in order of precedence:

1. Runtime parameters
2. Environment variables
3. Configuration files
   - `config.yaml` (base configuration)
   - `config.dev.yaml` (development overrides)
   - Excel parameter files

## Base Configuration (config.yaml)

```yaml
# Model settings
models:
  default_provider: "azure"  # or "openai" or "anthropic"
  default_model: "gpt-4o-mini"
  parameters:
    temperature: 0.0
    max_tokens: 1000
    top_p: 1.0
  
  providers:
    openai:
      available_models:
        gpt-4o-mini:
          max_tokens: 4096
          supports_functions: true
      
    azure:
      available_models:
        gpt-4o-mini:
          deployment_name: "gpt-4o-mini"
          max_tokens: 4096
          supports_functions: true
      api_version: "2024-02-15-preview"

# Language settings
languages:
  default_language: "en"
  languages:
    en:
      min_word_length: 3
      excluded_patterns: 
        - "^\\d+$"
        - "^[^a-zA-Z0-9]+$"
    fi:
      min_word_length: 3
      excluded_patterns: 
        - "^\\d+$"
        - "^[^a-zA-ZäöåÄÖÅ0-9]+$"
      voikko:
        paths:
          win32: "C:/scripts/Voikko"
          linux: "/usr/lib/x86_64-linux-gnu/libvoikko.so.1"

# Feature flags
features:
  use_caching: true
  batch_processing: true
```

## Excel Parameter Files

### General Parameters Sheet
```
Column Names:
- parameter: Parameter name
- value: Parameter value
- description: Parameter description

Required Parameters:
- max_keywords: Maximum keywords to extract
- language: Analysis language (en/fi)
- min_confidence: Minimum confidence threshold
- focus_on: Analysis focus area
```

### Categories Sheet
```
Column Names:
- category: Category name
- description: Category description
- keywords: Category-specific keywords
- threshold: Category confidence threshold
- parent: Optional parent category
```

### Keywords Sheet
```
Column Names:
- keyword: Keyword or phrase
- importance: Weight/importance (0-1)
- domain: Optional domain classification
```

## Environment Variables

Required environment variables:

```bash
# LLM Provider Keys (at least one required)
OPENAI_API_KEY=your-key-here
AZURE_OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Azure Configuration (if using Azure)
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Finnish Support (Windows)
VOIKKO_PATH=C:\scripts\Voikko
```

## Runtime Configuration

### Analyzer Initialization
```python
from src.semantic_analyzer import SemanticAnalyzer
from FileUtils import FileUtils

# Initialize with custom configuration
analyzer = SemanticAnalyzer(
    parameter_file="parameters.xlsx",
    file_utils=FileUtils(),
    config={
        "max_keywords": 10,
        "min_confidence": 0.3,
        "language": "en"
    }
)
```

## Best Practices

1. Configuration Files
   - Use `config.yaml` for base settings
   - Use `config.dev.yaml` for development
   - Keep sensitive data in environment variables

2. Parameter Files
   - Use separate files for different languages
   - Keep category thresholds between 0.3-0.8
   - Provide comprehensive category keywords

3. Security
   - Never commit API keys to version control
   - Use secure credential management
   - Validate all configuration inputs

## Configuration Validation

The analyzer validates configurations at multiple levels:

1. Parameter validation
2. LLM configuration validation
3. Language support validation
4. File path validation

For troubleshooting configuration issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).