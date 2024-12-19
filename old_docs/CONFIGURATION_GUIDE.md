# Configuration Guide

This guide covers the implemented configuration options for the Semantic Text Analyzer.

## Configuration Methods

The analyzer supports multiple configuration methods in order of precedence:

1. Runtime parameters
2. Environment variables
3. Configuration files
   - `config.yaml` (base configuration)
   - `config.dev.yaml` (development overrides)

## Core Configuration Files

### Base Configuration (`config.yaml`)

```yaml
# Project settings
project_name: "semantic-text-analyzer"
version: "0.4.5"
environment: "development"

# Logging configuration
logging:
  level: "INFO"  # Default logging level
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  file_path: "app.log"
  disable_existing_loggers: true

# Model settings
models:
  default_provider: "openai"  # or "azure" or "anthropic"
  default_model: "gpt-4o-mini"
  parameters:
    temperature: 0.0
    max_tokens: 1000
    top_p: 1.0
  
  providers:
    openai:
      available_models:
        gpt-4o-mini:
          description: "Fast and cost-effective for simpler tasks"
          max_tokens: 4096
          supports_functions: true
      api_type: "open_ai"
      
    azure:
      available_models:
        gpt-4o-mini:
          description: "Azure OpenAI GPT-4o-mini deployment"
          deployment_name: "gpt-4o-mini"
          max_tokens: 4096
          supports_functions: true
      api_type: "openai"
      api_version: "2024-02-15-preview"

# Language-specific settings
languages:
  fi:
    min_word_length: 3
    excluded_patterns: 
      - "^\\d+$"
      - "^[^a-zA-ZäöåÄÖÅ0-9]+$"
    voikko:
      paths:
        win32:
          lib_path: "C:/scripts/Voikko/libvoikko-1.dll"
          dict_path: "C:/scripts/Voikko"
        linux:
          lib_path: "/usr/lib/x86_64-linux-gnu/libvoikko.so.1"
          dict_path: "/usr/lib/voikko"
    preserve_compounds: true
```

### Development Configuration (`config.dev.yaml`)

```yaml
# Override main settings for development
environment: "development"

# Enhanced logging for development
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s [%(filename)s:  %(lineno)d] - %(message)s"

# Development-specific analyzer settings
analyzer:
  models:
    default_model: "gpt-4o-mini"
    parameters:
      temperature: 0.0
      max_tokens: 2000  # Increased for debugging
```

## Environment Variables

Currently implemented environment variables:

```bash
# LLM Provider Keys
OPENAI_API_KEY=your-key-here
AZURE_OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Azure Configuration
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Environment Selection
ENV=development  # or production
```

## Runtime Configuration

### Analyzer Initialization

```python
from src.semantic_analyzer import SemanticAnalyzer
from FileUtils import FileUtils
from src.config import ConfigManager

# Initialize shared components
file_utils = FileUtils()
config_manager = ConfigManager(file_utils=file_utils)

# Create analyzer with shared components
analyzer = SemanticAnalyzer(
    parameter_file="parameters_en.xlsx",  # or parameters_fi.xlsx
    file_utils=file_utils,
    config_manager=config_manager
)
```

## Category Configuration

Categories are defined in Excel parameter files with the following structure:

```
Sheet: Categories
Columns:
- category: Category name
- description: Detailed description
- keywords: Comma-separated keywords
- threshold: Confidence threshold (0.0-1.0)
- parent: Optional parent category
```

## Best Practices

1. **Configuration Management**
   - Use `config.yaml` for base settings
   - Use `config.dev.yaml` for development overrides
   - Use environment variables for credentials and environment selection

2. **Component Sharing**
   - Initialize FileUtils once and share the instance
   - Use ConfigManager for centralized configuration
   - Pass shared components to analyzers

3. **Logging**
   - Configure logging through ConfigManager
   - Use appropriate logging levels for different environments
   - Keep log messages informative but concise

4. **Security**
   - Store credentials in environment variables
   - Never commit sensitive data to version control
   - Use secure credential management in production

## Troubleshooting

Common configuration issues:

1. **Logging Level Issues**
   - Check the `logging.level` setting in your config files
   - Verify `ENV` environment variable is set correctly
   - Ensure logging is configured before other operations

2. **Initialization Problems**
   - Verify FileUtils is initialized only once
   - Check ConfigManager initialization order
   - Confirm parameter file paths are correct

3. **Model Settings**
   - Verify API keys are set in environment
   - Check model availability in provider config
   - Confirm endpoint configurations for Azure

For additional assistance, consult your system administrator or review the logs.