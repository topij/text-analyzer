# Configuration Guide

This guide covers all configuration options for the Semantic Text Analyzer, including model settings, analysis parameters, and integration options.

## Configuration Methods

The analyzer supports multiple configuration methods in order of precedence:

1. Runtime parameters
2. Environment variables
3. Configuration files (YAML)
4. Default values

## Core Configuration File

The main configuration file (`config.yaml`):

```yaml
# Project settings
project_name: "semantic-text-analyzer"
version: "1.0.0"
environment: "development"

# Logging configuration
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  file_path: "logs/app.log"
  disable_existing_loggers: false

# Model settings
models:
  default_provider: "azure"
  default_model: "gpt-4o-mini"
  parameters:
    temperature: 0.0
    max_tokens: 1000
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0
  
  providers:
    azure:
      available_models:
        gpt-4o-mini:
          description: "Fast and cost-effective model"
          deployment_name: "gpt-4o-mini"
          max_tokens: 4096
          supports_functions: true
      api_type: "azure"
      api_version: "2024-02-15-preview"
      
    anthropic:
      available_models:
        claude-3-sonnet:
          description: "Balanced performance model"
          max_tokens: 200000
          supports_functions: false

# Analysis configurations
analysis:
  keywords:
    max_keywords: 10
    min_keyword_length: 3
    include_compounds: true
    min_confidence: 0.3
    use_statistical: true
    compound_word_bonus: 0.2
    weights:
      statistical: 0.4
      llm: 0.6

  themes:
    max_themes: 3
    min_confidence: 0.5
    include_hierarchy: true
    require_evidence: true
    min_theme_length: 10

  categories:
    max_categories: 3
    min_confidence: 0.3
    require_evidence: true
    allow_overlap: true
    hierarchical: true

# Feature flags
features:
  use_caching: true
  use_async: true
  use_batching: true
  enable_finnish_support: true
  enable_compound_detection: true
  enable_spell_check: false

# Processing settings
processing:
  batch_size: 100
  max_retries: 3
  retry_delay: 1.0
  timeout: 30.0
  max_workers: 4
  chunk_size: 1000

# Cache settings
caching:
  enabled: true
  ttl: 3600  # 1 hour
  max_size: 1000
  backend: "memory"
  compression: true
```

## Environment Variables

Important environment variables:

```bash
# LLM Provider Keys
OPENAI_API_KEY=your-key-here
AZURE_OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Azure Configuration
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Language Support
VOIKKO_PATH=/path/to/voikko

# Analysis Configuration
SEMANTIC_ANALYZER_MAX_KEYWORDS=10
SEMANTIC_ANALYZER_MIN_CONFIDENCE=0.3
SEMANTIC_ANALYZER_BATCH_SIZE=100

# Feature Flags
SEMANTIC_ANALYZER_USE_CACHING=true
SEMANTIC_ANALYZER_USE_ASYNC=true
```

## Runtime Configuration

### Analyzer Initialization

```python
analyzer = SemanticAnalyzer(
    config={
        "max_keywords": 5,
        "min_confidence": 0.3,
        "analysis": {
            "keywords": {
                "include_compounds": True,
                "weights": {
                    "statistical": 0.4,
                    "llm": 0.6
                }
            },
            "themes": {
                "max_themes": 3,
                "min_confidence": 0.5
            }
        }
    }
)
```

### Language Processing Configuration

```python
# English configuration
en_config = {
    "min_word_length": 3,
    "excluded_patterns": [
        r"^\d+$",
        r"^[^a-zA-Z0-9]+$"
    ],
    "processing_options": {
        "use_pos_tagging": True,
        "handle_contractions": True
    }
}

# Finnish configuration
fi_config = {
    "min_word_length": 3,
    "excluded_patterns": [
        r"^\d+$",
        r"^[^a-zA-ZäöåÄÖÅ0-9]+$"
    ],
    "voikko": {
        "paths": {
            "win32": {
                "lib_path": "C:/scripts/Voikko/libvoikko-1.dll",
                "dict_path": "C:/scripts/Voikko"
            },
            "linux": {
                "lib_path": "/usr/lib/libvoikko.so.1",
                "dict_path": "/usr/lib/voikko"
            }
        }
    }
}
```

## Category Configuration

Define custom categories in Excel or YAML:

```yaml
categories:
  technical:
    description: "Technical content"
    keywords: ["software", "api", "data"]
    threshold: 0.6
    parent: null
  
  business:
    description: "Business content"
    keywords: ["revenue", "growth", "market"]
    threshold: 0.6
    parent: null
```

## Advanced Configuration 

### Caching Configuration (testing needed)

```python
caching_config = {
    "enabled": True,
    "backend": "redis",
    "ttl": 3600,
    "redis_config": {
        "host": "localhost",
        "port": 6379,
        "db": 0
    }
}
```

### FileUtils Configuration (testing needed)

```python
file_utils_config = {
    "storage_type": "azure",
    "azure_connection_string": "your-connection-string",
    "container_name": "your-container",
    "local_cache": True,
    "cache_dir": "cache"
}
```

### Logging Configuration

```python
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        }
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True
        }
    }
}
```

## Best Practices

1. **Configuration Organization**
   - Use YAML for static configuration
   - Use environment variables for credentials
   - Use runtime config for dynamic settings

2. **Security**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Implement proper access controls

3. **Performance**
   - Enable caching for repeated analyses
   - Configure appropriate batch sizes
   - Set reasonable timeouts

4. **Monitoring**
   - Enable appropriate logging levels
   - Configure error tracking
   - Set up performance monitoring

## Troubleshooting

Common configuration issues:

1. **Invalid Configuration**
   - Check YAML syntax
   - Verify environment variables
   - Validate parameter types

2. **Performance Issues**
   - Check batch sizes
   - Verify caching configuration
   - Monitor resource usage

3. **Integration Problems**
   - Verify API keys
   - Check endpoint configurations
   - Validate network settings

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).