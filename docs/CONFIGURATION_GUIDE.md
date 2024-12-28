# Configuration Guide

This guide covers configuration options for the Semantic Text Analyzer.

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

## Environment Configuration

The environment setup is managed through the `EnvironmentManager` class, which provides a centralized way to configure and manage the application environment.

### Basic Environment Setup

```python
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

# Create environment configuration
config = EnvironmentConfig(
    env_type="local",          # or "azure"
    log_level="INFO",          # Logging level
    config_dir="config",       # Configuration directory
    project_root=None,         # Auto-detected if None
)

# Initialize environment
env_manager = EnvironmentManager(config)
```

### Environment Variables

Create a `.env` file in your project root with these variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-key-here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Finnish Language Support
VOIKKO_PATH=path-to-voikko  # Required for Finnish
```

## Configuration Files

### Main Configuration (config.yaml)

```yaml
# Global settings
global:
  environment: "local"  # or "azure"
  log_level: "INFO"
  project_root: null    # Auto-detected if null

# Model settings
model:
  default_provider: "azure"  # or "openai"
  default_model: "gpt-4"
  temperature: 0.0
  max_tokens: 2000

# Logging settings
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
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

## Logging Configuration

Logging is managed through the `LoggingManager` class:

```python
from src.nb_helpers.logging_manager import LoggingManager

# Initialize logging manager
logging_manager = LoggingManager()

# Configure logging
logging_manager.configure_logging(
    level="INFO",
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up debug logging for specific component
logging_manager.setup_debug_logging("src.analyzers.keyword_analyzer")

# Verify logging setup
logging_manager.verify_logging_setup(show_hierarchy=True)
```

## Directory Structure

The project uses a standardized directory structure:

```
project_root/
├── data/
│   ├── raw/         # Input data
│   ├── interim/     # Intermediate files
│   ├── processed/   # Analysis results
│   ├── external/    # External resources
│   ├── parameters/  # Parameter files
│   └── config/      # Configuration files
└── src/            # Source code
```

## Verification and Troubleshooting

### Environment Verification

```python
# Verify environment setup
status = env_manager.verify_environment()
print("Environment Status:", status)

# Display current configuration
env_manager.display_configuration()

# Get LLM information
llm_info = env_manager.get_llm_info(analyzer, detailed=True)
print("LLM Configuration:", llm_info)
```

### Common Configuration Issues

1. **Environment Setup**:
   - Use `env_manager.verify_environment()` to check setup
   - Verify `.env` file is properly loaded
   - Check project root detection

2. **Logging**:
   - Use `logging_manager.verify_logging_setup()` to check configuration
   - Verify log levels are properly set
   - Check handler configuration

3. **LLM Configuration**:
   - Verify API keys in `.env`
   - Check model availability
   - Use `get_llm_info()` to verify settings

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