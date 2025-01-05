# Configuration Guide

This guide covers configuration options for the Semantic Text Analyzer.

## Configuration Methods

The analyzer supports multiple configuration methods in order of precedence:

1. Runtime parameters passed to components
2. Environment variables
3. Configuration files
   - Excel parameter files
   - Configuration directory files

## Environment Configuration

The environment setup is managed through the `EnvironmentManager` class, which provides a centralized way to configure and manage the application environment.

### Basic Environment Setup

```python
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Create environment configuration
config = EnvironmentConfig(
    log_level="INFO",          # Logging level
    config_dir="config",       # Configuration directory
    project_root=None,         # Auto-detected if None
    custom_directory_structure=None  # Optional custom structure
)

# Initialize environment
env_manager = EnvironmentManager(config)
```

### Environment Variables

Required environment variables in your `.env` file:

```bash
# OpenAI Configuration (Required - at least one of these)
OPENAI_API_KEY=your-key-here
# OR
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=your-endpoint

# Optional Environment Variables
APP_LOGGING_LEVEL=INFO  # Default: INFO
VOIKKO_PATH=path-to-voikko  # Required only for Finnish language support
```

## Parameter Configuration

The analyzer uses Excel parameter files for configuration. Here's a detailed breakdown of the parameters:

### Required Parameters

These parameters must be specified in the parameter file:

```yaml
# Sheet: General Parameters
parameter            | value   | description
--------------------|---------|-------------
language            | en/fi   | Analysis language code (en or fi)
max_keywords        | 10      | Maximum number of keywords to extract
min_keyword_length  | 3       | Minimum length for keywords
focus_on            | tech    | Analysis focus area (e.g., tech, business)
```

### Optional Parameters with Defaults

These parameters have default values but can be customized:

```yaml
# Sheet: General Parameters
parameter              | default | description
----------------------|---------|-------------
include_compounds      | true    | Include compound word analysis (Finnish only)
max_themes            | 3       | Maximum number of themes to extract
min_confidence        | 0.3     | Minimum confidence threshold for results
column_name_to_analyze | text    | Column name containing text to analyze
batch_size            | 3       | Number of texts to process in parallel
timeout               | 30.0    | Analysis timeout in seconds
```

### Analysis Settings

Optional settings to fine-tune the analysis:

```yaml
# Sheet: Analysis Settings
setting                  | value | description
------------------------|-------|-------------
theme_analysis.enabled  | true  | Enable/disable theme extraction
theme_analysis.min_confidence | 0.5 | Minimum confidence for themes
weights.statistical     | 0.4   | Weight for statistical analysis (0-1)
weights.llm            | 0.6   | Weight for LLM analysis (0-1)
```

### Category Configuration

Optional sheet for defining custom categories:

```yaml
# Sheet: Categories
category   | description        | keywords          | parent
-----------|-------------------|-------------------|--------
technical  | Technical content | api,data,system   | 
business   | Business content  | market,growth     | 
education  | Educational      | course,learn      | technical
```

## Parameter File Format

The parameter file supports both English and Finnish column names:

### English Column Names
- `parameter`: Parameter name
- `value`: Parameter value
- `description`: Parameter description

### Finnish Column Names
- `parametri`: Parameter name
- `arvo`: Parameter value
- `kuvaus`: Parameter description

## Example Parameter File

Here's a complete example of a parameter file structure:

```yaml
# Sheet: General Parameters
parameter            | value   | description
--------------------|---------|-------------
language            | en      | Analysis language
max_keywords        | 10      | Maximum keywords
min_keyword_length  | 3       | Minimum keyword length
focus_on           | tech    | Analysis focus
include_compounds   | true    | Include compound words
max_themes         | 3       | Maximum themes
min_confidence     | 0.3     | Confidence threshold

# Sheet: Analysis Settings
setting                  | value
------------------------|-------
theme_analysis.enabled  | true
theme_analysis.min_confidence | 0.5
weights.statistical     | 0.4
weights.llm            | 0.6

# Sheet: Categories
category   | description        | keywords
-----------|-------------------|----------
technical  | Technical content | api,data,system
business   | Business content  | market,growth
```

## Logging Configuration

Logging is configured automatically through the `EnvironmentManager`:

```python
# Log level is set via environment variable or config
log_level = os.getenv("APP_LOGGING_LEVEL", "INFO")

# Format is standardized
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

## Directory Structure

The default directory structure is:

```
project_root/
├── data/
│   ├── raw/          # Raw input data
│   ├── interim/      # Intermediate processing files
│   ├── processed/    # Final processed data
│   ├── config/       # Configuration files
│   ├── parameters/   # Parameter files
│   └── logs/         # Log files
├── notebooks/        # Jupyter notebooks
├── docs/            # Documentation
├── scripts/         # Utility scripts
├── src/             # Source code
├── reports/         # Analysis reports
└── models/          # Trained models
```

You can customize this structure by providing a `custom_directory_structure` to `EnvironmentConfig`.