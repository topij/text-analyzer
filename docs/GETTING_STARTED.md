# Getting Started with Semantic Text Analyzer

This guide will help you get up and running with the Semantic Text Analyzer.

## Prerequisites

1. Required software:
   - Python 3.9+
   - Conda (recommended) or virtualenv
   - Git

2. For Finnish language support:
   - Linux: `libvoikko-dev` and `voikko-fi` packages
   - Windows: Voikko installation from official site
   - macOS: `libvoikko` via Homebrew

## Basic Setup

1. Create and activate environment:
```bash
conda env create -f environment.yaml
conda activate semantic-analyzer
```

2. Install package:
```bash
pip install -e .
```

3. Configure environment:
Create a `.env` file in your project root with the following variables:
```bash
# App Configuration
APP_LOGGING_LEVEL=INFO

# OpenAI Configuration
OPENAI_API_KEY='your-key-here'

# Or Azure OpenAI
AZURE_OPENAI_API_KEY='your-key-here'
AZURE_OPENAI_ENDPOINT='your-endpoint'

# Finnish support (if needed)
VOIKKO_PATH='/path/to/voikko'  # Required for Finnish
```

4. Set up environment:
```python
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Initialize environment
config = EnvironmentConfig(
    log_level="INFO",
    config_dir="config"  # Optional: specify config directory
)
env_manager = EnvironmentManager(config)
```

## First Analysis

```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Set up environment
config = EnvironmentConfig(log_level="INFO")
env_manager = EnvironmentManager(config)
components = env_manager.get_components()

# Initialize analyzer with environment components
analyzer = SemanticAnalyzer(
    file_utils=components["file_utils"],
    config_manager=components["config_manager"]
)

# Analyze text
result = await analyzer.analyze(
    "Machine learning models process data efficiently.",
    analysis_types=["keywords", "themes", "categories"]
)

# Print results
print(f"Keywords: {result.keywords}")
print(f"Themes: {result.themes}")
print(f"Categories: {result.categories}")
```

## Excel Analysis

```python
# Initialize analyzer with parameter file
analyzer = SemanticAnalyzer(
    parameter_file="parameters.xlsx",
    file_utils=components["file_utils"]
)

# Run batch analysis
results = await analyzer.analyze_batch(
    texts=["Text 1", "Text 2", "Text 3"],
    batch_size=3,
    analysis_types=["keywords", "themes"]
)

# Save results
analyzer.save_results(
    results=results,
    output_file="results.xlsx"
)
```

## Parameter Configuration

The analyzer uses Excel parameter files for configuration. Required sheets:

1. General Parameters sheet:
```
parameter            | value | description
--------------------|-------|-------------
max_keywords        | 10    | Maximum keywords to extract
min_keyword_length  | 3     | Minimum keyword length
language           | en    | Analysis language
focus_on           | ...   | Analysis focus area
```

For full parameter configuration options, see [Configuration Guide](CONFIGURATION_GUIDE.md).

## Next Steps

- Check [Examples](EXAMPLES.md) for more usage patterns
- See [Configuration Guide](CONFIGURATION_GUIDE.md) for detailed settings
- Review [API Reference](API_REFERENCE.md) for complete functionality
- Read [Input Files](INPUT_FILES.md) for file format details
- Check [Troubleshooting](TROUBLESHOOTING.md) for common issues

## Common Issues

1. Language Support:
   - Verify Voikko installation for Finnish
   - Check language settings in parameters

2. LLM Connection:
   - Verify API keys in `.env`
   - Check Azure endpoint configuration
   - Ensure proper model access

3. Environment Setup:
   - Check `.env` file exists and is loaded
   - Verify logging level configuration
   - Ensure all required components are initialized

4. File Operations:
   - Check file paths and permissions
   - Verify parameter file format
   - Ensure output directories exist