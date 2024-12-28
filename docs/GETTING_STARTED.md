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

3. Optional:
   - Azure subscription (if using Azure OpenAI)

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
# OpenAI
OPENAI_API_KEY='your-key-here'

# Or Azure OpenAI
AZURE_OPENAI_API_KEY='your-key-here'
AZURE_OPENAI_ENDPOINT='your-endpoint'
AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment'

# Finnish support on Windows
VOIKKO_PATH="C:\scripts\Voikko"
```

4. Set up environment and verify installation:
```python
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

# Initialize environment with logging
config = EnvironmentConfig(log_level="INFO")
env_manager = EnvironmentManager(config)

# Verify setup
status = env_manager.verify_environment()
print("Environment Status:", status)
```

## First Analysis

```python
from src.semantic_analyzer import SemanticAnalyzer
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

# Set up environment
config = EnvironmentConfig(log_level="INFO")
env_manager = EnvironmentManager(config)

async def first_analysis():
    # Initialize analyzer
    analyzer = SemanticAnalyzer()
    
    # Analyze text
    result = await analyzer.analyze(
        "Machine learning models process data efficiently.",
        analysis_types=["keywords", "themes"]
    )
    
    # Print results
    print(f"Keywords: {result.keywords}")
    print(f"Themes: {result.themes}")
    
    # Display LLM info
    llm_info = env_manager.get_llm_info(analyzer, detailed=True)
    print("\nLLM Configuration:", llm_info)
```

## Excel Analysis

```python
# Set up environment
env_manager = EnvironmentManager(
    EnvironmentConfig(log_level="INFO")
)

# Create analyzer for Excel processing
analyzer = SemanticAnalyzer.from_excel(
    content_file="input.xlsx",
    parameter_file="parameters.xlsx"
)

# Run analysis
results_df = await analyzer.analyze_excel(
    analysis_types=["keywords", "themes"],
    batch_size=10,
    save_results=True,
    output_file="results.xlsx"
)
```

## Configuration

1. Basic parameter file structure (Excel):
   - General Parameters sheet: Basic settings
   - Categories sheet: Category definitions
   - Keywords sheet: Predefined keywords
   - Settings sheet: Analysis settings

2. Main configuration file (config.yaml):
```yaml
# Global settings
global:
  environment: "local"  # or "azure"
  log_level: "INFO"

# Model settings
model:
  default_provider: "azure"  # or "openai"
  default_model: "gpt-4"
  temperature: 0.0

# Logging settings
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
```

## Next Steps

- Review [Examples](EXAMPLES.md) for more usage patterns
- Check [Configuration Guide](CONFIGURATION_GUIDE.md) for detailed settings
- See [API Reference](API_REFERENCE.md) for complete functionality
- Review [Troubleshooting](TROUBLESHOOTING.md) for common issues

## Common Issues

1. Language Support:
   - Verify Voikko installation for Finnish support
   - Check language settings in parameters

2. LLM Connection:
   - Verify API keys are set in `.env` file
   - Check endpoint configuration for Azure
   - Use `env_manager.get_llm_info()` to verify LLM setup

3. Environment Setup:
   - Ensure environment variables are properly set in `.env`
   - Check logging configuration in config.yaml
   - Use `env_manager.verify_environment()` to check setup

4. File Operations:
   - Ensure proper file paths
   - Check file permissions
   - Verify directory structure with `env_manager.display_configuration()`