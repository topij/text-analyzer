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
```bash
# Create .env file or set environment variables
# OpenAI
export OPENAI_API_KEY='your-key-here'

# Or Azure OpenAI
export AZURE_OPENAI_API_KEY='your-key-here'
export AZURE_OPENAI_ENDPOINT='your-endpoint'
export AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment'

# Finnish support on Windows
setx VOIKKO_PATH "C:\scripts\Voikko"
```

4. Verify installation:
```python
from src.semantic_analyzer import verify_environment
verify_environment()
```

## First Analysis

```python
from src.semantic_analyzer import SemanticAnalyzer

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
```

## Excel Analysis

```python
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
models:
  default_provider: "azure"  # or "openai"
  default_model: "gpt-4o-mini"
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
   - Verify API keys are set
   - Check endpoint configuration for Azure

3. File Operations:
   - Ensure proper file paths
   - Check file permissions