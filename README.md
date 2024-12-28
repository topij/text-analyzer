# Semantic Text Analyzer

A Python-based text analysis toolkit providing keyword extraction, theme identification, and category classification with multi-language support.

## Key Features

- Keyword extraction with domain awareness and compound word support
- Theme identification with hierarchical relationships
- Category classification with configurable categories
- Support for English and Finnish languages
- Excel file processing for batch analysis
- LLM integration (OpenAI, Azure, and Anthropic) using Langchain
- Support for both local and Azure cloud operations (in part enabled by custom [FileUtils package](https://github.com/topij/FileUtils))
- Parameter management and content input through Excel files
- Centralized logging and environment management
- Type-safe configuration system

## Background and Use Cases

In my previous job, I frequently encountered situations where valuable textual data remained underutilized. Often, due to time constraints or resource limitations, decisions and/or actions were based on arbitrary highlights rather than comprehensive analysis of available text data.

This led me to develop a semantic text analyzer that combines Large Language Models (LLMs) and Natural Language Processing (NLP) techniques. The project served two purposes: to create a practical tool for better textual data analysis, and to deepen my understanding of LLMs, modern Python development tools and cloud services. 

Accessibility was a key priority in the design. To make the tool approachable for non-developers, all inputs (both content and parameters) can be provided through [Excel files](docs/INPUT_FILES.md), a format familiar to most business users.

**Example Use Cases:**
- Support ticket analysis and categorization
- Product information tagging using category analyzer
- Chatbot conversation analysis and insights extraction


## Quick Start

1. Create environment and install:
```bash
conda env create -f environment.yaml
conda activate semantic-analyzer
pip install -e .
```

2. Set up environment variables in `.env` file or export them:
```bash
# For OpenAI
OPENAI_API_KEY='your-key-here'

# For Azure OpenAI
AZURE_OPENAI_API_KEY='your-key-here'
AZURE_OPENAI_ENDPOINT='your-endpoint'
AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment'

# For Finnish support (Windows)
VOIKKO_PATH="C:\scripts\Voikko"
```

3. Basic usage:
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

# Set up the environment
config = EnvironmentConfig(log_level="INFO")
env_manager = EnvironmentManager(config)

async def analyze_text():
    analyzer = SemanticAnalyzer()
    result = await analyzer.analyze("Your text here")
    print(f"Keywords: {result.keywords}")
    print(f"Themes: {result.themes}")
```

## Example notebook
Basic operations are demonstrated in [demo notebook](notebooks/cross_env_analyzer_demo_nb.ipynb). For more information see the [documentation](docs/ANALYSIS_DEMO_DOC.md) page

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Basic setup and first steps
- [Installation Guide](docs/INSTALLATION_GUIDE.md) - Detailed installation instructions
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md) - Configuration options
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Examples](docs/EXAMPLES.md) - Usage examples
- [Output Formatting](docs/OUTPUT_FORMATTING.md) - Output format details
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Requirements

- Python 3.9+
- Required packages listed in environment.yaml
- For Finnish support: Voikko library
- Azure subscription (if using Azure OpenAI)

## Project Structure

```
semantic-text-analyzer/
├── data/              # Data directory managed by FileUtils
│   ├── raw/           # Original data files
│   ├── interim/       # Intermediate processing
│   ├── processed/     # Analysis results
│   ├── external/      # External data sources
│   ├── parameters/    # Analysis parameters
│   └── config/        # Config files
├── src/               # Source code
│   ├── config/            # Configuration management
│   ├── formatters/        # Output formatting
│   ├── loaders/           # Parameter loading and validation
│   ├── models/            # Data models and type definitions
│   ├── nb_helpers/        # Notebook helpers and environment management
│   ├── semantic_analyzer/ # Main analyzer interface
│   ├── analyzers/         # Analysis components
│   ├── core/              # Core functionality
│   └── utils/             # Utility functions
├── tests/             # Test suite
├── docs/              # Documentation
├── examples/          # Example scripts
└── notebooks/         # Jupyter notebooks
```

## Notes from the Author
This project was born out of everyday needs I encountered in data analysis and business development. Along the way, I used the development process as an opportunity to learn new concepts and explore LLMs, Langchain NLP, Python development and much more.

Please note that I am not a professional software developer, and, for example, the code has not been heavily optimized or cleaned up. My focus was on functionality rather than fine-tuning or optimization.

Active maintenance is not guaranteed, but I may occasionally revisit and update the code base when needed.
Let me know if you find this useful :-)


## License

MIT License