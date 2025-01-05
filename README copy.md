# Semantic Text Analyzer

A powerful Python-based text analysis toolkit that combines LLMs and NLP techniques to extract insights from text data. Supports both English and Finnish languages, with easy configuration through Excel files.

## Features

- **Text Analysis**
  - Keyword extraction with domain awareness
  - Theme identification with hierarchical relationships
  - Customizable category classification
  - Compound word support for Finnish language

- **Integration & Processing**
  - OpenAI and Azure OpenAI integration via Langchain
  - Excel-based parameter configuration
  - Batch processing support
  - Local and cloud operation support

- **Developer Experience**
  - Type-safe configuration system
  - Centralized environment management
  - Comprehensive logging
  - Excel-based input/output for business users

## Quick Start

### 1. Installation

```bash
# Create and activate environment
conda env create -f environment.yaml
conda activate semantic-analyzer

# Install package
pip install -e .
```

### 2. Configuration

Create a `.env` file with required API keys:

```bash
# Required: Choose one of these options
# Option 1: OpenAI
OPENAI_API_KEY='your-key-here'

# Option 2: Azure OpenAI
AZURE_OPENAI_API_KEY='your-key-here'
AZURE_OPENAI_ENDPOINT='your-endpoint'

# Optional: For Finnish language support
VOIKKO_PATH='/path/to/voikko'  # Required for Finnish
```

### 3. Basic Usage

```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def analyze_text():
    # Initialize environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Create analyzer
    analyzer = SemanticAnalyzer(
        file_utils=components["file_utils"],
        config_manager=components["config_manager"]
    )
    
    # Analyze text
    result = await analyzer.analyze(
        text="Your text here",
        analysis_types=["keywords", "themes", "categories"]
    )
    
    # Process results
    if result.success:
        print("\nKeywords:")
        for kw in result.keywords.keywords:
            print(f"• {kw.keyword} (score: {kw.score:.2f})")
        
        print("\nThemes:")
        for theme in result.themes.themes:
            print(f"• {theme.name} ({theme.confidence:.2f})")
```

### 4. Excel-based Analysis

```python
async def analyze_excel():
    # Initialize as above
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Create analyzer with parameters
    analyzer = SemanticAnalyzer(
        parameter_file="parameters.xlsx",
        file_utils=components["file_utils"]
    )
    
    # Process Excel file
    results = await analyzer.analyze_excel(
        excel_file="input.xlsx",
        analysis_types=["keywords", "themes"],
        save_results=True,
        output_file="results.xlsx"
    )
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Setup and first steps
- [Configuration](docs/CONFIGURATION_GUIDE.md) - Detailed configuration options
- [Examples](docs/EXAMPLES.md) - Code examples and use cases
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

For Finnish language support, see the [Voikko Installation Guide](docs/voikko_installation_guide.md).

## Requirements

- Python 3.9+
- OpenAI API key or Azure OpenAI credentials
- Voikko library (for Finnish language support)
- Required packages listed in `environment.yaml`

## Project Structure

```
semantic-text-analyzer/
├── data/              # Data files and configuration
│   ├── raw/          # Input data
│   ├── processed/    # Analysis results
│   ├── parameters/   # Parameter files
│   └── config/       # Configuration files
├── src/              # Source code
│   ├── analyzers/    # Analysis components
│   ├── core/         # Core functionality
│   ├── models/       # Data models
│   └── utils/        # Utilities
├── docs/             # Documentation
├── tests/            # Test suite
└── notebooks/        # Example notebooks
```

## Example Notebooks

See [demo notebook](notebooks/cross_env_analyzer_demo_nb.ipynb) for basic operations and examples.

## License

MIT License - See [LICENSE](LICENSE) for details.