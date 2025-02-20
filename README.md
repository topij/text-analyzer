# Semantic Text Analyzer

A Python-based text analysis toolkit that combines LLMs and NLP techniques to extract insights from text data. Supports both English and Finnish languages, with easy configuration through Excel files.

## Features

- **Advanced Semantic Analysis**
  - Integrated analysis workflow with theme-based context
  - Theme-enhanced keyword extraction and category matching
  - Hierarchical theme analysis with confidence scoring
  - Theme-based score adjustments for improved accuracy
  - Semantic similarity calculations between themes and categories

- **Text Analysis**
  - Keyword extraction with domain awareness
  - Theme identification with hierarchical relationships
  - Customizable category classification
  - Compound word support for Finnish language
  - Lite analyzer for faster processing with single LLM call

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

The easiest way to get started is to use the setup script:

```bash
# Run the setup script
scripts/setup_dev.sh

# Activate the environment
conda activate semantic-analyzer
```

The setup script will:
- Create and configure the conda environment
- Install required packages
- Set up NLTK data
- Configure Voikko for Finnish language support

Alternatively, you can perform the installation manually, but you may need to download and setup NLTK and Voikko library in addition to the basic environment. 

```bash
# Create and activate environment
conda env create -f environment.yaml
conda activate semantic-analyzer

# Install package
pip install -e .
```

### Platform Support

The package is tested on:
- macOS (Darwin) with Apple Silicon and Intel processors
- Linux (Ubuntu)

For macOS users, Voikko is installed automatically via Homebrew during setup. The library is located at `/opt/homebrew/lib/libvoikko.dylib`.

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
# Note: If using the setup script, Voikko is configured automatically
VOIKKO_LIBRARY_PATH='/opt/homebrew/lib/libvoikko.dylib'  # macOS default path
VOIKKO_DICT_PATH='/opt/homebrew/lib/voikko'  # macOS dictionary path
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
- [Lite Analyzer](docs/LITE_ANALYZER.md) - Guide for using the Lite Analyzer
- [Configuration](docs/CONFIGURATION_GUIDE.md) - Detailed configuration options
- [Examples](docs/EXAMPLES.md) - Code examples and use cases
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

For Finnish language support, see the [Voikko Installation Guide](docs/voikko_installation_guide.md).

## Requirements

- Python 3.9+
- OpenAI API key or Azure OpenAI credentials
- Required packages listed in `environment.yaml`

### Finnish Language Support (Voikko)
- **macOS**: Installed automatically via setup script using Homebrew
- **Linux**: Install `libvoikko` using your system's package manager
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libvoikko-dev
  
  # Fedora
  sudo dnf install libvoikko-devel
  ```

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

## Background and Use Cases

In my previous job, I frequently encountered situations where valuable textual data remained underutilized. Often, due to time constraints or resource limitations, decisions and/or actions were based on arbitrary highlights rather than comprehensive analysis of available text data.

This led me to develop a semantic text analyzer that combines Large Language Models (LLMs) and Natural Language Processing (NLP) techniques. The project served two purposes: to create a practical tool for better textual data analysis, and to deepen my understanding of LLMs, modern Python development tools and cloud services.

Accessibility was a key priority in the design. To make the tool approachable for non-developers, all inputs (both content and parameters) can be provided through [Excel files](docs/INPUT_FILES.md), a format familiar to most business users.

**Example Use Cases:**
- Support ticket analysis and categorization
- Product information tagging using category analyzer
- Chatbot conversation analysis and insights extraction

#### Note from the author
This project was born out of everyday needs I encountered in data analysis and business development. Along the way, I used the development process as an opportunity to learn new concepts and explore LLMs, Langchain NLP, Python development and much more.

Please note that I am not a professional software developer, and, for example, the code has not been heavily optimized or cleaned up. My focus was on functionality rather than fine-tuning or optimization.

Active maintenance is not guaranteed, but I may occasionally revisit and update the code base when needed.
Let me know if you find this useful :-)


## License

MIT License - See [LICENSE](LICENSE) for details.