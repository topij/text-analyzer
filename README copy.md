# Semantic Text Analyzer

A comprehensive toolkit for semantic text analysis with multi-language support (English and Finnish). The analyzer provides deep insights into text content through keywords, themes, and category classification using state-of-the-art language models.

## Key Features

- **Multi-Language Support**
  - English and Finnish language processing
  - Language-specific compound word handling
  - Domain-aware analysis for both languages

- **Analysis Types**
  - Keyword Analysis: Extract key terms with domain context
  - Theme Analysis: Identify main themes and their relationships
  - Category Classification: Classify content into predefined categories

- **File Management and Storage**
  - Integrated FileUtils for robust file operations
  - Support for local and cloud storage (Azure)
  - Automatic project structure management
  - Consistent file handling across environments

- **Advanced Capabilities**
  - Compound word detection and analysis
  - Domain-specific terminology recognition
  - Position-aware text analysis
  - Hierarchical theme relationships
  - Evidence-based categorization

- **Technical Features**
  - Async/await interface for modern applications
  - Batch processing support
  - Configurable caching
  - LLM provider flexibility (OpenAI, Azure OpenAI, Anthropic)
  - Extensive configuration options

- **Integration Options**
  - Azure ML workspace integration
  - Local and cloud deployment support
  - FileUtils for consistent file operations
  - Flexible storage backends

## Quick Start

```python
from semantic_analyzer import SemanticAnalyzer

# Initialize analyzer
analyzer = SemanticAnalyzer()

# Analyze text
result = await analyzer.analyze(
    text="Your text here",
    analysis_types=["keywords", "themes", "categories"]
)

# Access results
print(f"Keywords: {result.keywords}")
print(f"Themes: {result.themes}")
print(f"Categories: {result.categories}")
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Installation Guide](docs/INSTALLATION_GUIDE.md)
- [Azure Guide](docs/AZURE_GUIDE.md)
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)

## Requirements

- Python 3.9+
- Core Dependencies:
  - FileUtils (included in environment.yaml)
  - pandas>=2.0.0
  - pyyaml>=6.0
  - langchain and related packages

## Project Structure

The analyzer uses FileUtils to maintain the following project structure:

```
project_root/
├── data/
│   ├── raw/           # Original data files
│   ├── interim/       # Intermediate processing
│   ├── processed/     # Analysis results
│   ├── external/      # External data sources
│   ├── parameters/    # Analysis parameters
│   └── configurations/ # Config files
├── reports/
│   ├── figures/       # Visualizations
│   ├── tables/        # Analysis tables
│   └── outputs/       # Analysis outputs
├── models/            # Trained models
└── src/              # Source code
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-text-analyzer.git
cd semantic-text-analyzer
```

2. Create and activate conda environment:
```bash
# Create conda environment
conda env create -f environment.yaml

# Activate environment
conda activate semantic-analyzer
```

3. Install in development mode:
```bash
pip install -e .
```

### Optional Dependencies

For Finnish language support:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libvoikko1 voikko-fi

# Windows
# Download and install Voikko from https://voikko.puimula.org/windows.html
```

For Azure support:
```bash
pip install -e ".[azure]"
```

For development:
```bash
pip install -e ".[dev]"
```

See the [Installation Guide](docs/INSTALLATION_GUIDE.md) for detailed instructions.

## Configuration

The analyzer can be configured through:
- YAML configuration files
- Environment variables
- Runtime parameters

Example configuration:
```yaml
models:
  default_provider: "azure"
  default_model: "gpt-4o-mini"
  parameters:
    temperature: 0.0
    max_tokens: 1000
```

See [Configuration Guide](docs/CONFIGURATION_GUIDE.md) for detailed options.

## Usage Examples

### Basic Analysis
```python
from semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer()
result = await analyzer.analyze("Your text here")
```

### Batch Processing
```python
results = await analyzer.analyze_batch(
    texts=["Text 1", "Text 2", "Text 3"],
    batch_size=3
)
```

### Custom Configuration
```python
analyzer = SemanticAnalyzer(
    parameter_file="custom_parameters.xlsx",
    config={
        "max_keywords": 5,
        "min_confidence": 0.3
    }
)
```

## Development

To contribute:

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
```bash
pip install -e ".[dev]"
```
4. Run tests:
```bash
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.