# Semantic Text Analyzer

Advanced text analysis toolkit providing semantic understanding of text through keywords, themes, and categories.

## Features

- **Keyword Analysis**: Extract key terms using both statistical and LLM-based methods
- **Theme Detection**: Identify main themes and their relationships
- **Category Classification**: Classify text into predefined categories
- **Language Support**: English and Finnish language processing
- **Async Support**: Modern async/await interface
- **Flexible Configuration**: Extensive configuration options
- **LLM Integration**: Support for multiple LLM providers

## Installation

```bash
# Basic installation
pip install semantic-text-analyzer

# With Finnish language support
pip install semantic-text-analyzer[finnish]
```

## Quick Start

```python
from semantic_analyzer import SemanticAnalyzer

# Initialize analyzer
analyzer = SemanticAnalyzer()

# Analyze text
results = await analyzer.analyze(
    text="Your text here",
    analysis_types=["keywords", "themes", "categories"]
)

# Access results
print(f"Keywords: {results.keywords}")
print(f"Themes: {results.themes}")
print(f"Categories: {results.categories}")
```

## Migrating from keyword-extractor

This package is the successor to `keyword-extractor`, providing enhanced functionality and better modularity. See [Migration Guide](docs/migration.md) for details on upgrading.

## Documentation

Full documentation is available at [docs/](docs/):
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api.md)
- [Migration Guide](docs/migration.md)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.