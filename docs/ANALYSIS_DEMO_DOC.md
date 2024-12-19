# Semantic Text Analysis Demo Documentation

## Overview
This document provides a comprehensive guide to using the cross-environment semantic text analysis demo. The demo showcases text analysis capabilities in both local and Azure environments, with support for both single text analysis and batch Excel file processing.

[See the demo notebook](../notebooks/cross_env_analyzer_demo_nb.ipynb)

## Environment Setup

### Requirements
- Python 3.9+
- Required packages as specified in `environment.yaml`
- Proper environment variables for API access

### Initialization
```python
from src.nb_helpers.environment import setup_analysis_environment
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.formatting_config import OutputDetail, ExcelOutputConfig
```

The environment setup is handled through `setup_analysis_environment()`, which:
- Initializes FileUtils with appropriate storage settings
- Sets up logging configuration
- Returns initialized components dictionary

```python
components = setup_analysis_environment(
    env_type="local",  # or "azure"
    log_level="WARNING"
)
```

## Analyzer Configuration

The SemanticAnalyzer can be initialized with custom configuration:
```python
analyzer = SemanticAnalyzer(
    file_utils=file_utils,
    parameter_file="parameters_en.xlsx",
    format_config=ExcelOutputConfig(
        detail_level=OutputDetail.MINIMAL,
        include_confidence=True
    )
)
```

Key configuration options:
- `parameter_file`: Excel file containing analysis parameters
- `detail_level`: Controls output detail (MINIMAL, SUMMARY, DETAILED)
- `include_confidence`: Whether to include confidence scores in output

## Analysis Functions

### Single Text Analysis
The demo supports analyzing individual texts with specified language:
```python
async def analyze_text(text: str, language: str = "en") -> None:
```

Features:
- Supports multiple languages (currently implemented: en, fi)
- Returns keywords, themes, and categories
- Includes confidence scores for each result
- Handles errors gracefully with informative messages

### Excel File Processing
Batch processing of Excel files:
```python
async def process_excel(
    input_file: str,
    output_file: str = "analysis_results",
    content_column: str = "content",
    language_column: Optional[str] = None,
    analysis_types=["keywords", "themes", "category"],
    show_progress: bool = True,
    batch_size: int = 5
) -> None
```

Features:
- Processes Excel files with configurable content column
- Optional language column support
- Progress bar for batch processing
- Configurable batch size for performance optimization
- Saves results to Excel with formatted output

## Output Format

### Keyword Analysis
- List of extracted keywords with scores
- Domain classification (if available)
- Compound word detection

### Theme Analysis
- Main themes with confidence scores
- Theme descriptions
- Hierarchical relationships (if present)

### Category Analysis
- Category matches with confidence scores
- Evidence supporting categorization
- Relevance scoring

## Usage Examples

### Single Text Analysis
```python
texts = {
    "en": "Machine learning models analyze data efficiently.",
    "fi": "Koneoppimismallit analysoivat dataa tehokkaasti."
}

for lang, text in texts.items():
    await analyze_text(text, lang)
```

### Excel Processing
```python
await process_excel(
    input_file="test_content_short_en.xlsx",
    output_file="analysis_results",
    content_column="content",
    language_column="language",
    batch_size=1
)
```

## Environment Support

### Local Environment
- Uses local file system for storage
- Default configuration suitable for development

### Azure Environment
- Integrates with Azure storage
- Configured through environment variables
- Suitable for production deployment

## Error Handling
The demo includes comprehensive error handling:
- Input validation
- API error handling
- File operation error handling
- Informative error messages
