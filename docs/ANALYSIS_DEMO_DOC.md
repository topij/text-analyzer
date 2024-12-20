# Semantic Text Analysis Demo Documentation

## Overview
This document provides a guide to using the cross-environment semantic text analysis demo notebook. The demo showcases text analysis capabilities with support for both single text analysis and Excel file processing.

[See the demo notebook](../notebooks/cross_env_analyzer_demo_nb.ipynb)

## Environment Setup

### Requirements
- Python 3.9+
- Required packages as specified in `environment.yaml`
- API access credentials (OpenAI or Azure OpenAI)

### Initialization
```python
from src.nb_helpers.environment import setup_analysis_environment
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.formatting_config import OutputDetail, ExcelOutputConfig
```

The environment setup is handled through `setup_analysis_environment()`, which:
- Initializes FileUtils for file operations
- Sets up logging configuration
- Returns initialized components dictionary

```python
components = setup_analysis_environment(
    env_type="local",  # or "azure"
    log_level="ERROR",
    project_root=project_root
)
```
## Example files
You can create example files with `$ python scripts/test_data_generator.py`, For example: 
- `parameters_en.xlsx`, defines the parameters used in the analysis. This can be used as the basic template for other analyses
- `test_content_en.xlsx`, generic test content




## Analyzer Configuration

The SemanticAnalyzer is initialized with:
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

## LLM Configuration

The notebook supports viewing and changing LLM configurations:

```python
# View current LLM configuration
print(get_llm_info(analyzer))

# View available providers
print(get_available_providers(analyzer))

# Change provider (if needed)
change_llm_provider(analyzer, "azure", "gpt-4o-mini")
```

## Analysis Functions

### Single Text Analysis
Analyze individual texts with specified language:
```python
async def analyze_text(text: str, language: str = "en") -> None:
```

Features:
- Supports English and Finnish languages
- Returns keywords, themes, and categories
- Includes confidence scores
- Displays formatted results in the notebook

### Excel File Processing
Process Excel files containing multiple texts:
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
    input_file=content_file,
    output_file="analysis_results",
    content_column="content",
    language_column="language"
)
```

## Environment Support

### Local Environment
- Default environment for development defined in config.yaml and .env
- Uses local file system for storage using FileUtils

### Azure Environment
- Optional Azure integration available
- Can be enabled by in config.yaml or change in the notebook
- Requires appropriate Azure credentials and API keys
- Azure support both blob storage or local home directory cloud storage using FileUtils

## Output Information

The analysis results include:
- Keywords with confidence scores
- Identified themes with confidence scores
- Category matches with confidence scores
- Processing time information
