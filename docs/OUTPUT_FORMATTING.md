# Semantic Text Analyzer Formatting System

## Overview
The formatting system provides a flexible way to format analysis results for both console output and Excel files. It supports different levels of detail and customizable output configurations.

## Core Components

### Output Detail Levels
Three levels of detail are supported:
- **MINIMAL**: Basic output showing just core results
  - Keywords: `keyword (score)`
  - Themes: `theme_name (confidence)`
  - Categories: `category (confidence)`

- **SUMMARY**: Results with additional metadata
  - Keywords: `keyword (score) [domain]`
  - Themes: `theme_name (confidence): description`
  - Categories: `category (confidence): description`

- **DETAILED**: Full output with all metadata
  - Keywords: Includes compound words and domain groupings
  - Themes: Includes full descriptions and hierarchies
  - Categories: Includes descriptions and evidence

### Configuration Classes

#### BaseOutputConfig
Base configuration for all formatters:
```python
config = BaseOutputConfig(
    detail_level=OutputDetail.SUMMARY,  # Detail level
    batch_size=10,                      # Processing batch size
    include_confidence=True,            # Show confidence scores
    max_length=None                     # Max text length (if truncating)
)
```

#### ExcelOutputConfig
Extended configuration for Excel output:
```python
excel_config = ExcelOutputConfig(
    # Inherits BaseOutputConfig settings
    keywords_format=BaseColumnFormat(
        column_name="keywords",
        format_template="{keyword} ({confidence})",
        confidence_threshold=0.3,
        max_items=5
    ),
    themes_format=BaseColumnFormat(...),
    categories_format=BaseColumnFormat(...)
)
```

### Formatter Classes

#### BaseFormatter
Abstract base class defining the core formatting interface:
```python
class BaseFormatter:
    def format_output(
        self, 
        results: Dict[str, Any], 
        analysis_types: List[str]
    ) -> Any:
        """Format analysis results."""
        pass
```

#### ExcelAnalysisFormatter
Handles Excel-specific formatting and file output:
```python
formatter = ExcelAnalysisFormatter(
    file_utils=file_utils,
    config=excel_config
)

# Format results
formatted = formatter.format_output(
    results,
    ["keywords", "themes", "categories"]
)

# Save to Excel
formatter.save_excel_results(
    results_df,
    "output.xlsx",
    include_summary=True
)
```

## Usage Examples

### Basic Usage
```python
from src.utils.formatting_config import OutputDetail, ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter

# Create formatter
config = ExcelOutputConfig(
    detail_level=OutputDetail.MINIMAL,
    include_confidence=True
)
formatter = ExcelAnalysisFormatter(config=config)

# Format results
formatted = formatter.format_output(
    analysis_results,
    ["keywords", "themes", "categories"]
)
```

### Excel Output with Summary
```python
# Create DataFrame from results
results_df = pd.DataFrame(formatted_results)

# Save with summary sheet
formatter.save_excel_results(
    results_df,
    "analysis_results.xlsx",
    include_summary=True
)
```

### Different Detail Levels

#### Minimal Output
```python
config = ExcelOutputConfig(detail_level=OutputDetail.MINIMAL)
# Results: "keyword (0.95), another_keyword (0.85)"
```

#### Summary Output
```python
config = ExcelOutputConfig(detail_level=OutputDetail.SUMMARY)
# Results: "keyword (0.95) [technical]: description"
```

#### Detailed Output
```python
config = ExcelOutputConfig(detail_level=OutputDetail.DETAILED)
# Results include all metadata, evidence, and relationships
```

## Excel Output Format

### Sheet Structure
When saving to Excel, the following sheets are created:

1. **Analysis Results**
   - Content column
   - Keywords column
   - Themes column
   - Categories column

2. **Summary** (if include_summary=True)
   - Total Records
   - Records with Keywords
   - Records with Themes
   - Records with Categories

### Column Formats
Each column type follows its configured format:

- **Keywords**: `keyword (score) [domain]`
- **Themes**: `theme_name (confidence): description`
- **Categories**: `category (confidence): description`

## Configuration Options

### Detail Level Options
```python
from src.utils.formatting_config import OutputDetail

# Available options
OutputDetail.MINIMAL   # Basic results only
OutputDetail.SUMMARY   # Results with key metadata
OutputDetail.DETAILED  # Full results with all metadata
```

### Column Format Options
```python
BaseColumnFormat(
    column_name="column_name",    # Name in output
    format_template="template",   # How to format items
    included_fields=["fields"],   # Fields to include
    confidence_threshold=0.3,     # Min confidence
    max_items=5                   # Max items to show
)
```

## Limitations
- Currently only supports Excel file output
- Summary sheet is only available when saving to Excel
- Confidence scores are fixed to 2 decimal places
- Maximum items per column is enforced only in Excel output

## Best Practices
1. Always specify the detail level appropriate for your use case
2. Use include_summary=True when saving to Excel for better result overview
3. Set appropriate confidence thresholds to filter low-confidence results
4. Consider max_items setting to avoid overly long cells in Excel

## Error Handling
The formatting system handles common errors:
- Missing results return empty strings
- Invalid confidence scores are formatted as (0.00)
- File saving errors are logged and raised