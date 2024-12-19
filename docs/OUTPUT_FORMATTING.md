# Output Formatting Guide

## Overview

The Semantic Text Analyzer provides flexible output formatting through the `ExcelAnalysisFormatter` and related configuration classes.

## Detail Levels

Three levels of output detail are supported:

```python
from src.utils.formatting_config import OutputDetail

# Available levels
OutputDetail.MINIMAL    # Basic results only
OutputDetail.SUMMARY    # Results with key metadata
OutputDetail.DETAILED   # Full results with all metadata
```

### MINIMAL Output
- Keywords: `keyword (score)`
- Themes: `theme_name (confidence)`
- Categories: `category (confidence)`

### SUMMARY Output
- Keywords: `keyword (score) [domain]`
- Themes: `theme_name (confidence): description`
- Categories: `category (confidence): description`

### DETAILED Output
- Complete metadata
- Evidence and relationships
- Domain groupings
- Compound word information

## Configuration Classes

### BaseOutputConfig
```python
from src.utils.formatting_config import BaseOutputConfig

config = BaseOutputConfig(
    detail_level=OutputDetail.SUMMARY,
    batch_size=10,
    include_confidence=True,
    max_length=None
)
```

### ExcelOutputConfig
```python
from src.utils.formatting_config import ExcelOutputConfig

config = ExcelOutputConfig(
    keywords_format=BaseColumnFormat(
        column_name="keywords",
        format_template="{keyword} ({confidence})",
        confidence_threshold=0.3,
        max_items=5
    ),
    themes_format=BaseColumnFormat(
        column_name="themes",
        format_template="{name}: {description}",
        confidence_threshold=0.5,
        max_items=3
    ),
    categories_format=BaseColumnFormat(
        column_name="categories",
        format_template="{name} ({confidence})",
        confidence_threshold=0.3,
        max_items=3
    )
)
```

## Usage Examples

### Basic Formatting
```python
from src.utils.formatting_config import ExcelOutputConfig, OutputDetail
from src.excel_analysis.formatters import ExcelAnalysisFormatter

# Create formatter
config = ExcelOutputConfig(
    detail_level=OutputDetail.SUMMARY,
    include_confidence=True
)
formatter = ExcelAnalysisFormatter(config=config)

# Format results
formatted = formatter.format_output(
    results,
    ["keywords", "themes", "categories"]
)
```

### Excel Output
```python
# Create DataFrame from results
results_df = pd.DataFrame([formatted_results])

# Save with summary
formatter.save_excel_results(
    results_df,
    "analysis_results.xlsx",
    include_summary=True
)
```

## Excel Output Format

### Sheet Structure
1. Analysis Results Sheet:
   - Content column
   - Keywords column
   - Themes column
   - Categories column

2. Summary Sheet (optional):
   - Total records
   - Records with keywords
   - Records with themes
   - Records with categories

## Best Practices

1. Detail Level Selection:
   - Use MINIMAL for basic overviews
   - Use SUMMARY for general analysis
   - Use DETAILED for in-depth analysis

2. Excel Output:
   - Enable summary sheets for better overview
   - Set appropriate confidence thresholds
   - Configure maximum items per column

3. Confidence Scores:
   - Include for quantitative analysis
   - Format to 2 decimal places
   - Set meaningful thresholds

For implementation details, see [API_REFERENCE.md](API_REFERENCE.md).