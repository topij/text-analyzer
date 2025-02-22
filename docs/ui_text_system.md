# UI Text System Developer Guide

## Overview
The Text Analyzer UI implements a flexible, multilingual text management system that handles all UI texts dynamically. The system is built around the `UITextManager` class, which manages text content and translations through a YAML configuration file.

## Architecture

### Core Components

1. **UITextManager Class**
   - Located in `ui/text_manager.py`
   - Manages loading, storing, and retrieving UI texts
   - Handles language switching and text formatting
   - Provides export/import functionality for text management

2. **Configuration Storage**
   - Uses YAML format for text configuration
   - Stored in `data/config/ui_texts.yaml`
   - Hierarchical structure: categories → keys → languages

3. **FileUtils Integration**
   - Uses `FileUtils` class for file operations
   - Handles path resolution and file management
   - Ensures consistent file handling across the application

## Text Configuration Structure

```yaml
default_language: "en"
categories:
  titles: "Main UI titles and headers"
  messages: "Status and information messages"
  buttons: "Button labels"
  # ... other categories

texts:
  titles:
    main:
      en: "Text Analyzer"
      fi: "Tekstianalysaattori"
    # ... other titles
  messages:
    analyzing:
      en: "Analyzing text..."
      fi: "Analysoidaan tekstiä..."
    # ... other messages
```

## Usage in Code

### 1. Initializing Text Manager
```python
from text_manager import get_ui_text_manager
from FileUtils import FileUtils

file_utils = FileUtils()
text_manager = get_ui_text_manager(file_utils=file_utils)
```

### 2. Getting Translated Text
```python
# Basic text retrieval
text = text_manager.get_text("category", "key", language="en")

# With formatting parameters
text = text_manager.get_text(
    "messages", 
    "processing_text",
    language="en",
    current=1,
    total=10
)
```

### 3. Streamlit Integration
```python
# Initialize in session state
if 'text_manager' not in st.session_state:
    st.session_state.text_manager = get_ui_text_manager(file_utils=st.session_state.file_utils)

# Use in UI components
st.title(get_text("titles", "main", language=st.session_state.ui_language))
```

## Text Management Tools

### Export to Excel
```python
# Export texts for editing
text_manager.export_to_excel()
```

### Import from Excel
```python
# Import updated texts
text_manager.import_from_excel("path/to/updated_texts.xlsx")
```

## Adding New Texts

1. Add new category (if needed) in `categories` section
2. Add new text key under appropriate category
3. Provide translations for all supported languages
4. Update both YAML and Excel versions

Example:
```yaml
texts:
  new_category:
    new_key:
      en: "English text"
      fi: "Finnish text"
```

## Best Practices

1. **Text Organization**
   - Group related texts under appropriate categories
   - Use descriptive key names
   - Keep translations consistent across languages

2. **Code Integration**
   - Always use text manager for UI texts
   - Never hardcode text strings
   - Use formatting parameters for dynamic content

3. **Maintenance**
   - Keep Excel export up to date
   - Verify all texts have translations
   - Document new categories and keys

4. **Error Handling**
   - Provide fallback texts for missing translations
   - Log missing text keys for maintenance
   - Handle formatting errors gracefully

## Command Line Tools

### Text Converter Tool
Located in `ui/text_converter.py`:
```bash
# Export texts to Excel
python ui/text_converter.py export

# Import texts from Excel
python ui/text_converter.py import texts.xlsx
```

### Setup Tool
Located in `ui/setup_ui.py`:
```bash
# Initialize or update UI text configuration
python ui/setup_ui.py --force
```

## Verification Tools

### Check UI Texts
Located in `scripts/check_ui_texts.py`:
```bash
# Verify text configuration
python scripts/check_ui_texts.py
```

This tool checks for:
- Missing translations
- Incomplete categories
- Invalid text structure
- Consistency between YAML and Excel

## Adding New Languages

1. Add new language code to supported languages
2. Update YAML structure to include new language
3. Add translations for all text keys
4. Update export/import functionality
5. Add language to UI selector
6. Test all UI components with new language

## Troubleshooting

1. **Missing Texts**
   - Check YAML configuration
   - Verify text key exists
   - Ensure language code is correct
   - Check for typos in category/key names

2. **Format Errors**
   - Verify parameter names in format strings
   - Check parameter values passed to get_text
   - Ensure YAML syntax is valid

3. **File Issues**
   - Verify file permissions
   - Check file paths
   - Ensure FileUtils is properly initialized 