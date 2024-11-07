# Semantic Text Analyzer Test Suite
A comprehensive test suite for validating the semantic text analyzer functionality.

## Overview
This test suite covers parameter handling, text analysis capabilities, and error handling across different languages and configurations. The tests use pytest and support async operations.

## Test Categories

### 1. Parameter Loading Tests
Located in `tests/test_parameter_loading.py`

#### `test_parameter_loading_english`
- Validates loading of English parameter files
- Tests default values and overrides
- Checks parameter type validation
- **Key assertions:**
  - max_keywords = 8
  - language = "en"
  - Proper category loading

#### `test_parameter_loading_finnish`
- Tests Finnish parameter file loading
- Verifies language-specific settings
- **Key assertions:**
  - language = "fi"
  - column_name_to_analyze = "keskustelu"

#### `test_parameter_validation`
- Validates parameter constraints
- Tests boundary conditions
- **Key assertions:**
  - max_keywords range (1-20)
  - min_keyword_length range (2-10)
  - Valid language codes

#### `test_backward_compatibility`
- Tests compatibility with older parameter formats
- Verifies parameter conversion
- **Key assertions:**
  - Old format conversion
  - Default value handling

#### `test_parameter_integration`
- Tests parameter integration with analyzer
- Verifies configuration propagation
- **Key assertions:**
  - Parameter application in analyzers
  - Configuration inheritance

### 2. Semantic Analysis Tests
Located in `tests/test_semantic_analyzer.py`

#### `test_analysis_with_parameters`
- Tests complete analysis pipeline
- Verifies keyword extraction
- Validates category assignment
- **Key assertions:**
  - Keyword presence ("programming")
  - Keyword exclusions ("education")
  - Category confidence thresholds

#### `test_batch_analysis_with_parameters`
- Tests concurrent analysis of multiple texts
- Verifies consistent results
- **Key assertions:**
  - Result count matches input
  - Consistent parameter application
  - Proper error handling

#### `test_parameter_override`
- Tests runtime parameter overrides
- Validates parameter precedence
- **Key assertions:**
  - max_keywords override
  - Parameter persistence

#### `test_error_handling_with_parameters`
- Tests various error conditions
- Verifies graceful failure
- **Key assertions:**
  - Empty input handling
  - Invalid parameter handling
  - Proper error messages

## Test Data
- Sample texts in English and Finnish
- Test parameter files
- Mock LLM responses

## Test Fixtures
Located in `tests/conftest.py`

### Core Fixtures
- `file_utils`: FileUtils instance for test data management
- `analyzer_config`: Configuration for testing
- `mock_llm`: Mock LLM for testing
- `parameter_files`: Test parameter files
- `test_texts`: Sample texts for analysis
- `test_categories`: Test category definitions

### Setup Fixtures
- `setup_environment`: Test environment setup
- `setup_nltk`: NLTK data setup
- `event_loop`: Async event loop management

## Usage
Run the full test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_parameter_loading.py -v
pytest tests/test_semantic_analyzer.py -v
```

## Adding New Tests
1. Follow the existing test structure
2. Use appropriate fixtures
3. Include both positive and negative test cases
4. Document test purpose and assertions
5. Ensure async compatibility
6. Add proper error handling tests

## Best Practices
- Use descriptive test names
- Include adequate assertions
- Handle async operations properly
- Clean up test resources
- Document test requirements
- Use appropriate fixtures
- Include both positive and negative cases

## Notes
- Requires Python 3.9+
- Uses pytest-asyncio for async testing
- Includes NLTK data dependencies
- Handles Windows-specific asyncio behavior
