## Semantic Analyzer Test Suite Plan

1. Test Directory Structure
```
tests/
├── conftest.py               # Test fixtures and configuration
├── test_data/               # Test data files
│   ├── parameters_test.xlsx  # Test parameter files
│   ├── content_en.xlsx      # English test content
│   ├── content_fi.xlsx      # Finnish test content
│   └── expected_outputs/    # Expected analysis results
├── test_components/         # Individual component tests
│   ├── __init__.py
│   ├── test_keyword_analyzer.py
│   ├── test_theme_analyzer.py
│   └── test_category_analyzer.py
├── test_pipeline/          # Pipeline integration tests
│   ├── __init__.py
│   ├── test_english_pipeline.py
│   └── test_finnish_pipeline.py
└── test_utils/            # Test utilities and helpers
    ├── __init__.py
    ├── assertions.py      # Custom test assertions
    └── test_helpers.py    # Helper functions
```

2. Test Fixtures (conftest.py)
- Parameter fixtures (English/Finnish)
- Content fixtures (technical/business for both languages)
- Analyzer instances
- Language processor instances
- LLM mock responses

3. Test Categories
A. Component Tests (test_components/)
   - Keyword Analyzer:
     * Base form extraction
     * Compound word detection
     * Domain classification
     * Score calculation
     * Language-specific processing
     * Error handling

   - Theme Analyzer:
     * Theme identification
     * Theme hierarchy
     * Evidence extraction
     * Multilingual support
     * Error handling

   - Category Analyzer:
     * Category matching
     * Confidence scoring
     * Evidence gathering
     * Language handling
     * Error cases

B. Pipeline Tests (test_pipeline/)
   - Full pipeline integration
   - Language switching
   - Parameter handling
   - End-to-end processing
   - Error propagation

C. Performance Tests
   - Processing time benchmarks
   - Memory usage
   - Batch processing
   - Concurrent processing

4. Test Data Preparation
- Create standardized test cases
- Document expected outputs
- Cover edge cases
- Include error scenarios

5. Assertion Utilities (assertions.py)
- Custom assertions for analysis results
- Output format validation
- Score range checking
- Language-specific validations

6. Implementation Priority:
1. Basic component tests
2. Pipeline integration tests
3. Error handling tests
4. Performance tests
5. Edge cases