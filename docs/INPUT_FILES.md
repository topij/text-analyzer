# Input Files Documentation

This document describes the parameter file format used for configuring the semantic text analyzer.

## Parameter File (`parameters.xlsx`)

The parameter file configures the analysis settings. It supports both English and Finnish column names.

### Required Sheet: General Parameters

English sheet name: "General Parameters"  
Finnish sheet name: "yleiset säännöt"

Required columns:
```
English:
- parameter: Parameter name
- value: Parameter value
- description: Parameter description

Finnish:
- parametri: Parameter name
- arvo: Parameter value
- kuvaus: Parameter description
```

Required parameters:
```
parameter            | value | description
--------------------|-------|-------------
max_keywords        | 10    | Maximum keywords to extract
min_keyword_length  | 3     | Minimum keyword length
language           | en/fi | Analysis language
focus_on           | ...   | Analysis focus area
column_name_to_analyze | text | Name of content column
```

Optional parameters:
```
parameter            | value | description
--------------------|-------|-------------
include_compounds   | True  | Include compound word analysis
min_confidence     | 0.3   | Minimum confidence threshold
max_themes         | 3     | Maximum themes to identify
```

### Optional Sheets

#### 1. Predefined Keywords
English: "Predefined Keywords"  
Finnish: "haettavat avainsanat"
```
keyword          | importance | domain
-----------------|------------|--------
machine learning | 1.0       | technical
data analysis    | 0.9       | technical
```

#### 2. Excluded Keywords
English: "Excluded Keywords"  
Finnish: "älä käytä"
```
keyword | reason
--------|--------
the     | Common word
and     | Common word
```

#### 3. Categories
English: "Categories"  
Finnish: "kategoriat"
```
category   | description        | keywords
-----------|-------------------|----------------------
technical  | Technical content | api,data,system
```

#### 4. Domain Context
English: "Domain Context"  
Finnish: "aihepiirin lisätietot"
```
domain    | description | keywords
----------|-------------|----------
technical | ...         | ...
```

#### 5. Analysis Settings
English: "Analysis Settings"  
Finnish: "lisäasetukset"
```
setting                  | value | description
------------------------|-------|-------------
theme_analysis.enabled  | true  | Enable theme analysis
theme_analysis.min_confidence | 0.5 | Minimum confidence for themes
weights.statistical     | 0.4   | Weight for statistical analysis
weights.llm            | 0.6   | Weight for LLM analysis
```

### Default Values

When not specified in the parameter file, these default values are used:
- max_keywords: 10
- min_keyword_length: 3
- language: "en"
- include_compounds: true
- min_confidence: 0.3
- max_themes: 3
- column_name_to_analyze: "text"
- theme_analysis.enabled: true
- theme_analysis.min_confidence: 0.5
- weights.statistical: 0.4
- weights.llm: 0.6

### Validation Rules

1. Required parameters must be present in General Parameters sheet
2. Keywords must contain only allowed characters:
   - English: a-z, A-Z, 0-9, spaces, hyphens, underscores
   - Finnish: a-z, A-Z, ä, ö, å, Ä, Ö, Å, 0-9, spaces, hyphens, underscores
3. Numeric values must be within valid ranges:
   - Confidence thresholds: 0.0-1.0
   - Weights: 0.0-1.0 (must sum to 1.0)
   - max_keywords: positive integer
   - min_keyword_length: positive integer
4. Language must be either "en" or "fi"

### Important Notes

1. Only the General Parameters sheet is mandatory
2. All other sheets are optional but must follow the specified structure if included
3. Column names must match exactly (case-sensitive)
4. Use appropriate character encoding for language-specific characters
5. Keywords in lists should be comma-separated without spaces
6. Empty cells are allowed and will be ignored
7. Additional columns in sheets will be ignored
8. Sheet order does not matter