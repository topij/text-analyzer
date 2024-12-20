# Input Files Documentation

This document describes the two main input files required for the semantic text analysis tool: the content file and the parameters file.

## Content File (`test_content_{lang}.xlsx`)

The content file contains the texts to be analyzed. Separate files are maintained for different languages (e.g., `test_content_en.xlsx`, `test_content_fi.xlsx`).

### Required Structure
- File format: Excel (.xlsx)
- Required columns:
  - `language`: Language code ("en" or "fi")
  - `content`: The actual text to be analyzed

## Parameters File (`parameters_{lang}.xlsx`)

The parameters file configures the analysis settings. Each language has its own parameter file (e.g., `parameters_en.xlsx`, `parameters_fi.xlsx`).

### Required Sheet

#### General Parameters
English sheet name: "General Parameters"  
Finnish sheet name: "yleiset säännöt"

Required parameters:
```
parameter            | value | description
--------------------|-------|-------------
column_name_analyze | text  | Name of content column (REQUIRED)
focus_on            | ...   | Analysis focus area (REQUIRED)
max_keywords        | 10    | Maximum keywords to extract (REQUIRED)
```

Additional optional parameters:
```
parameter            | value | description
--------------------|-------|-------------
language            | en/fi | Analysis language
min_keyword_length  | 3     | Minimum length for keywords
include_compounds   | True  | Include compound word analysis
min_confidence     | 0.3   | Minimum confidence threshold
max_themes         | 3     | Maximum themes to identify
```

### Optional Sheets

The parameter file can include any of the following optional sheets:

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
category   | description        | keywords              | threshold
-----------|-------------------|----------------------|----------
technical  | Technical content | api,data,system     | 0.6
```

#### 4. Domain Context
English: "Domain Context"  
Finnish: "aihepiirin lisätietot"
```
name      | description | key_terms | context  | stopwords
----------|-------------|-----------|----------|----------
technical | ...         | ...       | ...      | ...
```

#### 5. Analysis Settings
English: "Analysis Settings"  
Finnish: "lisäasetukset"
```
setting                  | value | description
------------------------|-------|-------------
theme_analysis.min_conf | 0.5   | Minimum confidence for themes
weights.statistical     | 0.4   | Weight for statistical analysis
weights.llm            | 0.6   | Weight for LLM analysis
```

### Validation Rules
1. General Parameters sheet is required with three mandatory parameters:
   - column_name_analyze
   - focus_on
   - max_keywords
2. Keywords must contain only allowed characters (a-z, A-Z, ä, ö, å, Ä, Ö, Å, 0-9, spaces, hyphens, underscores)
3. Numeric values (thresholds, weights) must be within specified ranges
4. If weight parameters are provided, statistical and LLM weights must sum to 1.0

### Default Values
When not specified, the following default values are used:
- language: "en"
- min_keyword_length: 3
- include_compounds: True
- min_confidence: 0.3

### Important Notes
1. Only the General Parameters sheet with its three required parameters is mandatory
2. All other sheets are optional but must follow the specified structure if included
3. Column names must match exactly as specified
4. Language-specific files must use appropriate character sets (e.g., Finnish letters ä, ö, å)
5. Keyword lists use comma-separation without spaces