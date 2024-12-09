1. General Parameters (Sheet 1):
   | parameter | value | description (new) |
   |-----------|-------|------------------|
   | max_kws | 8 | Maximum keywords to extract |
   | max_themes | 3 | Maximum themes to identify |
   | focus_on | education... | Focus area for analysis |
   | language | en | Content language |
   | min_keyword_length | 3 | Minimum length for keywords |
   | min_confidence | 0.3 | Minimum confidence threshold |
   | include_compounds | true | Handle compound words |
   | column_name_to_analyze | text | Content column name |

2. Categories (Sheet 2):
   | category | description | keywords | threshold | parent |
   |----------|-------------|----------|-----------|---------|
   | education_type | Type of... | online,remote | 0.5 | education |
   | career_planning | Career... | career,job | 0.6 | careers |

3. Predefined Keywords (Sheet 3):
   | keyword | importance | domain | compound_parts |
   |---------|------------|--------|----------------|
   | programming | 1.0 | tech | |
   | verkkokurssi | 0.8 | education | verkko,kurssi |

4. Excluded Keywords (Sheet 4):
   | keyword | reason (new) |
   |---------|-------------|
   | education | too general |
   | study | ambiguous |

5. Analysis Settings (New Sheet 5):
   | setting | value | description |
   |---------|-------|-------------|
   | theme_analysis.enabled | true | Enable theme analysis |
   | theme_analysis.min_confidence | 0.5 | Theme confidence threshold |
   | statistical.weight | 0.4 | Weight for statistical analysis |
   | llm.weight | 0.6 | Weight for LLM analysis |
