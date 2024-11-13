# Category Analysis Implementation Plan

## 1. Core Components

### 1.1 Models & Schemas
```python
class CategoryInfo(BaseModel):
    """Information about a category match."""
    name: str
    confidence: float
    description: str
    evidence: List[str] = Field(default_factory=list)
    related_themes: List[str] = Field(default_factory=list)

class CategoryOutput(AnalyzerOutput):
    """Output model for category analysis."""
    categories: List[CategoryInfo]
    primary_category: Optional[str]
    category_confidence: Dict[str, float]
    evidence_mapping: Dict[str, List[str]]
```

### 1.2 Key Features
1. Category matching with confidence scores
2. Evidence collection for each category
3. Integration with themes and keywords
4. Hierarchical category support
5. Cross-validation between components

### 1.3 Integration Points
```python
# In SemanticAnalyzer
async def analyze(self, text: str) -> Dict[str, Any]:
    """Complete analysis pipeline."""
    # Get keyword analysis
    keyword_results = await self.keyword_analyzer.analyze(text)
    
    # Get theme analysis
    theme_results = await self.theme_analyzer.analyze(text)
    
    # Use keywords and themes for category analysis
    category_results = await self.category_analyzer.analyze(
        text,
        keywords=keyword_results.keywords,
        themes=theme_results.themes
    )
    
    return {
        "keywords": keyword_results,
        "themes": theme_results,
        "categories": category_results
    }
```

## 2. Implementation Steps

1. Create base category analyzer
2. Implement category matching logic
3. Add evidence collection
4. Integrate with themes
5. Add category validation
6. Implement confidence scoring
7. Add category hierarchy

Would you like me to start implementing any specific part? We can begin with the base CategoryAnalyzer class or look at how it integrates with the existing pipeline.

## Some key points to consider:

1. Like ThemeAnalyzer, CategoryAnalyzer will use:
   - LLM for category understanding
   - Confidence scoring
   - Integration with existing components

2. New features:
   - Evidence collection for verification
   - Category hierarchy
   - Cross-component validation

3. Integration considerations:
   - Using themes and keywords for better categorization
   - Shared confidence scoring
   - Common domain knowledge

Where would you like to start? I can:
1. Begin implementing the CategoryAnalyzer class
2. Show the integration points with existing components
3. Start with the models and schemas
4. Focus on a specific feature