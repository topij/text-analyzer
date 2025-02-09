# Usage Examples

Comprehensive examples for using the Semantic Text Analyzer.

## Basic Analysis

### Simple Text Analysis
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def analyze_text():
    # Set up environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer(
        file_utils=components["file_utils"],
        config_manager=components["config_manager"]
    )
    
    # Analyze with specific types
    result = await analyzer.analyze(
        text="""Machine learning models are trained using large datasets 
                to recognize patterns. The neural network architecture 
                includes multiple layers for feature extraction.""",
        analysis_types=["keywords", "themes", "categories"]
    )
    
    # Handle results
    if result.success:
        # Keywords
        if result.keywords:
            print("\nKeywords:")
            for kw in result.keywords.keywords:
                print(f"• {kw.keyword} (score: {kw.score:.2f})")
                if kw.domain:
                    print(f"  Domain: {kw.domain}")
        
        # Themes
        if result.themes:
            print("\nThemes:")
            for theme in result.themes.themes:
                print(f"• {theme.name} ({theme.confidence:.2f})")
                print(f"  Description: {theme.description}")
                if theme.parent_theme:
                    print(f"  Parent: {theme.parent_theme}")
        
        # Categories
        if result.categories:
            print("\nCategories:")
            for cat in result.categories.categories:
                print(f"• {cat.name} ({cat.confidence:.2f})")
                if cat.evidence:
                    print("  Evidence:")
                    for ev in cat.evidence:
                        print(f"    - {ev}")
    else:
        print(f"Analysis failed: {result.error}")
```

### Finnish Text Analysis
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def analyze_finnish_text():
    # Set up environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Initialize analyzer with Finnish parameters
    analyzer = SemanticAnalyzer(
        parameter_file="parameters_fi.xlsx",
        file_utils=components["file_utils"]
    )
    
    text = """Koneoppimismalleja koulutetaan suurilla datajoukolla 
              tunnistamaan kaavoja. Neuroverkon arkkitehtuuri 
              sisältää useita kerroksia."""
    
    result = await analyzer.analyze(
        text=text,
        analysis_types=["keywords", "themes"]
    )
    
    if result.success and result.keywords:
        # Show compound words
        print("\nCompound Words:")
        for kw in result.keywords.keywords:
            if kw.compound_parts:
                print(f"• {kw.keyword}")
                print(f"  Parts: {' + '.join(kw.compound_parts)}")
                print(f"  Score: {kw.score:.2f}")
```

### Batch Processing
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def process_batch():
    # Set up environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer(
        parameter_file="parameters.xlsx",
        file_utils=components["file_utils"]
    )
    
    # Prepare texts
    texts = [
        "First document to analyze",
        "Second document with different content",
        "Third document about something else"
    ]
    
    try:
        # Process batch
        results = await analyzer.analyze_batch(
            texts=texts,
            batch_size=3,
            analysis_types=["keywords", "themes"]
        )
        
        # Save results
        analyzer.save_results(
            results=results,
            output_file="results.xlsx"
        )
        
        print(f"Processed {len(results)} documents")
        
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
```

### Custom Categories
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Set up environment
env_manager = EnvironmentManager(EnvironmentConfig())
components = env_manager.get_components()

# Initialize analyzer with parameter file containing categories
analyzer = SemanticAnalyzer(
    parameter_file="parameters.xlsx",  # Define categories in Excel
    file_utils=components["file_utils"]
)

# Analyze with categories
result = await analyzer.analyze(
    text="Your text here",
    analysis_types=["categories"]
)

# Process category results
if result.success and result.categories:
    for cat in result.categories.categories:
        print(f"\nCategory: {cat.name}")
        print(f"Confidence: {cat.confidence:.2f}")
        if cat.description:
            print(f"Description: {cat.description}")
        if cat.evidence:
            print("Evidence:")
            for ev in cat.evidence:
                print(f"- {ev}")
```

### Parameter File Example
```yaml
# parameters.xlsx structure

# Sheet: General Parameters
parameter            | value | description
--------------------|-------|-------------
max_keywords        | 10    | Maximum keywords
min_keyword_length  | 3     | Minimum length
language           | en    | Language code
focus_on           | tech  | Analysis focus

# Sheet: Analysis Settings
setting                  | value
------------------------|-------
theme_analysis.enabled  | true
theme_analysis.min_confidence | 0.5
weights.statistical     | 0.4
weights.llm            | 0.6

# Sheet: Categories
category   | description        | keywords
-----------|-------------------|----------
technical  | Technical content | api,data,system
business   | Business content  | market,growth
```

### Excel Processing
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def process_excel():
    # Set up environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer(
        parameter_file="parameters.xlsx",
        file_utils=components["file_utils"]
    )
    
    # Prepare Excel file
    excel_file = "input.xlsx"
    
    try:
        # Process Excel
        results = await analyzer.analyze_excel(
            excel_file=excel_file,
            analysis_types=["keywords", "themes", "categories"],
            batch_size=10,
            save_results=True,
            output_file="results.xlsx"
        )
        
        print(f"Processed {len(results)} documents")
        
    except Exception as e:
        print(f"Excel processing failed: {str(e)}")
```

### Custom Configuration
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Set up environment
env_manager = EnvironmentManager(EnvironmentConfig())
components = env_manager.get_components()

# Initialize analyzer with custom configuration
config = {
    "analysis": {
        "keywords": {
            "max_keywords": 5,
            "min_keyword_length": 3,
            "include_compounds": True,
            "weights": {
                "statistical": 0.4,
                "llm": 0.6
            }
        },
        "themes": {
            "max_themes": 3,
            "min_confidence": 0.5
        },
        "categories": {
            "min_confidence": 0.3
        }
    }
}

analyzer = SemanticAnalyzer(
    config=config,
    file_utils=components["file_utils"]
)
```

### Error Handling
```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def robust_analysis(text: str):
    try:
        # Set up environment
        env_manager = EnvironmentManager(EnvironmentConfig())
        components = env_manager.get_components()
        
        # Initialize analyzer
        analyzer = SemanticAnalyzer(
            file_utils=components["file_utils"],
            config_manager=components["config_manager"]
        )
        
        # Analyze text
        result = await analyzer.analyze(text)
        
        if not result.success:
            print(f"Analysis failed: {result.error}")
            return
            
        if result.keywords.error:
            print(f"Keyword analysis error: {result.keywords.error}")
            
        if result.themes.error:
            print(f"Theme analysis error: {result.themes.error}")
            
        if result.categories.error:
            print(f"Category analysis error: {result.categories.error}")
            
        return result
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
```

### Using the Lite Analyzer
```python
from src.analyzers.lite_analyzer import LiteSemanticAnalyzer
from src.core.llm.factory import create_llm
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def analyze_with_lite():
    # Set up environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Create LLM instance
    llm = create_llm()
    
    # Initialize lite analyzer
    analyzer = LiteSemanticAnalyzer(
        llm=llm,
        language="en",
        file_utils=components["file_utils"],
        tfidf_weight=0.5,  # Balance between TF-IDF and LLM results
        custom_stop_words={"custom", "stop", "words"}  # Optional
    )
    
    # Analyze text
    result = await analyzer.analyze(
        text="""Machine learning models are trained using large datasets 
                to recognize patterns. The neural network architecture 
                includes multiple layers for feature extraction.""",
        analysis_types=["keywords", "themes", "categories"]  # Optional - defaults to all
    )
    
    # Process results
    if result.success:
        # Keywords with TF-IDF enhanced scoring
        print("\nKeywords:")
        for kw in result.keywords.keywords:
            print(f"• {kw.keyword} (score: {kw.score:.2f})")
            if kw.is_compound:
                print("  (Compound word)")
            if kw.frequency:
                print(f"  Frequency: {kw.frequency}")
        
        # Themes with hierarchy
        print("\nThemes:")
        for theme in result.themes.themes:
            print(f"• {theme.name} ({theme.confidence:.2f})")
        
        # Print theme hierarchy
        if result.themes.theme_hierarchy:
            print("\nTheme Hierarchy:")
            for main_theme, sub_themes in result.themes.theme_hierarchy.items():
                print(f"• {main_theme}")
                for sub in sub_themes:
                    print(f"  - {sub}")
        
        # Categories with evidence
        print("\nCategories:")
        for cat in result.categories.matches:
            print(f"• {cat.name} ({cat.confidence:.2f})")
            if cat.evidence:
                print("  Evidence:")
                for ev in cat.evidence:
                    print(f"    - {ev.text} (relevance: {ev.relevance:.2f})")
        
        # Performance metrics
        print(f"\nProcessing time: {result.processing_time:.2f}s")
    else:
        print(f"Analysis failed: {result.error}")
```

For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).