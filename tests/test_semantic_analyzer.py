# tests/test_semantic_analyzer.py

import pytest
import os
from typing import Dict, Any

from semantic_analyzer import SemanticAnalyzer, analyze_text
from . import SAMPLE_TEXTS

# Make sure we're using the test config
os.environ["SEMANTIC_ANALYZER_CONFIG"] = "tests/config"

@pytest.fixture
async def analyzer():
    """Create a test analyzer instance."""
    return SemanticAnalyzer(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        temperature=0.0
    )

@pytest.mark.asyncio
async def test_keyword_extraction():
    """Test keyword extraction functionality."""
    # Analyze technical text
    results = await analyze_text(
        SAMPLE_TEXTS["technical"],
        analysis_types=["keywords"],
        keyword_params={"max_keywords": 5}
    )
    
    # Check structure
    assert "keywords" in results
    keyword_result = results["keywords"]
    assert "keywords" in keyword_result
    assert "keyword_scores" in keyword_result
    
    # Check content
    keywords = keyword_result["keywords"]
    assert len(keywords) <= 5
    assert "python" in [k.lower() for k in keywords]
    assert "programming" in [k.lower() for k in keywords]
    
    # Check scores
    scores = keyword_result["keyword_scores"]
    assert all(0 <= score <= 1 for score in scores.values())

@pytest.mark.asyncio
async def test_theme_analysis():
    """Test theme analysis functionality."""
    results = await analyze_text(
        SAMPLE_TEXTS["business"],
        analysis_types=["themes"],
        theme_params={"max_themes": 3}
    )
    
    # Check structure
    assert "themes" in results
    theme_result = results["themes"]
    assert "themes" in theme_result
    assert "theme_descriptions" in theme_result
    
    # Check content
    themes = theme_result["themes"]
    assert len(themes) <= 3
    assert any("financial" in t.lower() or "business" in t.lower() for t in themes)

@pytest.mark.asyncio
async def test_category_classification():
    """Test category classification functionality."""
    categories = {
        "technical": "Technical content about programming and technology",
        "business": "Business and financial content",
        "general": "General or other content"
    }
    
    analyzer = SemanticAnalyzer(
        categories=categories,
        category_config={"min_confidence": 0.3}
    )
    
    # Test with technical text
    results = await analyzer.analyze(
        SAMPLE_TEXTS["technical"],
        analysis_types=["categories"]
    )
    
    # Check structure and content
    assert "categories" in results
    cat_result = results["categories"]
    assert "categories" in cat_result
    
    # Technical text should be classified as technical
    categories = cat_result["categories"]
    assert "technical" in categories
    assert categories["technical"] > 0.5  # High confidence for technical category

@pytest.mark.asyncio
async def test_finnish_support():
    """Test Finnish language support."""
    analyzer = SemanticAnalyzer(language="fi")
    
    results = await analyzer.analyze(
        SAMPLE_TEXTS["finnish"],
        analysis_types=["keywords", "themes"]
    )
    
    # Check keyword extraction
    assert "keywords" in results
    keywords = results["keywords"]["keywords"]
    
    # Should find Finnish compound words
    assert any("ohjelmisto" in k.lower() for k in keywords)
    assert any("kehittäjä" in k.lower() for k in keywords)
    
    # Check language detection
    assert results["keywords"]["language"] == "fi"

@pytest.mark.asyncio
async def test_batch_analysis():
    """Test batch analysis functionality."""
    texts = [
        SAMPLE_TEXTS["technical"],
        SAMPLE_TEXTS["business"]
    ]
    
    analyzer = SemanticAnalyzer()
    results = await analyzer.analyze_batch(
        texts,
        analysis_types=["keywords", "themes"]
    )
    
    # Check structure
    assert len(results) == 2
    for result in results:
        assert "keywords" in result
        assert "themes" in result

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling."""
    # Test with empty text
    results = await analyze_text(
        "",
        analysis_types=["keywords"]
    )
    assert results["keywords"].get("error") is not None
    
    # Test with invalid language
    with pytest.raises(ValueError):
        SemanticAnalyzer(language="invalid")
    
    # Test with invalid analysis type
    with pytest.raises(ValueError):
        await analyze_text(
            "test",
            analysis_types=["invalid_type"]
        )

@pytest.mark.asyncio
async def test_configuration():
    """Test configuration handling."""
    # Test with custom parameters
    analyzer = SemanticAnalyzer(
        keyword_config={
            "min_keyword_length": 5,
            "max_keywords": 3
        },
        theme_config={
            "min_confidence": 0.7
        }
    )
    
    results = await analyzer.analyze(
        SAMPLE_TEXTS["technical"],
        analysis_types=["keywords"]
    )
    
    # Check if configuration is respected
    keywords = results["keywords"]["keywords"]
    assert len(keywords) <= 3  # max_keywords setting
    assert all(len(k) >= 5 for k in keywords)  # min_keyword_length setting

def test_version():
    """Test version information."""
    import semantic_analyzer
    assert hasattr(semantic_analyzer, '__version__')
    assert isinstance(semantic_analyzer.__version__, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])