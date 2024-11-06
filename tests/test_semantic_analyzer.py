# tests/test_semantic_analyzer.py

import logging
from typing import Any, Dict

import pytest

from src.semantic_analyzer import SemanticAnalyzer

from . import SAMPLE_TEXTS


@pytest.mark.asyncio
async def test_keyword_extraction(analyzer_config, mock_llm):
    """Test keyword extraction functionality."""
    analyzer = SemanticAnalyzer(config=analyzer_config, llm=mock_llm)

    results = await analyzer.analyze(SAMPLE_TEXTS["technical"], analysis_types=["keywords"])

    # Check structure
    assert "keywords" in results
    keyword_result = results["keywords"]
    assert "keywords" in keyword_result
    assert "keyword_scores" in keyword_result

    # Check content
    keywords = keyword_result["keywords"]
    assert len(keywords) >= 3
    assert "python" in [k.lower() for k in keywords]
    assert "programming" in [k.lower() for k in keywords]

    # Check scores
    scores = keyword_result["keyword_scores"]
    assert all(0 <= score <= 1 for score in scores.values())


@pytest.mark.asyncio
async def test_theme_analysis(analyzer_config, mock_llm):
    """Test theme analysis functionality."""
    analyzer = SemanticAnalyzer(config=analyzer_config, llm=mock_llm)

    results = await analyzer.analyze(SAMPLE_TEXTS["business"], analysis_types=["themes"])

    assert "themes" in results
    theme_result = results["themes"]
    assert "themes" in theme_result
    assert "theme_descriptions" in theme_result

    themes = theme_result["themes"]
    assert len(themes) > 0
    assert any("business" in t.lower() or "financial" in t.lower() for t in themes)


@pytest.mark.asyncio
async def test_category_classification(analyzer_config, mock_llm, test_categories):
    """Test category classification functionality."""
    analyzer = SemanticAnalyzer(config=analyzer_config, llm=mock_llm, categories=test_categories)

    # Test with technical text
    results = await analyzer.analyze(SAMPLE_TEXTS["technical"], analysis_types=["categories"])

    assert "categories" in results
    cat_result = results["categories"]
    assert "categories" in cat_result

    categories = cat_result["categories"]
    assert "technical" in categories
    assert categories["technical"] > 0.5


@pytest.mark.asyncio
async def test_finnish_support(analyzer_config, mock_llm, voikko_path):
    """Test Finnish language support."""
    if not voikko_path:
        pytest.skip("VOIKKO_PATH not set, skipping Finnish tests")

    analyzer = SemanticAnalyzer(config=analyzer_config, llm=mock_llm, language="fi")

    results = await analyzer.analyze(SAMPLE_TEXTS["finnish"], analysis_types=["keywords", "themes"])

    assert "keywords" in results
    keywords = results["keywords"]["keywords"]

    # Should find Finnish compound words
    assert any("ohjelmisto" in k.lower() for k in keywords)
    assert any("kehittäjä" in k.lower() for k in keywords)

    assert results["keywords"]["language"] == "fi"


@pytest.mark.asyncio
async def test_batch_analysis(analyzer_config, mock_llm):
    """Test batch analysis functionality."""
    analyzer = SemanticAnalyzer(config=analyzer_config, llm=mock_llm)

    texts = [SAMPLE_TEXTS["technical"], SAMPLE_TEXTS["business"]]

    results = await analyzer.analyze_batch(texts, analysis_types=["keywords", "themes"])

    assert len(results) == 2
    for result in results:
        assert "keywords" in result
        assert "themes" in result


@pytest.mark.asyncio
async def test_result_saving(analyzer_config, mock_llm, test_data_dir, cleanup_test_files):
    """Test result saving functionality."""
    analyzer = SemanticAnalyzer(config=analyzer_config, llm=mock_llm)

    results = await analyzer.analyze(SAMPLE_TEXTS["technical"], analysis_types=["keywords", "themes"])

    # Save results
    saved_path = analyzer_config.save_results(results, "test_results.yaml")

    # Load and verify results
    loaded_results = analyzer_config.file_utils.load_yaml(saved_path)
    assert "keywords" in loaded_results
    assert "themes" in loaded_results


@pytest.mark.asyncio
async def test_error_handling(analyzer_config):
    """Test error handling."""
    analyzer = SemanticAnalyzer(config=analyzer_config)

    # Test with empty text
    results = await analyzer.analyze("", analysis_types=["keywords"])
    assert results["keywords"].get("error") is not None

    # Test with invalid language
    with pytest.raises(ValueError):
        SemanticAnalyzer(config=analyzer_config, language="invalid")

    # Test with invalid analysis type
    with pytest.raises(ValueError):
        await analyzer.analyze("test", analysis_types=["invalid_type"])


@pytest.mark.asyncio
async def test_configuration(analyzer_config, mock_llm):
    """Test configuration handling."""
    # Test with custom keyword settings
    analyzer = SemanticAnalyzer(
        config=analyzer_config,
        llm=mock_llm,
        keyword_config={"min_keyword_length": 5, "max_keywords": 3},
    )

    results = await analyzer.analyze(SAMPLE_TEXTS["technical"], analysis_types=["keywords"])

    keywords = results["keywords"]["keywords"]
    assert len(keywords) <= 3
    assert all(len(k) >= 5 for k in keywords)


def test_file_utils_integration(file_utils):
    """Test FileUtils integration."""
    # Test data path access
    data_path = file_utils.get_data_path("raw")
    assert data_path.exists()

    # Test configuration loading
    config = file_utils.config
    assert "semantic_analyzer" in config

    # Test directory structure
    assert (data_path.parent / "configurations").exists()
    assert (data_path.parent / "processed").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
