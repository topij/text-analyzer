# tests/unit/test_analyzers/test_category_analyzer.py
import pytest
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from FileUtils import FileUtils
from src.core.config import AnalyzerConfig

from src.config.manager import ConfigManager
from src.core.language_processing import create_text_processor

import json

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.loaders.models import CategoryConfig
from src.schemas import CategoryMatch, Evidence, CategoryOutput
from tests.helpers.mock_llms.category_mock import CategoryMockLLM


@pytest.fixture
def mock_llm() -> CategoryMockLLM:
    """Create mock LLM instance."""
    return CategoryMockLLM()


@pytest.fixture(scope="module")
def category_configs() -> Dict[str, CategoryConfig]:
    """Create test category configurations."""
    return {
        "Machine Learning": CategoryConfig(
            description="Machine learning and AI technology content",
            keywords=["machine learning", "neural network", "data"],
            threshold=0.7,
        ),
        "Data Science": CategoryConfig(
            description="Data processing and analysis content",
            keywords=["data", "preprocessing", "feature engineering"],
            threshold=0.7,
        ),
        "Financial Analysis": CategoryConfig(
            description="Financial metrics and performance",
            keywords=["revenue", "financial", "profit", "growth"],
            threshold=0.7,
        ),
        "Market Strategy": CategoryConfig(
            description="Market and business strategy",
            keywords=["market", "strategy", "growth"],
            threshold=0.7,
        ),
    }


@pytest.fixture
def test_analyzer(
    mock_llm: CategoryMockLLM,
    analyzer_config: AnalyzerConfig,
    file_utils: FileUtils,
) -> CategoryAnalyzer:
    """Create CategoryAnalyzer with mock LLM for testing."""
    return CategoryAnalyzer(
        llm=mock_llm,
        config=analyzer_config.config.get("analysis", {}),
        language_processor=create_text_processor(language="en"),
    )


# @pytest.fixture
# def analyzer(
#     self,
#     mock_llm: BaseChatModel,
#     config_manager: ConfigManager,  # Updated fixture
# ) -> CategoryAnalyzer:
#     """Create analyzer with mock LLM and config."""
#     return CategoryAnalyzer(
#         llm=mock_llm,
#         config=config_manager.get_config(),  # Updated config access
#         language_processor=create_text_processor(language="en"),
#     )


# @pytest.fixture
# def fi_analyzer(
#     mock_llm: CategoryMockLLM,
#     config_manager: ConfigManager,
#     category_configs: Dict[str, CategoryConfig],
# ) -> CategoryAnalyzer:
#     """Create Finnish analyzer with mock LLM and test categories."""
#     return CategoryAnalyzer(
#         categories=category_configs,
#         llm=mock_llm,
#         config={**config_manager.get_config("analysis", {}), "language": "fi"},
#         language_processor=create_text_processor(language="fi"),
#     )


@pytest.fixture
def test_analyzer(
    mock_llm: CategoryMockLLM, analyzer_config: AnalyzerConfig
) -> CategoryAnalyzer:
    """Create CategoryAnalyzer with mock LLM for testing."""
    return CategoryAnalyzer(
        llm=mock_llm,
        config=analyzer_config.config.get("analysis", {}),
        language_processor=create_text_processor(language="en"),
    )


@pytest.fixture
def fi_analyzer(
    mock_llm: CategoryMockLLM, analyzer_config: AnalyzerConfig
) -> CategoryAnalyzer:
    """Create Finnish CategoryAnalyzer with mock LLM for testing."""
    return CategoryAnalyzer(
        llm=mock_llm,
        config=analyzer_config.config.get("analysis", {}),
        language_processor=create_text_processor(language="fi"),
    )


class TestCategoryAnalyzer:
    """Tests for category analysis functionality."""

    def _validate_category_result(self, result: CategoryOutput) -> None:
        """Validate category analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert len(result.categories) > 0, "No categories found in result"

        for category in result.categories:
            assert isinstance(
                category, CategoryMatch
            ), f"Invalid category type: {type(category)}"
            assert category.name, "Category missing name"
            assert (
                0 <= category.confidence <= 1.0
            ), f"Invalid confidence score: {category.confidence}"
            assert category.description, "Category missing description"

            if category.evidence:
                for evidence in category.evidence:
                    assert isinstance(
                        evidence, Evidence
                    ), f"Invalid evidence type: {type(evidence)}"
                    assert evidence.text, "Evidence missing text"
                    assert (
                        0 <= evidence.relevance <= 1.0
                    ), f"Invalid evidence relevance: {evidence.relevance}"

    @pytest.mark.asyncio
    async def test_technical_category_analysis(
        self,
        test_analyzer: CategoryAnalyzer,  # Use the fixture name test_analyzer
    ):
        """Test category analysis of technical content."""
        text = """Machine learning models are trained using large datasets.
                Neural network architecture includes multiple layers.
                Data preprocessing and feature engineering are crucial steps."""

        result = await test_analyzer.analyze(text)
        self._validate_category_result(result)

        # Verify specific categories
        categories = {cat.name.lower() for cat in result.categories}
        assert (
            "machine learning" in categories
        ), "Machine Learning category not found"
        assert "data science" in categories, "Data Science category not found"

    def _validate_category_result(self, result):
        """Validate category analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert result.categories, "No categories found in result"

        for category in result.categories:
            assert category.name, "Category missing name"
            assert (
                0 <= category.confidence <= 1.0
            ), f"Invalid confidence score: {category.confidence}"
            assert isinstance(
                category.evidence, list
            ), "Category evidence should be a list"

    @pytest.mark.asyncio
    async def test_business_category_analysis(
        self, test_analyzer: CategoryAnalyzer
    ):
        """Test category analysis of business content."""
        text = """Q3 financial results show 15% revenue growth.
                Market expansion strategy focuses on emerging sectors.
                Improved profit margins and market performance."""

        result = await test_analyzer.analyze(text)
        self._validate_category_result(result)

        categories = {cat.name.lower() for cat in result.categories}
        assert (
            "financial analysis" in categories
        ), "Financial Analysis category not found"
        assert (
            "market strategy" in categories
        ), "Market Strategy category not found"

    @pytest.mark.asyncio
    async def test_finnish_technical_analysis(
        self, fi_analyzer: CategoryAnalyzer
    ):
        """Test category analysis of Finnish technical content."""
        text = """Koneoppimismalleja koulutetaan suurilla datajoukolla.
                 Neuroverkon arkkitehtuuri sisältää useita kerroksia.
                 Datan esikäsittely ja piirteiden suunnittelu ovat tärkeitä."""

        result = await fi_analyzer.analyze(text)
        self._validate_category_result(result)

        categories = {cat.name.lower() for cat in result.categories}
        assert (
            "machine learning" in categories
        ), "Machine Learning category not found"
        assert "data science" in categories, "Data Science category not found"

    @pytest.mark.asyncio
    async def test_finnish_business_analysis(
        self, fi_analyzer: CategoryAnalyzer
    ):
        """Test category analysis of Finnish business content."""
        text = """Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun.
                 Markkinalaajennusstrategia keskittyy uusiin sektoreihin."""

        result = await fi_analyzer.analyze(text)
        self._validate_category_result(result)

        categories = {cat.name.lower() for cat in result.categories}
        assert (
            "financial analysis" in categories
        ), "Financial Analysis category not found"
        assert (
            "market strategy" in categories
        ), "Market Strategy category not found"

    @pytest.mark.asyncio
    async def test_error_handling(self, test_analyzer: CategoryAnalyzer):
        """Test error handling for invalid inputs."""
        # Empty input
        result = await test_analyzer.analyze("")
        assert not result.success
        assert result.error is not None
        assert "Empty input" in result.error

        # None input
        with pytest.raises(ValueError) as exc_info:
            await test_analyzer.analyze(None)
        assert "Input text cannot be None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_confidence_thresholds(self, test_analyzer: CategoryAnalyzer):
        """Test confidence score thresholds."""
        text = "Machine learning models process data using neural networks."

        # Test with high threshold
        test_analyzer.min_confidence = 0.8
        result = await test_analyzer.analyze(text)
        assert result.success

        for category in result.categories:
            assert (
                category.confidence >= 0.8
            ), f"Category {category.name} confidence {category.confidence} below threshold 0.8"

        # Test with lower threshold
        test_analyzer.min_confidence = 0.3
        result = await test_analyzer.analyze(text)
        assert (
            len(result.categories) > 0
        ), "No categories returned with low threshold"

    @pytest.mark.asyncio
    async def test_evidence_validation(self, test_analyzer: CategoryAnalyzer):
        """Test evidence validation and relevance scores."""
        text = """Machine learning models perform complex data analysis.
                 Neural networks enable advanced pattern recognition.
                 Data preprocessing improves model accuracy."""

        result = await test_analyzer.analyze(text)
        self._validate_category_result(result)

        for category in result.categories:
            assert (
                category.evidence
            ), f"No evidence found for category {category.name}"
            for evidence in category.evidence:
                assert evidence.text.strip(), "Empty evidence text found"
                assert (
                    0 < evidence.relevance <= 1.0
                ), f"Invalid evidence relevance score: {evidence.relevance}"
