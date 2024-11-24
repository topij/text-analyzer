# tests/test_components/test_category_analyzer.py

import pytest
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

from src.analyzers.category_analyzer import CategoryAnalyzer, CategoryOutput
from src.schemas import CategoryMatch, Evidence
from src.core.language_processing import create_text_processor
from src.loaders.models import CategoryConfig
from tests.helpers.mock_llms import CategoryMockLLM


# Move fixtures outside the test class
@pytest.fixture(scope="module")
def test_content() -> Dict[str, Dict[str, str]]:
    """Provide test content."""
    return {
        "en": {
            "technical": """Machine learning models are trained using large datasets to recognize patterns.
                        Neural network architecture includes multiple layers for feature extraction.
                        Data preprocessing and feature engineering are crucial steps.""",
            "business": """Q3 financial results show 15% revenue growth and improved profit margins.
                        Customer acquisition costs decreased while retention rates increased.
                        Market expansion strategy focuses on emerging technology sectors.""",
        },
        "fi": {
            "technical": """Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja.
                        Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen.
                        Datan esikäsittely ja piirteiden suunnittelu ovat keskeisiä vaiheita.""",
            "business": """Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet.
                        Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani.
                        Markkinalaajennusstrategia keskittyy nouseviin teknologiasektoreihin.""",
        },
    }


@pytest.fixture(scope="module")
def category_analyzer_en() -> CategoryAnalyzer:
    """Create English category analyzer with mock LLM."""
    categories = {
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
            description="Financial performance and metrics content",
            keywords=["revenue", "profit", "financial", "growth"],
            threshold=0.7,
        ),
        "Market Strategy": CategoryConfig(
            description="Market and business strategy content",
            keywords=["market", "strategy", "customer", "acquisition"],
            threshold=0.7,
        ),
    }

    return CategoryAnalyzer(
        categories=categories,
        llm=CategoryMockLLM(),
        config={
            "min_confidence": 0.3,
            "language": "en",
        },
        language_processor=create_text_processor(language="en"),
    )


@pytest.fixture(scope="module")
def category_analyzer_fi() -> CategoryAnalyzer:
    """Create Finnish category analyzer with mock LLM."""
    categories = {
        "Koneoppiminen": CategoryConfig(
            description="Koneoppimisen ja tekoälyn teknologia",
            keywords=["koneoppiminen", "neuroverkko", "tekoäly"],
            threshold=0.7,
        ),
        "Data-analyysi": CategoryConfig(
            description="Datan käsittely ja analysointi",
            keywords=["data", "analyysi", "käsittely"],
            threshold=0.7,
        ),
    }

    return CategoryAnalyzer(
        categories=categories,
        llm=CategoryMockLLM(),
        config={
            "min_confidence": 0.3,
            "language": "fi",
        },
        language_processor=create_text_processor(language="fi"),
    )


class TestCategoryAnalyzer:
    """Tests for category analysis functionality."""

    def _validate_category_output(self, result: Union[Dict, BaseModel]) -> None:
        """Validate category analysis output with improved error messages."""
        print(f"\nValidating category output: {result}")

        # Early error check with detailed message
        if not result.success:
            error_msg = (
                f"Analysis failed: {result.error}"
                if result.error
                else "Analysis failed without error message"
            )
            assert result.error is None, error_msg
            return

        # Get categories
        categories = result.categories
        print(f"Found categories: {[cat.name for cat in categories]}")
        assert len(categories) > 0, "No categories found in result"

        # Validate each category
        for category in categories:
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
    async def test_english_technical_categories(
        self, category_analyzer_en: CategoryAnalyzer, test_content: Dict
    ) -> None:
        """Test category extraction from English technical content."""
        result = await category_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_category_output(result)

        # Check specific categories with debug output
        category_names = {cat.name.lower() for cat in result.categories}
        expected_categories = {"machine learning", "data science"}

        print(f"\nFound categories: {category_names}")
        print(f"Expected categories: {expected_categories}")

        assert any(
            cat in category_names for cat in expected_categories
        ), f"Expected categories {expected_categories} not found in {category_names}"

    @pytest.mark.asyncio
    async def test_english_business_categories(
        self, category_analyzer_en: CategoryAnalyzer, test_content: Dict
    ) -> None:
        """Test category extraction from English business content."""
        result = await category_analyzer_en.analyze(
            test_content["en"]["business"]
        )
        self._validate_category_output(result)

        category_names = {cat.name.lower() for cat in result.categories}
        expected_categories = {"financial analysis", "market strategy"}
        assert any(
            cat in category_names for cat in expected_categories
        ), f"Expected categories {expected_categories} not found in {category_names}"

    @pytest.mark.asyncio
    async def test_finnish_technical_categories(
        self, category_analyzer_fi: CategoryAnalyzer, test_content: Dict
    ) -> None:
        """Test category extraction from Finnish technical content."""
        result = await category_analyzer_fi.analyze(
            test_content["fi"]["technical"]
        )
        self._validate_category_output(result)

        category_names = {cat.name.lower() for cat in result.categories}
        expected_categories = {"koneoppiminen", "data-analyysi"}
        assert any(
            cat in category_names for cat in expected_categories
        ), f"Expected categories {expected_categories} not found in {category_names}"

    @pytest.mark.asyncio
    async def test_error_handling(
        self, category_analyzer_en: CategoryAnalyzer
    ) -> None:
        """Test error handling for invalid inputs."""
        # Empty input
        empty_result = await category_analyzer_en.analyze("")
        assert empty_result.success is False
        assert empty_result.error is not None
        assert "Empty input" in empty_result.error

        # None input
        with pytest.raises(ValueError) as exc_info:
            await category_analyzer_en.analyze(None)
        assert "Input text cannot be None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_confidence_thresholds(
        self, category_analyzer_en: CategoryAnalyzer, test_content: Dict
    ) -> None:
        """Test confidence score thresholds."""
        # Test with high threshold
        category_analyzer_en.config["min_confidence"] = 0.8
        result = await category_analyzer_en.analyze(
            test_content["en"]["technical"]
        )

        for category in result.categories:
            assert (
                category.confidence >= 0.8
            ), f"Category {category.name} confidence {category.confidence} below threshold 0.8"

        # Test with lower threshold
        category_analyzer_en.config["min_confidence"] = 0.3
        result = await category_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        assert (
            len(result.categories) > 0
        ), "No categories returned with low threshold"
