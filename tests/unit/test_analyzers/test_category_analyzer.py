# tests/unit/test_analyzers/test_category_analyzer.py

import pytest
from typing import Dict, Any

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import CategoryOutput, CategoryMatch, Evidence
from src.loaders.models import CategoryConfig
from tests.helpers.mock_llms.category_mock import CategoryMockLLM
from tests.helpers.config import create_test_config


class TestCategoryAnalyzer:
    """Tests for category analysis functionality."""

    @pytest.fixture
    def test_categories(self) -> Dict[str, CategoryConfig]:
        """Provide test category configurations."""
        return {
            "Technical": CategoryConfig(
                description="Technical content",
                keywords=["software", "api", "data"],
                threshold=0.6,
            ),
            "Business": CategoryConfig(
                description="Business content",
                keywords=["revenue", "growth", "market"],
                threshold=0.6,
            ),
            "Educational": CategoryConfig(
                description="Educational content",
                keywords=["learning", "teaching", "training"],
                threshold=0.5,
                parent="Technical",
            ),
        }

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM instance."""
        return CategoryMockLLM()

    @pytest.fixture
    def analyzer(self, mock_llm, test_analyzer_config, test_categories):
        """Create analyzer with mock LLM and test config."""
        return CategoryAnalyzer(
            llm=mock_llm,
            config=test_analyzer_config.get_analyzer_config("categories"),
            language_processor=create_text_processor(language="en"),
            categories=test_categories,
        )

    def _validate_category_result(self, result: CategoryOutput) -> None:
        """Validate category analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert result.categories, "No categories found in result"

        for category in result.categories:
            assert isinstance(category, CategoryMatch)
            assert category.name, "Category missing name"
            assert 0 <= category.confidence <= 1.0
            assert category.evidence, "Category missing evidence"

            # Validate evidence
            for evidence in category.evidence:
                assert isinstance(evidence, Evidence)
                assert evidence.text, "Evidence missing text"
                assert 0 <= evidence.relevance <= 1.0

    @pytest.mark.asyncio
    async def test_technical_category_analysis(self, analyzer):
        """Test categorization of technical content."""
        text = """Machine learning models are trained using large datasets.
                Neural networks enable complex pattern recognition.
                API endpoints handle data validation."""

        result = await analyzer.analyze(text)
        self._validate_category_result(result)

        # Verify technical categorization
        categories = {cat.name for cat in result.categories}
        assert "Technical" in categories
        assert any(
            cat.confidence >= 0.6
            for cat in result.categories
            if cat.name == "Technical"
        )

    @pytest.mark.asyncio
    async def test_business_category_analysis(self, analyzer):
        """Test categorization of business content."""
        text = """Q3 financial results show 15% revenue growth.
                Market expansion strategy focuses on emerging sectors.
                Customer acquisition metrics improved."""

        result = await analyzer.analyze(text)
        self._validate_category_result(result)

        # Verify business categorization
        categories = {cat.name for cat in result.categories}
        assert "Business" in categories
        assert any(
            cat.confidence >= 0.6
            for cat in result.categories
            if cat.name == "Business"
        )

    @pytest.mark.asyncio
    async def test_configuration_handling(
        self, test_analyzer_config, mock_llm, test_categories
    ):
        """Test configuration handling."""
        config = test_analyzer_config.get_analyzer_config("categories")
        config["min_confidence"] = 0.7

        analyzer = CategoryAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="en"),
            categories=test_categories,
        )

        assert analyzer.min_confidence == 0.7

        # Test with analysis
        text = "Technical implementation of machine learning API."
        result = await analyzer.analyze(text)
        assert all(cat.confidence >= 0.7 for cat in result.categories)

    @pytest.mark.asyncio
    async def test_category_hierarchy(self, analyzer):
        """Test category hierarchical relationships."""
        text = """The machine learning course provides hands-on training.
                Students learn API development through practical exercises."""

        result = await analyzer.analyze(text)
        self._validate_category_result(result)

        # Verify hierarchy
        categories = {cat.name: cat for cat in result.categories}
        if "Educational" in categories and "Technical" in categories:
            edu_cat = categories["Educational"]
            tech_cat = categories["Technical"]
            assert edu_cat.confidence <= tech_cat.confidence

    @pytest.mark.asyncio
    async def test_finnish_language(
        self, test_analyzer_config, mock_llm, test_categories
    ):
        """Test Finnish language support."""
        fi_analyzer = CategoryAnalyzer(
            llm=mock_llm,
            config=test_analyzer_config.get_analyzer_config("categories"),
            language_processor=create_text_processor(language="fi"),
            categories=test_categories,
        )

        text = """Koneoppimismallit analysoivat dataa tehokkaasti.
                 Järjestelmän rajapinta käsittelee tiedon validoinnin."""

        result = await fi_analyzer.analyze(text)
        self._validate_category_result(result)
        assert result.language == "fi"
        assert "Technical" in {cat.name for cat in result.categories}

    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling cases."""
        # Test empty input
        result = await analyzer.analyze("")
        assert not result.success
        assert "Empty input" in result.error

        # Test None input
        with pytest.raises(ValueError, match="Input text cannot be None"):
            await analyzer.analyze(None)

    @pytest.mark.asyncio
    async def test_evidence_validation(self, analyzer):
        """Test evidence validation in categories."""
        text = """This sophisticated machine learning system utilizes
                neural networks for pattern recognition. The API endpoints
                ensure proper data validation and processing."""

        result = await analyzer.analyze(text)
        self._validate_category_result(result)

        # Verify evidence
        for category in result.categories:
            assert (
                category.evidence
            ), f"No evidence for category {category.name}"
            for evidence in category.evidence:
                assert evidence.text, "Evidence missing text"
                assert (
                    0 <= evidence.relevance <= 1.0
                ), "Invalid evidence relevance score"
