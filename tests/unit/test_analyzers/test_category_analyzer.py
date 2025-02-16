# tests/unit/test_analyzers/test_category_analyzer.py

import pytest
from typing import Dict, Any

from src.analyzers.category_analyzer import CategoryAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import CategoryOutput, CategoryMatch, Evidence
from src.loaders.models import CategoryConfig
from tests.helpers.mock_llms.category_mock import CategoryMockLLM
from tests.helpers.config import create_test_config
from langchain_core.messages import BaseMessage, HumanMessage

import logging

logger = logging.getLogger(__name__)


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
    def analyzer(self, mock_llm, test_environment_manager, test_categories):
        """Create analyzer with mock LLM and test config."""
        config = {
            "min_confidence": 0.3,
            "language": "en",
            "focus_on": "general content analysis",
        }
        
        return CategoryAnalyzer(
            llm=mock_llm,
            config=config,
            categories=test_categories,
        )

    @pytest.fixture
    def debug_analyzer_setup(self, analyzer):
        """Debug helper to verify analyzer setup."""
        logger.info(f"Analyzer LLM type: {type(analyzer.llm)}")
        logger.info(f"Analyzer chain type: {type(analyzer.chain)}")
        logger.info(f"Analyzer categories: {analyzer.categories}")

        # Test the mock directly
        mock_llm = analyzer.llm
        test_msg = HumanMessage(
            content="""Q3 financial results show 15% revenue growth.
            Market expansion strategy focuses on emerging sectors.
            Customer acquisition metrics improved."""
        )

        mock_response = mock_llm._get_mock_response([test_msg])
        logger.info(
            f"Direct mock response categories: {[cat.name for cat in mock_response.categories]}"
        )
        return analyzer

    @pytest.mark.asyncio
    async def test_business_category_analysis(self, analyzer):
        """Test categorization of business content."""
        # Updated test text to be more explicitly business-focused
        text = """Q3 financial results show 15% revenue growth.
                Market expansion strategy focuses on emerging sectors.
                Customer acquisition metrics improved."""  # Pure business text, no technical terms

        logger.info("Starting business content analysis")
        result = await analyzer.analyze(text)
        self._validate_category_result(result)

        # Verify business categorization
        categories = {cat.name for cat in result.categories}
        logger.info(f"Found categories: {categories}")
        assert (
            "Business" in categories
        ), f"Expected 'Business' category but found: {categories}"

    def _validate_category_result(self, result: CategoryOutput) -> None:
        """Validate category analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert len(result.categories) > 0, "No categories returned"

        for cat in result.categories:
            logger.debug(f"Validating category: {cat.name}")
            assert cat.confidence >= 0.0 and cat.confidence <= 1.0
            assert len(cat.evidence) > 0, f"No evidence for category {cat.name}"

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
    async def test_configuration_handling(
        self, test_environment_manager, test_categories
    ):
        """Test configuration handling."""
        config = {
            "min_confidence": 0.3,
            "language": "en",
            "focus_on": "general content analysis",
        }
        
        # Create separate mock LLM instances for each analyzer
        mock_llm_1 = CategoryMockLLM()
        mock_llm_2 = CategoryMockLLM()
        
        analyzer = CategoryAnalyzer(
            llm=mock_llm_1,
            config=config,
            categories=test_categories,
        )

        # Test with custom config
        custom_config = {
            "min_confidence": 0.5,
            "language": "en",
            "focus_on": "technical analysis",
        }
        
        analyzer_custom = CategoryAnalyzer(
            llm=mock_llm_2,
            config=custom_config,
            categories=test_categories,
        )

        text = """The software development process requires careful planning
                and implementation of best practices."""

        # Test both analyzers
        result = await analyzer.analyze(text)
        result_custom = await analyzer_custom.analyze(text)

        assert result.success
        assert result_custom.success
        assert result.categories != result_custom.categories, "Expected different results with different configs"

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
        self, test_environment_manager, mock_llm, test_categories
    ):
        """Test Finnish language support."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        
        config = {
            "min_confidence": 0.3,
            "language": "fi",
            "focus_on": "general content analysis",
        }
        
        logger.debug(f"Creating analyzer with config: {config}")
        language_processor = create_text_processor(language="fi", file_utils=file_utils)
        logger.debug(f"Created language processor with language: {language_processor.language}")
        
        analyzer = CategoryAnalyzer(
            llm=mock_llm,
            config=config,
            categories=test_categories,
            language_processor=language_processor,
        )
        logger.debug(f"Analyzer config: {analyzer.config}")
        logger.debug(f"Analyzer language: {analyzer._get_language()}")

        text = """
        Tekoäly ja koneoppiminen ovat tärkeitä teknologioita.
        Ohjelmistokehitys vaatii paljon osaamista.
        """

        result = await analyzer.analyze(text)
        logger.debug(f"Analysis result: {result}")
        assert result.success
        assert result.language == "fi"

        # Check Finnish categories
        categories = {cat.name.lower() for cat in result.categories}
        assert "technical" in categories, "Expected category not found"

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

    def _log_category_analysis(self, result):
        """Helper to log category analysis details."""
        logger.debug("Category Analysis Results:")
        logger.debug(f"Success: {result.success}")
        logger.debug(f"Error: {result.error}")
        logger.debug(f"Language: {result.language}")
        if hasattr(result, "categories"):
            logger.debug(f"Categories: {result.categories}")
            if hasattr(result.categories, "matches"):
                logger.debug(f"Category matches: {result.categories.matches}")

    def _validate_category_result(self, result: CategoryOutput) -> None:
        """Validate category analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert result.categories, "No categories found in result"
        logger.debug(
            f"Found categories: {[cat.name for cat in result.categories]}"
        )
