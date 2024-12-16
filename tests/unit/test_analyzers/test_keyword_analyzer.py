import pytest
from typing import Dict, Any

from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import KeywordAnalysisResult, KeywordInfo
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM
from tests.helpers.config import (
    create_test_config,
)  # Remove test_analyzer_config import


class TestKeywordAnalyzer:
    """Tests for keyword analysis functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM instance."""
        return KeywordMockLLM()

    @pytest.fixture
    def analyzer(
        self, mock_llm, test_analyzer_config
    ):  # test_analyzer_config comes from conftest.py
        """Create analyzer with mock LLM and test config."""
        config = test_analyzer_config.get_analyzer_config("keywords")
        return KeywordAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="en"),
        )

    def _validate_keyword_result(self, result: KeywordAnalysisResult) -> None:
        """Validate keyword analysis result structure."""
        assert result.success
        assert len(result.keywords) > 0
        for kw in result.keywords:
            assert isinstance(kw, KeywordInfo)
            assert kw.keyword
            assert 0 <= kw.score <= 1.0
            if kw.domain:
                assert kw.domain in ["technical", "business"]

    @pytest.mark.asyncio
    async def test_technical_keyword_extraction(self, analyzer):
        """Test extraction of technical keywords."""
        text = "The machine learning model uses neural networks."
        result = await analyzer.analyze(text)

        # Validate result structure
        self._validate_keyword_result(result)

        # Check for technical keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "machine learning" in keywords
        assert "neural network" in keywords

        # Verify correct domain
        assert any(kw.domain == "technical" for kw in result.keywords)

    @pytest.mark.asyncio
    async def test_empty_input(self, analyzer):
        """Test analyzer behavior with empty input."""
        result = await analyzer.analyze("")
        assert not result.success
        assert len(result.keywords) == 0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_none_input(self, analyzer):
        """Test analyzer behavior with None input."""
        with pytest.raises(ValueError):
            await analyzer.analyze(None)

    @pytest.mark.asyncio
    async def test_finnish_language(self, test_analyzer_config, mock_llm):
        """Test Finnish language support."""
        config = test_analyzer_config.get_analyzer_config("keywords")

        # Create Finnish analyzer
        fi_analyzer = KeywordAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="fi"),
        )

        text = "Koneoppimismalli käyttää neuroverkkoja."
        result = await fi_analyzer.analyze(text)

        self._validate_keyword_result(result)
        assert result.language == "fi"

        # Check Finnish keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "koneoppimismalli" in keywords
        assert "neuroverkko" in keywords

    @pytest.mark.asyncio
    async def test_configuration_handling(self, test_analyzer_config, mock_llm):
        """Test configuration handling."""
        config = test_analyzer_config.get_analyzer_config("keywords")
        config.update(
            {
                "max_keywords": 5,
                "min_confidence": 0.4,
                "weights": {"statistical": 0.3, "llm": 0.7},
            }
        )

        analyzer = KeywordAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="en"),
        )

        assert analyzer.max_keywords == 5
        assert analyzer.min_confidence == 0.4
        assert analyzer.weights["statistical"] == 0.3
        assert analyzer.weights["llm"] == 0.7

        # Test with analysis
        text = "Machine learning model performance improved."
        result = await analyzer.analyze(text)
        assert len(result.keywords) <= 5  # Respects max_keywords setting

    @pytest.mark.asyncio
    async def test_business_keyword_extraction(self, analyzer):
        """Test extraction of business keywords."""
        text = "Revenue growth increased by 20% with improved market share."
        result = await analyzer.analyze(text)

        self._validate_keyword_result(result)

        # Check for business keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "revenue growth" in keywords
        assert "market share" in keywords

        # Verify business domain
        assert any(kw.domain == "business" for kw in result.keywords)

        # Verify compound words
        assert len(result.compound_words) > 0
        assert "revenue growth" in result.compound_words
