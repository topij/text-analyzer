# tests/unit/test_analyzers/test_keyword_analyzer.py

import json
import logging
import pytest
from typing import Any, Dict
import logging
from pydantic import BaseModel

from src.analyzers.keyword_analyzer import KeywordAnalyzer, KeywordOutput
from src.core.language_processing import create_text_processor
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def transform_analyzer_output(result: Any) -> KeywordOutput:
    """Transform analyzer output to KeywordOutput for testing."""
    try:
        # Handle AIMessage output from mock LLMs
        if hasattr(result, "content"):
            data = json.loads(result.content)
            return KeywordOutput(**data)

        # Handle direct KeywordOutput
        if isinstance(result, KeywordOutput):
            return result

        # Handle dictionary output
        if isinstance(result, dict):
            return KeywordOutput(**result)

        # Handle JSON string output
        if isinstance(result, str):
            data = json.loads(result)
            return KeywordOutput(**data)

        raise ValueError(f"Unexpected output type: {type(result)}")

    except Exception as e:
        logger.error(f"Error transforming test output: {e}")
        return KeywordOutput(
            keywords=[],
            compound_words=[],
            domain_keywords={},
            language="en",
            success=False,
            error=str(e),
        )


class TestKeywordAnalyzer:
    """Tests for keyword analysis functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM instance."""
        return KeywordMockLLM()

    @pytest.fixture
    def analyzer(self, mock_llm, test_analyzer_config):
        """Create analyzer with mock LLM and test config."""
        config = test_analyzer_config.get_analyzer_config("keywords")
        return KeywordAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="en"),
        )

    def _validate_keyword_result(self, result: KeywordOutput) -> None:
        """Validate keyword analysis result structure."""
        assert result.success
        assert len(result.keywords) > 0
        for kw in result.keywords:
            assert kw.keyword
            assert 0 <= kw.score <= 1.0
            if kw.domain:
                assert kw.domain in ["technical", "business"]

    @pytest.mark.asyncio
    async def test_technical_keyword_extraction(self, analyzer):
        """Test extraction of technical keywords."""
        text = "The machine learning model uses neural networks."
        raw_result = await analyzer.analyze(text)
        print(f"raw_result: {raw_result}, result type: {type(raw_result)}")
        # Transform output for testing
        result = transform_analyzer_output(raw_result)

        print(f"Transformed result: {result}, result type: {type(result)}")

        # Validate result structure
        self._validate_keyword_result(result)

        # Check for technical keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "machine learning" in keywords
        assert "neural network" in keywords

        # Verify technical domain
        assert any(kw.domain == "technical" for kw in result.keywords)

    @pytest.mark.asyncio
    async def test_empty_input(self, analyzer):
        """Test analyzer behavior with empty input."""
        raw_result = await analyzer.analyze("")
        result = transform_analyzer_output(raw_result)
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
        raw_result = await fi_analyzer.analyze(text)
        result = transform_analyzer_output(raw_result)

        self._validate_keyword_result(result)
        assert result.language == "fi"

        # Check Finnish keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "koneoppimismalli" in keywords
        assert "neuroverkko" in keywords
