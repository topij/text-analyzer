# tests/unit/test_analyzers/test_keyword_analyzer.py

import json
import logging
import pytest
from typing import Any, Dict
from pydantic import BaseModel

from src.analyzers.keyword_analyzer import KeywordAnalyzer, KeywordOutput
from src.core.language_processing import create_text_processor
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM
from tests.conftest import test_environment_manager

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
    def analyzer(self, mock_llm, test_environment_manager):
        """Create analyzer with mock LLM and test config."""
        config = {
            "min_confidence": 0.3,
            "max_keywords": 10,
            "min_keyword_length": 3,
            "include_compounds": True,
            "language": "en",
            "focus_on": "technical content analysis",
        }
        
        return KeywordAnalyzer(
            llm=mock_llm,
            config=config,
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
        """Test keyword extraction for technical content."""
        text = """Machine learning models are trained using large datasets.
                Neural networks enable complex pattern recognition."""
        result = await analyzer.analyze(text)
        
        assert result.success
        assert len(result.keywords) > 0
        assert any("machine learning" in kw.keyword.lower() for kw in result.keywords)

    @pytest.mark.asyncio
    async def test_empty_input(self, analyzer):
        """Test handling of empty input."""
        result = await analyzer.analyze("")
        assert not result.success
        assert "Empty input text" in result.error

    @pytest.mark.asyncio
    async def test_none_input(self, analyzer):
        """Test handling of None input."""
        with pytest.raises(ValueError, match="Input text cannot be None"):
            await analyzer.analyze(None)

    @pytest.mark.asyncio
    async def test_finnish_language(self, mock_llm, test_environment_manager):
        """Test Finnish language support."""
        config = {
            "min_confidence": 0.3,
            "max_keywords": 10,
            "min_keyword_length": 3,
            "include_compounds": True,
            "language": "fi",
            "focus_on": "general content analysis",
        }
        
        analyzer = KeywordAnalyzer(
            llm=mock_llm,
            config=config,
        )

        text = """
        Tekoäly ja koneoppiminen ovat tärkeitä teknologioita.
        Ohjelmistokehitys vaatii paljon osaamista.
        """

        result = await analyzer.analyze(text)
        self._validate_keyword_result(result)

        # Verify Finnish keywords
        keywords = {kw.keyword.lower() for kw in result.keywords}
        assert "tekoäly" in keywords, "Expected Finnish keyword not found"
        assert "koneoppiminen" in keywords, "Expected Finnish keyword not found"
