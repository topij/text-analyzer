# tests/unit/test_analyzers/test_keyword_analyzer.py

import json
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from src.core.config import AnalyzerConfig

from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import KeywordAnalysisResult, KeywordInfo
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM


class TestKeywordAnalyzer:
    @pytest.fixture
    def mock_llm(self) -> KeywordMockLLM:
        """Create mock LLM instance."""
        return KeywordMockLLM()

    @pytest.fixture
    def analyzer(
        self, mock_llm: KeywordMockLLM, analyzer_config: AnalyzerConfig
    ) -> KeywordAnalyzer:
        """Create analyzer with mock LLM and config."""
        return KeywordAnalyzer(
            llm=mock_llm,
            config=analyzer_config.config.get("analysis", {}),
            language_processor=create_text_processor(language="en"),
        )

    # @pytest.fixture
    # def analyzer(self, mock_llm: KeywordMockLLM) -> KeywordAnalyzer:
    #     """Create analyzer with mock LLM."""
    #     return KeywordAnalyzer(
    #         llm=mock_llm,
    #         config={
    #             "max_keywords": 10,
    #             "min_keyword_length": 3,
    #             "include_compounds": True,
    #             "weights": {"statistical": 0.4, "llm": 0.6},
    #         },
    #         language_processor=create_text_processor(language="en"),
    #     )

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
    async def test_technical_keyword_extraction(
        self,
        analyzer: KeywordAnalyzer,
        mock_llm: KeywordMockLLM,
        analyzer_config: AnalyzerConfig,
    ):
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

        # Check mock LLM was called correctly
        assert "machine learning" in mock_llm.get_last_call().lower()

    @pytest.mark.asyncio
    async def test_business_keyword_extraction(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordMockLLM
    ):
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

    @pytest.mark.asyncio
    async def test_empty_input(self, analyzer: KeywordAnalyzer):
        """Test analyzer behavior with empty input."""
        result = await analyzer.analyze("")
        assert not result.success
        assert len(result.keywords) == 0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_none_input(self, analyzer: KeywordAnalyzer):
        """Test analyzer behavior with None input."""
        with pytest.raises(ValueError):
            await analyzer.analyze(None)

    @pytest.mark.asyncio
    async def test_compound_words(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordMockLLM
    ):
        """Test identification of compound words."""
        text = "The machine learning framework uses neural networks."
        result = await analyzer.analyze(text)

        # Verify compound words were identified
        assert len(result.compound_words) > 0
        assert "machine learning" in result.compound_words

        # Check compound parts in keywords
        for kw in result.keywords:
            if kw.keyword == "machine learning":
                assert kw.compound_parts == ["machine", "learning"]

    @pytest.mark.asyncio
    async def test_finnish_technical_keyword_extraction(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordMockLLM
    ):
        """Test extraction of Finnish technical keywords."""
        text = "Koneoppimismalli käyttää neuroverkkoja datan analysointiin."

        # Create Finnish analyzer
        fi_analyzer = KeywordAnalyzer(
            llm=mock_llm,
            config={
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_compounds": True,
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            language_processor=create_text_processor(language="fi"),
        )

        result = await fi_analyzer.analyze(text)

        # Validate result structure
        self._validate_keyword_result(result)

        # Check for Finnish technical keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "koneoppimismalli" in keywords
        assert "neuroverkko" in keywords

        # Verify compound words were detected
        assert "koneoppimismalli" in result.compound_words

        # Check domain classification
        assert any(kw.domain == "technical" for kw in result.keywords)

        # Verify language
        assert result.language == "fi"

    @pytest.mark.asyncio
    async def test_finnish_business_keyword_extraction(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordMockLLM
    ):
        """Test extraction of Finnish business keywords."""
        text = (
            "Liikevaihdon kasvu oli 20% ja markkinaosuus parani huomattavasti."
        )

        # Create Finnish analyzer
        fi_analyzer = KeywordAnalyzer(
            llm=mock_llm,
            config={
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_compounds": True,
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            language_processor=create_text_processor(language="fi"),
        )

        result = await fi_analyzer.analyze(text)

        # Validate result structure
        self._validate_keyword_result(result)

        # Check for Finnish business keywords
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert "liikevaihdon kasvu" in keywords
        assert "markkinaosuus" in keywords

        # Verify compound word handling
        assert any(
            "liike" in parts
            for kw in result.keywords
            for parts in ([kw.compound_parts] if kw.compound_parts else [])
        )

        # Check domain classification
        assert any(kw.domain == "business" for kw in result.keywords)

        # Verify language
        assert result.language == "fi"

    @pytest.mark.asyncio
    async def test_finnish_compound_words(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordMockLLM
    ):
        """Test handling of Finnish compound words."""
        text = "Tietokantajärjestelmä hyödyntää koneoppimisalgoritmeja."

        # Create Finnish analyzer
        fi_analyzer = KeywordAnalyzer(
            llm=mock_llm,
            config={
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_compounds": True,
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            language_processor=create_text_processor(language="fi"),
        )

        result = await fi_analyzer.analyze(text)

        # Check compound words were identified
        assert len(result.compound_words) > 0
        assert any(
            word in ["tietokantajärjestelmä", "koneoppimisalgoritmi"]
            for word in result.compound_words
        )

        # Verify compound parts
        for kw in result.keywords:
            if kw.keyword == "tietokantajärjestelmä":
                assert kw.compound_parts and len(kw.compound_parts) >= 2
                assert (
                    "tieto" in kw.compound_parts or "kanta" in kw.compound_parts
                )

    # TODO: Add more tests for Finnish keyword extraction
    # - test_finnish_mixed_domain_content
    # - test_finnish_stopword_handling
    # - test_finnish_edge_cases
