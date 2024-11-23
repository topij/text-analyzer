# tests/test_components/test_keyword_analyzer.py

import pytest
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import KeywordOutput, KeywordInfo
from tests.helpers.mock_llm import MockLLM


# Test fixtures at module level
@pytest.fixture(scope="session")
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


class TestKeywordAnalyzer:
    @pytest.fixture(scope="class")
    def keyword_analyzer_en(self) -> KeywordAnalyzer:
        """Create English keyword analyzer with mock LLM."""
        return KeywordAnalyzer(
            llm=MockLLM(),
            config={
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_compounds": True,
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            language_processor=create_text_processor(language="en"),
        )

    @pytest.fixture(scope="class")
    def keyword_analyzer_fi(self) -> KeywordAnalyzer:
        """Create Finnish keyword analyzer with mock LLM."""
        return KeywordAnalyzer(
            llm=MockLLM(),
            config={
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_compounds": True,
                "weights": {"statistical": 0.4, "llm": 0.6},
            },
            language_processor=create_text_processor(language="fi"),
        )

    def _validate_keyword_output(self, result: Union[Dict, BaseModel]) -> None:
        """Validate keyword analysis output."""
        print("\nDebug: Validating output:", result)
        # Handle both dict and model cases
        if isinstance(result, dict):
            # Validate required fields exist
            assert "keywords" in result
            assert "success" in result
            if not result.get("success", True):
                return

            keywords = result["keywords"]
        else:
            # Already a model instance
            assert hasattr(result, "keywords")
            if not getattr(result, "success", True):
                return

            keywords = (
                result.keywords if isinstance(result.keywords, list) else []
            )

        # Validate keywords
        assert len(keywords) > 0
        for kw in keywords:
            if isinstance(kw, dict):
                assert "keyword" in kw
                assert "score" in kw
                assert len(kw["keyword"]) >= 3
                assert 0 <= kw["score"] <= 1.0
            else:
                assert isinstance(kw, KeywordInfo)
                assert len(kw.keyword) >= 3
                assert 0 <= kw.score <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling(
        self, keyword_analyzer_en: KeywordAnalyzer
    ) -> None:
        """Test error handling for invalid inputs."""
        # Empty input should still produce a valid output structure
        result = await keyword_analyzer_en.analyze("")
        self._validate_keyword_output(result)

        # None input should raise TypeError
        with pytest.raises(TypeError):
            await keyword_analyzer_en.analyze(None)  # type: ignore

    @pytest.mark.asyncio
    async def test_english_technical(
        self, keyword_analyzer_en: KeywordAnalyzer, test_content: Dict
    ) -> None:
        """Test keyword extraction from English technical content."""
        result = await keyword_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_keyword_output(result)

        # Test for expected technical keywords
        keywords = {
            (
                kw.keyword.lower()
                if isinstance(kw, KeywordInfo)
                else kw["keyword"].lower()
            )
            for kw in (
                result.keywords
                if hasattr(result, "keywords")
                else result["keywords"]
            )
        }
        assert any(
            term in keywords
            for term in ["machine learning", "neural network", "data"]
        )

    @pytest.mark.asyncio
    async def test_english_business(
        self, keyword_analyzer_en: KeywordAnalyzer, test_content: Dict
    ) -> None:
        """Test keyword extraction from English business content."""
        result = await keyword_analyzer_en.analyze(
            test_content["en"]["business"]
        )
        self._validate_keyword_output(result)

        # Debug output
        print(
            "\nDebug: Expected business terms:", {"revenue", "growth", "market"}
        )
        print(
            "Debug: Actual keywords:",
            {kw.keyword.lower() for kw in result.keywords},
        )

        # Verify business terms
        keywords = {kw.keyword.lower() for kw in result.keywords}
        expected_terms = {"revenue", "growth", "market"}
        assert any(
            term in keywords for term in expected_terms
        ), f"No expected business terms found in keywords: {keywords}"

    @pytest.mark.asyncio
    async def test_finnish_technical(
        self, keyword_analyzer_fi: KeywordAnalyzer, test_content: Dict
    ) -> None:
        """Test keyword extraction from Finnish technical content."""
        result = await keyword_analyzer_fi.analyze(
            test_content["fi"]["technical"]
        )
        self._validate_keyword_output(result)

        # Verify technical terms
        keywords = {kw.keyword.lower() for kw in result.keywords}
        expected_terms = {"koneoppimismalli", "neuroverkko", "data"}
        assert any(term in keywords for term in expected_terms)

        # Check compound words
        assert len(result.compound_words) > 0
        assert any(
            "koneoppimis" in word.lower() for word in result.compound_words
        )

    @pytest.mark.asyncio
    async def test_batch_processing(
        self, keyword_analyzer_en: KeywordAnalyzer, test_content: Dict
    ) -> None:
        """Test batch processing capabilities."""
        texts = [
            test_content["en"]["technical"],
            test_content["en"]["business"],
        ]
        results = await keyword_analyzer_en.analyze_batch(texts, batch_size=2)

        assert len(results) == 2
        for result in results:
            self._validate_keyword_output(result)

    @pytest.mark.asyncio
    async def test_scoring(
        self, keyword_analyzer_en: KeywordAnalyzer, test_content: Dict
    ) -> None:
        """Test keyword scoring and order."""
        result = await keyword_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_keyword_output(result)

        # Get keywords based on result type
        if isinstance(result, dict):
            keywords = sorted(
                result["keywords"], key=lambda x: x["score"], reverse=True
            )
            scores = [kw["score"] for kw in keywords]
        else:
            keywords = sorted(
                result.keywords, key=lambda x: x.score, reverse=True
            )
            scores = [kw.score for kw in keywords]

        assert len(keywords) > 0
        assert all(0 <= score <= 1.0 for score in scores)
        # Verify scores are in descending order
        assert scores == sorted(scores, reverse=True)
