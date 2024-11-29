import json
import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import Field, ConfigDict
import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.analyzers.keyword_analyzer import KeywordAnalyzer, KeywordOutput
from src.schemas import KeywordInfo

logger = logging.getLogger(__name__)


class KeywordAnalyzerMockLLM(BaseChatModel):
    """Purpose-built mock LLM for testing KeywordAnalyzer."""

    response_mode: str = Field(default="standard")
    call_history: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, response_mode: str = "standard", **kwargs):
        super().__init__(**kwargs)
        # Convert these to instance variables to avoid Pydantic validation
        self._responses = {
            "technical": {
                "keywords": [
                    {
                        "keyword": "cloud infrastructure",
                        "score": 0.9,
                        "domain": "technical",
                        "compound_parts": ["cloud", "infrastructure"],
                    },
                    {
                        "keyword": "microservices",
                        "score": 0.85,
                        "domain": "technical",
                    },
                ],
                "compound_words": ["cloud infrastructure"],
                "domain_keywords": {
                    "technical": ["cloud infrastructure", "microservices"]
                },
            },
            "business": {
                "keywords": [
                    {
                        "keyword": "revenue growth",
                        "score": 0.9,
                        "domain": "business",
                        "compound_parts": ["revenue", "growth"],
                    },
                    {
                        "keyword": "market share",
                        "score": 0.85,
                        "domain": "business",
                    },
                ],
                "compound_words": ["revenue growth", "market share"],
                "domain_keywords": {
                    "business": ["revenue growth", "market share"]
                },
            },
            "default": {
                "keywords": [],
                "compound_words": [],
                "domain_keywords": {},
                "success": True,
                "language": "en",
            },
        }

    def get_call_history(self) -> List[str]:
        """Get history of calls made to this mock."""
        return self.call_history

    def _get_response_for_text(self, text: str) -> Dict[str, Any]:
        """Get appropriate response based on text content."""
        # Track the call
        self.call_history.append(text)

        if self.response_mode == "error":
            raise ValueError("Simulated LLM error")
        elif self.response_mode == "empty":
            return self._responses["default"]

        # Standard response mode
        if any(
            term in text.lower()
            for term in ["cloud", "technical", "microservices"]
        ):
            return self._responses["technical"]
        elif any(
            term in text.lower() for term in ["revenue", "business", "market"]
        ):
            return self._responses["business"]
        return self._responses["default"]

    response_mode: str = Field(default="standard")
    call_history: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate mock response with proper signature."""
        logger.debug(f"Mock LLM _generate called with messages: {messages}")

        text = messages[-1].content if messages else ""
        self.call_history.append(text)

        response = self._get_response_for_text(text)
        message = AIMessage(content=json.dumps(response))
        generation = ChatGeneration(message=message)

        logger.debug(f"Mock LLM returning response: {response}")
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "keyword-analyzer-mock"


@pytest.fixture(scope="function")
def mock_llm() -> KeywordAnalyzerMockLLM:
    """Create mock LLM instance."""
    return KeywordAnalyzerMockLLM()


@pytest.fixture(scope="function")
def error_llm() -> KeywordAnalyzerMockLLM:
    """Create mock LLM that simulates errors."""
    return KeywordAnalyzerMockLLM(response_mode="error")


@pytest.fixture(scope="function")
def analyzer(mock_llm: KeywordAnalyzerMockLLM) -> KeywordAnalyzer:
    """Create KeywordAnalyzer with mock LLM."""
    config = {
        "language": "en",
        "min_confidence": 0.1,
        "max_keywords": 10,
        "include_compounds": True,
        "clustering": {
            "similarity_threshold": 0.85,
            "max_cluster_size": 3,
            "boost_factor": 1.2,
            "domain_bonus": 0.1,
            "min_cluster_size": 2,
            "max_relation_distance": 2,
        },
    }

    return KeywordAnalyzer(llm=mock_llm, config=config)


# def analyzer(mock_llm: KeywordAnalyzerMockLLM) -> KeywordAnalyzer:
#     """Create KeywordAnalyzer with mock LLM."""
#     config = {
#         "language": "en",
#         "min_confidence": 0.1,
#         "max_keywords": 10,
#         "include_compounds": True,
#         "clustering": {
#             "similarity_threshold": 0.85,
#             "max_cluster_size": 3,
#             "boost_factor": 1.2,
#             "domain_bonus": 0.1,
#             "min_cluster_size": 2,
#             "max_relation_distance": 2,
#         },
#     }

#     analyzer = KeywordAnalyzer(llm=mock_llm, config=config)

#     # Add debug logging
#     original_analyze = analyzer.analyze

#     async def debug_analyze(text: str) -> KeywordOutput:
#         logger.debug(f"Analyzing text: {text}")
#         try:
#             result = await original_analyze(text)
#             logger.debug(f"Analysis result: {result}")
#             return result
#         except Exception as e:
#             logger.error(f"Analysis error: {str(e)}", exc_info=True)
#             raise

#     analyzer.analyze = debug_analyze
#     return analyzer


class TestKeywordAnalyzer:
    """Test suite for KeywordAnalyzer."""

    @pytest.mark.asyncio
    async def test_technical_keyword_extraction(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordAnalyzerMockLLM
    ):
        """Test extraction of technical keywords."""
        text = "Our cloud infrastructure deployment includes microservices."
        result = await analyzer.analyze(text)

        # Verify LLM was called correctly
        assert mock_llm.get_call_history()[-1] == text

        # Verify result structure
        assert isinstance(result, KeywordOutput)
        assert result.success
        assert len(result.keywords) > 0

        # Verify technical keywords
        cloud_infra = next(
            k for k in result.keywords if k.keyword == "cloud infrastructure"
        )
        assert cloud_infra.score == 0.9
        assert cloud_infra.domain == "technical"
        assert cloud_infra.compound_parts == ["cloud", "infrastructure"]

        microservices = next(
            k for k in result.keywords if k.keyword == "microservices"
        )
        assert microservices.score == 0.85
        assert microservices.domain == "technical"

    @pytest.mark.asyncio
    async def test_business_keyword_extraction(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordAnalyzerMockLLM
    ):
        """Test extraction of business keywords."""
        text = "The revenue growth exceeded expectations, improving our market share."
        result = await analyzer.analyze(text)

        assert mock_llm.get_call_history()[-1] == text
        assert isinstance(result, KeywordOutput)
        assert result.success
        assert len(result.keywords) > 0

        revenue = next(
            k for k in result.keywords if k.keyword == "revenue growth"
        )
        assert revenue.score == 0.9
        assert revenue.domain == "business"
        assert revenue.compound_parts == ["revenue", "growth"]

        market = next(k for k in result.keywords if k.keyword == "market share")
        assert market.score == 0.85
        assert market.domain == "business"

    @pytest.mark.asyncio
    async def test_empty_input(self, analyzer: KeywordAnalyzer):
        """Test analyzer behavior with empty input."""
        result = await analyzer.analyze("")
        assert len(result.keywords) == 0

    @pytest.mark.asyncio
    async def test_none_input(self, analyzer: KeywordAnalyzer):
        """Test analyzer behavior with None input."""
        with pytest.raises(ValueError, match="Input text cannot be None"):
            await analyzer.analyze(None)

    @pytest.mark.asyncio
    async def test_very_short_input(self, analyzer: KeywordAnalyzer):
        """Test analyzer behavior with very short input."""
        result = await analyzer.analyze("hi")
        assert len(result.keywords) == 0

    @pytest.mark.asyncio
    async def test_compound_words(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordAnalyzerMockLLM
    ):
        """Test identification of compound words."""
        text = "The cloud infrastructure shows good revenue growth patterns."
        result = await analyzer.analyze(text)

        assert mock_llm.get_call_history()[-1] == text
        assert isinstance(result, KeywordOutput)
        assert result.success
        assert len(result.compound_words) >= 1
        assert "cloud infrastructure" in result.compound_words

    @pytest.mark.asyncio
    async def test_domain_classification(
        self, analyzer: KeywordAnalyzer, mock_llm: KeywordAnalyzerMockLLM
    ):
        """Test domain classification of keywords."""
        text = "The cloud infrastructure deployment boosted our revenue growth."
        result = await analyzer.analyze(text)

        assert mock_llm.get_call_history()[-1] == text
        assert isinstance(result, KeywordOutput)
        assert result.success
        assert len(result.keywords) > 0

        domains = {k.domain for k in result.keywords if k.domain}
        assert "technical" in domains
        assert "business" in domains
