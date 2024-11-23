# tests/helpers/mock_llm.py

from typing import Any, List, Dict, Optional, Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import json
import logging

logger = logging.getLogger(__name__)


class MockLLM(BaseChatModel):
    """Mock LLM for testing."""

    def _detect_content_type(self, message: str) -> tuple[str, str]:
        """Detect content type and language from message."""
        message = message.lower()

        # Language detection
        is_finnish = any(
            term in message
            for term in [
                "koneoppimis",
                "neuroverko",
                "datan",
                "piirteiden",
                "taloudellinen",
                "liikevaihto",
            ]
        )
        language = "fi" if is_finnish else "en"

        # Content type detection
        is_technical = any(
            term in message
            for term in [
                "machine learning",
                "neural",
                "data",
                "koneoppimis",
                "neuroverko",
                "datan",
            ]
        )

        is_business = any(
            term in message
            for term in [
                "financial",
                "revenue",
                "market",
                "business",
                "taloudellinen",
                "liikevaihto",
                "markkina",
            ]
        )

        content_type = (
            "technical"
            if is_technical
            else "business" if is_business else "unknown"
        )

        logger.debug(
            f"Detected language: {language}, content type: {content_type}"
        )
        return language, content_type

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate mock response."""
        content = self._get_mock_response(messages)
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _get_mock_response(self, messages: Sequence[BaseMessage]) -> str:
        """Get appropriate mock response based on input and language."""
        # Get the last message content
        last_message = messages[-1].content if messages else ""

        # Empty input
        if not last_message:
            return json.dumps(
                {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "success": True,
                    "language": "en",
                    "error": None,
                }
            )

        # Detect content type and language
        language, content_type = self._detect_content_type(last_message)
        logger.debug(f"Message content: {last_message[:100]}...")

        # Finnish technical content
        if language == "fi" and content_type == "technical":
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "koneoppimismalli",
                            "score": 0.9,
                            "domain": "technical",
                            "compound_parts": ["kone", "oppimis", "malli"],
                        },
                        {
                            "keyword": "neuroverkko",
                            "score": 0.85,
                            "domain": "technical",
                            "compound_parts": ["neuro", "verkko"],
                        },
                        {
                            "keyword": "data",
                            "score": 0.8,
                            "domain": "technical",
                            "compound_parts": None,
                        },
                    ],
                    "compound_words": ["koneoppimismalli", "neuroverkko"],
                    "domain_keywords": {
                        "technical": ["koneoppimismalli", "neuroverkko", "data"]
                    },
                    "success": True,
                    "language": language,
                    "error": None,
                }
            )

        # English technical content
        elif content_type == "technical":
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "machine learning",
                            "score": 0.9,
                            "domain": "technical",
                            "compound_parts": ["machine", "learning"],
                        },
                        {
                            "keyword": "neural network",
                            "score": 0.85,
                            "domain": "technical",
                            "compound_parts": ["neural", "network"],
                        },
                        {
                            "keyword": "data",
                            "score": 0.8,
                            "domain": "technical",
                            "compound_parts": None,
                        },
                    ],
                    "compound_words": ["machine learning", "neural network"],
                    "domain_keywords": {
                        "technical": [
                            "machine learning",
                            "neural network",
                            "data",
                        ]
                    },
                    "success": True,
                    "language": language,
                    "error": None,
                }
            )

        # Business content
        elif content_type == "business":
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "revenue",
                            "score": 0.9,
                            "domain": "business",
                            "compound_parts": None,
                        },
                        {
                            "keyword": "growth",
                            "score": 0.85,
                            "domain": "business",
                            "compound_parts": None,
                        },
                        {
                            "keyword": "market",
                            "score": 0.8,
                            "domain": "business",
                            "compound_parts": None,
                        },
                    ],
                    "compound_words": [],
                    "domain_keywords": {
                        "business": ["revenue", "growth", "market"]
                    },
                    "success": True,
                    "language": language,
                    "error": None,
                }
            )

        # Default response
        return json.dumps(
            {
                "keywords": [],
                "compound_words": [],
                "domain_keywords": {},
                "success": True,
                "language": language,
                "error": None,
            }
        )

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"mock_param": True}
