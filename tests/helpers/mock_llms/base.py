# tests/helpers/mock_llms/base.py

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


class BaseMockLLM(BaseChatModel):
    """Base class for all mock LLMs with simplified implementation."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate mock response."""
        try:
            logger.debug(
                f"MockLLM received message: {messages[-1].content[:100]}..."
            )
            content = self._get_mock_response(messages)
            logger.debug(
                f"MockLLM returning content: {content[:100] if len(content) > 100 else content}"
            )

            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Error in mock LLM: {e}")
            raise

    def _detect_content_type(self, message: str) -> Tuple[str, str]:
        """Detect language and content type from message."""
        message = message.lower()

        # Language detection - Finnish indicators
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

        # Content type detection
        is_technical = any(
            term in message
            for term in [
                # English technical terms
                "machine learning",
                "neural",
                "data",
                # Finnish technical terms
                "koneoppimis",
                "neuroverko",
                "datan",
            ]
        )

        is_business = any(
            term in message
            for term in [
                # English business terms
                "financial",
                "revenue",
                "market",
                "profit",
                # Finnish business terms
                "liikevaihto",
                "tulos",
                "markkina",
            ]
        )

        language = "fi" if is_finnish else "en"
        content_type = (
            "technical"
            if is_technical
            else "business" if is_business else "unknown"
        )

        return language, content_type

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Abstract method to be implemented by specific mock LLMs."""
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {"mock_type": self.__class__.__name__}
