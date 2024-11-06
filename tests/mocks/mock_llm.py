# tests/mocks/mock_llm.py

import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockLLM(BaseChatModel):
    """Mock LLM for testing."""

    def get_mock_response(self, text: str) -> Dict[str, Any]:
        """Get mock response based on text content."""
        text = text.lower()

        if "technical" in text:
            return {
                "keywords": ["python", "programming", "language"],
                "keyword_scores": {"python": 0.9, "programming": 0.8, "language": 0.7},
                "themes": ["software development", "programming languages"],
                "categories": {"technical": 0.9, "general": 0.1},
            }
        elif "business" in text:
            return {
                "keywords": ["revenue", "growth", "acquisition"],
                "keyword_scores": {"revenue": 0.9, "growth": 0.8, "acquisition": 0.7},
                "themes": ["business performance", "finance"],
                "categories": {"business": 0.9, "general": 0.1},
            }
        else:
            return {
                "keywords": ["test", "mock"],
                "keyword_scores": {"test": 0.9, "mock": 0.8},
                "themes": ["testing"],
                "categories": {"general": 1.0},
            }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response."""
        # Combine all message content
        text = " ".join(msg.content for msg in messages)

        # Get mock response
        response = self.get_mock_response(text)

        # Create chat generation
        message = ChatMessage(content=json.dumps(response), role="assistant")
        generation = ChatGeneration(message=message, generation_info={"finish_reason": "stop"})

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of generate."""
        return self._generate(messages, stop, run_manager, **kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Combine LLM outputs."""
        return {"finish_reason": "stop"}

    @property
    def _llm_type(self) -> str:
        """Get LLM type."""
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {"name": "mock_llm"}

    def get_num_tokens(self, text: str) -> int:
        """Get number of tokens."""
        return len(text.split())
