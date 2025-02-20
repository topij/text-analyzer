# tests/helpers/mock_llms/base.py

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel
from langchain_core.callbacks import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)


class BaseMockLLM(BaseChatModel):
    """Base class for all mock LLMs with structured output support."""

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self):
        super().__init__()
        # Store state in a protected dict instead of direct attributes
        self._state = {
            "call_history": [],
            "current_output_class": None
        }

    @property
    def _call_history(self):
        return self._state["call_history"]
    
    @_call_history.setter
    def _call_history(self, value):
        self._state["call_history"] = value

    @property
    def _current_output_class(self):
        return self._state["current_output_class"]
    
    @_current_output_class.setter
    def _current_output_class(self, value):
        self._state["current_output_class"] = value

    def with_structured_output(
        self, output_class: Type[BaseModel]
    ) -> "BaseMockLLM":
        """Configure for structured output to match real LLM behavior."""
        logger.debug(f"Setting up structured output for class: {output_class}")
        self._current_output_class = output_class
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate mock response."""
        try:
            mock_response = self._get_mock_response(messages)
            
            # If we have a structured output class configured
            if self._current_output_class is not None:
                # If the response is already an instance of the expected class, use it directly
                if isinstance(mock_response, self._current_output_class):
                    structured_output = mock_response
                # Otherwise, try to convert it to the expected class
                elif isinstance(mock_response, (dict, BaseModel)):
                    structured_output = self._current_output_class(**mock_response)
                else:
                    raise ValueError(f"Unexpected response type: {type(mock_response)}")
                
                # Return the structured output wrapped in an AIMessage
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",  # Empty content since we're using structured output
                                additional_kwargs={"structured_output": structured_output}
                            )
                        )
                    ]
                )
            
            # For non-structured output, ensure we have a proper message
            if isinstance(mock_response, BaseMessage):
                message = mock_response
            elif isinstance(mock_response, str):
                message = AIMessage(content=mock_response)
            elif isinstance(mock_response, BaseModel):
                # For Pydantic models, use their json() method
                message = AIMessage(content=mock_response.json())
            elif isinstance(mock_response, dict):
                message = AIMessage(content=json.dumps(mock_response))
            else:
                # Convert any other type to string and wrap in AIMessage
                message = AIMessage(content=str(mock_response))
            
            # Ensure the message has content
            if not hasattr(message, 'content') or not message.content:
                message = AIMessage(content=str(mock_response))
            
            return ChatResult(generations=[ChatGeneration(message=message)])
            
        except Exception as e:
            logger.error(f"Error in mock LLM generation: {str(e)}", exc_info=True)
            raise

    def _detect_content_type(self, message: str) -> Tuple[str, str]:
        """Detect language and content type from message."""
        message = message.lower()

        # Language detection
        is_finnish = any(
            term in message
            for term in [
                "koneoppimis",
                "neuroverko",
                "datan",
                "taloudellinen",
                "liikevaihto",
            ]
        )

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

    def _get_mock_response(self, messages: List[BaseMessage]) -> Any:
        """Abstract method to be implemented by specific mock LLMs."""
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"mock_type": self.__class__.__name__}
