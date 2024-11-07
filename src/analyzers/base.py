# src/analyzers/base.py

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field


class AnalyzerOutput(BaseModel):
    """Base output model for all analyzers."""
    language: str = Field(default="unknown")
    error: Optional[str] = None
    success: bool = Field(default=True)

    def dict(self) -> Dict[str, Any]:
        """Convert to dict preserving structure."""
        base = super().model_dump()
        # Ensure error handling is consistent
        if not self.success:
            return {"error": base["error"]}
        return base


class TextAnalyzer(ABC):
    """Abstract base class for all text analyzers."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize analyzer with optional LLM and config."""
        from src.core.llm.factory import create_llm

        self.llm = llm or create_llm()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.chain = self._create_chain()

    @abstractmethod
    def _create_chain(self) -> RunnableSequence:
        """Create the LangChain processing chain.
        Must be implemented by subclasses to define their specific processing logic.
        """
        pass

    @abstractmethod
    async def analyze(self, text: str, **kwargs) -> AnalyzerOutput:
        """Perform analysis on text.

        Args:
            text: Input text to analyze
            **kwargs: Additional analysis parameters

        Returns:
            AnalyzerOutput: Analysis results
        """
        pass

    def _detect_language(self, text: str) -> str:
        """Detect text language.

        Args:
            text: Input text

        Returns:
            str: Language code (e.g., 'en', 'fi')
        """
        try:
            from langdetect import detect

            return detect(text)
        except Exception:
            return "unknown"

    def _validate_input(self, text: str) -> Optional[str]:
        """Validate input text.

        Args:
            text: Input text to validate

        Returns:
            Optional[str]: Error message if validation fails, None otherwise
        """
        if not text:
            return "Empty input text"
        if not isinstance(text, str):
            return f"Invalid input type: {type(text)}, expected str"
        if len(text.strip()) == 0:
            return "Input text contains only whitespace"
        return None

    def _handle_error(self, error: str) -> AnalyzerOutput:
        """Create error output.

        Args:
            error: Error message

        Returns:
            AnalyzerOutput: Output with error information
        """
        return AnalyzerOutput(confidence=0.0, language="unknown", error=str(error))

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process and validate LLM output."""
        try:
            # Handle AIMessage output
            if hasattr(output, "content"):
                content = output.content
            elif isinstance(output, str):
                content = output
            else:
                self.logger.error(f"Unexpected output type: {type(output)}")
                return {"error": f"Unexpected output type: {type(output)}"}

            # Parse JSON content
            try:
                if isinstance(content, str):
                    return json.loads(content)
                return content
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
                return {"error": "Failed to parse LLM output"}
                
        except Exception as e:
            self.logger.error(f"Error processing LLM output: {e}")
            return {"error": str(e)}