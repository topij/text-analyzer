# src/analyzers/base.py

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

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


class TextSection:
    """Represents a section of text with position information."""
    def __init__(self, text: str, position: str, weight: float = 1.0):
        self.text = text
        self.position = position  # e.g., 'title', 'first_para', 'body', 'last_para'
        self.weight = weight

class BaseAnalyzer:
    """Base class for text analyzers with shared functionality."""
    
    DEFAULT_POSITION_WEIGHTS = {
        'title': 1.5,
        'first_para': 1.3,
        'last_para': 1.2,
        'body': 1.0
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def _split_text_sections(self, text: str) -> List[TextSection]:
        """Split text into weighted sections based on position."""
        if not text:
            return []

        # Get weights from config or use defaults
        weights = self.config.get('position_weights', self.DEFAULT_POSITION_WEIGHTS)
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return [TextSection(text, 'body', weights['body'])]

        sections = []
        
        # Handle title (first line if it looks like a title)
        first_line = paragraphs[0]
        if len(first_line) <= 100 and not first_line.endswith('.'):
            sections.append(TextSection(first_line, 'title', weights['title']))
            paragraphs = paragraphs[1:]

        # Handle remaining paragraphs
        for i, para in enumerate(paragraphs):
            if i == 0 and paragraphs:
                sections.append(TextSection(para, 'first_para', weights['first_para']))
            elif i == len(paragraphs) - 1 and i > 0:
                sections.append(TextSection(para, 'last_para', weights['last_para']))
            else:
                sections.append(TextSection(para, 'body', weights['body']))

        return sections

    def _get_word_positions(self, word: str, sections: List[TextSection]) -> List[Tuple[str, float]]:
        """Get all positions and weights where a word appears."""
        positions = []
        word_pattern = r'\b' + re.escape(word) + r'\b'
        
        for section in sections:
            if re.search(word_pattern, section.text, re.IGNORECASE):
                positions.append((section.position, section.weight))
                
        return positions

    def _calculate_position_score(self, word: str, sections: List[TextSection]) -> float:
        """Calculate position-based importance score for a word."""
        positions = self._get_word_positions(word, sections)
        if not positions:
            return 1.0  # Default weight if word not found
            
        # Use maximum weight from all positions where word appears
        return max(weight for _, weight in positions)

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove standalone punctuation
        text = re.sub(r'\s+([.,!?;:])\s+', ' ', text)
        
        return text.strip()

    def _validate_input(self, text: str) -> Optional[str]:
        """Validate input text."""
        if not text:
            return "Empty input text"
        if not isinstance(text, str):
            return f"Invalid input type: {type(text)}, expected str"
        if len(text.strip()) == 0:
            return "Input text contains only whitespace"
        return None

    def _create_error_output(self, error: str) -> Dict[str, Any]:
        """Create standardized error output."""
        return {
            "error": str(error),
            "success": False,
            "language": "unknown"
        }