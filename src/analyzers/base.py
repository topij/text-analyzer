import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class TextSection:
    """Represents a section of text with position information."""
    content: str
    start: int
    end: int
    weight: float = 1.0

class AnalyzerOutput(BaseModel):
    """Base output model for all analyzers."""
    language: str = Field(default="unknown")
    error: Optional[str] = None
    success: bool = Field(default=True)

    def dict(self) -> Dict[str, Any]:
        """Convert to dict preserving structure."""
        base = super().model_dump()
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
        """Create the LangChain processing chain."""
        pass

    @abstractmethod
    async def analyze(self, text: str, **kwargs) -> AnalyzerOutput:
        """Perform analysis on text."""
        pass

    def _detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "unknown"

    def _validate_input(self, text: str) -> Optional[str]:
        """Validate input text."""
        if not text:
            return "Empty input text"
        if not isinstance(text, str):
            return f"Invalid input type: {type(text)}, expected str"
        if len(text.strip()) == 0:
            return "Input text contains only whitespace"
        return None

    def _handle_error(self, error: str) -> AnalyzerOutput:
        """Create error output."""
        return AnalyzerOutput(
            error=str(error),
            success=False,
            language="unknown"
        )

    def _get_text_sections(self, text: str, section_size: int = 200) -> List[TextSection]:
        """Split text into weighted sections."""
        sections = []
        text = text.strip()
        
        # Handle short texts
        if len(text) <= section_size:
            return [TextSection(text, 0, len(text), 1.0)]
            
        # Split into sections with position-based weights
        for i in range(0, len(text), section_size):
            section = text[i:i + section_size]
            # Higher weights for start and end sections
            weight = 1.2 if i == 0 else 1.1 if i + section_size >= len(text) else 1.0
            sections.append(TextSection(section, i, i + len(section), weight))
            
        return sections

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process raw LLM output into dictionary format.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Dict[str, Any]: Processed output
        """
        try:
            # Extract content from different output types
            if hasattr(output, "content"):
                content = output.content
            elif isinstance(output, str):
                content = output
            elif isinstance(output, dict):
                return output
            else:
                self.logger.error(f"Unexpected output type: {type(output)}")
                return {}

            # Clean and parse JSON
            try:
                import json
                # Remove any potential prefix/suffix text
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                    
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error: {e}\nContent: {content}")
                return {}

        except Exception as e:
            self.logger.error(f"Error processing LLM output: {e}")
            return {}