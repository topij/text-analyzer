# src/analyzers/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

class AnalyzerOutput(BaseModel):
    """Base output model for all analyzers."""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = Field(default="unknown")
    error: Optional[str] = None

    class Config:
        extra = "allow"

class TextAnalyzer(ABC):
    """Abstract base class for all text analyzers."""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize analyzer with optional LLM and config."""
        from src.core.llm.factory import create_llm
        
        self.llm = llm or create_llm()
        self.config = config or {}
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
        return AnalyzerOutput(
            confidence=0.0,
            language="unknown",
            error=str(error)
        )