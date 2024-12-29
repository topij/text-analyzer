# src.analyzer.base.py

# TODO(test-infrastructure): Current analyzer implementations modify production code to handle
# mock LLM responses. This should be refactored to handle the mock responses in the test code
# instead. Potential approaches:
#
# 1. Create mock LLMs that better match real LLM behavior by returning the correct types
# 2. Add a testing-specific adapter layer to transform mock responses
# 3. Investigate if LangChain provides better testing utilities
#
# The current implementation works but mixes test requirements with production code.
#
# Current workarounds:
# - analyzer.analyze() methods include AIMessage handling
# - Mock LLM responses are converted to structured output in the analyzer
# - Test framework includes response transformation utilities
#
# Priority: Medium
# Impact: Test maintainability and production code clarity


import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

from src.core.llm.factory import create_llm
from src.loaders.parameter_handler import ParameterHandler
from src.core.language_processing.base import BaseTextProcessor
from src.config.manager import ConfigManager
from src.core.managers import EnvironmentManager

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


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        parameter_handler: Optional[ParameterHandler] = None,
    ):
        from src.core.llm.factory import create_llm

        self.llm = llm or create_llm()
        self.config = config or {}
        self.parameter_handler = parameter_handler
        if self.parameter_handler:
            self.parameters = self.parameter_handler.get_parameters()
            self.config.update(self.parameters.model_dump())

        self.chain = self._create_chain()

    @abstractmethod
    def _create_chain(self) -> RunnableSequence:
        """Create analysis chain."""
        pass

    @abstractmethod
    async def analyze(self, text: str, **kwargs) -> AnalyzerOutput:
        """Perform analysis."""
        pass


class TextAnalyzer(ABC):
    """Abstract base class for all text analyzers."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize analyzer with optional LLM and config."""
        # Get config from EnvironmentManager if not provided
        if llm is None:
            try:
                environment = EnvironmentManager.get_instance()
                components = environment.get_components()
                analyzer_config = components["config_manager"]
            except RuntimeError:
                raise ValueError(
                    "EnvironmentManager must be initialized before creating TextAnalyzer. "
                    "Initialize EnvironmentManager first or provide config and LLM."
                )
            
            llm = create_llm(config=analyzer_config)

            # Merge analyzer config with provided config if any
            if config is None:
                config = {}
            config = {**analyzer_config.config.get("analysis", {}), **config}

        self.llm = llm
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Create chain after all initialization is done
        self.chain = self._create_chain()

    @abstractmethod
    def _create_chain(self) -> RunnableSequence:
        """Create the LangChain processing chain."""
        pass

    @abstractmethod
    async def analyze(self, text: str, **kwargs) -> AnalyzerOutput:
        """Perform analysis on text."""
        pass

    def _get_language(self) -> str:
        """Get current language with better fallback."""
        if self.language_processor:
            logger.debug(f"Getting language from language_processor: {self.language_processor.language}")
            return self.language_processor.language
        if "language" in self.config:
            logger.debug(f"Getting language from config: {self.config['language']}")
            return self.config["language"]
        logger.debug("No language found, using 'unknown'")
        return "unknown"

    def _prepare_result(self, result: Any, output_type: str) -> Any:
        """Prepare result with proper language."""
        if hasattr(result, "__dict__"):
            result.language = self._get_language()
        elif isinstance(result, dict):
            result["language"] = self._get_language()
        return result

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
            error=str(error), success=False, language="unknown"
        )

    def _get_text_sections(
        self, text: str, section_size: int = 200
    ) -> List[TextSection]:
        """Split text into weighted sections."""
        sections = []
        text = text.strip()

        # Handle short texts
        if len(text) <= section_size:
            return [TextSection(text, 0, len(text), 1.0)]

        # Split into sections with position-based weights
        for i in range(0, len(text), section_size):
            section = text[i : i + section_size]
            # Higher weights for start and end sections
            weight = (
                1.2 if i == 0 else 1.1 if i + section_size >= len(text) else 1.0
            )
            sections.append(TextSection(section, i, i + len(section), weight))

        return sections

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process raw LLM output into dictionary format."""
        try:
            logger.debug(f"Raw LLM output type: {type(output)}")
            logger.debug(f"Raw LLM output content: {output}")

            # Extract content from different output types
            if hasattr(output, "content"):
                content = output.content
                logger.debug(f"Extracted content: {content}")
            elif isinstance(output, str):
                content = output
            elif isinstance(output, dict):
                return output
            else:
                logger.error(f"Unexpected output type: {type(output)}")
                return {}

            # Clean and parse JSON
            try:
                # Remove any potential prefix/suffix text
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                if not content:
                    logger.error("Empty content after cleanup")
                    return {}

                logger.debug(f"Cleaned content for parsing: {content}")
                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}\nContent: {content}")
                # Try to extract data if it's embedded in text
                try:
                    # Look for JSON-like structure
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        logger.debug(
                            f"Attempting to parse extracted JSON: {json_str}"
                        )
                        return json.loads(json_str)
                except:
                    pass
                return {}

        except Exception as e:
            logger.error(f"Error processing LLM output: {e}")
            return {}
