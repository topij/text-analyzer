# src/analyzers/theme_analyzer.py

import json
import logging
from typing import Any, Dict, List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

from src.config.manager import ConfigManager
from src.core.config import AnalyzerConfig
from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel

from src.analyzers.base import AnalyzerOutput, TextAnalyzer, TextSection
from src.core.language_processing.base import BaseTextProcessor
from src.core.llm.factory import create_llm
from src.schemas import ThemeInfo

logger = logging.getLogger(__name__)


# class ThemePattern(BaseModel):
#     """Pattern for theme identification."""

#     pattern: str
#     domain: Optional[str]
#     weight: float = 1.0


class ThemeEvidence(BaseModel):
    """Evidence supporting a theme identification."""

    text: str
    relevance: float
    keywords: List[str] = Field(default_factory=list)


class ThemeOutput(AnalyzerOutput):
    """Output model for theme analysis."""

    themes: List[ThemeInfo] = Field(default_factory=list)
    theme_hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    evidence: Dict[str, List[ThemeEvidence]] = Field(default_factory=dict)


class ThemeAnalyzer(TextAnalyzer):
    """Analyzes text to identify main themes and topics with language support."""

    # # Domain-specific theme patterns
    # DOMAIN_PATTERNS = {
    #     "technical": [
    #         ThemePattern(
    #             pattern="cloud|infrastructure|deployment",
    #             domain="technical",
    #             weight=1.2,
    #         ),
    #         ThemePattern(
    #             pattern="api|integration|microservice",
    #             domain="technical",
    #             weight=1.1,
    #         ),
    #     ],
    #     "business": [
    #         ThemePattern(
    #             pattern="revenue|growth|profit|market",
    #             domain="business",
    #             weight=1.2,
    #         ),
    #         ThemePattern(
    #             pattern="customer|acquisition|retention",
    #             domain="business",
    #             weight=1.1,
    #         ),
    #     ],
    # }

    # # Finnish domain patterns
    # FINNISH_DOMAIN_PATTERNS = {
    #     "technical": [
    #         ThemePattern(
    #             pattern="pilvi|infrastruktuuri|käyttöönotto",
    #             domain="technical",
    #             weight=1.2,
    #         ),
    #         ThemePattern(
    #             pattern="rajapinta|integraatio|mikropalvelu",
    #             domain="technical",
    #             weight=1.1,
    #         ),
    #     ],
    #     "business": [
    #         ThemePattern(
    #             pattern="liikevaihto|kasvu|tuotto|markkina",
    #             domain="business",
    #             weight=1.2,
    #         ),
    #         ThemePattern(
    #             pattern="asiakas|hankinta|pysyvyys",
    #             domain="business",
    #             weight=1.1,
    #         ),
    #     ],
    # }

    # def __init__(
    #     self,
    #     llm=None,
    #     config: Optional[Dict[str, Any]] = None,
    #     language_processor: Optional[BaseTextProcessor] = None,
    # ):
    #     super().__init__(llm, config)
    #     self.language_processor = language_processor
    #     self.llm = llm or create_llm()
    #     self.chain = self._create_chain()

    # def __init__(
    #     self,
    #     llm: Optional[BaseChatModel] = None,
    #     config: Optional[Dict] = None,
    #     language_processor: Optional[BaseTextProcessor] = None,
    # ):
    #     """Initialize analyzer with configuration and language processing.

    #     Args:
    #         llm: Optional LLM instance (will create using factory if None)
    #         config: Optional configuration dictionary
    #         language_processor: Optional language processor instance
    #     """
    #     # Initialize analyzer config if not provided in config dict
    #     if llm is None:
    #         analyzer_config = ConfigManager()
    #         llm = create_llm(config=analyzer_config)

    #         # Merge analyzer config with provided config if any
    #         if config is None:
    #             config = {}
    #         config = {**analyzer_config.config.get("analysis", {}), **config}

    #     # Call parent init with LLM and config
    #     super().__init__(llm, config)

    #     # Set up language processor
    #     self.language_processor = language_processor

    #     # Create processing chain
    #     self.chain = self._create_chain()

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
    ):
        """Initialize analyzer with new configuration system.

        Args:
            llm: Optional LLM instance (will create using factory if None)
            config: Optional configuration dictionary
            language_processor: Optional language processor instance
        """
        # Initialize analyzer config if not provided
        if llm is None:
            analyzer_config = AnalyzerConfig()
            llm = create_llm(config=analyzer_config)

            # Merge analyzer config with provided config if any
            if config is None:
                config = {}
            config = {**analyzer_config.config.get("analysis", {}), **config}

        # Call parent init with LLM and config
        super().__init__(llm, config)

        # Set up language processor
        self.language_processor = language_processor

        # Initialize components with config
        self.min_confidence = config.get("min_confidence", 0.3)
        self.max_themes = config.get("max_themes", 3)
        self.focus_on = config.get("focus_on", "general content analysis")

        # Create processing chain
        self.chain = self._create_chain()
        logger.debug("ThemeAnalyzer initialized with new configuration system")

    # def _initialize_patterns(self) -> Dict[str, List[ThemePattern]]:
    #     """Initialize language-specific patterns."""
    #     if self.language_processor and self.language_processor.language == "fi":
    #         return self.FINNISH_DOMAIN_PATTERNS
    #     return self.DOMAIN_PATTERNS

    def _create_chain(self) -> RunnableSequence:
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a theme analysis expert. For the given text:
            1. Identify specific themes that emerge from the actual content
            2. Provide detailed descriptions based on text evidence
            3. Calculate confidence scores based on supporting evidence
            4. Extract relevant keywords that support each theme
            
            Focus on themes that are clearly supported by the text content.
            Do not generate generic themes that could apply to any text.""",
                ),
                (
                    "human",
                    """Analyze this text to identify specific themes:
            {text}
            
            Language: {language}
            Required: Extract themes with descriptions, confidence scores, and supporting keywords.
            Base all themes on actual content and evidence from the text.""",
                ),
            ]
        )

        return (
            {
                "text": RunnablePassthrough(),
                "language": lambda _: self._get_language(),
            }
            | template
            | self.llm.with_structured_output(ThemeOutput)
        )

    # def _create_chain(self) -> RunnableSequence:
    #     """Create enhanced LangChain processing chain."""
    #     template = ChatPromptTemplate.from_messages(
    #         [
    #             (
    #                 "system",
    #                 """You are a theme analysis expert. Analyze text to identify main themes,
    #         their hierarchical relationships, and supporting evidence. Consider:
    #         1. Main themes and subthemes
    #         2. Evidence supporting each theme
    #         3. Domain-specific context
    #         4. Cross-theme relationships""",
    #             ),
    #             (
    #                 "human",
    #                 """Analyze this text and identify themes:
    #         Language: {language}}
    #         Text: {text}}

    #         Guidelines:
    #         - Maximum themes: {max_themes}}
    #         - Focus area: {focus_area}}
    #         - Minimum confidence: {min_confidence}}

    #         Return in this format:
    #         {{
    #             "themes": [
    #                 {{
    #                     "name": "theme_name",
    #                     "description": "detailed description",
    #                     "confidence": 0.95,
    #                     "keywords": ["key1", "key2"],
    #                     "parent_theme": "optional_parent"
    #                 }}
    #             ],
    #             "theme_hierarchy": {{
    #                 "parent_theme": ["child_theme1", "child_theme2"]
    #             }}
    #         }}""",
    #             ),
    #         ]
    #     )

    #     return (
    #         {
    #             "text": RunnablePassthrough(),
    #             "language": lambda _: (
    #                 self.language_processor.language
    #                 if self.language_processor
    #                 else "en"
    #             ),
    #             "max_themes": lambda _: self.max_themes,
    #             "focus_area": lambda _: self.focus_on,
    #             "min_confidence": lambda _: self.min_confidence,
    #         }
    #         | template
    #         | self.llm
    #         | self._post_process_llm_output
    #     )

    # def _create_chain(self) -> RunnableSequence:
    #     """Create LangChain processing chain."""
    #     template = ChatPromptTemplate.from_messages(
    #         [
    #             (
    #                 "system",
    #                 """You are a theme analysis expert. Analyze the text to identify main themes,
    #             their hierarchical relationships, and supporting evidence.
    #             Consider the language and domain context in your analysis.

    #             Pay special attention to:
    #             1. Main themes and subthemes
    #             2. Evidence supporting each theme
    #             3. Domain-specific terminology
    #             4. Cross-theme relationships

    #             Return results in this exact JSON format:
    #             {{
    #                 "themes": [
    #                     {{
    #                         "name": "theme_name",
    #                         "description": "detailed description",
    #                         "confidence": 0.95,
    #                         "keywords": ["key1", "key2"],
    #                         "domain": "technical/business",
    #                         "parent_theme": "optional_parent"
    #                     }}
    #                 ],
    #                 "evidence": {{
    #                     "theme_name": [
    #                         {{
    #                             "text": "relevant text",
    #                             "relevance": 0.9,
    #                             "keywords": ["key1", "key2"]
    #                         }}
    #                     ]
    #                 }},
    #                 "relationships": {{
    #                     "theme_name": ["related_theme1", "related_theme2"]
    #                 }}
    #             }}""",
    #             ),
    #             (
    #                 "human",
    #                 """Analyze this text and identify themes:
    #             Language: {language}
    #             Text: {text}

    #             Guidelines:
    #             - Maximum themes: {max_themes}
    #             - Consider these key terms: {key_terms}
    #             - Focus on domain: {focus_domain}
    #             - Minimum confidence: {min_confidence}
    #             """,
    #             ),
    #         ]
    #     )

    #     return (
    #         {
    #             "text": RunnablePassthrough(),
    #             "language": lambda _: (
    #                 self.language_processor.language
    #                 if self.language_processor
    #                 else "en"
    #             ),
    #             "max_themes": lambda _: self.config.get("max_themes", 3),
    #             "key_terms": self._get_key_terms,
    #             "focus_domain": lambda _: self.config.get(
    #                 "focus_on", "general"
    #             ),
    #             "min_confidence": lambda _: self.config.get(
    #                 "min_confidence", 0.3
    #             ),
    #         }
    #         | template
    #         | self.llm
    #         | self._post_process_llm_output
    #     )

    def _get_key_terms(self, text: str) -> str:
        """Get key terms for theme identification."""
        if not self.language_processor:
            return ""

        try:
            # Get word frequencies for context
            words = self.language_processor.tokenize(text)
            if not words:
                return ""

            # Filter and process key terms
            key_terms = []
            for word in words:
                if len(word) < 3:
                    continue

                # Get base form
                base = self.language_processor.get_base_form(word)
                if not base or self.language_processor.is_stop_word(base):
                    continue

                # Check for compound words
                if self.language_processor.is_compound_word(word):
                    parts = self.language_processor.get_compound_parts(word)
                    if parts:
                        key_terms.extend(parts)

                key_terms.append(base)

            # Return unique terms
            return ", ".join(set(key_terms))

        except Exception as e:
            logger.error(f"Error getting key terms: {e}")
            return ""

    def _validate_themes(
        self, themes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate theme data."""
        validated = []
        for theme in themes:
            if not isinstance(theme, dict):
                continue

            # Ensure required fields
            if "name" not in theme or "confidence" not in theme:
                continue

            # Validate confidence
            theme["confidence"] = float(theme["confidence"])
            if theme["confidence"] < 0 or theme["confidence"] > 1:
                continue

            # Ensure description
            if "description" not in theme:
                theme["description"] = theme["name"]

            # Ensure keywords list
            if "keywords" not in theme:
                theme["keywords"] = []

            validated.append(theme)

        return validated

    def _validate_hierarchy(
        self, hierarchy: Dict[str, List[str]], themes: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Validate theme hierarchy."""
        theme_names = {theme["name"] for theme in themes}
        validated = {}

        for parent, children in hierarchy.items():
            if parent not in theme_names:
                continue

            valid_children = [
                child
                for child in children
                if child in theme_names and child != parent
            ]

            if valid_children:
                validated[parent] = valid_children

        return validated

    def _validate_input(self, text: Any) -> Optional[str]:
        """Validate input text with proper type checking."""
        if text is None:
            raise ValueError("Input text cannot be None")

        if not isinstance(text, str):
            raise TypeError(
                f"Invalid input type: expected str, got {type(text)}"
            )

        text = text.strip()
        if not text:
            return "Empty input text"

        if len(text) < self.config.get("min_text_length", 10):
            return "Input text too short for meaningful analysis"

        return None

    # temporary fix for the test failure
    async def analyze(self, text: str) -> ThemeOutput:
        """Analyze text with AIMessage handling."""
        if text is None:
            raise ValueError("Input text cannot be None")

        if not text:
            return ThemeOutput(
                themes=[],
                theme_hierarchy={},
                language=self._get_language(),
                success=False,
                error="Empty input text",
            )

        try:
            logger.debug("ThemeAnalyzer.analyze: Starting analysis")
            result = await self.chain.ainvoke(text)
            logger.debug(
                f"ThemeAnalyzer.analyze: Chain result type: {type(result)}"
            )
            logger.debug(f"ThemeAnalyzer.analyze: Chain result: {result}")

            # Handle AIMessage from mock LLMs
            if hasattr(result, "content"):
                try:
                    data = json.loads(result.content)
                    logger.debug(
                        f"ThemeAnalyzer.analyze: Parsed JSON data: {data}"
                    )
                    result = ThemeOutput(**data)
                except Exception as e:
                    logger.error(f"Error parsing AIMessage content: {e}")
                    return ThemeOutput(
                        themes=[],
                        theme_hierarchy={},
                        language=self._get_language(),
                        success=False,
                        error=f"Error parsing response: {str(e)}",
                    )

            # Filter themes by confidence
            if getattr(result, "themes", None):
                result.themes = [
                    theme
                    for theme in result.themes
                    if theme.confidence >= self.min_confidence
                ]

                # Limit number of themes
                result.themes = result.themes[: self.max_themes]

            return result

        except Exception as e:
            logger.error(f"ThemeAnalyzer.analyze: Exception occurred: {e}")
            return ThemeOutput(
                themes=[],
                theme_hierarchy={},
                language=self._get_language(),
                success=False,
                error=str(e),
            )

    # don't remove, this is the working production version
    # async def analyze(self, text: str) -> ThemeOutput:
    #     """Analyze text with structured output validation."""
    #     if text is None:
    #         raise ValueError("Input text cannot be None")

    #     if not text:
    #         return ThemeOutput(
    #             themes=[],
    #             theme_hierarchy={},
    #             language=self._get_language(),
    #             success=False,
    #             error="Empty input text",
    #         )

    #     try:
    #         result = await self.chain.ainvoke(text)

    #         # Filter themes by confidence
    #         result.themes = [
    #             theme
    #             for theme in result.themes
    #             if theme.confidence >= self.min_confidence
    #         ]

    #         # Limit number of themes
    #         result.themes = result.themes[: self.max_themes]

    #         return result

    #     except Exception as e:
    #         logger.error(f"Theme analysis failed: {str(e)}")
    #         return ThemeOutput(
    #             themes=[],
    #             theme_hierarchy={},
    #             language=self._get_language(),
    #             success=False,
    #             error=str(e),
    #         )

    def _process_llm_response(self, response: Dict[str, Any]) -> ThemeOutput:
        """Process LLM response into ThemeOutput with enhanced error handling."""
        try:
            if not response or not isinstance(response, dict):
                raise ValueError(f"Invalid response format: {response}")

            themes = []
            for theme_data in response.get("themes", []):
                theme = ThemeInfo(
                    name=theme_data["name"],
                    description=theme_data["description"],
                    confidence=float(theme_data["confidence"]),
                    keywords=theme_data.get("keywords", []),
                    parent_theme=theme_data.get("parent_theme"),
                )
                themes.append(theme)

            # Get hierarchy
            hierarchy = response.get("theme_hierarchy", {})

            return ThemeOutput(
                themes=themes,
                theme_hierarchy=hierarchy,
                language=response.get("language", self._get_language()),
                success=True,
            )

        except Exception as e:
            logger.error(f"Error processing themes: {e}")
            return ThemeOutput(
                themes=[],
                theme_hierarchy={},
                language=self._get_language(),
                error=str(e),
                success=False,
            )

    def _adjust_confidence(
        self, base_confidence: float, domain: str, keywords: List[str]
    ) -> float:
        """Adjust confidence based on domain and language context."""
        confidence = base_confidence

        # Get relevant patterns
        domain_patterns = self.patterns.get(domain, [])
        if not domain_patterns:
            return confidence

        # Check keyword matches
        matches = 0
        total_weight = 0
        for pattern in domain_patterns:
            if any(kw.lower() in pattern.pattern for kw in keywords):
                matches += 1
                total_weight += pattern.weight

        if matches:
            # Adjust confidence based on pattern matches
            confidence *= 1.0 + (total_weight / matches) * 0.1

        return min(confidence, 1.0)

    def _get_language(self) -> str:
        """Get current language."""
        return (
            self.language_processor.language
            if self.language_processor
            else "en"
        )

    # def _get_language(self) -> str:
    #     """Get current language."""
    #     return (
    #         self.language_processor.language
    #         if self.language_processor
    #         else "unknown"
    #     )

    def display_themes(self, results: ThemeOutput) -> None:
        """Display theme analysis results."""
        if not results.success:
            print(f"Analysis failed: {results.error}")
            return

        print("\nTheme Analysis Results")
        print("=" * 50)

        for theme in results.themes:
            print(f"\nTheme: {theme.name}")
            print("-" * 20)
            print(f"Description: {theme.description}")
            print(f"Confidence: {theme.confidence:.2f}")

            if theme.keywords:
                print(f"Keywords: {', '.join(theme.keywords)}")

            if theme.parent_theme:
                print(f"Parent Theme: {theme.parent_theme}")

            # Show evidence
            if theme.name in results.evidence:
                print("\nEvidence:")
                for e in results.evidence[theme.name]:
                    print(f"- {e.text} (relevance: {e.relevance:.2f})")

        # Show hierarchy
        if results.theme_hierarchy:
            print("\nTheme Hierarchy:")
            for parent, children in results.theme_hierarchy.items():
                print(f"{parent} -> {', '.join(children)}")
