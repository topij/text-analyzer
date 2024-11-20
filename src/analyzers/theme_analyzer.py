# src/analyzers/theme_analyzer.py

from typing import Any, Dict, List, Optional, Set
import logging
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import TextAnalyzer, AnalyzerOutput, TextSection
from src.core.language_processing.base import BaseTextProcessor
from src.schemas import ThemeInfo

logger = logging.getLogger(__name__)


class ThemePattern(BaseModel):
    """Pattern for theme identification."""

    pattern: str
    domain: Optional[str]
    weight: float = 1.0


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

    # Domain-specific theme patterns
    DOMAIN_PATTERNS = {
        "technical": [
            ThemePattern(
                pattern="cloud|infrastructure|deployment",
                domain="technical",
                weight=1.2,
            ),
            ThemePattern(
                pattern="api|integration|microservice",
                domain="technical",
                weight=1.1,
            ),
        ],
        "business": [
            ThemePattern(
                pattern="revenue|growth|profit|market",
                domain="business",
                weight=1.2,
            ),
            ThemePattern(
                pattern="customer|acquisition|retention",
                domain="business",
                weight=1.1,
            ),
        ],
    }

    # Finnish domain patterns
    FINNISH_DOMAIN_PATTERNS = {
        "technical": [
            ThemePattern(
                pattern="pilvi|infrastruktuuri|käyttöönotto",
                domain="technical",
                weight=1.2,
            ),
            ThemePattern(
                pattern="rajapinta|integraatio|mikropalvelu",
                domain="technical",
                weight=1.1,
            ),
        ],
        "business": [
            ThemePattern(
                pattern="liikevaihto|kasvu|tuotto|markkina",
                domain="business",
                weight=1.2,
            ),
            ThemePattern(
                pattern="asiakas|hankinta|pysyvyys",
                domain="business",
                weight=1.1,
            ),
        ],
    }

    def __init__(
        self,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
    ):
        """Initialize theme analyzer with language support."""
        super().__init__(llm, config)
        self.language_processor = language_processor
        self.patterns = self._initialize_patterns()
        self.chain = self._create_chain()

    def _initialize_patterns(self) -> Dict[str, List[ThemePattern]]:
        """Initialize language-specific patterns."""
        if self.language_processor and self.language_processor.language == "fi":
            return self.FINNISH_DOMAIN_PATTERNS
        return self.DOMAIN_PATTERNS

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a theme analysis expert. Analyze the text to identify main themes,
                their hierarchical relationships, and supporting evidence.
                Consider the language and domain context in your analysis.
                
                Pay special attention to:
                1. Main themes and subthemes
                2. Evidence supporting each theme
                3. Domain-specific terminology
                4. Cross-theme relationships
                
                Return results in this exact JSON format:
                {{
                    "themes": [
                        {{
                            "name": "theme_name",
                            "description": "detailed description",
                            "confidence": 0.95,
                            "keywords": ["key1", "key2"],
                            "domain": "technical/business",
                            "parent_theme": "optional_parent"
                        }}
                    ],
                    "evidence": {{
                        "theme_name": [
                            {{
                                "text": "relevant text",
                                "relevance": 0.9,
                                "keywords": ["key1", "key2"]
                            }}
                        ]
                    }},
                    "relationships": {{
                        "theme_name": ["related_theme1", "related_theme2"]
                    }}
                }}""",
                ),
                (
                    "human",
                    """Analyze this text and identify themes:
                Language: {language}
                Text: {text}
                
                Guidelines:
                - Maximum themes: {max_themes}
                - Consider these key terms: {key_terms}
                - Focus on domain: {focus_domain}
                - Minimum confidence: {min_confidence}
                """,
                ),
            ]
        )

        return (
            {
                "text": RunnablePassthrough(),
                "language": lambda _: (
                    self.language_processor.language
                    if self.language_processor
                    else "en"
                ),
                "max_themes": lambda _: self.config.get("max_themes", 3),
                "key_terms": self._get_key_terms,
                "focus_domain": lambda _: self.config.get(
                    "focus_on", "general"
                ),
                "min_confidence": lambda _: self.config.get(
                    "min_confidence", 0.3
                ),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

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

    async def analyze(self, text: str) -> ThemeOutput:
        """Analyze text to identify themes."""
        if error := self._validate_input(text):
            return ThemeOutput(
                error=error, success=False, language=self._get_language()
            )

        try:
            # Get LLM analysis
            results = await self.chain.ainvoke(text)

            # Process themes
            themes = []
            evidence = {}
            hierarchy = {}

            for theme_data in results.get("themes", []):
                # Create theme
                theme = ThemeInfo(
                    name=theme_data["name"],
                    description=theme_data["description"],
                    confidence=float(theme_data["confidence"]),
                    keywords=theme_data.get("keywords", []),
                    parent_theme=theme_data.get("parent_theme"),
                )

                # Apply domain-specific adjustments
                if domain := theme_data.get("domain"):
                    theme.confidence = self._adjust_confidence(
                        theme.confidence, domain, theme.keywords
                    )

                themes.append(theme)

                # Store evidence
                if theme.name in results.get("evidence", {}):
                    evidence[theme.name] = [
                        ThemeEvidence(**e)
                        for e in results["evidence"][theme.name]
                    ]

                # Build hierarchy
                if theme.parent_theme:
                    hierarchy.setdefault(theme.parent_theme, []).append(
                        theme.name
                    )

            return ThemeOutput(
                themes=themes,
                theme_hierarchy=hierarchy,
                evidence=evidence,
                language=self._get_language(),
                success=True,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return ThemeOutput(
                error=str(e), success=False, language=self._get_language()
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
            else "unknown"
        )

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
