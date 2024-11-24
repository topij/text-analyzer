# src/analyzers/category_analyzer.py
from typing import Any, Dict, List, Optional, Set
import logging
import json
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import TextAnalyzer, AnalyzerOutput
from src.core.language_processing.base import BaseTextProcessor
from src.loaders.models import CategoryConfig
from src.schemas import CategoryMatch, Evidence

logger = logging.getLogger(__name__)


class CategoryEvidence(BaseModel):
    """Enhanced evidence with keyword matching."""

    text: str
    relevance: float
    matched_keywords: List[str] = Field(default_factory=list)
    context: Optional[str] = None


class CategoryOutput(AnalyzerOutput):
    """Enhanced output model for category analysis."""

    categories: List[CategoryMatch] = Field(default_factory=list)
    evidence: Dict[str, List[CategoryEvidence]] = Field(default_factory=dict)
    category_hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    related_themes: Dict[str, List[str]] = Field(default_factory=dict)


class CategoryAnalyzer(TextAnalyzer):
    """Analyzes text to classify it into predefined categories with language support."""

    # Domain-specific boosts for categories
    DOMAIN_BOOSTS = {
        "technical": {
            "keywords": {
                "cloud": 1.2,
                "api": 1.2,
                "infrastructure": 1.2,
                "deployment": 1.1,
                "integration": 1.1,
                "microservices": 1.2,
                "monitoring": 1.1,
                "automation": 1.1,
            },
            "patterns": [
                r"cloud[\-\s]native",
                r"micro[\-\s]services?",
                r"dev[\-\s]?ops",
            ],
        },
        "business": {
            "keywords": {
                "revenue": 1.2,
                "growth": 1.1,
                "market": 1.1,
                "customer": 1.1,
                "sales": 1.1,
                "profit": 1.2,
                "recurring": 1.1,
                "retention": 1.1,
            },
            "patterns": [
                r"market[\-\s]share",
                r"customer[\-\s]acquisition",
                r"recurring[\-\s]revenue",
            ],
        },
    }

    # Finnish domain boosts
    FINNISH_DOMAIN_BOOSTS = {
        "technical": {
            "keywords": {
                "pilvi": 1.2,
                "rajapinta": 1.2,
                "infrastruktuuri": 1.2,
                "käyttöönotto": 1.1,
                "integraatio": 1.1,
                "mikropalvelu": 1.2,
                "monitorointi": 1.1,
                "automaatio": 1.1,
            },
            "patterns": [
                r"pilvi[\-\s]?palvelu",
                r"mikro[\-\s]?palvelu",
                r"dev[\-\s]?ops",
            ],
        },
        "business": {
            "keywords": {
                "liikevaihto": 1.2,
                "kasvu": 1.1,
                "markkina": 1.1,
                "asiakas": 1.1,
                "myynti": 1.1,
                "tuotto": 1.2,
                "toistuva": 1.1,
                "pysyvyys": 1.1,
            },
            "patterns": [
                r"markkina[\-\s]?osuus",
                r"asiakas[\-\s]?hankinta",
                r"toistuva[\-\s]?laskutus",
            ],
        },
    }

    def __init__(
        self,
        categories: Dict[str, CategoryConfig],
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
    ):
        """Initialize category analyzer with language support."""
        super().__init__(llm, config)
        self.categories = categories
        self.language_processor = language_processor
        self.boosts = self._initialize_boosts()
        self.min_confidence = config.get("min_confidence", 0.3)
        self.chain = self._create_chain()

    def _initialize_boosts(self) -> Dict:
        """Initialize language-specific domain boosts."""
        if self.language_processor and self.language_processor.language == "fi":
            return self.FINNISH_DOMAIN_BOOSTS
        return self.DOMAIN_BOOSTS

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a classification expert focusing on technical and business content analysis.
                Analyze the given text and classify it into the provided categories. Consider:
                1. Direct keyword matches
                2. Contextual relevance
                3. Domain-specific terminology
                4. Evidence supporting classification
                
                Return results in this exact JSON format:
                {{
                    "categories": [
                        {{
                            "category": "category_name",
                            "confidence": 0.95,
                            "explanation": "Detailed explanation",
                            "evidence": [
                                {{
                                    "text": "relevant text",
                                    "relevance": 0.9,
                                    "matched_keywords": ["keyword1", "keyword2"],
                                    "context": "surrounding context"
                                }}
                            ],
                            "themes": ["related_theme1", "related_theme2"]
                        }}
                    ],
                    "relationships": {{
                        "category_name": ["related_category1", "related_category2"]
                    }}
                }}""",
                ),
                (
                    "human",
                    """Analyze this text and classify into the following categories:
                Categories: {categories_json}
                
                Text: {text}
                Language: {language}
                
                Guidelines:
                - Minimum confidence: {min_confidence}
                - Consider these key terms: {key_terms}
                - Focus on these domains: {focus_domains}
                """,
                ),
            ]
        )

        return (
            {
                "text": RunnablePassthrough(),
                "categories_json": self._format_categories,
                "language": lambda _: (
                    self.language_processor.language
                    if self.language_processor
                    else "en"
                ),
                "min_confidence": lambda _: self.min_confidence,
                "key_terms": self._get_key_terms,
                "focus_domains": lambda _: self.config.get(
                    "focus_on", ["technical", "business"]
                ),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

    def _format_categories(self, _: Any) -> str:
        """Format categories for prompt."""
        formatted = []
        for name, config in self.categories.items():
            cat_info = {
                "name": name,
                "description": config.description,
                "keywords": config.keywords,
                "threshold": config.threshold,
            }
            formatted.append(cat_info)
        return json.dumps(formatted, indent=2)

    def _get_key_terms(self, text: str) -> str:
        """Get key terms for classification."""
        if not self.language_processor:
            return ""

        try:
            words = self.language_processor.tokenize(text)
            if not words:
                return ""

            # Process words for key terms
            key_terms = []
            for word in words:
                # Get base form
                base = self.language_processor.get_base_form(word)
                if not base or self.language_processor.is_stop_word(base):
                    continue

                # Handle compound words
                if self.language_processor.is_compound_word(word):
                    parts = self.language_processor.get_compound_parts(word)
                    if parts:
                        key_terms.extend(parts)

                key_terms.append(base)

            return ", ".join(set(key_terms))

        except Exception as e:
            logger.error(f"Error getting key terms: {e}")
            return ""

    def _calculate_confidence(
        self, base_confidence: float, category: str, evidence: List[Dict]
    ) -> float:
        """Calculate final confidence score with domain awareness."""
        confidence = base_confidence

        # Get category config
        config = self.categories[category]

        # Get domain and boost config
        domain = "technical" if "technical" in category.lower() else "business"
        domain_boosts = self.boosts.get(domain, {})

        try:
            # Check keyword matches in evidence
            keyword_matches = 0
            total_boost = 0

            for e in evidence:
                matched = e.get("matched_keywords", [])
                for kw in matched:
                    kw_lower = kw.lower()
                    # Apply keyword boosts
                    if boost := domain_boosts["keywords"].get(kw_lower, 1.0):
                        keyword_matches += 1
                        total_boost += boost

                    # Check patterns
                    for pattern in domain_boosts.get("patterns", []):
                        if re.search(pattern, kw_lower):
                            total_boost += 0.1

            # Apply keyword match boost
            if keyword_matches:
                avg_boost = total_boost / keyword_matches
                confidence *= min(1.0 + (avg_boost - 1.0) * 0.5, 1.2)

            # Apply threshold adjustment
            if confidence > config.threshold:
                confidence = min(confidence * 1.1, 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")

        return min(max(confidence, 0.0), 1.0)

    def _get_language(self) -> str:
        """Get current language."""
        return (
            self.language_processor.language
            if self.language_processor
            else "unknown"
        )

    def _process_llm_response(self, response: Dict) -> List[CategoryMatch]:
        """Process LLM response into CategoryMatch objects with debug output."""
        print(f"\nProcessing response: {response}")
        categories = []

        for cat in response.get("categories", []):
            try:
                print(f"\nProcessing category: {cat}")
                # Convert evidence list
                evidence_list = []
                raw_evidence = cat.pop("evidence", [])
                for ev in raw_evidence:
                    if isinstance(ev, dict):
                        evidence_list.append(
                            Evidence(
                                text=ev.get("text", ""),
                                relevance=float(ev.get("relevance", 0.8)),
                            )
                        )
                    elif isinstance(ev, str):
                        evidence_list.append(Evidence(text=ev, relevance=0.8))

                # Create CategoryMatch with mapped fields
                category = CategoryMatch(
                    name=cat.get("name", ""),
                    confidence=float(cat.get("confidence", 0.0)),
                    description=cat.get("description", ""),
                    evidence=evidence_list,
                    themes=cat.get("themes", []),
                )
                print(f"Created category: {category}")
                categories.append(category)

            except Exception as e:
                print(f"Error processing category: {str(e)}")
                continue

        print(f"\nFinal categories: {categories}")
        return categories

    async def analyze(self, text: str) -> CategoryOutput:
        """Analyze text and categorize it."""
        if text is None:
            raise ValueError("Input text cannot be None")

        if not text:
            return CategoryOutput(
                error="Empty input text",
                success=False,
                language=(
                    self.language_processor.language
                    if self.language_processor
                    else "unknown"
                ),
            )

        try:
            language = (
                self.language_processor.language
                if self.language_processor
                else self._detect_language(text)
            )

            # Get LLM analysis
            response = await self.chain.ainvoke(text)
            if not response or not isinstance(response, dict):
                return CategoryOutput(
                    error="Invalid LLM response",
                    success=False,
                    language=language,
                )

            # Process categories
            categories = self._process_llm_response(response)

            return CategoryOutput(
                categories=categories, language=language, success=True
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return CategoryOutput(
                error=str(e),
                success=False,
                language=language if "language" in locals() else "unknown",
            )

    def display_categories(self, results: CategoryOutput) -> None:
        """Display category analysis results."""
        if not hasattr(results, "categories") or not results.categories:
            print("No categories found.")
            return

        print("\nCategory Analysis Results")
        print("=" * 50)

        for category in results.categories:
            print(f"\nCategory: {category.name}")
            print("-" * 20)
            print(f"Confidence: {category.confidence:.2f}")

            if category.description:
                print(f"Description: {category.description}")

            if category.evidence:
                print("\nEvidence:")
                for ev in category.evidence:
                    print(f"- {ev.text} (relevance: {ev.relevance:.2f})")

            if category.themes:
                print(f"\nRelated Themes: {', '.join(category.themes)}")
