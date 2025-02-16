# src/analyzers/keyword_analyzer.py

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config.manager import ConfigManager
from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel
from src.core.language_processing.base import BaseTextProcessor
from src.core.config import AnalyzerConfig

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import Field

from src.analyzers.base import AnalyzerOutput, TextAnalyzer, TextSection
from src.schemas import KeywordAnalysisResult, KeywordInfo, ThemeContext

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"


@dataclass
class DomainTerms:
    terms: Set[str]
    boost_factor: float = 1.0


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""

    keywords: List[KeywordInfo] = Field(default_factory=list)
    compound_words: List[str] = Field(default_factory=list)
    domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility


class KeywordAnalyzer(TextAnalyzer):
    """Analyzes text to extract keywords with position-aware weighting."""

    # Base domain terms (English)
    DOMAIN_TERMS = {
        DomainType.TECHNICAL: DomainTerms(
            {
                "cloud",
                "infrastructure",
                "platform",
                "kubernetes",
                "deployment",
                "pipeline",
                "integration",
                "monitoring",
                "microservices",
                "api",
                "devops",
                "architecture",
                "latency",
                "throughput",
                "availability",
                "reliability",
            },
            boost_factor=1.2,
        ),
        DomainType.BUSINESS: DomainTerms(
            {
                "revenue",
                "cost",
                "profit",
                "margin",
                "growth",
                "efficiency",
                "performance",
                "optimization",
                "strategy",
                "operations",
                "analytics",
                "metrics",
            },
            boost_factor=1.15,
        ),
    }

    FINNISH_DOMAIN_TERMS = {
        DomainType.TECHNICAL: DomainTerms(
            {
                # Cloud/Infrastructure
                "pilvipalvelu",
                "pilvipohjainen",
                "mikropalvelu",
                "infrastruktuuri",
                "konttiteknologia",
                "skaalautuvuus",
                # Integration/APIs
                "rajapinta",
                "integraatio",
                "käyttöönotto",
                # Performance/Ops
                "monitorointi",
                "vikasietoisuus",
                "kuormantasaus",
                "suorituskyky",
                "automaatio",
                "palvelutaso",
            },
            boost_factor=1.2,
        ),
        DomainType.BUSINESS: DomainTerms(
            {
                # Financial
                "liikevaihto",
                "tuotto",
                "kustannus",
                "markkinaosuus",
                "toistuvaislaskutus",
                "vuosineljännes",
                "investointi",
                # Customer
                "asiakaspysyvyys",
                "asiakashankinta",
                "asiakaskokemus",
                "asiakassegmentti",
                "asiakaskanta",
                # Market
                "markkinaosuus",
                "kilpailuasema",
                "markkina-asema",
                "markkinasegmentti",
                "markkinajohtaja",
            },
            boost_factor=1.15,
        ),
    }

    GENERIC_TERMS = {
        "fi": {
            "prosentti",
            "määrä",
            "osuus",
            "kasvaa",
            "vahvistua",
            "tehostua",
            "parantua",
            "merkittävä",
            "huomattava",
            "tärkeä",
            "kasvu",
            "vahva",
            "heikko",
            "suuri",
            "pieni",
        },
        "en": {
            "percent",
            "amount",
            "increase",
            "improve",
            "significant",
            "important",
            "growth",
            "strong",
            "weak",
            "large",
            "small",
        },
    }

    DEFAULT_WEIGHTS = {
        "statistical": 0.4,
        "llm": 0.6,
        "compound_bonus": 0.2,
        "domain_bonus": 0.15,
        "length_factor": 0.1,
        "generic_penalty": 0.3,
        "position_boost": 0.1,
        "frequency_factor": 0.15,
        "cluster_boost": 0.2,
    }

    # Technical term variations for better matching
    TERM_VARIATIONS = {
        "api": ["apis", "rest-api", "api-driven"],
        "cloud": ["cloud-based", "cloud-native", "multi-cloud"],
        "devops": ["dev-ops", "devsecops"],
        "data": ["dataset", "database", "datastore"],
        "ai": ["artificial-intelligence", "machine-learning", "ml"],
    }

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
    ):
        """Initialize analyzer with structured output support."""
        # Initialize analyzer config if not provided
        if llm is None:
            analyzer_config = AnalyzerConfig()
            llm = create_llm(config=analyzer_config)

            # Merge analyzer config with provided config
            if config is None:
                config = {}
            config = {**analyzer_config.config.get("analysis", {}), **config}

        super().__init__(llm, config)
        self.language_processor = language_processor

        # Initialize components with config
        self.weights = self._initialize_weights(config)
        self.min_confidence = config.get("min_confidence", 0.3)
        self.max_keywords = config.get("max_keywords", 10)

        # Create chain with structured output
        self.chain = self._create_chain()
        logger.debug(
            "KeywordAnalyzer initialized with new configuration system"
        )

    def _initialize_weights(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Initialize and validate weights."""
        weights = {
            "statistical": 0.4,
            "llm": 0.6,
            "compound_bonus": 0.2,
            "domain_bonus": 0.15,
        }
        if "weights" in config:
            weights.update(config["weights"])

        # Validate source weights sum to 1.0
        source_sum = weights["statistical"] + weights["llm"]
        if abs(source_sum - 1.0) > 0.001:
            logger.warning(
                f"Source weights sum to {source_sum}, normalizing..."
            )
            weights["statistical"] /= source_sum
            weights["llm"] /= source_sum

        return weights

    def _initialize_clustering_config(self, config: Optional[Dict]) -> Dict:
        """Initialize clustering configuration."""
        if not config:  # Handle None or empty config
            return {
                "similarity_threshold": 0.85,
                "max_cluster_size": 3,
                "boost_factor": 1.2,
                "domain_bonus": 0.1,
                "min_cluster_size": 2,
                "max_relation_distance": 2,
            }
        return config.get(
            "clustering",
            {
                "similarity_threshold": 0.85,
                "max_cluster_size": 3,
                "boost_factor": 1.2,
                "domain_bonus": 0.1,
                "min_cluster_size": 2,
                "max_relation_distance": 2,
            },
        )

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain with theme context support."""
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a keyword analysis expert. For the given text:
1. Identify specific and meaningful keywords
2. Consider compound words and technical terms
3. Calculate relevance scores
4. Group keywords by domain relevance

{theme_context_prompt}

Focus on keywords that are clearly supported by the text content.
Do not generate generic keywords that could apply to any text."""
            ),
            (
                "human",
                """Analyze this text to identify keywords:
{text}

Language: {language}
{theme_context_str}

Required: Extract keywords with scores, identify compound words, and consider domain relevance.
Base all keywords on actual content and evidence from the text."""
            ),
        ])

        return (
            {
                "text": RunnablePassthrough(),
                "language": lambda _: self._get_language(),
                "theme_context_str": lambda x: self._format_theme_context(x.get("theme_context") if isinstance(x, dict) else None),
                "theme_context_prompt": lambda x: self._get_theme_context_prompt(x.get("theme_context") if isinstance(x, dict) else None),
            }
            | template
            | self.llm.with_structured_output(KeywordOutput)
        )

    def _format_theme_context(self, theme_context: Optional[ThemeContext]) -> str:
        """Format theme context for prompt if available."""
        if not theme_context or not theme_context.main_themes:
            return ""

        context_parts = ["Theme Context:"]
        
        # Add main themes with descriptions
        context_parts.append("\nMain Themes:")
        for theme in theme_context.main_themes:
            desc = theme_context.theme_descriptions.get(theme, "")
            conf = theme_context.theme_confidence.get(theme, 0.0)
            context_parts.append(f"- {theme} (confidence: {conf:.2f}): {desc}")
        
        # Add theme hierarchy if available
        if theme_context.theme_hierarchy:
            context_parts.append("\nTheme Relationships:")
            for parent, children in theme_context.theme_hierarchy.items():
                context_parts.append(f"- {parent} -> {', '.join(children)}")
        
        # Add theme keywords if available
        if theme_context.theme_keywords:
            context_parts.append("\nTheme Keywords:")
            for theme, keywords in theme_context.theme_keywords.items():
                if keywords:
                    context_parts.append(f"- {theme}: {', '.join(keywords)}")
        
        return "\n".join(context_parts)

    def _adjust_score_with_themes(
        self,
        keyword: str,
        base_score: float,
        theme_context: Optional[ThemeContext]
    ) -> float:
        """Adjust keyword score based on theme relevance."""
        if not theme_context or not theme_context.main_themes:
            return base_score

        max_theme_relevance = 0.0
        keyword_lower = keyword.lower()

        # Check direct keyword matches in theme keywords
        for theme, keywords in theme_context.theme_keywords.items():
            if any(kw.lower() in keyword_lower or keyword_lower in kw.lower() for kw in keywords):
                theme_conf = theme_context.theme_confidence.get(theme, 0.0)
                max_theme_relevance = max(max_theme_relevance, theme_conf)

        # Check relevance to theme descriptions
        for theme, desc in theme_context.theme_descriptions.items():
            if keyword_lower in desc.lower():
                theme_conf = theme_context.theme_confidence.get(theme, 0.0)
                max_theme_relevance = max(max_theme_relevance, theme_conf * 0.8)  # Slightly lower weight for description matches

        # Apply theme relevance boost (up to 30% boost for highly relevant keywords)
        theme_boost = 1.0 + (max_theme_relevance * 0.3)
        return min(base_score * theme_boost, 1.0)

    def _get_theme_context_prompt(self, theme_context: Optional[ThemeContext]) -> str:
        """Get theme-specific prompt section based on context availability."""
        if theme_context and theme_context.main_themes:
            return """If theme context is provided:
1. Use themes to guide keyword identification
2. Ensure keywords align with and support the identified themes
3. Consider theme hierarchy in scoring
4. Look for keywords that bridge multiple themes"""
        return ""  # Return empty string if no theme context

    def _calculate_score(
        self,
        word: str,
        base_score: float,
        domain: Optional[str] = None,
        compound_parts: Optional[List[str]] = None,
        theme_context: Optional[ThemeContext] = None,
    ) -> float:
        """Calculate final score with optional theme context consideration."""
        # Start with base statistical/positional score
        score = base_score

        # Apply domain-specific adjustments
        if domain:
            score *= 1.2  # 20% boost for domain-relevant terms

        # Adjust for compound words
        if compound_parts:
            score *= 1.1  # 10% boost for compound words

        # Apply theme-based scoring only if theme context is available
        if theme_context and theme_context.main_themes:
            score = self._adjust_score_with_themes(word, score, theme_context)

        # Ensure score stays in valid range
        return min(max(score, 0.0), 1.0)

    def _create_error_result(self) -> KeywordAnalysisResult:
        """Create error result."""
        return KeywordAnalysisResult(
            keywords=[],
            compound_words=[],
            domain_keywords={},
            language=self._get_language(),
            success=False,
            error="Analysis failed",
        )

    def _handle_error(self, error: str) -> KeywordOutput:
        """Create error output that matches model requirements."""
        return KeywordOutput(
            keywords=[],
            domain_keywords={},
            error=str(error),
            success=False,
            language=self._get_language(),
        )

    def _post_process_llm_output(self, response: Any) -> Dict[str, Any]:
        """Process LLM response and apply user-defined parameters as guidance."""
        try:
            # Parse base response
            parsed = self._parse_response(response)
            
            # Process keywords with user preferences as guidance
            processed_keywords = []
            for kw in parsed.get("keywords", []):
                keyword = kw.get("keyword", "").lower()
                score = kw.get("score", 0.0)
                
                # Skip only exact matches with excluded keywords
                if any(keyword == excluded.lower() for excluded in self.config.excluded_keywords):
                    continue
                    
                # Boost score for predefined keywords to prioritize them
                if any(predefined.lower() in keyword for predefined in self.config.predefined_keywords):
                    score = min(score * 1.2, 1.0)  # 20% boost for predefined keywords
                
                # Include keyword if it's either predefined or has sufficient score
                if score >= self.min_confidence:
                    processed_keywords.append({
                        "keyword": kw.get("keyword"),
                        "score": score,
                        "domain": kw.get("domain"),
                        "compound_parts": kw.get("compound_parts", []),
                        "is_predefined": any(predefined.lower() in keyword for predefined in self.config.predefined_keywords)
                    })
            
            # Sort keywords by score, but ensure predefined keywords get priority
            processed_keywords.sort(key=lambda x: (not x["is_predefined"], -x["score"]))
            
            # Remove the temporary is_predefined field before returning
            for kw in processed_keywords:
                kw.pop("is_predefined", None)
            
            return {
                "keywords": processed_keywords[:self.max_keywords],
                "domain_keywords": parsed.get("domain_keywords", {}),
                "language": parsed.get("language", self._get_language()),
                "success": parsed.get("success", True)
            }
            
        except Exception as e:
            logger.error(f"Error post-processing response: {e}")
            return self._handle_error(str(e))

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract content from responses
            if hasattr(response, "content"):
                data = json.loads(response.content)
            elif isinstance(response, str):
                data = json.loads(response)
            elif isinstance(response, dict):
                data = response
            else:
                return {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "success": True,
                    "language": self._get_language(),
                }

            # Transform keyword format to match Pydantic model
            transformed_keywords = []
            for kw in data.get("keywords", []):
                # Handle both string and dict formats for keywords
                if isinstance(kw, str):
                    transformed = {
                        "keyword": kw,
                        "score": 0.5,
                        "domain": None,
                        "compound_parts": [],
                    }
                else:
                    transformed = {
                        "keyword": kw.get("keyword", kw.get("text", "")),
                        "score": kw.get("score", kw.get("confidence", 0.5)),
                        "domain": kw.get("domain"),
                        "compound_parts": kw.get("compound_parts", []),
                    }
                transformed_keywords.append(transformed)

            return {
                "keywords": transformed_keywords,
                "compound_words": data.get("compound_words", []),
                "domain_keywords": data.get("domain_keywords", {}),
                "language": data.get("language", self._get_language()),
                "success": data.get("success", True)
            }

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "keywords": [],
                "compound_words": [],
                "domain_keywords": {},
                "success": False,
                "language": self._get_language(),
                "error": str(e)
            }

    async def analyze(self, text: str) -> KeywordOutput:
        """Analyze text with proper AIMessage handling."""
        if text is None:
            raise ValueError("Input text cannot be None")

        if not text:
            return KeywordOutput(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language=self._get_language(),
                success=False,
                error="Empty input text",
            )

        try:
            logger.debug("KeywordAnalyzer.analyze: Starting analysis")
            result = await self.chain.ainvoke(text)
            logger.debug(
                f"KeywordAnalyzer.analyze: Chain result type: {type(result)}"
            )
            logger.debug(f"KeywordAnalyzer.analyze: Chain result: {result}")

            # Handle AIMessage from mock LLMs
            if hasattr(result, "content"):
                try:
                    data = json.loads(result.content)
                    logger.debug(
                        f"KeywordAnalyzer.analyze: Parsed JSON data: {data}"
                    )
                    result = KeywordOutput(**data)
                except Exception as e:
                    logger.error(f"Error parsing AIMessage content: {e}")
                    return KeywordOutput(
                        keywords=[],
                        compound_words=[],
                        domain_keywords={},
                        language=self._get_language(),
                        success=False,
                        error=f"Error parsing response: {str(e)}",
                    )

            # Filter keywords by confidence and apply lemmatization
            if getattr(result, "keywords", None):
                processed_keywords = []
                for kw in result.keywords:
                    if kw.score >= self.min_confidence:
                        # Apply lemmatization if language processor is available
                        if self.language_processor:
                            # Handle hyphenated compound words
                            if "-" in kw.keyword:
                                # Split by hyphen, lemmatize parts separately, then rejoin
                                parts = kw.keyword.split("-")
                                lemmatized_parts = [
                                    self.language_processor.get_base_form(part)
                                    for part in parts
                                ]
                                lemmatized_keyword = "-".join(lemmatized_parts)
                            else:
                                # Regular lemmatization for non-hyphenated words
                                lemmatized_keyword = self.language_processor.get_base_form(kw.keyword)
                            
                            kw.keyword = lemmatized_keyword
                            
                            # Handle compound parts similarly
                            if hasattr(kw, 'compound_parts') and kw.compound_parts:
                                processed_parts = []
                                for part in kw.compound_parts:
                                    if "-" in part:
                                        # Handle hyphenated compound parts
                                        subparts = part.split("-")
                                        lemmatized_subparts = [
                                            self.language_processor.get_base_form(subpart)
                                            for subpart in subparts
                                        ]
                                        processed_parts.append("-".join(lemmatized_subparts))
                                    else:
                                        processed_parts.append(
                                            self.language_processor.get_base_form(part)
                                        )
                                kw.compound_parts = processed_parts
                        
                        processed_keywords.append(kw)

                # Limit number of keywords
                result.keywords = processed_keywords[:self.max_keywords]

                # Update domain keywords
                if hasattr(result, "domain_keywords"):
                    result.domain_keywords = self._group_by_domain(
                        result.keywords
                    )

            return result

        except Exception as e:
            logger.error(f"KeywordAnalyzer.analyze: Exception occurred: {e}")
            return KeywordOutput(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language=self._get_language(),
                success=False,
                error=str(e),
            )

    def _group_by_domain(self, keywords: List[KeywordInfo]) -> Dict[str, List[str]]:
        """Group keywords by their domain."""
        domain_groups: Dict[str, List[str]] = defaultdict(list)
        
        for kw in keywords:
            if kw.domain:
                domain_groups[kw.domain].append(kw.keyword)
        
        return dict(domain_groups)