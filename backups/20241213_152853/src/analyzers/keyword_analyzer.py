# src/analyzers/keyword_analyzer.py
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.config import AnalyzerConfig
from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel
from src.core.language_processing.base import BaseTextProcessor

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import Field

from src.analyzers.base import AnalyzerOutput, TextAnalyzer, TextSection
from src.schemas import KeywordAnalysisResult, KeywordInfo

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
    compound_words: List[str] = Field(default_factory=list)  # Add this field
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
        config: Optional[Dict] = None,
        language_processor: Optional[BaseTextProcessor] = None,
    ):
        """Initialize analyzer with configuration and language processing.

        Args:
            llm: Optional LLM instance (will create using factory if None)
            config: Optional configuration dictionary
            language_processor: Optional language processor instance
        """
        # Initialize analyzer config if not provided in config dict
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
        self.weights = self._initialize_weights(config)
        self.clustering_config = self._initialize_clustering_config(config)

        # Initialize internal state
        self._frequency_cache = {}
        self._current_text = ""

        # Create processing chain
        self.chain = self._create_chain()

    def _initialize_weights(self, config: Optional[Dict]) -> Dict[str, float]:
        """Initialize and validate weights."""
        weights = self.DEFAULT_WEIGHTS.copy()
        if config and "weights" in config:
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

    # def _initialize_clustering_config(self, config: Optional[Dict]) -> Dict:
    #     """Initialize clustering configuration."""
    #     return config.get(
    #         "clustering",
    #         {
    #             "similarity_threshold": 0.85,
    #             "max_cluster_size": 3,
    #             "boost_factor": 1.2,
    #             "domain_bonus": 0.1,
    #             "min_cluster_size": 2,
    #             "max_relation_distance": 2,
    #         },
    #     )

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
        """Create LangChain processing chain."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a keyword extraction expert specialized in identifying 
            compound terms and technical concepts. Extract keywords with attention to:
            1. Multi-word technical terms
            2. Domain-specific compound phrases
            3. Technical and business terminology
            Return ONLY valid JSON format.""",
                ),
                (
                    "human",
                    """Analyze this text and extract keywords:
            Text: {text}
            Guidelines:
            - Maximum keywords: {max_keywords}
            - Consider these statistical keywords: {statistical_keywords}
            - Identify technical compound terms
            - Classify by domain (technical/business)
            
            Return in this exact format:
            {{
                "keywords": [
                    {{
                        "keyword": "term",
                        "score": 0.95,
                        "domain": "technical",
                        "compound_parts": ["part1", "part2"]
                    }}
                ],
                "compound_phrases": [
                    {{
                        "phrase": "term",
                        "parts": ["part1", "part2"],
                        "domain": "technical"
                    }}
                ]
            }}""",
                ),
            ]
        )

        return (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.config.get("max_keywords", 10),
                "statistical_keywords": lambda x: ", ".join(
                    self._get_statistical_keywords(x)
                ),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

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

    async def analyze(self, text: str) -> KeywordOutput:
        """Analyze text to extract keywords with proper error handling."""
        try:
            # Validate input first
            if text is None:
                raise ValueError("Input text cannot be None")

            # Validate other input conditions
            if error := self._validate_input(text):
                return KeywordOutput(
                    keywords=[],
                    compound_words=[],
                    domain_keywords={},
                    error=error,
                    success=False,
                    language=self._get_language(),
                )

            # Get LLM analysis
            logger.debug(f"Starting keyword analysis for text: {text[:100]}...")
            response = await self.chain.ainvoke(text)

            # Process response
            if response is None:
                return KeywordOutput(
                    keywords=[],
                    compound_words=[],
                    domain_keywords={},
                    error="No response from LLM",
                    success=False,
                    language=self._get_language(),
                )

            # Extract compound words from keywords with compound parts
            compound_words = [
                kw.get("keyword")
                for kw in response.get("keywords", [])
                if kw.get("compound_parts")
            ] or response.get(
                "compound_words", []
            )  # Fallback to explicit compound_words if provided

            # Create output
            return KeywordOutput(
                keywords=response.get("keywords", []),
                compound_words=compound_words,  # Use extracted compound words
                domain_keywords=response.get("domain_keywords", {}),
                language=response.get("language", self._get_language()),
                success=True,
            )

        except ValueError as e:
            # Re-raise ValueError for input validation
            raise
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return KeywordOutput(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                error=str(e),
                success=False,
                language=self._get_language(),
            )

    def _validate_input(self, text: str) -> Optional[str]:
        """Validate input text."""
        if not isinstance(text, str):
            raise ValueError(
                f"Invalid input type: expected str, got {type(text)}"
            )

        text = text.strip()
        if not text:
            return "Empty input text"

        if len(text) < self.config.get("min_keyword_length", 3):
            return "Input text too short for meaningful analysis"

        return None

    def _get_compound_parts(self, word: str) -> Optional[List[str]]:
        """Get compound parts safely."""
        try:
            if not self.language_processor:
                return None

            parts = self.language_processor.get_compound_parts(word)
            if parts:
                logger.debug(f"Identified compound parts for {word}: {parts}")
            return parts
        except Exception as e:
            logger.debug(f"Error getting compound parts for {word}: {e}")
            return None

    async def _analyze_with_llm(
        self, text: str, statistical_keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Get keyword analysis from LLM."""
        try:
            # Use the LangChain chain to get LLM response
            response = await self.chain.ainvoke(text)
            logger.debug(f"Got LLM response for keyword analysis")

            if not isinstance(response, dict) or "keywords" not in response:
                logger.warning("Invalid LLM response format")
                return []

            # Convert response keywords to KeywordInfo objects
            keywords = []
            for kw in response.get("keywords", []):
                if isinstance(kw, dict) and "keyword" in kw:
                    keywords.append(
                        KeywordInfo(
                            keyword=kw["keyword"],
                            score=float(kw.get("score", 0.5)),
                            domain=kw.get("domain"),
                            compound_parts=kw.get("compound_parts"),
                        )
                    )
                    logger.debug(f"Processed LLM keyword: {kw['keyword']}")

            return keywords

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return []

    async def _analyze_internal(self, text: str) -> KeywordOutput:
        """Internal analysis implementation."""
        if error := self._validate_input(text):
            return self._handle_error(error)

        try:
            # Get statistical and LLM keywords
            sections = self._split_text_sections(text)
            statistical_keywords = self._analyze_statistically(text, sections)
            llm_keywords = await self._analyze_with_llm(
                text, statistical_keywords
            )

            # Process and combine results
            all_keywords = self._combine_keywords(
                statistical_keywords, llm_keywords
            )

            # Apply clustering and filtering
            clustered_keywords = self._cluster_keywords(all_keywords)
            final_keywords = self._filter_keywords(clustered_keywords)

            # Group by domain
            domain_keywords = self._group_by_domain(final_keywords)

            return KeywordOutput(
                keywords=final_keywords,
                domain_keywords=domain_keywords,
                language=self._get_language(),
                success=True,
            )

        except Exception as e:
            logger.error(f"Internal analysis failed: {str(e)}", exc_info=True)
            return self._handle_error(str(e))

    def _get_statistical_keywords(self, text: str) -> List[str]:
        """Get keywords for LLM prompt."""
        sections = self._split_text_sections(text)
        keywords = self._analyze_statistically(text, sections)
        return [k.keyword for k in keywords[:5]]  # Top 5 for prompt

    def is_compound_word(self, word: str) -> bool:
        """Check if word is a true compound."""
        parts = self._get_compound_parts(word)
        return bool(parts and len(parts) > 1)

    def _update_combined_keywords(
        self,
        combined: Dict[str, KeywordInfo],
        keyword: KeywordInfo,
        weight: float,
    ) -> None:
        """Update combined keywords with weighted scoring."""
        if keyword.keyword in combined:
            # Take highest weighted score
            existing = combined[keyword.keyword]
            score = max(
                existing.score * self.weights["statistical"],
                keyword.score * weight,
            )
        else:
            score = keyword.score * weight

        # Process compound words
        compound_parts = None
        if self.language_processor:
            if self.language_processor.is_compound_word(keyword.keyword):
                compound_parts = self.language_processor.get_compound_parts(
                    keyword.keyword
                )

        # Calculate final score
        final_score = self._calculate_score(
            keyword.keyword, score, keyword.domain, compound_parts
        )

        combined[keyword.keyword] = KeywordInfo(
            keyword=keyword.keyword,
            score=final_score,
            domain=keyword.domain,
            compound_parts=compound_parts,
        )

    def _process_word(self, word: str) -> Optional[KeywordInfo]:
        """Process word with compound handling."""
        if not self.language_processor:
            return None

        # Get base form
        base_form = self.language_processor.get_base_form(word)
        if not base_form:
            return None

        # Process as potential compound word
        compound_parts = None
        if self.config.get("include_compounds", True):
            if hasattr(self.language_processor, "process_compound_word"):
                compound_parts = self.language_processor.process_compound_word(
                    base_form
                )
                if compound_parts:
                    logger.debug(f"Compound word {base_form}: {compound_parts}")

        # Detect domain
        domain = self._detect_domain(base_form)

        return KeywordInfo(
            keyword=base_form,
            score=0.0,  # Initial score, will be updated
            domain=domain,
            compound_parts=compound_parts,
        )

    def _analyze_statistically(
        self, text: str, sections: List[TextSection]
    ) -> List[KeywordInfo]:
        """Extract keywords using statistical analysis."""
        try:
            if not self.language_processor:
                return []

            # Get word frequencies
            words = self._extract_candidate_words(text)
            word_freq = Counter(words)

            # Calculate scores
            keywords = []
            max_freq = max(word_freq.values()) if word_freq else 1

            for word, freq in word_freq.items():
                if not self._is_valid_keyword(word):
                    continue

                # Calculate base score
                base_score = freq / max_freq
                position_score = self._calculate_position_score(word, sections)

                # Calculate final score
                final_score = self._calculate_score(
                    word,
                    base_score * position_score,
                    self._detect_domain(word),
                    self._get_compound_parts(word),
                )

                if final_score >= self.config.get("min_confidence", 0.1):
                    keywords.append(
                        KeywordInfo(
                            keyword=word,
                            score=final_score,
                            domain=self._detect_domain(word),
                        )
                    )

            return sorted(keywords, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return []

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output into standardized format."""
        logger.debug(f"Processing LLM output type {type(output)}: {output}")

        try:
            if hasattr(output, "content"):
                content = output.content
                logger.debug(f"Extracted content: {content}")

                data = json.loads(content)
                logger.debug(f"Parsed JSON result: {data}")

            elif isinstance(output, dict):
                data = output
            else:
                return {"keywords": [], "compound_phrases": []}

            # Process keywords
            keywords = []
            for kw in data.get("keywords", []):
                if isinstance(kw, dict) and "keyword" in kw:
                    keywords.append(
                        {
                            "keyword": kw["keyword"],
                            "score": float(kw.get("score", 0.5)),
                            "domain": kw.get("domain"),
                            "compound_parts": kw.get("compound_parts"),
                        }
                    )

            return {
                "keywords": keywords,
                "compound_phrases": data.get("compound_phrases", []),
            }

        except Exception as e:
            logger.error(f"Error processing LLM output: {e}")
            return {"keywords": [], "compound_phrases": []}

    # def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
    #     """Process LLM output with logging."""
    #     logger.debug(f"Processing LLM output type {type(output)}: {output}")
    #     try:
    #         content = output.content if hasattr(output, "content") else output
    #         logger.debug(f"Extracted content: {content}")
    #         result = json.loads(content)
    #         logger.debug(f"Parsed JSON result: {result}")
    #         return result
    #     except Exception as e:
    #         logger.error(f"Error processing output: {e}")
    #         raise

    def _calculate_score(
        self,
        keyword: str,
        base_score: float,
        domain: Optional[str] = None,
        compound_parts: Optional[List[str]] = None,
    ) -> float:
        """Calculate final keyword score with improved compound handling."""
        score = base_score
        is_finnish = (
            self.language_processor
            and getattr(self.language_processor, "language", "") == "fi"
        )

        # Domain boost based on language
        if domain:
            terms = (
                self.FINNISH_DOMAIN_TERMS if is_finnish else self.DOMAIN_TERMS
            )
            domain_terms = terms.get(domain)
            if domain_terms and keyword.lower() in domain_terms.terms:
                score *= domain_terms.boost_factor
                logger.debug(
                    f"Applied domain boost ({domain_terms.boost_factor}) to {keyword}"
                )

        # Compound word handling
        if compound_parts and len(compound_parts) > 1:
            # Check if it's a valid compound with proper parts
            if (
                self.language_processor
                and self.language_processor.is_compound_word(keyword)
            ):
                # More conservative compound bonus for Finnish
                if is_finnish:
                    compound_bonus = self.weights["compound_bonus"] * 0.8
                else:
                    compound_bonus = self.weights["compound_bonus"]

                score *= 1.0 + compound_bonus
                logger.debug(f"Applied compound bonus to {keyword}")

        # Technical term boost
        if self.language_processor and hasattr(
            self.language_processor, "is_technical_term"
        ):
            if self.language_processor.is_technical_term(keyword):
                score *= 1.1
                logger.debug(f"Applied technical term boost to {keyword}")

        return min(score, 1.0)

    def _detect_domain(self, word: str) -> Optional[str]:
        """Detect domain for a keyword."""
        word_lower = word.lower()

        # Check direct matches
        for domain, terms in self.DOMAIN_TERMS.items():
            if word_lower in terms.terms:
                return domain

        # Check variations
        for domain, terms in self.DOMAIN_TERMS.items():
            for base_term in terms.terms:
                if base_term in self.TERM_VARIATIONS:
                    if word_lower in self.TERM_VARIATIONS[base_term]:
                        return domain

        return None

    # def _get_compound_parts(self, word: str) -> Optional[List[str]]:
    #     """Get compound word parts with strict Finnish compound rules."""
    #     if not self.language_processor:
    #         return None

    #     if "-" in word:
    #         parts = [p.strip() for p in word.split("-") if p.strip()]
    #         return parts if len(parts) > 1 else None

    #     if self.language_processor.language == "fi" and hasattr(
    #         self.language_processor, "voikko"
    #     ):
    #         # Check if it's just a generic term
    #         if word.lower() in self.GENERIC_TERMS.get(
    #             self.language_processor.language, set()
    #         ):
    #             return None

    #         try:
    #             analyses = self.language_processor.voikko.analyze(word)
    #             if not analyses:
    #                 return None

    #             for analysis in analyses:
    #                 structure = analysis.get("STRUCTURE", "")
    #                 wordbases = analysis.get("WORDBASES", "")

    #                 # Skip if not a compound structure
    #                 if "=" not in structure[1:]:
    #                     continue

    #                 # Skip derivational forms
    #                 if "+" not in wordbases or "+(" in wordbases:
    #                     continue

    #                 parts = []
    #                 for part in wordbases.split("+"):
    #                     if "(" in part and not part.startswith("+"):
    #                         base = part.split("(")[0].strip()
    #                         if len(base) > 2:
    #                             parts.append(base)

    #                 if len(parts) > 1:
    #                     return parts

    #             return None

    #         except Exception as e:
    #             logger.debug(f"Voikko analysis failed for {word}: {e}")
    #             return None

    #     elif self.language_processor.language == "en":
    #         # English handling remains the same
    #         patterns = {
    #             "base": ["cloud", "data", "web", "api", "micro", "dev"],
    #             "suffix": [
    #                 "based",
    #                 "driven",
    #                 "oriented",
    #                 "aware",
    #                 "ready",
    #                 "ops",
    #             ],
    #         }
    #         word_lower = word.lower()

    #         for base in patterns["base"]:
    #             for suffix in patterns["suffix"]:
    #                 if (
    #                     word_lower == f"{base}{suffix}"
    #                     or word_lower == f"{base}-{suffix}"
    #                 ):
    #                     return [base, suffix]

    #         if " " in word:
    #             parts = word.split()
    #             return parts if len(parts) > 1 else None

    #     return None

    def _combine_keywords(
        self, statistical: List[KeywordInfo], llm: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Combine and deduplicate keywords from different sources."""
        combined = {}

        # Process statistical keywords
        for kw in statistical:
            self._update_combined_keywords(
                combined, kw, self.weights["statistical"]
            )

        # Process LLM keywords
        for kw in llm:
            self._update_combined_keywords(combined, kw, self.weights["llm"])

        return list(combined.values())

    def _group_by_domain(
        self, keywords: List[KeywordInfo]
    ) -> Dict[str, List[str]]:
        """Group keywords by their domains."""
        domains: Dict[str, List[str]] = defaultdict(list)
        for kw in keywords:
            if kw.domain:
                domains[kw.domain].append(kw.keyword)
        return dict(domains)

    def _get_language(self) -> str:
        """Get current language."""
        return (
            self.language_processor.language
            if self.language_processor
            else "unknown"
        )

    async def analyze_batch(
        self, texts: List[str], batch_size: int = 3, timeout: float = 30.0
    ) -> List[KeywordAnalysisResult]:
        """Process multiple texts with controlled concurrency."""
        import asyncio

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts
            batch = texts[i : i + batch_size]
            try:
                tasks = [self.analyze(text) for text in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add error results for failed batch
                error_results = [self._create_error_result() for _ in batch]
                results.extend(error_results)

        return results

    def _split_text_sections(
        self, text: str, section_size: int = 200
    ) -> List[TextSection]:
        """Split text into weighted sections for analysis."""
        sections = []
        text = text.strip()

        # Handle short texts
        if len(text) <= section_size:
            return [TextSection(text, 0, len(text), 1.0)]

        # Split into sections with position-based weights
        for i in range(0, len(text), section_size):
            section_text = text[i : i + section_size]

            # Higher weights for start and end sections
            weight = (
                1.2 if i == 0 else 1.1 if i + section_size >= len(text) else 1.0
            )

            sections.append(
                TextSection(
                    content=section_text,
                    start=i,
                    end=i + len(section_text),
                    weight=weight,
                )
            )

        return sections

    def _calculate_position_score(
        self, word: str, sections: List[TextSection]
    ) -> float:
        """Calculate position-based score for a word."""
        if not sections:
            return 1.0

        weights = []
        word_lower = word.lower()

        for section in sections:
            if word_lower in section.content.lower():
                weights.append(section.weight)

        return sum(weights) / len(weights) if weights else 1.0

    def _get_keyword_frequency(self, keyword: str) -> int:
        """Get frequency of keyword in the current text."""
        if keyword not in self._frequency_cache:
            if self.language_processor:
                # Use language processor for accurate counting
                base_form = self.language_processor.get_base_form(keyword)
                tokens = self.language_processor.tokenize(self._current_text)

                # Count base form matches
                freq = sum(
                    1
                    for token in tokens
                    if self.language_processor.get_base_form(token) == base_form
                )

                # Add matches for original form
                freq += self._current_text.lower().count(keyword.lower())

                # Check compound variations if applicable
                compound_parts = self._get_compound_parts(keyword)
                if compound_parts:
                    part_freq = sum(
                        self._current_text.lower().count(part.lower())
                        for part in compound_parts
                    )
                    freq = max(freq, part_freq)
            else:
                # Simple case-insensitive count
                freq = self._current_text.lower().count(keyword.lower())

            self._frequency_cache[keyword] = freq

        return self._frequency_cache[keyword]

    def _extract_candidate_words(self, text: str) -> List[str]:
        """Extract candidate words for keyword analysis."""
        if not self.language_processor:
            return []

        candidates = []
        tokens = self.language_processor.tokenize(text)

        for token in tokens:
            base_form = self.language_processor.get_base_form(token)
            if self._is_valid_keyword(base_form):
                candidates.append(base_form)

        return candidates

    def _is_valid_keyword(self, word: str) -> bool:
        """Check if word is a valid keyword candidate."""
        if not word or len(word) < self.config.get("min_keyword_length", 3):
            return False

        if self.language_processor:
            if self.language_processor.is_stop_word(word.lower()):
                return False

            # Additional checks can be added here
            if any(c.isdigit() for c in word):
                return False

            if not any(c.isalnum() for c in word):
                return False

        return True

    def _cluster_keywords(
        self, keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Group similar keywords into clusters."""
        if not keywords:
            return []

        clusters = defaultdict(list)
        processed = set()

        # Sort by score for better clustering
        sorted_keywords = sorted(keywords, key=lambda x: x.score, reverse=True)

        for kw in sorted_keywords:
            if kw.keyword in processed:
                continue

            cluster = []
            base_form = (
                self.language_processor.get_base_form(kw.keyword)
                if self.language_processor
                else kw.keyword.lower()
            )

            # Find related keywords
            for other in sorted_keywords:
                if other.keyword in processed:
                    continue

                if self._are_keywords_related(
                    kw.keyword, other.keyword, base_form
                ):
                    cluster.append(other)
                    processed.add(other.keyword)

            if cluster:
                clusters[base_form] = cluster

        # Adjust scores based on clustering
        return self._adjust_cluster_scores(clusters)

    def _are_keywords_related(self, kw1: str, kw2: str, base_form: str) -> bool:
        """Check if keywords are related."""
        # Exact match after normalization
        if kw1.lower() == kw2.lower():
            return True

        # Base form match
        other_base = (
            self.language_processor.get_base_form(kw2)
            if self.language_processor
            else kw2.lower()
        )
        if base_form == other_base:
            return True

        # Check for compound word relationships
        if self.language_processor:
            kw1_parts = set(self._get_compound_parts(kw1) or [])
            kw2_parts = set(self._get_compound_parts(kw2) or [])
            if kw1_parts and kw2_parts and (kw1_parts & kw2_parts):
                return True

        # Technical term variations
        for base_term, variations in self.TERM_VARIATIONS.items():
            if (kw1.lower() in variations or kw1.lower() == base_term) and (
                kw2.lower() in variations or kw2.lower() == base_term
            ):
                return True

        return False

    def _adjust_cluster_scores(
        self, clusters: Dict[str, List[KeywordInfo]]
    ) -> List[KeywordInfo]:
        """Adjust scores based on cluster relationships."""
        result = []
        boost_factor = self.clustering_config["boost_factor"]
        domain_bonus = self.clustering_config["domain_bonus"]

        for base_form, cluster in clusters.items():
            # Sort cluster by score
            cluster.sort(key=lambda x: x.score, reverse=True)

            # Get primary domain from highest scoring keywords
            domains = Counter(k.domain for k in cluster[:3] if k.domain)
            primary_domain = domains.most_common(1)[0][0] if domains else None

            # Process each keyword in cluster
            for i, kw in enumerate(cluster):
                # Calculate position-based boost
                position_boost = (
                    boost_factor
                    if i == 0
                    else 1.0 + (boost_factor - 1.0) * (1.0 - (i / len(cluster)))
                )

                # Calculate domain boost
                domain_boost = (
                    1.0 + domain_bonus if kw.domain == primary_domain else 1.0
                )

                # Apply compound boost if applicable
                compound_boost = (
                    1.0 + 0.1 * len(kw.compound_parts)
                    if getattr(kw, "compound_parts", None)
                    else 1.0
                )

                # Calculate final score
                final_score = min(
                    kw.score * position_boost * domain_boost * compound_boost,
                    1.0,
                )

                result.append(
                    KeywordInfo(
                        keyword=kw.keyword,
                        score=final_score,
                        domain=kw.domain,
                        compound_parts=getattr(kw, "compound_parts", None),
                    )
                )

        return sorted(result, key=lambda x: x.score, reverse=True)

    def _evaluate_quality(self, word: str) -> float:
        """Enhanced quality evaluation with better business term handling."""
        if not self.language_processor:
            return 1.0

        word_lower = word.lower()
        quality_score = 1.0

        # Check if it's a business term
        if hasattr(self.language_processor, "BUSINESS_TERMS"):
            if word_lower in self.language_processor.BUSINESS_TERMS:
                return 1.0  # Always give full score to business terms

        # Get base form and check for None (filtered verbs)
        base_form = self.language_processor.get_base_form(word)
        if base_form is None:
            return 0.0  # Filter out verbs completely

        # Process other terms
        pos_tag = self.language_processor.get_pos_tag(word)
        if pos_tag:
            if pos_tag.startswith("VB"):
                return 0.0  # Filter out remaining verbs

            pos_penalties = {
                "JJ": 0.7,  # Adjective
                "RB": 0.3,  # Adverb
            }
            if pos_tag[:2] in pos_penalties:
                quality_score *= pos_penalties[pos_tag[:2]]

        # Compound word handling
        if self.language_processor.is_compound_word(word):
            parts = self.language_processor.get_compound_parts(word)
            if parts:
                # Check if any part is a business term
                if any(
                    part in self.language_processor.BUSINESS_TERMS
                    for part in parts
                ):
                    quality_score *= 1.2

        return quality_score

    def _filter_keywords(
        self, keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Enhanced keyword filtering."""
        filtered = []
        seen_bases = set()

        for kw in keywords:
            # Get base form
            base = self.language_processor.get_base_form(kw.keyword.lower())

            # Skip filtered words (like verbs)
            if base is None:
                continue

            if base in seen_bases:
                continue

            # Business terms get priority
            if (
                hasattr(self.language_processor, "BUSINESS_TERMS")
                and kw.keyword.lower() in self.language_processor.BUSINESS_TERMS
            ):
                filtered.append(kw)
                seen_bases.add(base)
                continue

            # Quality evaluation for other terms
            quality = self._evaluate_quality(kw.keyword)
            if quality > 0.0:  # Any word that wasn't filtered
                kw.score *= quality
                filtered.append(kw)
                seen_bases.add(base)

        # Sort by score and apply limit
        max_keywords = self.config.get("max_keywords", 10)
        return sorted(filtered, key=lambda x: x.score, reverse=True)[
            :max_keywords
        ]
