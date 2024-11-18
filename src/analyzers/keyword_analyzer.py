# src/analyzers/keyword_analyzer.py

import logging
from typing import Any, Dict, List, Optional, Set
from collections import Counter
from math import log
import asyncio

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import (
    TextAnalyzer,
    AnalyzerOutput,
    TextSection,
)  # Changed from BaseAnalyzer
from src.schemas import KeywordAnalysisResult, KeywordInfo

from pydantic import Field

logger = logging.getLogger(__name__)


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""

    keywords: List[KeywordInfo] = Field(default_factory=list)
    domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)


class KeywordAnalyzer(TextAnalyzer):
    """Analyzes text to extract keywords with position-aware weighting."""

    def __init__(
        self,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor=None,
    ):
        """Initialize KeywordAnalyzer with complete configuration."""
        super().__init__(llm, config)
        self.language_processor = language_processor
        self.llm = llm

        # Define all weights with clear purposes
        self.weights = {
            # Source weights (sum should be 1.0)
            "statistical": 0.4,
            "llm": 0.6,
            # Bonus/penalty factors
            "compound_bonus": 0.2,  # Bonus for compound words
            "domain_bonus": 0.15,  # Bonus for domain-specific terms
            "length_factor": 0.1,  # Bonus for multi-word terms
            "generic_penalty": 0.3,  # Penalty for common/generic terms
            # Additional weights for fine-tuning
            "technical_bonus": 0.25,  # Extra boost for technical terms
            "position_boost": 0.1,  # Boost for terms at start/end
            "frequency_factor": 0.15,  # Impact of term frequency
            "cluster_boost": 0.2,  # Boost for terms in strong clusters
        }

        # Update weights from config if provided
        if config and "weights" in config:
            self.weights.update(config["weights"])

        # Validate source weights sum to 1.0
        source_sum = self.weights["statistical"] + self.weights["llm"]
        if abs(source_sum - 1.0) > 0.001:
            logger.warning(
                f"Source weights sum to {source_sum}, normalizing..."
            )
            self.weights["statistical"] /= source_sum
            self.weights["llm"] /= source_sum

        # Define generic terms to be filtered or penalized
        self.generic_terms = {
            # Common verbs
            "reduce",
            "increase",
            "decrease",
            "improve",
            "change",
            "update",
            "implement",
            "use",
            "make",
            "create",
            "modify",
            "perform",
            # Common adjectives
            "new",
            "current",
            "existing",
            "previous",
            "next",
            "good",
            "better",
            "best",
            "high",
            "low",
            "various",
            # Common nouns
            "project",
            "system",
            "process",
            "department",
            "part",
            "type",
            "kind",
            "way",
            "thing",
            "item",
            # Common business words
            "meeting",
            "report",
            "update",
            "status",
            "progress",
        }

        # Domain-specific terms that override generic penalties
        self.domain_specific_terms = {
            "technical": {
                # Development terms
                "implementation",
                "deployment",
                "integration",
                "architecture",
                "framework",
                "platform",
                # Infrastructure terms
                "system",
                "server",
                "network",
                "database",
                "cloud",
                "container",
                "instance",
                # Process terms
                "pipeline",
                "workflow",
                "automation",
                "monitoring",
                "logging",
                "testing",
            },
            "business": {
                # Financial terms
                "cost",
                "revenue",
                "growth",
                "profit",
                "margin",
                "investment",
                "budget",
                "forecast",
                # Strategic terms
                "strategy",
                "initiative",
                "objective",
                "goal",
                "market",
                "customer",
                "competitor",
                # Process terms
                "process",
                "optimization",
                "efficiency",
                "performance",
                "metrics",
                "analytics",
            },
        }

        # Clustering configuration
        self.clustering_config = config.get(
            "clustering",
            {
                "similarity_threshold": 0.85,
                "max_cluster_size": 3,
                "boost_factor": 1.2,
                "domain_bonus": 0.1,  # Additional boost for same-domain clusters
                "min_cluster_size": 2,  # Minimum size for cluster formation
                "max_relation_distance": 2,  # Maximum steps for term relationship
            },
        )

        # Domain configuration
        self.domain_keywords = config.get("domain_keywords", {})

        # Technical term variations for clustering
        self.term_variations = {
            "api": ["apis", "rest-api", "api-driven", "api-first", "api-based"],
            "cloud": [
                "cloud-based",
                "cloud-native",
                "multi-cloud",
                "cloud-enabled",
            ],
            "devops": [
                "dev-ops",
                "devsecops",
                "devops-driven",
                "devops-enabled",
            ],
            "data": [
                "dataset",
                "database",
                "datastore",
                "data-driven",
                "big-data",
            ],
            "micro": ["microservice", "micro-service", "microservices-based"],
            "ai": [
                "artificial-intelligence",
                "machine-learning",
                "ml",
                "deep-learning",
            ],
        }
        self._frequency_cache = {}
        self._current_text = ""

        # Initialize chain
        self.chain = self._create_chain()
        logger.debug(f"Initialized with weights: {self.weights}")
        logger.debug(f"Initialized with config: {self.config}")

    def _is_technical_variation(self, term: str) -> bool:
        """Check if term is a technical variation."""
        term_lower = term.lower()
        return any(
            term_lower in variations or term_lower == base
            for base, variations in self.term_variations.items()
        )

    def _get_technical_base(self, term: str) -> Optional[str]:
        """Get base form of technical term if it exists."""
        term_lower = term.lower()
        for base, variations in self.term_variations.items():
            if term_lower in variations or term_lower == base:
                return base

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a keyword extraction expert. Extract important keywords and phrases from text.
                Focus on: technical terms, domain-specific vocabulary, important concepts, named entities, and compound phrases.
                Return ONLY valid JSON with keywords and optional domain groupings.""",
                ),
                (
                    "human",
                    """Analyze this text and extract keywords:

                Text: {text}
                
                Guidelines:
                - Maximum keywords: {max_keywords}
                - Consider statistical keywords: {statistical_keywords}
                - Minimum length: {min_length} characters
                - Focus area: {focus_area}
                
                Return the results in this exact JSON format:
                {{"keywords": [
                    {{"keyword": "example term", "score": 0.95, "domain": "technical"}},
                    {{"keyword": "another term", "score": 0.85, "domain": "business"}}
                ],
                "domain_keywords": {{
                    "technical": ["term1", "term2"],
                    "business": ["term3", "term4"]
                }}}}""",
                ),
            ]
        )

        chain = (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.config.get("max_keywords", 10),
                "statistical_keywords": lambda x: ", ".join(
                    self._get_statistical_keywords(x)
                ),
                "min_length": lambda _: self.config.get(
                    "min_keyword_length", 3
                ),
                "focus_area": lambda _: self.config.get(
                    "focus_on", "general topics"
                ),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    def _split_text_sections(
        self, text: str, section_size: int = 200
    ) -> List[TextSection]:
        """Split text into weighted sections.

        Args:
            text: Input text
            section_size: Target size for each section

        Returns:
            List of TextSection objects with position-based weights
        """
        sections = []
        text = text.strip()

        # Handle short texts
        if len(text) <= section_size:
            return [TextSection(text, 0, len(text), 1.0)]

        # Split into sections with position-based weights
        for i in range(0, len(text), section_size):
            section_text = text[i : i + section_size]

            # Higher weights for start and end sections
            if i == 0:  # First section
                weight = 1.2
            elif i + section_size >= len(text):  # Last section
                weight = 1.1
            else:  # Middle sections
                weight = 1.0

            sections.append(
                TextSection(
                    content=section_text,
                    start=i,
                    end=i + len(section_text),
                    weight=weight,
                )
            )

        return sections

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output specifically for keyword analysis.

        Args:
            output: Raw LLM output

        Returns:
            Dict[str, Any]: Processed keyword output
        """
        # First use base processing to get JSON
        data = super()._post_process_llm_output(output)
        if not data:
            return {"keywords": [], "domain_keywords": {}}

        try:
            # Process keywords with validation
            keywords = []
            for kw in data.get("keywords", []):
                if isinstance(kw, dict) and "keyword" in kw:
                    keywords.append(
                        {
                            "keyword": kw["keyword"],
                            "score": float(kw.get("score", 0.5)),
                            "domain": kw.get("domain"),
                        }
                    )

            return {
                "keywords": keywords,
                "domain_keywords": data.get("domain_keywords", {}),
            }

        except Exception as e:
            self.logger.error(f"Error processing keyword output: {e}")
            return {"keywords": [], "domain_keywords": {}}

    def _analyze_statistically(
        self, text: str, sections: Optional[List[TextSection]] = None
    ) -> List[KeywordInfo]:
        """Extract keywords using statistical analysis."""
        try:
            if not self.language_processor:
                return []

            # Extract word frequencies
            words = self._extract_candidate_words(text)
            word_freq = Counter(words)

            # Calculate max frequency for normalization
            max_freq = max(word_freq.values()) if word_freq else 1

            # Process each word
            keywords = []
            for word, freq in word_freq.items():
                if not self._is_valid_keyword(word):
                    continue

                # Calculate base statistical score
                base_score = freq / max_freq

                # Calculate position score (1.0 if no sections provided)
                position_score = 1.0
                if sections:
                    # Find all occurrences and their weights
                    weights = []
                    for section in sections:
                        if word.lower() in section.content.lower():
                            weights.append(section.weight)
                    if weights:
                        position_score = sum(weights) / len(weights)

                # Calculate final score
                final_score = min(base_score * position_score, 1.0)

                if final_score >= self.config.get("min_confidence", 0.1):
                    keywords.append(
                        KeywordInfo(
                            keyword=word,
                            score=final_score,
                            domain=self._detect_domain(word),
                        )
                    )

            return sorted(keywords, key=lambda x: x.score, reverse=True)[
                : self.config.get("max_keywords", 10)
            ]

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return []

    async def analyze(self, text: str) -> KeywordAnalysisResult:
        """Analyze text to extract keywords."""
        logger.debug("Starting keyword analysis")
        try:
            # Get internal analysis results
            logger.debug(f"Analyzing text of length {len(text)}")
            output = await self._analyze_internal(text)

            result = KeywordAnalysisResult(
                keywords=[k for k in output.keywords],
                compound_words=[],  # Fill if needed
                domain_keywords=output.domain_keywords,
                language=(
                    self.language_processor.language
                    if self.language_processor
                    else "unknown"
                ),
                success=True,
            )
            logger.debug(
                f"Analysis complete. Found {len(result.keywords)} keywords"
            )
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return KeywordAnalysisResult(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language="unknown",
                success=False,
                error=str(e),
            )

    def analyze_statistically(
        self, text: str, sections: List[TextSection]
    ) -> List[KeywordInfo]:
        """Extract keywords using statistical analysis with position weighting."""
        keywords = []

        # Get word frequencies
        words = self._extract_candidate_words(text)
        word_freq = Counter(words)

        # Calculate max frequency for normalization
        max_freq = max(word_freq.values()) if word_freq else 1

        for word, freq in word_freq.items():
            if not self._is_valid_keyword(word):
                continue

            # Calculate base statistical score
            base_score = freq / max_freq

            # Apply position weighting
            position_score = self._calculate_position_score(word, sections)
            final_score = min(base_score * position_score, 1.0)  # Cap at 1.0

            if final_score >= self.config.get("min_confidence", 0.1):
                keywords.append(
                    KeywordInfo(
                        keyword=word,
                        score=final_score,
                        domain=self._detect_domain(word),
                    )
                )

        return sorted(keywords, key=lambda x: x.score, reverse=True)[
            : self.config.get("max_keywords", 10)
        ]

    def _get_keyword_frequency(self, keyword: str) -> int:
        """Get frequency of keyword in the analyzed text."""
        if keyword not in self._frequency_cache:
            # Use language processor to get normalized frequency if available
            if self.language_processor:
                tokens = self.language_processor.tokenize(self._current_text)
                base_form = self.language_processor.get_base_form(keyword)

                # Count occurrences of base form
                freq = sum(
                    1
                    for token in tokens
                    if self.language_processor.get_base_form(token) == base_form
                )

                # Count occurrences of original form
                freq += self._current_text.lower().count(keyword.lower())

                # Get compound word variations if applicable
                if self.language_processor.is_compound_word(keyword):
                    compound_parts = self.language_processor.get_compound_parts(
                        keyword
                    )
                    if compound_parts:
                        # Add partial matches for compound words
                        part_freq = sum(
                            self._current_text.lower().count(part.lower())
                            for part in compound_parts
                        )
                        freq = max(freq, part_freq)
            else:
                # Simple case insensitive count if no language processor
                freq = self._current_text.lower().count(keyword.lower())

            self._frequency_cache[keyword] = freq

        return self._frequency_cache[keyword]

    def _extract_candidate_words(self, text: str) -> List[str]:
        """Extract candidate words for keyword analysis."""
        if not self.language_processor:
            return []

        # Tokenize and process
        tokens = self.language_processor.tokenize(text)
        candidates = []

        for token in tokens:
            # Get base form
            base_form = self.language_processor.get_base_form(token)

            # Check if word should be kept
            if self.language_processor.should_keep_word(base_form):
                candidates.append(base_form)

        return candidates

    def _is_valid_keyword(self, word: str) -> bool:
        """Check if word is a valid keyword candidate."""
        if not word:
            return False

        # Length check
        min_length = self.config.get("min_keyword_length", 3)
        if len(word) < min_length:
            return False

        # Stop word check
        if self.language_processor and self.language_processor.is_stop_word(
            word
        ):
            return False

        return True

    def _detect_domain(self, word: str) -> Optional[str]:
        """Detect domain for a keyword."""
        # Check each domain's keywords
        for domain, keywords in self.domain_keywords.items():
            if word in keywords:
                return domain

        # Check compound words
        if self.language_processor:
            try:
                if self.language_processor.is_compound_word(word):
                    parts = self.language_processor.get_compound_parts(word)
                    for part in parts:
                        for domain, keywords in self.domain_keywords.items():
                            if part in keywords:
                                return domain
            except Exception as e:
                logger.debug(f"Compound word check failed for {word}: {e}")

        return None

    def _get_statistical_keywords(self, text: str) -> List[str]:
        """Get statistical keywords for LLM prompt."""
        sections = self._split_text_sections(text)
        keywords = self._analyze_statistically(text, sections)
        return [k.keyword for k in keywords[:5]]  # Top 5 for the prompt

    async def _analyze_with_llm(
        self, text: str, statistical_keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Get keyword analysis from LLM."""
        if not self.llm:
            return []

        try:
            # Use the LangChain chain to get LLM response
            response = await self.chain.ainvoke(text)

            if isinstance(response, dict) and "keywords" in response:
                return [
                    KeywordInfo(**kw) if isinstance(kw, dict) else kw
                    for kw in response["keywords"]
                ]
            return []

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return []

    def _process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process raw LLM output."""
        try:
            # Handle different output types
            if hasattr(output, "content"):
                content = output.content
            elif isinstance(output, str):
                content = output
            elif isinstance(output, dict):
                return output
            else:
                return {"keywords": []}

            # Parse JSON content
            import json

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"keywords": []}

        except Exception as e:
            logger.error(f"Error processing LLM output: {e}")
            return {"keywords": []}

    def _cluster_keywords(
        self, keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Group similar keywords into clusters and adjust scores."""
        if not keywords:
            return []

        clusters: Dict[str, List[KeywordInfo]] = {}
        processed_keywords = set()

        for kw in keywords:
            if kw.keyword in processed_keywords:
                continue

            cluster = []
            base_form = (
                self.language_processor.get_base_form(kw.keyword)
                if self.language_processor
                else kw.keyword.lower()
            )

            # Find related keywords
            for other in keywords:
                if other.keyword in processed_keywords:
                    continue

                if self._are_keywords_related(
                    kw.keyword, other.keyword, base_form
                ):
                    cluster.append(other)
                    processed_keywords.add(other.keyword)

            if cluster:
                clusters[base_form] = cluster

        # Adjust scores based on clusters
        return self._adjust_cluster_scores(clusters)

    def _are_keywords_related(self, kw1: str, kw2: str, base_form: str) -> bool:
        """Check if keywords are related using multiple methods."""
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

        # Common substring (for compound words)
        if len(kw1) > 3 and len(kw2) > 3:
            if kw1.lower() in kw2.lower() or kw2.lower() in kw1.lower():
                return True

        # Check domain relationship
        kw1_domain = self._detect_domain(kw1)
        kw2_domain = self._detect_domain(kw2)
        if kw1_domain and kw1_domain == kw2_domain:
            if self._check_domain_relationship(kw1, kw2, kw1_domain):
                return True

        return False

    def _check_domain_relationship(
        self, kw1: str, kw2: str, domain: str
    ) -> bool:
        """Check if keywords are related within their domain."""
        # Get domain-specific relationships
        domain_pairs = {
            "technical": [
                ("ai", "artificial intelligence"),
                ("ml", "machine learning"),
                ("api", "interface"),
            ],
            "business": [
                ("roi", "return on investment"),
                ("kpi", "key performance indicator"),
                ("b2b", "business to business"),
            ],
        }

        pairs = domain_pairs.get(domain, [])
        kw1_lower = kw1.lower()
        kw2_lower = kw2.lower()

        return any((kw1_lower in pair and kw2_lower in pair) for pair in pairs)

    def _adjust_cluster_scores(
        self, clusters: Dict[str, List[KeywordInfo]]
    ) -> List[KeywordInfo]:
        """Adjust scores based on cluster relationships."""
        boost_factor = self.clustering_config.get("boost_factor", 1.2)
        result = []

        for base_form, cluster in clusters.items():
            if not cluster:
                continue

            # Sort cluster by base score
            cluster.sort(key=lambda x: x.score, reverse=True)

            # Boost scores based on cluster size and position
            for i, kw in enumerate(cluster):
                # Primary term gets full boost
                if i == 0:
                    boost = boost_factor
                # Related terms get diminishing boost
                else:
                    boost = 1.0 + (boost_factor - 1.0) * (
                        1.0 - (i / len(cluster))
                    )

                # Create new keyword with adjusted score
                result.append(
                    KeywordInfo(
                        keyword=kw.keyword,
                        score=min(kw.score * boost, 1.0),
                        domain=kw.domain,
                    )  # Cap at 1.0
                )

        return sorted(result, key=lambda x: x.score, reverse=True)

    def _calculate_confidence_score(
        self,
        word: str,
        base_score: float,
        position_score: float,
        domain: Optional[str] = None,
        compound_parts: Optional[List[str]] = None,
        frequency: Optional[int] = None,
    ) -> float:
        """Calculate confidence score using all weight factors."""
        logger.debug(
            f"Calculating score for '{word}' with base score {base_score}"
        )

        score = base_score * position_score
        word_lower = word.lower()

        # Technical/business term handling
        technical_terms = {
            "api",
            "cloud",
            "deployment",
            "pipeline",
            "devops",
            "kubernetes",
            "docker",
            "microservices",
            "scalability",
            "infrastructure",
        }
        business_terms = {
            "revenue",
            "cost",
            "project",
            "strategy",
            "market",
            "business",
            "stakeholder",
            "roi",
            "optimization",
            "efficiency",
        }

        # Domain-specific handling
        is_domain_specific = False
        if domain:
            if domain == "technical" and word_lower in technical_terms:
                is_domain_specific = True
                score *= 1.2  # Higher boost for core technical terms
            elif domain == "business" and word_lower in business_terms:
                is_domain_specific = True
                score *= 1.15  # Standard boost for business terms
            else:
                # Apply regular domain boost
                score *= 1.0 + self.weights["domain_bonus"]
            logger.debug(f"After domain boost: {score}")

        # Generic term penalty if not domain-specific
        if not is_domain_specific and word_lower in self.generic_terms:
            penalty = 1.0 - self.weights["generic_penalty"]
            score *= penalty
            logger.debug(f"After generic penalty ({penalty}): {score}")

        # Compound word handling with context
        if compound_parts and len(compound_parts) > 1:
            compound_score = 1.0
            # Check if compound parts are technical/business terms
            if any(part.lower() in technical_terms for part in compound_parts):
                compound_score *= 1.15
            if any(part.lower() in business_terms for part in compound_parts):
                compound_score *= 1.1
            # Apply compound bonus based on number of parts
            compound_bonus = self.weights["compound_bonus"] * (
                len(compound_parts) - 1
            )
            compound_score *= 1.0 + compound_bonus
            score *= compound_score
            logger.debug(f"After compound boost: {score}")

        # Length bonus for multi-word terms
        if (
            " " in word
            or "-" in word
            or (compound_parts and len(compound_parts) > 1)
        ):
            length_boost = 1.0 + self.weights["length_factor"]
            score *= length_boost
            logger.debug(f"After length boost: {score}")

        # Frequency impact if provided
        if frequency is not None and frequency > 1:
            freq_factor = 1.0 + (
                self.weights["frequency_factor"]
                * min(frequency / 10.0, 1.0)  # Cap frequency impact
            )
            score *= freq_factor
            logger.debug(f"After frequency boost ({freq_factor}): {score}")

        # Position boost from section weighting
        position_boost = 1.0 + (
            self.weights["position_boost"] * (position_score - 1.0)
        )
        score *= position_boost
        logger.debug(f"After position boost: {score}")

        # Normalize to 0-1 range
        final_score = min(score, 1.0)
        logger.debug(f"Final score for '{word}': {final_score}")

        return final_score

    # Internal async methods

    async def _analyze_internal(self, text: str) -> KeywordOutput:
        """Internal analysis method returning KeywordOutput."""
        if error := self._validate_input(text):
            return KeywordOutput(error=error, success=False, language="unknown")

        try:
            logger.debug("Starting internal analysis")

            # Reset and initialize tracking
            self._frequency_cache = {}
            self._current_text = text

            language = (
                self.language_processor.language
                if self.language_processor
                else "unknown"
            )

            # Split text into weighted sections
            sections = self._split_text_sections(text)

            # Get keywords with both methods
            statistical_keywords = self._analyze_statistically(text, sections)
            llm_keywords = await self._analyze_with_llm(
                text, statistical_keywords
            )

            # Process compound words
            compound_words = []
            if self.language_processor:
                logger.debug("Processing compound words")
                processed_compounds = set()  # Avoid duplicates

                # Check both statistical and LLM keywords for compounds
                for keyword in set(
                    kw.keyword for kw in statistical_keywords + llm_keywords
                ):
                    if keyword.lower() in processed_compounds:
                        continue

                    if self.language_processor.is_compound_word(keyword):
                        parts = self.language_processor.get_compound_parts(
                            keyword
                        )
                        if parts and len(parts) > 1:
                            logger.debug(
                                f"Found compound word: {keyword} -> {parts}"
                            )
                            compound_words.append(keyword)
                            processed_compounds.add(keyword.lower())

            # Combine results with weight-based scoring
            combined_keywords = {}

            # Process statistical keywords
            for kw in statistical_keywords:
                base_score = kw.score * self.weights["statistical"]
                if kw.keyword not in combined_keywords:
                    combined_keywords[kw.keyword] = {
                        "score": base_score,
                        "domain": kw.domain,
                        "compound_parts": None,
                    }

            # Process LLM keywords
            for kw in llm_keywords:
                base_score = kw.score * self.weights["llm"]
                if kw.keyword in combined_keywords:
                    # Take max score if keyword already exists
                    combined_keywords[kw.keyword]["score"] = max(
                        combined_keywords[kw.keyword]["score"], base_score
                    )
                else:
                    combined_keywords[kw.keyword] = {
                        "score": base_score,
                        "domain": kw.domain,
                        "compound_parts": None,
                    }

            # Apply compound word information and calculate final scores
            final_keywords = []
            for keyword, info in combined_keywords.items():
                compound_parts = None
                if keyword in compound_words:
                    compound_parts = self.language_processor.get_compound_parts(
                        keyword
                    )

                # Calculate final score with all weights
                final_score = self._calculate_confidence_score(
                    keyword,
                    info["score"],
                    1.0,  # default position score
                    info["domain"],
                    compound_parts,
                    frequency=self._get_keyword_frequency(keyword),
                )

                final_keywords.append(
                    KeywordInfo(
                        keyword=keyword,
                        score=final_score,
                        domain=info["domain"],
                        compound_parts=compound_parts,
                    )
                )

            # Apply clustering to get final results
            clustered_keywords = self._cluster_keywords(final_keywords)

            # Sort by score and apply max keywords limit
            final_keywords = sorted(
                clustered_keywords, key=lambda x: x.score, reverse=True
            )[: self.config.get("max_keywords", 10)]

            # Group keywords by domain
            domain_keywords = {}
            for kw in final_keywords:
                if kw.domain:
                    if kw.domain not in domain_keywords:
                        domain_keywords[kw.domain] = []
                    domain_keywords[kw.domain].append(kw.keyword)

            logger.debug(f"Found {len(compound_words)} compound words")
            logger.debug(
                f"Analysis complete. Found {len(final_keywords)} keywords"
            )

            return KeywordOutput(
                keywords=final_keywords,
                compound_words=compound_words,
                domain_keywords=domain_keywords,
                language=language,
                success=True,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            self._frequency_cache = {}  # Reset cache on error
            self._current_text = ""
            return KeywordOutput(
                error=str(e), success=False, language="unknown"
            )

    def _cluster_keywords(
        self, keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Group similar keywords into clusters with improved handling."""
        if not keywords:
            return []

        clusters: Dict[str, List[KeywordInfo]] = {}
        processed_keywords = set()

        # Sort keywords by initial score for better cluster seeds
        sorted_keywords = sorted(keywords, key=lambda x: x.score, reverse=True)

        for kw in sorted_keywords:
            if kw.keyword in processed_keywords:
                continue

            cluster = []
            base_form = (
                self.language_processor.get_base_form(kw.keyword)
                if self.language_processor
                else kw.keyword.lower()
            )

            # Find related keywords
            for other in sorted_keywords:
                if other.keyword in processed_keywords:
                    continue

                if self._are_keywords_related(
                    kw.keyword, other.keyword, base_form
                ):
                    cluster.append(other)
                    processed_keywords.add(other.keyword)

            if cluster:
                clusters[base_form] = cluster

        return self._adjust_cluster_scores(clusters)

    def _are_keywords_related(self, kw1: str, kw2: str, base_form: str) -> bool:
        """Check if keywords are related using multiple methods."""
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
            kw1_parts = set(self.language_processor.get_compound_parts(kw1))
            kw2_parts = set(self.language_processor.get_compound_parts(kw2))
            if kw1_parts & kw2_parts:  # If they share any parts
                return True

        # Technical term variations
        tech_variations = {
            "api": ["apis", "rest-api", "api-driven"],
            "cloud": ["cloud-based", "cloud-native", "multi-cloud"],
            "devops": ["dev-ops", "devsecops", "devops-driven"],
            # Add more variations as needed
        }

        for base_term, variations in tech_variations.items():
            if (kw1.lower() in variations or kw1.lower() == base_term) and (
                kw2.lower() in variations or kw2.lower() == base_term
            ):
                return True

        return False

    def _adjust_cluster_scores(
        self, clusters: Dict[str, List[KeywordInfo]]
    ) -> List[KeywordInfo]:
        """Adjust scores based on cluster relationships with improved domain handling."""
        result = []
        boost_factor = self.clustering_config.get("boost_factor", 1.2)
        domain_bonus = self.clustering_config.get("domain_bonus", 0.1)

        for base_form, cluster in clusters.items():
            if not cluster:
                continue

            # Sort cluster by base score
            cluster.sort(key=lambda x: x.score, reverse=True)

            # Get primary domain from highest scoring keywords
            primary_domain = Counter(k.domain for k in cluster[:3]).most_common(
                1
            )[0][0]

            # Process each keyword in cluster
            for i, kw in enumerate(cluster):
                # Calculate position-based boost
                if i == 0:  # Primary term
                    position_boost = boost_factor
                else:  # Related terms get diminishing boost
                    position_boost = 1.0 + (boost_factor - 1.0) * (
                        1.0 - (i / len(cluster))
                    )

                # Calculate domain boost
                domain_boost = 1.0
                if kw.domain == primary_domain:
                    domain_boost += domain_bonus

                # Apply compound word bonus if applicable
                compound_boost = 1.0
                if (
                    hasattr(kw, "compound_parts")
                    and kw.compound_parts
                    and len(kw.compound_parts) > 1
                ):
                    compound_bonus = 0.1 * (len(kw.compound_parts) - 1)
                    compound_boost += compound_bonus

                # Calculate final score
                final_score = min(
                    kw.score * position_boost * domain_boost * compound_boost,
                    1.0,
                )

                # Create new keyword with adjusted score
                result.append(
                    KeywordInfo(
                        keyword=kw.keyword,
                        score=final_score,
                        domain=kw.domain,
                        compound_parts=kw.compound_parts,
                    )
                )

        return sorted(result, key=lambda x: x.score, reverse=True)

    def _process_compound_word(self, word: str) -> Dict[str, Any]:
        """Process compound word to get its parts and domain."""
        if not self.language_processor:
            return {"parts": None, "domain": None}

        try:
            if self.language_processor.is_compound_word(word):
                logger.debug(f"Processing compound word: {word}")
                parts = self.language_processor.get_compound_parts(word)
                logger.debug(f"Found parts: {parts}")
                # Check if any part indicates a domain
                for part in parts:
                    if domain := self._detect_domain(part):
                        return {"parts": parts, "domain": domain}
                return {"parts": parts, "domain": None}
        except Exception as e:
            logger.error(f"Error processing compound word {word}: {e}")

        return {"parts": None, "domain": self._detect_domain(word)}

    def _combine_results(
        self,
        statistical_keywords: List[KeywordInfo],
        llm_keywords: List[KeywordInfo],
    ) -> List[KeywordInfo]:
        """Combine statistical and LLM results with clustering."""
        combined = {}

        # Process statistical keywords
        for kw in statistical_keywords:
            base_score = kw.score * self.weights["statistical"]
            self._update_combined_keywords(combined, kw, base_score)

        # Process LLM keywords
        for kw in llm_keywords:
            base_score = kw.score * self.weights["llm"]
            self._update_combined_keywords(combined, kw, base_score)

        # Convert to list and apply clustering
        initial_results = list(combined.values())
        clustered_results = self._cluster_keywords(initial_results)

        # Sort by score and apply max keywords limit
        return sorted(clustered_results, key=lambda x: x.score, reverse=True)[
            : self.config.get("max_keywords", 10)
        ]

    def _update_combined_keywords(
        self,
        combined: Dict[str, KeywordInfo],
        keyword: KeywordInfo,
        base_score: float,
    ) -> None:
        """Update combined keywords dictionary with proper weight handling."""
        if keyword.keyword in combined:
            # Take highest weighted score if keyword already exists
            existing = combined[keyword.keyword]
            score = max(existing.score, base_score)
        else:
            score = base_score

        # Process compound words
        compound_parts = None
        if self.language_processor:
            try:
                if self.language_processor.is_compound_word(keyword.keyword):
                    compound_parts = self.language_processor.get_compound_parts(
                        keyword.keyword
                    )
                    logger.debug(
                        f"Found compound word: {keyword.keyword} -> {compound_parts}"
                    )
            except Exception as e:
                logger.debug(
                    f"Error processing compound word {keyword.keyword}: {e}"
                )

        # Calculate final score with all weights
        final_score = self._calculate_confidence_score(
            keyword.keyword,
            score,
            1.0,  # default position score
            keyword.domain,
            compound_parts,
            frequency=self._get_keyword_frequency(keyword.keyword),
        )

        combined[keyword.keyword] = KeywordInfo(
            keyword=keyword.keyword,
            score=final_score,
            domain=keyword.domain,
            compound_parts=compound_parts,
        )

    # Helper for batch processing
    async def analyze_batch(
        self, texts: List[str], batch_size: int = 3, timeout: float = 30.0
    ) -> List[KeywordAnalysisResult]:
        """Process multiple texts with controlled concurrency."""
        results = []

        async def process_batch(
            batch: List[str],
        ) -> List[KeywordAnalysisResult]:
            tasks = [asyncio.create_task(self.analyze(text)) for text in batch]
            batch_results = []

            for task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=timeout)
                    batch_results.append(result)
                except asyncio.TimeoutError:
                    batch_results.append(
                        KeywordAnalysisResult(
                            keywords=[],
                            compound_words=[],
                            domain_keywords={},
                            language="unknown",
                            success=False,
                            error="Analysis timed out",
                        )
                    )
                except Exception as e:
                    batch_results.append(
                        KeywordAnalysisResult(
                            keywords=[],
                            compound_words=[],
                            domain_keywords={},
                            language="unknown",
                            success=False,
                            error=str(e),
                        )
                    )
                finally:
                    if not task.done():
                        task.cancel()

            return batch_results

        # Process in smaller batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                batch_results = await asyncio.wait_for(
                    process_batch(batch), timeout=timeout * len(batch)
                )
                results.extend(batch_results)
            except asyncio.TimeoutError:
                results.extend(
                    [
                        KeywordAnalysisResult(
                            keywords=[],
                            compound_words=[],
                            domain_keywords={},
                            language="unknown",
                            success=False,
                            error="Batch processing timed out",
                        )
                        for _ in batch
                    ]
                )
            except Exception as e:
                results.extend(
                    [
                        KeywordAnalysisResult(
                            keywords=[],
                            compound_words=[],
                            domain_keywords={},
                            language="unknown",
                            success=False,
                            error=str(e),
                        )
                        for _ in batch
                    ]
                )

        return results
