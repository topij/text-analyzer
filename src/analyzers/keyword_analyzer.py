# src/analyzers/keyword_analyzer.py

import logging
from typing import Any, Dict, List, Optional, Set
from collections import Counter
from math import log

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import TextAnalyzer, AnalyzerOutput, TextSection  # Changed from BaseAnalyzer
from src.schemas import KeywordAnalysisResult, KeywordInfo

from pydantic import Field

logger = logging.getLogger(__name__)


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""

    keywords: List[KeywordInfo] = Field(default_factory=list)
    domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)


class KeywordAnalyzer(TextAnalyzer):
    """Analyzes text to extract keywords with position-aware weighting."""

    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None, language_processor=None):
        super().__init__(llm, config)
        self.language_processor = language_processor
        self.llm = llm

        # Initialize with default weights first
        self.weights = {
            "statistical": 0.4,
            "llm": 0.6,
            "compound_bonus": 0.2,
            "domain_bonus": 0.15,
            "length_factor": 0.1,
            "generic_penalty": 0.3,
        }

        # Update with any provided weights from config
        if config and "weights" in config:
            self.weights.update(config["weights"])

        # Refine generic terms to be more context-aware
        self.generic_terms = {
            # Common verbs
            "reduce",
            "increase",
            "decrease",
            "improve",
            "change",
            "update",
            # Common adjectives
            "new",
            "current",
            "existing",
            "previous",
            "next",
            # Common nouns (reduced list)
            "project",
            "system",
            "process",
            "department",
        }

        # Add domain-specific exceptions
        self.domain_specific_terms = {
            "technical": {"implementation", "deployment", "integration"},
            "business": {"cost", "revenue", "growth"},
        }

        self.chain = self._create_chain()
        logger.debug(f"Initialized with weights: {self.weights}")

        # Domain configuration
        self.domain_keywords = config.get("domain_keywords", {})

        # Clustering configuration
        self.clustering_config = config.get(
            "clustering", {"similarity_threshold": 0.85, "max_cluster_size": 3, "boost_factor": 1.2}
        )

        logger.debug(f"Initialized with config: {self.config}")

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
                "statistical_keywords": lambda x: ", ".join(self._get_statistical_keywords(x)),
                "min_length": lambda _: self.config.get("min_keyword_length", 3),
                "focus_area": lambda _: self.config.get("focus_on", "general topics"),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    def _split_text_sections(self, text: str, section_size: int = 200) -> List[TextSection]:
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

            sections.append(TextSection(content=section_text, start=i, end=i + len(section_text), weight=weight))

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
                        {"keyword": kw["keyword"], "score": float(kw.get("score", 0.5)), "domain": kw.get("domain")}
                    )

            return {"keywords": keywords, "domain_keywords": data.get("domain_keywords", {})}

        except Exception as e:
            self.logger.error(f"Error processing keyword output: {e}")
            return {"keywords": [], "domain_keywords": {}}

    def _analyze_statistically(self, text: str, sections: Optional[List[TextSection]] = None) -> List[KeywordInfo]:
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
                    keywords.append(KeywordInfo(keyword=word, score=final_score, domain=self._detect_domain(word)))

            return sorted(keywords, key=lambda x: x.score, reverse=True)[: self.config.get("max_keywords", 10)]

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return []

    async def analyze(self, text: str) -> KeywordAnalysisResult:
        """Analyze text to extract keywords."""
        logger.debug("Starting keyword analysis")  # Verify this appears
        try:
            # Get internal analysis results
            logger.debug(f"Analyzing text of length {len(text)}")
            output = await self._analyze_internal(text)

            # Convert to KeywordAnalysisResult
            result = KeywordAnalysisResult(
                keywords=[k for k in output.keywords],
                compound_words=[],  # Fill if needed
                domain_keywords=output.domain_keywords,
                language=self.language_processor.language if self.language_processor else "unknown",
                success=True,
            )
            logger.debug(f"Analysis complete. Found {len(result.keywords)} keywords")
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return KeywordAnalysisResult(
                keywords=[], compound_words=[], domain_keywords={}, language="unknown", success=False, error=str(e)
            )

    def analyze_statistically(self, text: str, sections: List[TextSection]) -> List[KeywordInfo]:
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
                keywords.append(KeywordInfo(keyword=word, score=final_score, domain=self._detect_domain(word)))

        return sorted(keywords, key=lambda x: x.score, reverse=True)[: self.config.get("max_keywords", 10)]

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
        if self.language_processor and self.language_processor.is_stop_word(word):
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

    async def _analyze_with_llm(self, text: str, statistical_keywords: List[KeywordInfo]) -> List[KeywordInfo]:
        """Extract keywords using LLM with statistical hints."""
        if not self.llm:
            return []

        try:
            # Get LLM response using the chain
            response = await self.chain.ainvoke(text)

            # Extract keywords from response
            if isinstance(response, dict) and "keywords" in response:
                return [KeywordInfo(**kw) if isinstance(kw, dict) else kw for kw in response["keywords"]]

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

    def _cluster_keywords(self, keywords: List[KeywordInfo]) -> List[KeywordInfo]:
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
                self.language_processor.get_base_form(kw.keyword) if self.language_processor else kw.keyword.lower()
            )

            # Find related keywords
            for other in keywords:
                if other.keyword in processed_keywords:
                    continue

                if self._are_keywords_related(kw.keyword, other.keyword, base_form):
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
        other_base = self.language_processor.get_base_form(kw2) if self.language_processor else kw2.lower()
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

    def _check_domain_relationship(self, kw1: str, kw2: str, domain: str) -> bool:
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

    def _adjust_cluster_scores(self, clusters: Dict[str, List[KeywordInfo]]) -> List[KeywordInfo]:
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
                    boost = 1.0 + (boost_factor - 1.0) * (1.0 - (i / len(cluster)))

                # Create new keyword with adjusted score
                result.append(
                    KeywordInfo(keyword=kw.keyword, score=min(kw.score * boost, 1.0), domain=kw.domain)  # Cap at 1.0
                )

        return sorted(result, key=lambda x: x.score, reverse=True)

    def _calculate_confidence_score(
        self,
        word: str,
        base_score: float,
        position_score: float,
        domain: Optional[str] = None,
        compound_parts: Optional[List[str]] = None,
    ) -> float:
        logger.debug(f"Calculating score for '{word}' with base score {base_score}")

        score = base_score * position_score

        # More nuanced generic term handling
        word_lower = word.lower()
        words = word_lower.split()

        # Check if the term is domain-specific despite being "generic"
        is_domain_specific = False
        if domain:
            domain_terms = self.domain_specific_terms.get(domain, set())
            if any(term in domain_terms for term in words):
                is_domain_specific = True
                logger.debug(f"Term '{word}' contains domain-specific parts")

        # Only apply generic penalty if not domain-specific
        if not is_domain_specific and (
            word_lower in self.generic_terms or any(part in self.generic_terms for part in words)
        ):
            penalty = self.weights.get("generic_penalty", 0.3)
            score *= 1.0 - penalty
            logger.debug(f"Applied generic penalty ({penalty}) to '{word}', new score: {score}")

        # Domain relevance boost
        if domain and domain not in ("general", None):
            domain_boost = 1.0 + self.weights.get("domain_bonus", 0.15)
            score *= domain_boost
            logger.debug(f"Applied domain boost ({domain_boost}) for '{domain}', new score: {score}")

        # Compound word bonus
        if compound_parts and len(compound_parts) > 1:
            compound_bonus = self.weights.get("compound_bonus", 0.2)
            compound_boost = 1.0 + (compound_bonus * (len(compound_parts) - 1))
            score *= compound_boost
            logger.debug(
                f"Applied compound boost ({compound_boost}) for {len(compound_parts)} parts, new score: {score}"
            )
        else:
            # Length factor (prefer multi-word terms slightly)
            word_count = len(word.split())
            if word_count > 1:
                length_factor = self.weights.get("length_factor", 0.1)
                length_boost = 1.0 + (length_factor * (word_count - 1))
                score *= length_boost
                logger.debug(f"Applied length boost ({length_boost}) for {word_count} words, new score: {score}")

        # Normalize to 0-1 range
        final_score = min(score, 1.0)
        logger.debug(f"Final score for '{word}': {final_score}")
        return final_score

    async def _analyze_internal(self, text: str) -> KeywordOutput:
        """Internal analysis method returning KeywordOutput."""
        if error := self._validate_input(text):
            return KeywordOutput(error=error, success=False, language="unknown")

        try:
            logger.debug("Starting internal analysis")
            language = self.language_processor.language if self.language_processor else "unknown"

            # Split text into weighted sections
            sections = self._split_text_sections(text)

            # Get keywords with both methods
            statistical_keywords = self._analyze_statistically(text, sections)
            llm_keywords = await self._analyze_with_llm(text, statistical_keywords)

            # Process compound words first
            compound_words = []
            if self.language_processor:
                logger.debug("Processing compound words")
                for keyword in set(kw.keyword for kw in statistical_keywords + llm_keywords):
                    if self.language_processor.is_compound_word(keyword):
                        parts = self.language_processor.get_compound_parts(keyword)
                        if parts and len(parts) > 1:
                            logger.debug(f"Found compound word: {keyword} -> {parts}")
                            compound_words.append(keyword)

            # Combine results with compound word information
            final_keywords = self._combine_results(statistical_keywords, llm_keywords)

            # Add compound parts information to keywords
            for kw in final_keywords:
                if kw.keyword in compound_words:
                    kw.compound_parts = self.language_processor.get_compound_parts(kw.keyword)
                    logger.debug(f"Added compound parts for {kw.keyword}: {kw.compound_parts}")

            # Group keywords by domain
            domain_keywords = {}
            for kw in final_keywords:
                if kw.domain:
                    if kw.domain not in domain_keywords:
                        domain_keywords[kw.domain] = []
                    domain_keywords[kw.domain].append(kw.keyword)

            logger.debug(f"Found {len(compound_words)} compound words")
            return KeywordOutput(
                keywords=final_keywords,
                compound_words=compound_words,  # Add the compound words list
                domain_keywords=domain_keywords,
                language=language,
                success=True,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return KeywordOutput(error=str(e), success=False, language="unknown")

    def _combine_results(
        self, statistical_keywords: List[KeywordInfo], llm_keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Combine statistical and LLM results with improved scoring."""
        combined = {}

        # Get weights
        stat_weight = self.weights.get("statistical", 0.4)
        llm_weight = self.weights.get("llm", 0.6)

        # Process statistical keywords
        for kw in statistical_keywords:
            compound_info = self._process_compound_word(kw.keyword)
            combined[kw.keyword] = KeywordInfo(
                keyword=kw.keyword,
                score=self._calculate_confidence_score(
                    kw.keyword, kw.score * stat_weight, 1.0, compound_info["domain"], compound_info["parts"]
                ),
                domain=compound_info["domain"] or kw.domain,
                compound_parts=compound_info["parts"],
            )

        # Process LLM keywords similarly
        for kw in llm_keywords:
            compound_info = self._process_compound_word(kw.keyword)
            if kw.keyword in combined:
                existing = combined[kw.keyword]
                base_score = existing.score + (kw.score * llm_weight)
            else:
                base_score = kw.score * llm_weight

            combined[kw.keyword] = KeywordInfo(
                keyword=kw.keyword,
                score=self._calculate_confidence_score(
                    kw.keyword, base_score, 1.0, compound_info["domain"] or kw.domain, compound_info["parts"]
                ),
                domain=compound_info["domain"] or kw.domain,
                compound_parts=compound_info["parts"],
            )

        # Apply clustering to the results
        clustered = self._cluster_keywords(list(combined.values()))
        # Filter out low-scoring single words that are part of higher-scoring compounds
        compounds = {k.keyword for k in clustered if k.compound_parts and len(k.compound_parts) > 1}
        min_single_word_score = 0.4

        filtered = []
        for kw in clustered:
            # Keep compound words
            if kw.keyword in compounds:
                filtered.append(kw)
            # Keep single words only if they meet score threshold
            elif kw.score >= min_single_word_score:
                filtered.append(kw)
        # Sort by score and return top keywords
        return sorted(filtered, key=lambda x: x.score, reverse=True)[: self.config.get("max_keywords", 10)]

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
