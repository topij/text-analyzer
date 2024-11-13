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
    
    def __init__(
        self,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor=None
    ):
        super().__init__(llm, config)
        self.language_processor = language_processor
        self.llm = llm
        self.chain = self._create_chain()
        
        # Load or set default weights
        self.weights = config.get("weights", {
            "statistical": 0.4,
            "llm": 0.6
        })
        
        # Domain configuration
        self.domain_keywords = config.get("domain_keywords", {})
        
        # Clustering configuration
        self.clustering_config = config.get("clustering", {
            "similarity_threshold": 0.85,
            "max_cluster_size": 3,
            "boost_factor": 1.2
        })



    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain."""
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a keyword extraction expert. Extract important keywords and phrases from text.
                Consider:
                - Technical terms and domain-specific vocabulary
                - Important concepts and themes
                - Named entities and proper nouns
                - Compound terms and multi-word phrases"""
            ),
            (
                "human",
                """Extract keywords from this text:
                {text}
                
                Guidelines:
                - Extract up to {max_keywords} keywords
                - Consider these statistical keywords: {statistical_keywords}
                - Min keyword length: {min_length} characters
                - Focus on: {focus_area}
                
                Return in JSON format with these fields:
                {{
                    "keywords": [
                        {{
                            "keyword": "example term",
                            "score": 0.95,
                            "domain": "technical"
                        }}
                    ],
                    "domain_keywords": {{
                        "technical": ["term1", "term2"],
                        "business": ["term3", "term4"]
                    }}
                }}"""
            )
        ])

        chain = (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.config.get("max_keywords", 10),
                "statistical_keywords": lambda x: self._get_statistical_keywords(x),
                "min_length": lambda _: self.config.get("min_keyword_length", 3),
                "focus_area": lambda _: self.config.get("focus_on", "general topics")
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    async def analyze(self, text: str) -> KeywordAnalysisResult:
        """Analyze text to extract keywords. Returns schema model for API use."""
        # Get internal analysis results
        output = await self._analyze_internal(text)
        # Convert to schema model
        return output.to_schema()

    async def _analyze_internal(self, text: str) -> KeywordOutput:
        """Internal analysis method returning KeywordOutput."""
        # Validate input
        if error := self._validate_input(text):
            return KeywordOutput(
                error=error,
                success=False,
                language="unknown"
            )

        try:
            # Detect language
            language = (
                self.language_processor.language 
                if self.language_processor 
                else "unknown"
            )

            # Split text into weighted sections
            sections = self._split_text_sections(text)

            # Get keywords with both methods
            statistical_keywords = self._analyze_statistically(text, sections)
            llm_keywords = await self._analyze_with_llm(text, statistical_keywords)
            final_keywords = self._combine_results(statistical_keywords, llm_keywords)

            # Group keywords by domain
            domain_keywords = {}
            for kw in final_keywords:
                if kw.domain:
                    if kw.domain not in domain_keywords:
                        domain_keywords[kw.domain] = []
                    domain_keywords[kw.domain].append(kw.keyword)

            return KeywordOutput(
                keywords=[k.dict() for k in final_keywords],
                domain_keywords=domain_keywords,
                language=language,
                success=True
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return KeywordOutput(
                error=str(e),
                success=False,
                language="unknown"
            )

    def analyze_statistically(
        self, 
        text: str,
        sections: List[TextSection]
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
                keywords.append(KeywordInfo(
                    keyword=word,
                    score=final_score,
                    domain=self._detect_domain(word)
                ))
        
        return sorted(
            keywords,
            key=lambda x: x.score,
            reverse=True
        )[:self.config.get("max_keywords", 10)]

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

    async def _analyze_with_llm(
        self,
        text: str,
        statistical_keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Extract keywords using LLM with statistical hints."""
        if not self.llm:
            return []
            
        try:
            # Get LLM response using the chain
            response = await self.chain.ainvoke(text)
            
            # Extract keywords from response
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
            base_form = self.language_processor.get_base_form(kw.keyword) if self.language_processor else kw.keyword.lower()
            
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
            ]
        }
        
        pairs = domain_pairs.get(domain, [])
        kw1_lower = kw1.lower()
        kw2_lower = kw2.lower()
        
        return any(
            (kw1_lower in pair and kw2_lower in pair)
            for pair in pairs
        )

    def _adjust_cluster_scores(
        self, 
        clusters: Dict[str, List[KeywordInfo]]
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
                    boost = 1.0 + (boost_factor - 1.0) * (1.0 - (i / len(cluster)))
                    
                # Create new keyword with adjusted score
                result.append(KeywordInfo(
                    keyword=kw.keyword,
                    score=min(kw.score * boost, 1.0),  # Cap at 1.0
                    domain=kw.domain
                ))
        
        return sorted(result, key=lambda x: x.score, reverse=True)

    def _calculate_confidence_score(
        self, 
        word: str,
        base_score: float,
        position_score: float,
        domain: Optional[str] = None
    ) -> float:
        """Calculate enhanced confidence score."""
        # Base importance
        score = base_score * position_score
        
        # Domain relevance boost
        if domain:
            domain_boost = 1.2  # Configurable
            score *= domain_boost
            
        # Length factor (prefer multi-word terms slightly)
        word_count = len(word.split())
        if word_count > 1:
            length_boost = 1.0 + (0.1 * (word_count - 1))  # Small boost per word
            score *= length_boost
            
        # Normalize to 0-1 range
        return min(score, 1.0)

    def _combine_results(
        self,
        statistical_keywords: List[KeywordInfo],
        llm_keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Combine statistical and LLM results with clustering."""
        combined = {}
        
        # Get weights
        stat_weight = self.weights.get("statistical", 0.4)
        llm_weight = self.weights.get("llm", 0.6)
        
        # Combine initial keywords
        for kw in statistical_keywords:
            combined[kw.keyword] = KeywordInfo(
                keyword=kw.keyword,
                score=kw.score * stat_weight,
                domain=kw.domain
            )
            
        for kw in llm_keywords:
            if kw.keyword in combined:
                existing = combined[kw.keyword]
                combined[kw.keyword] = KeywordInfo(
                    keyword=kw.keyword,
                    score=self._calculate_confidence_score(
                        kw.keyword,
                        existing.score + (kw.score * llm_weight),
                        1.0,  # Base position score
                        existing.domain or kw.domain
                    ),
                    domain=existing.domain or kw.domain
                )
            else:
                combined[kw.keyword] = KeywordInfo(
                    keyword=kw.keyword,
                    score=kw.score * llm_weight,
                    domain=kw.domain
                )
        
        # Apply clustering to combined results
        clustered = self._cluster_keywords(list(combined.values()))
        
        # Return top keywords
        return clustered[:self.config.get("max_keywords", 10)]