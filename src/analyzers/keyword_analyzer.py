# src/analyzers/keyword_analyzer.py

import logging
from typing import Any, Dict, List, Optional, Set
from collections import Counter
from math import log

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import BaseAnalyzer, TextSection, AnalyzerOutput
from src.schemas import KeywordAnalysisResult, KeywordInfo

from pydantic import Field

logger = logging.getLogger(__name__)


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""
    keywords: List[Dict[str, Any]] = Field(default_factory=list)
    domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)

    def dict(self) -> Dict[str, Any]:
        """Convert to dict with proper structure."""
        if self.error:
            return {
                "keywords": {
                    "error": self.error,
                    "success": False,
                    "language": self.language
                }
            }

        return {
            "keywords": {
                "keywords": self.keywords,
                "domain_keywords": self.domain_keywords,
                "success": self.success,
                "language": self.language
            }
        }

    def to_schema(self) -> KeywordAnalysisResult:
        """Convert to schema model."""
        if not self.success:
            return KeywordAnalysisResult(
                keywords=[],
                language=self.language,
                success=False,
                error=self.error
            )
            
        return KeywordAnalysisResult(
            keywords=[KeywordInfo(**k) for k in self.keywords],
            language=self.language,
            success=True
        )

class KeywordAnalyzer(BaseAnalyzer):
    """Analyzes text to extract keywords with position-aware weighting."""

    def __init__(
        self,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor=None
    ):
        """Initialize analyzer with LLM and config."""
        super().__init__(config)
        self.llm = llm
        self.language_processor = language_processor
        self.chain = self._create_chain()
        
        # Load or set default weights
        self.weights = config.get("weights", {
            "statistical": 0.4,
            "llm": 0.6
        })
        
        # Load domain keywords if available
        self.domain_keywords = config.get("domain_keywords", {})

    def _create_chain(self) -> RunnableSequence:
        """Create the LangChain processing chain."""
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a keyword extraction expert. Extract important keywords and phrases from text.
                Consider:
                - Technical terms and domain-specific vocabulary
                - Important concepts and themes
                - Named entities and proper nouns
                - Compound terms and multi-word phrases
                
                Return results in JSON format with these exact fields:
                {{
                    "keywords": [
                        {{
                            "keyword": "example_keyword",
                            "score": 0.95,
                            "domain": "technical"
                        }}
                    ]
                }}""",
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
                """,
            ),
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
            | self._process_llm_output
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

    def _analyze_statistically(
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

    def _combine_results(
        self,
        statistical_keywords: List[KeywordInfo],
        llm_keywords: List[KeywordInfo]
    ) -> List[KeywordInfo]:
        """Combine statistical and LLM results."""
        combined = {}
        
        # Get weights
        stat_weight = self.weights.get("statistical", 0.4)
        llm_weight = self.weights.get("llm", 0.6)
        
        # Combine statistical keywords
        for kw in statistical_keywords:
            combined[kw.keyword] = KeywordInfo(
                keyword=kw.keyword,
                score=kw.score * stat_weight,
                domain=kw.domain
            )
            
        # Combine LLM keywords
        for kw in llm_keywords:
            if kw.keyword in combined:
                # Update existing keyword
                existing = combined[kw.keyword]
                combined[kw.keyword] = KeywordInfo(
                    keyword=kw.keyword,
                    score=min(
                        existing.score + (kw.score * llm_weight),
                        1.0  # Cap at 1.0
                    ),
                    domain=existing.domain or kw.domain
                )
            else:
                # Add new keyword
                combined[kw.keyword] = KeywordInfo(
                    keyword=kw.keyword,
                    score=kw.score * llm_weight,
                    domain=kw.domain
                )
                
        # Sort by score and limit to max keywords
        return sorted(
            combined.values(),
            key=lambda x: x.score,
            reverse=True
        )[:self.config.get("max_keywords", 10)]