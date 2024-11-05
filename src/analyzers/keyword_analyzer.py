# src/analyzers/keyword_analyzer.py

from typing import List, Dict, Any, Optional, Set
import logging
from collections import Counter
import math
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

from .base import TextAnalyzer, AnalyzerOutput
from src.core.language_processing.base import BaseTextProcessor

logger = logging.getLogger(__name__)

class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""
    keywords: List[str] = Field(default_factory=list)
    keyword_scores: Dict[str, float] = Field(default_factory=dict)
    statistical_keywords: List[str] = Field(default_factory=list)
    compound_words: List[str] = Field(default_factory=list)

class KeywordAnalyzer(TextAnalyzer):
    """Analyzes text to extract keywords using both statistical and LLM methods."""
    
    def __init__(
        self,
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
        custom_stop_words: Optional[Set[str]] = None
    ):
        """Initialize the keyword analyzer.
        
        Args:
            llm: Language model to use
            config: Configuration parameters
            language_processor: Processor for language-specific operations
            custom_stop_words: Additional stop words to use
        """
        super().__init__(llm, config)
        self.language_processor = language_processor
        self.custom_stop_words = custom_stop_words or set()
        self.min_keyword_length = config.get('min_keyword_length', 3)
        self.max_keywords = config.get('max_keywords', 10)
    
    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain for keyword extraction."""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are a keyword extraction expert. Extract important keywords and phrases from text.
            Consider:
            - Technical terms and domain-specific vocabulary
            - Important concepts and themes
            - Named entities and proper nouns
            - Compound terms and multi-word phrases
            
            Return them in this format:
            {
                "keywords": ["keyword1", "keyword2", ...],
                "compound_words": ["compound1", "compound2", ...],
                "confidence_scores": {"keyword1": 0.95, "keyword2": 0.85, ...}
            }"""),
            ("human", """Extract keywords from this text:
            {text}
            
            Guidelines:
            - Extract up to {max_keywords} keywords
            - Consider these statistical keywords: {statistical_keywords}
            - Min keyword length: {min_length} characters
            - Focus on: {focus}
            
            Return in the specified JSON format.""")
        ])
        
        # Create processing chain
        chain = (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.max_keywords,
                "statistical_keywords": self._extract_statistical_keywords,
                "min_length": lambda _: self.min_keyword_length,
                "focus": lambda _: self.config.get('focus', 'general topics')
            }
            | template 
            | self.llm
            | self._post_process_llm_output
        )
        
        return chain
    
    async def analyze(self, text: str, **kwargs) -> KeywordOutput:
        """Analyze text to extract keywords.
        
        Args:
            text: Input text to analyze
            **kwargs: Additional parameters
            
        Returns:
            KeywordOutput: Extraction results
        """
        # Validate input
        if error := self._validate_input(text):
            return self._handle_error(error)
        
        try:
            # Get language-specific processing if available
            processed_text = (
                self.language_processor.process_text(text)
                if self.language_processor
                else text
            )
            
            # Extract statistical keywords
            statistical_keywords = self._extract_statistical_keywords(processed_text)
            
            # Get LLM-based analysis
            llm_results = await self.chain.ainvoke(processed_text)
            
            # Combine and score results
            combined_keywords = self._combine_keywords(
                statistical_keywords,
                llm_results.get("keywords", []),
                llm_results.get("confidence_scores", {})
            )
            
            return KeywordOutput(
                keywords=combined_keywords,
                keyword_scores=self._calculate_final_scores(combined_keywords),
                statistical_keywords=statistical_keywords,
                compound_words=llm_results.get("compound_words", []),
                language=self._detect_language(text)
            )
            
        except Exception as e:
            logger.error(f"Error in keyword analysis: {str(e)}", exc_info=True)
            return self._handle_error(f"Keyword analysis failed: {str(e)}")
    
    def _extract_statistical_keywords(self, text: str) -> List[str]:
        """Extract keywords using statistical methods.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Statistically significant keywords
        """
        # Use TF-IDF or similar statistical methods
        # This could be moved to a separate StatisticalKeywordExtractor class
        words = self._tokenize(text)
        freq = Counter(words)
        
        # Calculate TF-IDF like scores
        scores = {
            word: freq[word] * math.log(len(words) / (freq[word] + 1))
            for word in freq
            if len(word) >= self.min_keyword_length
            and word not in self.custom_stop_words
        }
        
        # Sort by score and take top keywords
        return [
            word for word, _ in sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_keywords]
        ]
    
    def _combine_keywords(
        self,
        statistical: List[str],
        llm: List[str],
        confidence_scores: Dict[str, float]
    ) -> List[str]:
        """Combine keywords from different sources.
        
        Args:
            statistical: Statistically extracted keywords
            llm: LLM-extracted keywords
            confidence_scores: Confidence scores for LLM keywords
            
        Returns:
            List[str]: Combined and ranked keywords
        """
        combined = {}
        
        # Add statistical keywords
        for kw in statistical:
            combined[kw] = {"score": 0.5, "sources": ["statistical"]}
        
        # Add LLM keywords with confidence scores
        for kw in llm:
            if kw in combined:
                combined[kw]["score"] = max(
                    combined[kw]["score"],
                    confidence_scores.get(kw, 0.5)
                )
                combined[kw]["sources"].append("llm")
            else:
                combined[kw] = {
                    "score": confidence_scores.get(kw, 0.5),
                    "sources": ["llm"]
                }
        
        # Sort by score and number of sources
        sorted_keywords = sorted(
            combined.items(),
            key=lambda x: (x[1]["score"], len(x[1]["sources"])),
            reverse=True
        )
        
        return [kw for kw, _ in sorted_keywords]
    
    def _calculate_final_scores(self, keywords: List[str]) -> Dict[str, float]:
        """Calculate final confidence scores for keywords."""
        # This could be enhanced with more sophisticated scoring
        return {kw: 1.0 - (i / len(keywords)) for i, kw in enumerate(keywords)}
    
    def _post_process_llm_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate LLM output."""
        if isinstance(output, str):
            import json
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM output as JSON")
                return {"keywords": [], "compound_words": [], "confidence_scores": {}}
        
        return {
            "keywords": output.get("keywords", []),
            "compound_words": output.get("compound_words", []),
            "confidence_scores": output.get("confidence_scores", {})
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.language_processor:
            return self.language_processor.tokenize(text)
        # Simple fallback tokenization
        return text.lower().split()