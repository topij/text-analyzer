# src/analyzers/keyword_analyzer.py

import json
import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional  # , Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import Field  # BaseModel

from src.core.language_processing.base import BaseTextProcessor

from .base import AnalyzerOutput, TextAnalyzer

logger = logging.getLogger(__name__)


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""

    keywords: List[str] = Field(default_factory=list)
    keyword_scores: Dict[str, float] = Field(default_factory=dict)
    compound_words: List[str] = Field(default_factory=list)
    domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    success: bool = Field(default=True)
    language: str = Field(default="unknown")
    error: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dict preserving structure."""
        if self.error:
            return {"keywords": {"error": self.error, "success": False, "language": self.language}}

        return {
            "keywords": {
                "keywords": self.keywords,
                "keyword_scores": self.keyword_scores,
                "compound_words": self.compound_words,
                "domain_keywords": self.domain_keywords,
                "success": self.success,
                "language": self.language,
            }
        }


class KeywordAnalyzer(TextAnalyzer):
    """Analyzes text to extract keywords using statistical and LLM methods."""

    def __init__(
        self, llm=None, config: Optional[Dict[str, Any]] = None, language_processor: Optional[BaseTextProcessor] = None
    ):
        # def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None, language_processor=None):
        """Initialize the keyword analyzer.

        Args:
            llm: Language model to use
            config: Configuration parameters including:
                - max_keywords: Maximum keywords to extract
                - min_keyword_length: Minimum keyword length
                - include_compounds: Whether to detect compound words
                - excluded_keywords: Set of keywords to exclude
                - predefined_keywords: Dictionary of predefined keywords with importance
                - min_confidence: Minimum confidence threshold
                - weights: Weights for statistical vs LLM analysis
            language_processor: Language-specific text processor
        """
        super().__init__(llm, config)
        self.language_processor = language_processor
        self.max_keywords = self.config.get("max_keywords", 10)
        self.min_keyword_length = self.config.get("min_keyword_length", 3)
        self.predefined_keywords = self.config.get("predefined_keywords", {})
        self.excluded_keywords = self.config.get("excluded_keywords", set())
        self.weights = self.config.get("weights", {"statistical": 0.4, "llm": 0.6})
        self.min_confidence = self.config.get("min_confidence", 0.3)

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain for keyword extraction."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a keyword extraction expert. Extract important keywords and phrases from text.
            Return results in JSON format with these exact fields:
            {{
                "keywords": ["keyword1", "keyword2", ...],
                "keyword_scores": {{"keyword1": 0.9, "keyword2": 0.8, ...}},
                "compound_words": ["word1", "word2"],
                "domain_keywords": {{"domain1": ["kw1", "kw2"], ...}}
            }}""",
                ),
                (
                    "human",
                    """Extract keywords from this text:
            {text}
            
            Guidelines:
            - Max keywords: {max_keywords}
            - Statistical keywords to consider: {statistical_keywords}
            - Min length: {min_length} characters
            - Focus on: {focus}""",
                ),
            ]
        )

        # Create chain with proper output handling
        chain = (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.config.get("max_keywords", 8),
                "statistical_keywords": lambda x: self._get_statistical_keywords_str(x),
                "min_length": lambda _: self.config.get("min_keyword_length", 3),
                "focus": lambda _: self.config.get("focus", "general"),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    async def analyze(self, text: str, **kwargs) -> KeywordOutput:
        """Analyze text to extract keywords."""
        try:
            if kwargs:
                self.config.update(kwargs)

            # Dynamically set keyword limit
            self.max_keywords = self._get_keyword_limit(text)

            # Get statistical keywords
            statistical_scores = self._extract_statistical_keywords(text)
            
            # Get LLM keywords
            llm_results = await self.chain.ainvoke(text)
            
            # Combine results (weighted)
            combined_scores = self._combine_keywords(
                statistical_scores,
                llm_results.get("keyword_scores", {}),
                self.weights
            )
            
            # Sort by score and limit
            sorted_keywords = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            keywords = [k for k, _ in sorted_keywords[:self.max_keywords]]
            scores = {k: s for k, s in sorted_keywords[:self.max_keywords]}
            
            return KeywordOutput(
                keywords=keywords,
                keyword_scores=scores,
                compound_words=llm_results.get("compound_words", []),
                domain_keywords=llm_results.get("domain_keywords", {}),
                language=self.language_processor.language,
                success=True
            )

        except Exception as e:
            logger.error(f"Keyword analysis failed: {e}", exc_info=True)
            return KeywordOutput(
                keywords=[],
                keyword_scores={},
                compound_words=[],
                domain_keywords={},
                error=str(e),
                success=False,
                language=self.language_processor.language
            )

    def _get_statistical_keywords_str(self, text: str) -> str:
        """Get statistical keywords as a formatted string."""
        keywords = self._extract_statistical_keywords(text)
        return ", ".join(f"'{k}'" for k in sorted(keywords.keys(), key=keywords.get, reverse=True)[:5])

    def _extract_statistical_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords using enhanced statistical methods."""
        if not text:
            return {}

        # Split into sentences for position weighting
        sentences = text.split('.')
        sentence_tokens = [
            self.language_processor.tokenize(sent.strip())
            for sent in sentences if sent.strip()
        ]

        # Calculate position weights (earlier sentences get higher weight)
        sent_weights = [1.0 - (i * 0.1) for i in range(len(sentences))]
        sent_weights = [max(0.5, w) for w in sent_weights]  # Minimum weight 0.5

        # Get word frequencies with position weighting
        word_scores = {}
        total_words = 0
        
        for sent_tokens, weight in zip(sentence_tokens, sent_weights):
            # Get base forms
            base_forms = [
                self.language_processor.get_base_form(token)
                for token in sent_tokens
                if self.language_processor.should_keep_word(token)
            ]
            
            # Count with position weight
            for word in base_forms:
                if len(word) >= self.min_keyword_length:
                    word_scores[word] = word_scores.get(word, 0) + weight
                    total_words += 1

        # Calculate TF-IDF-like scores
        if total_words > 0:
            for word in word_scores:
                tf = word_scores[word] / total_words
                # Simple IDF approximation based on word length and uniqueness
                idf = math.log(1 + len(word)) * (1 + int(word_scores[word] == 1))
                word_scores[word] = tf * idf

            # Normalize scores
            max_score = max(word_scores.values()) if word_scores else 1.0
            word_scores = {
                word: score / max_score 
                for word, score in word_scores.items()
            }

        return word_scores
    
    def _get_keyword_limit(self, text: str) -> int:
        """Dynamically determine number of keywords based on text length."""
        # Count sentences
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Base calculation on sentence count
        if sentence_count <= 2:
            limit = max(3, min(5, sentence_count * 2))
        elif sentence_count <= 5:
            limit = max(5, min(8, sentence_count * 1.5))
        else:
            limit = min(self.max_keywords, sentence_count)
            
        return int(limit)

    def _combine_keywords(
        self, statistical: Dict[str, float], llm: Dict[str, float], weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine keywords from different sources with weights."""
        combined = {}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Combine scores
        for keyword, score in statistical.items():
            combined[keyword] = score * weights["statistical"]

        for keyword, score in llm.items():
            if keyword in combined:
                combined[keyword] += score * weights["llm"]
            else:
                combined[keyword] = score * weights["llm"]

        return combined

    def _apply_predefined_keywords(self, keywords: Dict[str, float], text: str) -> Dict[str, float]:
        """Apply predefined keywords with their importance."""
        # Check for predefined keywords in text
        for keyword, config in self.predefined_keywords.items():
            if self._keyword_in_text(keyword, text):
                keywords[keyword] = max(keywords.get(keyword, 0), config.importance)

            # Check compound parts if available
            if config.compound_parts:
                if all(self._keyword_in_text(part, text) for part in config.compound_parts):
                    keywords[keyword] = max(keywords.get(keyword, 0), config.importance)

        return keywords

    def _keyword_in_text(self, keyword: str, text: str) -> bool:
        """Check if keyword is present in text."""
        if self.language_processor:
            return self.language_processor.contains_word(text, keyword)
        return keyword.lower() in text.lower()

    def _get_domain_keywords(self, keywords: List[str], llm_domains: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Group keywords by domain."""
        domains = {}

        # Add predefined domain keywords
        for keyword in keywords:
            if keyword in self.predefined_keywords:
                domain = self.predefined_keywords[keyword].domain
                if domain:
                    domains.setdefault(domain, []).append(keyword)

        # Add LLM-detected domain keywords
        for domain, words in llm_domains.items():
            existing = set(domains.get(domain, []))
            domains[domain] = list(existing | set(w for w in words if w in keywords))

        return domains

    def _get_compound_words(self, text: str, keywords: List[tuple[str, float]], llm_compounds: List[str]) -> List[str]:
        """Extract compound words from text."""
        if not self.config.get("include_compounds", True):
            return []

        compounds = set()

        # Get predefined compounds
        for keyword, _ in keywords:
            if keyword in self.predefined_keywords:
                if self.predefined_keywords[keyword].compound_parts:
                    compounds.add(keyword)

        # Get language-specific compounds
        if self.language_processor and hasattr(self.language_processor, "get_compound_words"):
            compounds.update(self.language_processor.get_compound_words(text))

        # Add LLM-detected compounds
        compounds.update(c for c in llm_compounds if self._keyword_in_text(c, text))

        return list(compounds)

    def _create_empty_output(self) -> Dict[str, Any]:
        """Create empty output structure."""
        return {"keywords": [], "keyword_scores": {}, "compound_words": [], "domain_keywords": {}}

    def _process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process raw LLM output into standard format."""
        try:
            # First use base class processing
            base_output = super()._post_process_llm_output(output)
            if "error" in base_output:
                return base_output

            # Extract components
            keywords = base_output.get("keywords", [])
            scores = base_output.get("keyword_scores", {})
            compounds = base_output.get("compound_words", [])
            domains = base_output.get("domain_keywords", {})

            # Validate and normalize scores
            if scores and max(scores.values()) > 0:
                max_score = max(scores.values())
                scores = {k: v / max_score for k, v in scores.items()}

            return {
                "keywords": keywords,
                "keyword_scores": scores,
                "compound_words": compounds,
                "domain_keywords": domains,
            }

        except Exception as e:
            self.logger.error(f"Error processing LLM output: {e}")
            return {"error": str(e)}

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output into standardized format."""
        try:
            # Get content from AIMessage or string
            if hasattr(output, "content"):
                content = output.content
            elif isinstance(output, str):
                content = output
            elif isinstance(output, dict):
                return output
            else:
                self.logger.error(f"Unexpected output type: {type(output)}")
                return self._create_empty_output()

            # Parse JSON content
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                # Ensure required fields with defaults
                empty = self._create_empty_output()
                data["keywords"] = data.get("keywords", empty["keywords"])
                data["keyword_scores"] = data.get("keyword_scores", empty["keyword_scores"])
                data["compound_words"] = data.get("compound_words", empty["compound_words"])
                data["domain_keywords"] = data.get("domain_keywords", empty["domain_keywords"])

                return data

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
                return self._create_empty_output()

        except Exception as e:
            self.logger.error(f"Error processing output: {e}")
            return self._create_empty_output()

    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate analyzer parameters."""
        if "max_keywords" in params and (not isinstance(params["max_keywords"], int) or params["max_keywords"] < 1):
            raise ValueError("max_keywords must be a positive integer")

        if "min_keyword_length" in params and (
            not isinstance(params["min_keyword_length"], int) or params["min_keyword_length"] < 1
        ):
            raise ValueError("min_keyword_length must be a positive integer")