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
        template = ChatPromptTemplate.from_messages([
            ("system", """You are a keyword extraction expert. Extract important keywords and phrases from text.
            Do NOT include common stopwords or generic terms.
            Consider the text's context and focus on meaningful terms.
            
            Return results in JSON format with these exact fields:
            {{
                "keywords": ["keyword1", "keyword2", ...],
                "keyword_scores": {{"keyword1": 0.9, "keyword2": 0.8, ...}},
                "compound_words": ["word1+word2", ...],
                "domain_keywords": {{"domain1": ["kw1", "kw2"], ...}}
            }}"""),  # Note the double curly braces to escape JSON template
            
            ("human", """Extract keywords from this text, focusing on {focus}.
            Text: {text}
            
            Guidelines:
            - Max keywords: {max_keywords}
            - Statistical keywords to consider: {statistical_keywords}
            - Min length: {min_length} characters
            - Excluded words: {excluded_words}
            - Focus on business/technical terms""")
        ])

        chain = (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.config.get("max_keywords", 8),
                "statistical_keywords": lambda x: self._get_statistical_keywords_str(x),
                "min_length": lambda _: self.config.get("min_keyword_length", 3),
                "focus": lambda _: self.config.get("focus", "general"),
                "excluded_words": lambda _: list(
                    self.language_processor._stop_words.union(
                        self.language_processor.excluded_keywords
                    ) if self.language_processor else set()
                )
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

            # Get LLM keywords
            llm_results = await self.chain.ainvoke(text)

            # Ensure default structure even on error
            if "error" in llm_results:
                return KeywordOutput(
                    keywords=[],
                    keyword_scores={},
                    compound_words=[],
                    domain_keywords={},
                    error=llm_results["error"],
                    success=False,
                    language=self.language_processor.language if self.language_processor else "unknown",
                )

            # Get statistical keywords and combine results
            statistical_keywords = self._extract_statistical_keywords(text)
            combined_keywords = self._combine_keywords(
                statistical_keywords, llm_results.get("keyword_scores", {}), self.weights
            )

            max_keywords = self.config.get("max_keywords", 8)
            return KeywordOutput(
                keywords=list(combined_keywords.keys())[:max_keywords],
                keyword_scores=combined_keywords,
                compound_words=llm_results.get("compound_words", []),
                domain_keywords=llm_results.get("domain_keywords", {}),
                success=True,
                language=self.language_processor.language if self.language_processor else "unknown",
            )

        except Exception as e:
            self.logger.error(f"Keyword analysis failed: {e}", exc_info=True)
            return KeywordOutput(
                keywords=[],
                keyword_scores={},
                compound_words=[],
                domain_keywords={},
                error=str(e),
                success=False,
                language=self.language_processor.language if self.language_processor else "unknown",
            )

    def _get_statistical_keywords_str(self, text: str) -> str:
        """Get statistical keywords as a formatted string."""
        keywords = self._extract_statistical_keywords(text)
        return ", ".join(f"'{k}'" for k in sorted(keywords.keys(), key=keywords.get, reverse=True)[:5])

    def _extract_statistical_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords using statistical methods."""
        # Use language processor if available
        if self.language_processor:
            # Tokenize
            words = self.language_processor.tokenize(text)
            logger.debug(f"Initial tokens: {len(words)}")
            
            # Filter and get base forms
            filtered_words = []
            for word in words:
                # Skip if it should be excluded
                if not self.language_processor.should_exclude_word(word):
                    base_form = self.language_processor.get_base_form(word)
                    if self.language_processor.should_keep_word(word, base_form):
                        filtered_words.append(base_form)
            
            logger.debug(f"Words after filtering: {len(filtered_words)}")
            
            # Calculate frequencies
            freq = Counter(filtered_words)
        else:
            base_forms = text.lower().split()
            freq = Counter(base_forms)
        
        # Calculate scores
        total_words = len(filtered_words if self.language_processor else base_forms)
        scores = {}
        
        for word, count in freq.items():
            if len(word) >= self.config.get("min_keyword_length", 3):
                tf = count / total_words
                idf = math.log(total_words / (count + 1))
                scores[word] = tf * idf
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            scores = {k: v/max_score for k, v in scores.items()}
            
        logger.debug(f"Extracted statistical keywords: {list(scores.keys())}")
        return scores

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
