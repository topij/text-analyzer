# src/analyzers/keyword_analyzer.py

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

from src.core.language_processing.base import BaseTextProcessor

from .base import AnalyzerOutput, TextAnalyzer

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
        custom_stop_words: Optional[Set[str]] = None,
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
        self.min_keyword_length = config.get("min_keyword_length", 3)
        self.max_keywords = config.get("max_keywords", 10)

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain for keyword extraction."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a keyword extraction expert. Extract important keywords and phrases from text.
            Consider:
            - Technical terms and domain-specific vocabulary
            - Important concepts and themes
            - Named entities and proper nouns
            - Compound terms and multi-word phrases
            
            Return them in this format:
            {{
                "keywords": ["keyword1", "keyword2", ...],
                "compound_words": ["compound1", "compound2", ...],
                "confidence_scores": {{"keyword1": 0.95, "keyword2": 0.85, ...}}
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
            - Focus on: {focus}
            
            Return in the specified JSON format.""",
                ),
            ]
        )

        # Create processing chain
        chain = (
            {
                "text": RunnablePassthrough(),
                "max_keywords": lambda _: self.max_keywords,
                "statistical_keywords": self._extract_statistical_keywords,
                "min_length": lambda _: self.min_keyword_length,
                "focus": lambda _: self.config.get("focus", "general topics"),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    async def analyze(self, text: str, **kwargs) -> KeywordOutput:
        """Analyze text to extract keywords."""
        if not text:
            return self._handle_error("Empty text")

        try:
            # Ensure text is a string
            text_str = str(text)
            logger.debug(f"Processing text length: {len(text_str)}")

            # Process text
            if self.language_processor:
                processed_text = self.language_processor.process_text(text_str)
                logger.debug("Text processed by language processor")
            else:
                processed_text = text_str

            # Extract statistical keywords
            try:
                statistical_keywords = self._extract_statistical_keywords(processed_text)
                logger.debug(f"Found {len(statistical_keywords)} statistical keywords")
            except Exception as e:
                logger.error(f"Statistical keyword extraction failed: {e}")
                statistical_keywords = []

            # Prepare chain input
            chain_input = {
                "text": text_str,
                "max_keywords": self.max_keywords,
                "statistical_keywords": statistical_keywords,
                "min_length": self.min_keyword_length,
                "focus": self.config.get("focus", "general topics"),
            }

            # Get LLM results
            try:
                llm_results = await self.chain.ainvoke(chain_input)
                logger.debug("LLM analysis completed")
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                llm_results = {
                    "keywords": statistical_keywords,
                    "compound_words": [],
                    "confidence_scores": {k: 0.5 for k in statistical_keywords},
                }

            # Process results
            processed_results = self._post_process_llm_output(llm_results)

            # Handle Finnish compounds if applicable
            keywords = list(processed_results["keywords"])
            if (
                hasattr(self.language_processor, "COMPOUND_MAPPINGS")
                and hasattr(self.language_processor, "language")
                and self.language_processor.language == "fi"
            ):

                text_lower = text_str.lower()
                compound_words = set()

                for (
                    compound,
                    parts,
                ) in self.language_processor.COMPOUND_MAPPINGS.items():
                    if compound in text_lower:
                        compound_words.add(compound)
                        compound_words.update(parts)

                if compound_words:
                    keywords.extend(compound_words)
                    # Update confidence scores
                    for word in compound_words:
                        processed_results["confidence_scores"][word] = 1.0

            # Ensure unique keywords while preserving order
            unique_keywords = list(dict.fromkeys(keywords))

            return KeywordOutput(
                keywords=unique_keywords,
                keyword_scores=processed_results["confidence_scores"],
                statistical_keywords=statistical_keywords,
                compound_words=processed_results.get("compound_words", []),
                language=self._detect_language(text_str),
            )

        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            error_output = self._handle_error(f"Keyword analysis failed: {str(e)}")
            # Ensure consistent structure even in error case
            if not hasattr(error_output, "keywords"):
                error_output.keywords = []
            return error_output

    def _extract_statistical_keywords(self, text: str) -> List[str]:
        """Extract keywords using statistical methods."""
        if not text:
            return []

        try:
            # Ensure text is a string and tokenize
            if isinstance(text, (list, tuple)):
                words = [str(w) for w in text]
            else:
                text_str = str(text)
                words = self._tokenize(text_str)

            # Filter words
            valid_words = [
                word.lower()
                for word in words
                if (
                    len(word) >= self.min_keyword_length
                    and (not self.language_processor or not self.language_processor.is_stop_word(word.lower()))
                )
            ]

            if not valid_words:
                return []

            # Count frequencies
            freq = Counter(valid_words)
            total_words = float(len(valid_words))  # Ensure float division

            # Calculate scores (TF-IDF)
            scores = {}
            for word, count in freq.items():
                try:
                    tf = float(count) / total_words
                    idf = math.log(total_words / (float(count) + 1.0))
                    scores[word] = tf * idf
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating score for {word}: {e}")
                    scores[word] = 0.0

            # Sort and return top keywords
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[: self.max_keywords]]

        except Exception as e:
            logger.error(f"Error extracting statistical keywords: {e}")
            return []

    def _combine_keywords(
        self,
        statistical: List[str],
        llm: List[str],
        confidence_scores: Dict[str, float],
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
                combined[kw]["score"] = max(combined[kw]["score"], confidence_scores.get(kw, 0.5))
                combined[kw]["sources"].append("llm")
            else:
                combined[kw] = {
                    "score": confidence_scores.get(kw, 0.5),
                    "sources": ["llm"],
                }

        # Sort by score and number of sources
        sorted_keywords = sorted(
            combined.items(),
            key=lambda x: (x[1]["score"], len(x[1]["sources"])),
            reverse=True,
        )

        return [kw for kw, _ in sorted_keywords]

    def _calculate_final_scores(self, keywords: List[str]) -> Dict[str, float]:
        """Calculate final confidence scores for keywords."""
        # This could be enhanced with more sophisticated scoring
        return {kw: 1.0 - (i / len(keywords)) for i, kw in enumerate(keywords)}

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output to standardized format."""
        try:
            # Convert message content to dict
            if hasattr(output, "content"):
                import json

                try:
                    parsed = json.loads(output.content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM output as JSON")
                    return {
                        "keywords": [],
                        "compound_words": [],
                        "confidence_scores": {},
                    }
            elif isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM output as JSON")
                    return {
                        "keywords": [],
                        "compound_words": [],
                        "confidence_scores": {},
                    }
            else:
                parsed = output

            # Ensure required fields are present
            return {
                "keywords": parsed.get("keywords", []),
                "compound_words": parsed.get("compound_words", []),
                "confidence_scores": parsed.get("confidence_scores", {}),
            }

        except Exception as e:
            logger.error(f"Error processing LLM output: {str(e)}")
            return {"keywords": [], "compound_words": [], "confidence_scores": {}}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.language_processor:
            return self.language_processor.tokenize(text)
        # Simple fallback tokenization
        return text.lower().split()
