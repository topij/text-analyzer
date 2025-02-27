# tests/helpers/mock_llms/keyword_mock.py

import logging
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.schemas import KeywordInfo, KeywordOutput, KeywordAnalysisResult
from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class KeywordMockLLM(BaseMockLLM):
    """Mock LLM for keyword analysis testing."""

    def _get_mock_response(
        self, messages: List[BaseMessage]
    ) -> KeywordOutput:
        """Get keyword-specific mock response."""
        last_message = messages[-1].content if messages else ""

        if not last_message or not last_message.strip():
            return KeywordOutput(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language="en",
                success=False,
                error="Empty input text"
            )

        language, content_type = self._detect_content_type(last_message)

        # Get appropriate response based on language and content
        response_dict = (
            self._get_finnish_response(content_type)
            if language == "fi"
            else self._get_english_response(content_type)
        )

        # Convert keywords to KeywordInfo objects
        keywords = [KeywordInfo(**kw) for kw in response_dict["keywords"]]
        response_dict["keywords"] = keywords

        # Ensure success and error fields are set
        response_dict["success"] = True
        response_dict["error"] = None

        # Convert to KeywordOutput
        return KeywordOutput(**response_dict)

    def _get_english_response(self, content_type: str) -> Dict[str, Any]:
        """Get English keyword response."""
        if content_type == "technical":
            return {
                "keywords": [
                    {
                        "keyword": "machine learning",
                        "score": 0.9,
                        "domain": "technical",
                        "compound_parts": ["machine", "learning"],
                        "frequency": 1,
                        "metadata": {}
                    },
                    {
                        "keyword": "neural network",
                        "score": 0.85,
                        "domain": "technical",
                        "compound_parts": ["neural", "network"],
                        "frequency": 1,
                        "metadata": {}
                    },
                    {
                        "keyword": "data",
                        "score": 0.8,
                        "domain": "technical",
                        "frequency": 1,
                        "metadata": {}
                    },
                ],
                "compound_words": ["machine learning", "neural network"],
                "domain_keywords": {
                    "technical": ["machine learning", "neural network", "data"]
                },
                "language": "en",
                "success": True,
                "error": None
            }
        else:  # business content
            return {
                "keywords": [
                    {
                        "keyword": "revenue growth",
                        "score": 0.9,
                        "domain": "business",
                        "compound_parts": ["revenue", "growth"],
                        "frequency": 1,
                        "metadata": {}
                    },
                    {
                        "keyword": "market share",
                        "score": 0.85,
                        "domain": "business",
                        "compound_parts": ["market", "share"],
                        "frequency": 1,
                        "metadata": {}
                    },
                ],
                "compound_words": ["revenue growth", "market share"],
                "domain_keywords": {
                    "business": ["revenue growth", "market share"]
                },
                "language": "en",
                "success": True,
                "error": None
            }

    def _get_finnish_response(self, content_type: str) -> Dict[str, Any]:
        """Get Finnish keyword response."""
        if content_type == "technical":
            return {
                "keywords": [
                    {
                        "keyword": "tekoäly",
                        "score": 0.9,
                        "domain": "technical",
                        "frequency": 1,
                        "metadata": {}
                    },
                    {
                        "keyword": "koneoppiminen",
                        "score": 0.85,
                        "domain": "technical",
                        "compound_parts": ["kone", "oppiminen"],
                        "frequency": 1,
                        "metadata": {}
                    },
                    {
                        "keyword": "ohjelmistokehitys",
                        "score": 0.8,
                        "domain": "technical",
                        "compound_parts": ["ohjelmisto", "kehitys"],
                        "frequency": 1,
                        "metadata": {}
                    },
                ],
                "compound_words": ["koneoppiminen", "ohjelmistokehitys"],
                "domain_keywords": {
                    "technical": ["tekoäly", "koneoppiminen", "ohjelmistokehitys"]
                },
                "language": "fi",
                "success": True,
                "error": None
            }
        else:  # business content
            return {
                "keywords": [
                    {
                        "keyword": "liikevaihdon kasvu",
                        "score": 0.9,
                        "domain": "business",
                        "compound_parts": ["liike", "vaihto", "kasvu"],
                        "frequency": 1,
                        "metadata": {}
                    },
                    {
                        "keyword": "markkinaosuus",
                        "score": 0.85,
                        "domain": "business",
                        "compound_parts": ["markkina", "osuus"],
                        "frequency": 1,
                        "metadata": {}
                    },
                ],
                "compound_words": ["liikevaihdon kasvu", "markkinaosuus"],
                "domain_keywords": {
                    "business": ["liikevaihdon kasvu", "markkinaosuus"]
                },
                "language": "fi",
                "success": True,
                "error": None
            }

    def _detect_content_type(self, message: str) -> Tuple[str, str]:
        """Detect language and content type from message."""
        message = message.lower()

        # Finnish language detection - check first
        finnish_terms = {
            "koneoppimis",
            "neuroverko",
            "järjestelmä",
            "rajapinta",
            "validointi",
            "käsittelee",
            "analysoivat",
            "mallit",
            "dataa",
            "tiedon",
            "tehokkaasti",
            "tekoäly",  # Add common Finnish test terms
            "ohjelmistokehitys",
            "teknologia",  # Add more terms from test
            "osaamista",
        }

        # Check for Finnish text presence
        finnish_matches = [term for term in finnish_terms if term in message]
        is_finnish = bool(finnish_matches)
        language = "fi" if is_finnish else "en"
        logger.debug(
            f"Language detection - Finnish terms found: {finnish_matches}"
        )
        logger.debug(f"Language detection - Using language: {language}")

        # Business indicators by language
        business_indicators = {
            "en": [
                "revenue growth",
                "financial results",
                "market expansion",
                "customer acquisition",
                "q3",
                "quarterly",
            ],
            "fi": [
                "liikevaihdon kasvu",
                "taloudelliset tulokset",
                "markkinalaajuus",
                "asiakashankinta",
            ],
        }

        # Technical indicators by language
        technical_indicators = {
            "en": [
                "machine learning",
                "neural network",
                "trained",
                "datasets",
                "api",
                "system",
            ],
            "fi": [
                "koneoppimis",
                "neuroverko",
                "data",
                "järjestelmä",
                "rajapinta",
                "validointi",
                "analysoida",
                "tekoäly",  # Add technical Finnish terms
                "teknologia",
                "ohjelmistokehitys",
            ],
        }

        # Count matches for detected language
        tech_count = sum(
            1 for term in technical_indicators[language] if term in message
        )
        business_count = sum(
            1 for term in business_indicators[language] if term in message
        )

        logger.debug(
            f"Language {language}: Technical matches: {tech_count}, Business matches: {business_count}"
        )
        content_type = (
            "business" if business_count > tech_count else "technical"
        )
        logger.debug(f"Detected content type: {content_type}")

        return language, content_type
