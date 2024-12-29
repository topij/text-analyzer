# tests/helpers/mock_llms/keyword_mock.py

import logging
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.schemas import KeywordInfo, KeywordOutput
from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class KeywordMockLLM(BaseMockLLM):
    """Mock LLM for keyword analysis testing."""

    def _get_mock_response(
        self, messages: List[BaseMessage]
    ) -> str:  # Change return type to str
        """Get keyword-specific mock response as JSON string."""
        last_message = messages[-1].content if messages else ""

        if not last_message:
            # Return directly as JSON string
            return json.dumps(
                {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "language": "en",
                    "success": False,
                    "error": "Empty input text",
                }
            )

        language, content_type = self._detect_content_type(last_message)

        # Get appropriate response
        response = (
            self._get_finnish_response(content_type)
            if language == "fi"
            else self._get_english_response(content_type)
        )

        # Return JSON string
        return (
            response.model_dump_json()
            if isinstance(response, BaseModel)
            else json.dumps(response)
        )

    def _get_english_response(self, content_type: str) -> KeywordOutput:
        """Get English keyword response."""
        if content_type == "technical":
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="machine learning",
                        score=0.9,
                        domain="technical",
                        compound_parts=["machine", "learning"],
                    ),
                    KeywordInfo(
                        keyword="neural network",
                        score=0.85,
                        domain="technical",
                        compound_parts=["neural", "network"],
                    ),
                    KeywordInfo(keyword="data", score=0.8, domain="technical"),
                ],
                compound_words=["machine learning", "neural network"],
                domain_keywords={
                    "technical": ["machine learning", "neural network", "data"]
                },
                language="en",
                success=True,
            )
        else:  # business content
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="revenue growth",
                        score=0.9,
                        domain="business",
                        compound_parts=["revenue", "growth"],
                    ),
                    KeywordInfo(
                        keyword="market share",
                        score=0.85,
                        domain="business",
                        compound_parts=["market", "share"],
                    ),
                ],
                compound_words=["revenue growth", "market share"],
                domain_keywords={
                    "business": ["revenue growth", "market share"]
                },
                language="en",
                success=True,
            )

    def _get_finnish_response(self, content_type: str) -> KeywordOutput:
        """Get Finnish keyword response."""
        if content_type == "technical":
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="tekoäly",
                        score=0.9,
                        domain="technical",
                    ),
                    KeywordInfo(
                        keyword="koneoppiminen",
                        score=0.85,
                        domain="technical",
                        compound_parts=["kone", "oppiminen"],
                    ),
                    KeywordInfo(
                        keyword="ohjelmistokehitys",
                        score=0.8,
                        domain="technical",
                        compound_parts=["ohjelmisto", "kehitys"],
                    ),
                ],
                compound_words=["koneoppiminen", "ohjelmistokehitys"],
                domain_keywords={
                    "technical": ["tekoäly", "koneoppiminen", "ohjelmistokehitys"]
                },
                language="fi",
                success=True,
            )
        else:  # business content
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="liikevaihdon kasvu",
                        score=0.9,
                        domain="business",
                        compound_parts=["liike", "vaihto", "kasvu"],
                    ),
                    KeywordInfo(
                        keyword="markkinaosuus",
                        score=0.85,
                        domain="business",
                        compound_parts=["markkina", "osuus"],
                    ),
                ],
                compound_words=["liikevaihdon kasvu", "markkinaosuus"],
                domain_keywords={
                    "business": ["liikevaihdon kasvu", "markkinaosuus"]
                },
                language="fi",
                success=True,
            )

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
