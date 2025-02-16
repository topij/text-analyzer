# tests/helpers/mock_llms/category_mock.py

import logging
from typing import Any, List, Dict, Tuple, Optional
import json

from langchain_core.messages import BaseMessage
from src.schemas import (
    CategoryMatch,
    CategoryOutput,
    CategoryAnalysisResult,
    Evidence
)
from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class CategoryMockLLM(BaseMockLLM):
    """Mock LLM for category analysis testing."""

    def __init__(self):
        super().__init__()
        self._config = {}  # Use private attribute instead of pydantic field

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

        return language, content_type

    def _get_mock_response(
        self, messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Get category-specific mock response."""
        last_message = messages[-1].content if messages else ""

        if not last_message or not last_message.strip():
            return {
                "categories": [],
                "language": "en",
                "success": False,
                "error": "Empty input text"
            }

        # Get min_confidence from config, default to 0.3 if not set
        min_confidence = self._config.get("min_confidence", 0.3)
        
        # Adjust base confidence based on min_confidence
        # For higher min_confidence, we'll return fewer but higher confidence matches
        base_confidence = max(0.85 if min_confidence <= 0.3 else 0.95, min_confidence + 0.1)

        language, content_type = self._detect_content_type(last_message)
        logger.debug(f"Using min_confidence: {min_confidence}, base_confidence: {base_confidence}")

        # Get appropriate response based on language and content
        return (
            self._get_finnish_response(content_type, base_confidence)
            if language == "fi"
            else self._get_english_response(content_type, base_confidence)
        )

    def _get_english_response(self, content_type: str, base_confidence: float) -> Dict[str, Any]:
        """Get English category response."""
        logger.debug(
            f"CategoryMockLLM: Getting English response for content_type={content_type}, base_confidence={base_confidence}"
        )

        if content_type == "technical":
            return {
                "categories": [
                    {
                        "name": "Technical",
                        "confidence": base_confidence,
                        "description": "Technical content with software and API focus",
                        "evidence": [
                            {
                                "text": "Machine learning models are trained using large datasets",
                                "relevance": min(base_confidence + 0.05, 1.0),
                            },
                            {
                                "text": "Neural networks enable complex pattern recognition",
                                "relevance": base_confidence,
                            },
                        ],
                        "themes": ["Software Development", "Machine Learning"],
                    },
                ],
                "language": "en",
                "success": True
            }
        else:  # business content
            return {
                "categories": [
                    {
                        "name": "Business",
                        "confidence": base_confidence,
                        "description": "Business metrics and financial analysis",
                        "evidence": [
                            {
                                "text": "Q3 financial results show 15% revenue growth",
                                "relevance": min(base_confidence + 0.05, 1.0),
                            },
                            {
                                "text": "Market expansion strategy focuses on emerging sectors",
                                "relevance": base_confidence,
                            },
                        ],
                        "themes": ["Financial Performance", "Market Strategy"],
                    },
                ],
                "language": "en",
                "success": True
            }

    def _get_finnish_response(self, content_type: str, base_confidence: float) -> Dict[str, Any]:
        """Get Finnish category response."""
        logger.debug(
            f"CategoryMockLLM: Getting Finnish response for content_type={content_type}, base_confidence={base_confidence}"
        )

        if content_type == "technical":
            return {
                "categories": [
                    {
                        "name": "Technical",
                        "confidence": base_confidence,
                        "description": "Tekninen sisältö ja ohjelmistokehitys",
                        "evidence": [
                            {
                                "text": "Koneoppimismallit analysoivat dataa tehokkaasti",
                                "relevance": min(base_confidence + 0.05, 1.0),
                            },
                            {
                                "text": "Järjestelmän rajapinta käsittelee tiedon validoinnin",
                                "relevance": base_confidence,
                            },
                        ],
                        "themes": ["Ohjelmistokehitys", "Data-analyysi"],
                    },
                ],
                "language": "fi",
                "success": True
            }
        else:  # business content
            return {
                "categories": [
                    {
                        "name": "Business",
                        "confidence": base_confidence,
                        "description": "Liiketoiminnan mittarit ja talousanalyysi",
                        "evidence": [
                            {
                                "text": "Q3 taloudelliset tulokset osoittavat 15% kasvun",
                                "relevance": min(base_confidence + 0.05, 1.0),
                            },
                            {
                                "text": "Markkinalaajuuden strategia keskittyy uusiin sektoreihin",
                                "relevance": base_confidence,
                            },
                        ],
                        "themes": ["Taloudellinen suorituskyky", "Markkinastrategia"],
                    },
                ],
                "language": "fi",
                "success": True
            }
