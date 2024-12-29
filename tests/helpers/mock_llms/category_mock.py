# tests/helpers/mock_llms/category_mock.py

import logging
from typing import Any, List, Dict, Tuple
import json

from langchain_core.messages import BaseMessage
from src.schemas import CategoryMatch, CategoryOutput, Evidence
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

    def _get_mock_response(self, messages: List[BaseMessage]) -> CategoryOutput:
        """Get category-specific mock response with structured output."""
        last_message = messages[-1].content if messages else ""
        logger.debug(f"CategoryMockLLM: Processing message: {last_message}")

        # Extract config from the messages
        for msg in messages:
            if isinstance(msg.content, str) and "config" in msg.content.lower():
                try:
                    # Look for config in the message content
                    config_str = msg.content.split("config=", 1)[1].strip()
                    # Remove any trailing text after the config JSON
                    config_str = config_str.split("\n")[0].strip()
                    # Parse the config
                    self._config.update(json.loads(config_str))
                    logger.debug(f"Found config in message: {self._config}")
                except Exception as e:
                    logger.warning(f"Failed to parse config from message: {e}")

        if not last_message:
            return CategoryOutput(
                categories=[],
                language="en",
                success=False,
                error="Empty input text",
            )

        language, content_type = self._detect_content_type(last_message)
        logger.debug(
            f"Detected language={language}, content_type={content_type}"
        )

        if language == "fi":
            response = self._get_finnish_response(content_type)
        else:
            response = self._get_english_response(content_type)

        # Adjust confidence based on focus
        if "focus_on" in self._config:
            focus = self._config["focus_on"].lower()
            for cat in response.categories:
                if focus == "technical analysis" and cat.name == "Technical":
                    cat.confidence *= 1.1  # Boost technical confidence
                elif focus == "business analysis" and cat.name == "Business":
                    cat.confidence *= 1.1  # Boost business confidence
                else:
                    cat.confidence *= 0.9  # Reduce other confidences

                # Ensure confidence stays in valid range
                cat.confidence = min(1.0, max(0.0, cat.confidence))

        # Filter by min_confidence if set
        if "min_confidence" in self._config:
            min_conf = float(self._config["min_confidence"])
            response.categories = [
                cat for cat in response.categories 
                if cat.confidence >= min_conf
            ]

        return response

    def _get_english_response(self, content_type: str) -> CategoryOutput:
        """Get English category response."""
        logger.debug(
            f"CategoryMockLLM: Getting English response for content_type={content_type}"
        )

        if content_type == "technical":
            return CategoryOutput(
                categories=[
                    CategoryMatch(
                        name="Technical",
                        confidence=0.75,  # Lower base confidence
                        description="Technical content with software and API focus",
                        evidence=[
                            Evidence(
                                text="Machine learning models are trained using large datasets",
                                relevance=0.9,
                            ),
                            Evidence(
                                text="Neural networks enable complex pattern recognition",
                                relevance=0.85,
                            ),
                        ],
                        themes=["Software Development", "Machine Learning"],
                    ),
                ],
                language="en",
                success=True,
            )
        else:  # business content
            return CategoryOutput(
                categories=[
                    CategoryMatch(
                        name="Business",
                        confidence=0.75,  # Lower base confidence
                        description="Business metrics and financial analysis",
                        evidence=[
                            Evidence(
                                text="Q3 financial results show 15% revenue growth",
                                relevance=0.95,
                            ),
                            Evidence(
                                text="Market expansion strategy focuses on emerging sectors",
                                relevance=0.90,
                            ),
                        ],
                        themes=["Financial Performance", "Market Strategy"],
                    ),
                ],
                language="en",
                success=True,
            )

    def _get_finnish_response(self, content_type: str) -> CategoryOutput:
        """Get Finnish category response."""
        logger.debug(
            f"CategoryMockLLM: Getting Finnish response for content_type={content_type}"
        )

        if content_type == "technical":
            return CategoryOutput(
                categories=[
                    CategoryMatch(
                        name="Technical",
                        confidence=0.75,  # Lower base confidence
                        description="Tekninen sisältö ja ohjelmistokehitys",
                        evidence=[
                            Evidence(
                                text="Koneoppimismallit analysoivat dataa tehokkaasti",
                                relevance=0.9,
                            ),
                            Evidence(
                                text="Järjestelmän rajapinta käsittelee tiedon validoinnin",
                                relevance=0.85,
                            ),
                        ],
                        themes=["Ohjelmistokehitys", "Data-analyysi"],
                    ),
                ],
                language="fi",
                success=True,
            )
        else:  # business content
            return CategoryOutput(
                categories=[
                    CategoryMatch(
                        name="Business",
                        confidence=0.75,  # Lower base confidence
                        description="Liiketoiminnan mittarit ja talousanalyysi",
                        evidence=[
                            Evidence(
                                text="Q3 taloudelliset tulokset osoittavat 15% kasvun",
                                relevance=0.95,
                            ),
                            Evidence(
                                text="Markkinalaajuuden strategia keskittyy uusiin sektoreihin",
                                relevance=0.90,
                            ),
                        ],
                        themes=["Taloudellinen suorituskyky", "Markkinastrategia"],
                    ),
                ],
                language="fi",
                success=True,
            )
