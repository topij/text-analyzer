# tests/helpers/mock_llms/category_mock.py

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage

from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class CategoryMockLLM(BaseMockLLM):
    """Mock LLM for category analysis testing."""

    def __init__(self):
        """Initialize with call tracking."""
        super().__init__()
        self._call_history = []

    def _detect_content_type(self, message: str) -> Tuple[str, str]:
        """Detect language and content type from message with improved classification."""
        message = message.lower()

        # Language detection - Finnish indicators
        is_finnish = any(
            term in message
            for term in [
                "koneoppimis",
                "neuroverko",
                "datan",
                "piirteiden",
                "taloudellinen",
                "liikevaihto",
                "markkinalaaje",
            ]
        )

        # Score-based classification
        technical_score = sum(
            1
            for term in [
                "machine learning",
                "neural network",
                "data",
                "preprocessing",
                "feature engineering",
                "architecture",
                "model",
                "layers",
                # Finnish technical terms
                "koneoppimis",
                "neuroverko",
                "datan",
                "esikäsittely",
                "piirteiden",
                "arkkitehtuuri",
                "malli",
            ]
            if term in message
        )

        business_score = sum(
            1
            for term in [
                "financial",
                "revenue",
                "market",
                "profit",
                "q3",
                "growth",
                "strategy",
                "margins",
                # Finnish business terms
                "liikevaihto",
                "tulos",
                "markkina",
                "strategia",
                "kasvu",
                "taloudelliset",
            ]
            if term in message
        )

        # Determine content type based on scores
        language = "fi" if is_finnish else "en"
        if technical_score > business_score:
            content_type = "technical"
        elif business_score > 0:
            content_type = "business"
        else:
            content_type = (
                "technical"  # Default to technical if no clear signal
            )

        logger.debug(
            f"Content detection - Technical score: {technical_score}, Business score: {business_score}"
        )
        logger.debug(
            f"Selected content type: {content_type}, Language: {language}"
        )

        return language, content_type

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get category-specific mock response with improved error handling."""
        try:
            last_message = messages[-1].content if messages else ""
            self._call_history.append(last_message)

            # Basic validation
            if not last_message:
                return json.dumps(
                    {"categories": [], "success": True, "language": "en"}
                )

            # Get language and content type
            language, content_type = self._detect_content_type(last_message)

            logger.debug(
                f"Detected language: {language}, content type: {content_type}"
            )

            # Get appropriate response
            if language == "fi":
                response = self._get_finnish_response(content_type)
            else:
                response = self._get_english_response(content_type)

            # Parse and ensure success is set
            try:
                data = json.loads(response)
                data["success"] = True
                return json.dumps(data)
            except json.JSONDecodeError:
                return json.dumps(
                    {"categories": [], "success": True, "language": language}
                )

        except Exception as e:
            logger.error(f"Error in mock LLM: {e}")
            return json.dumps(
                {"categories": [], "success": True, "language": "en"}
            )

    def _get_default_response(self, language: str, content_type: str) -> str:
        """Get default response when category parsing fails."""
        categories = [
            {
                "category": (
                    "Technical" if content_type == "technical" else "Business"
                ),
                "confidence": 0.85,
                "explanation": (
                    "General technical content"
                    if content_type == "technical"
                    else "General business content"
                ),
                "evidence": [{"text": "Default evidence", "relevance": 0.8}],
                "themes": ["Default Theme"],
            }
        ]
        return json.dumps(
            {"categories": categories, "success": True, "language": language}
        )

    def _parse_categories_from_prompt(self, message: str) -> Dict[str, Dict]:
        """Extract categories from the prompt message."""
        try:
            # Find categories list in the prompt
            start_marker = "Categories:"
            start_idx = message.find(start_marker)
            if start_idx == -1:
                return {}

            # Find the section between Categories: and the next prompt section
            text_markers = ["Language:", "Guidelines:", "Text:"]
            end_idx = len(message)
            for marker in text_markers:
                marker_idx = message.find(marker, start_idx + len(start_marker))
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx

            categories_text = message[
                start_idx + len(start_marker) : end_idx
            ].strip()

            # Debug log the extracted text
            logger.debug(
                f"Extracted categories text: {categories_text[:100]}..."
            )

            # Try to parse the text as a Python literal first (safer than eval)
            try:
                import ast

                categories = ast.literal_eval(categories_text)
            except:
                # If that fails, try cleaning and parsing as JSON
                cleaned_text = categories_text.replace("'", '"')
                # Remove any trailing commas before closing brackets
                cleaned_text = cleaned_text.replace(",]", "]").replace(
                    ",}", "}"
                )
                # Remove newlines and extra whitespace
                cleaned_text = " ".join(cleaned_text.split())
                categories = json.loads(cleaned_text)

            # Verify we got a list
            if not isinstance(categories, list):
                logger.warning("Categories not in expected list format")
                return {}

            # Convert to required format
            return {
                cat["name"]: {
                    "description": cat.get("description", ""),
                    "keywords": cat.get("keywords", []),
                    "threshold": cat.get("threshold", 0.7),
                }
                for cat in categories
                if isinstance(cat, dict) and "name" in cat
            }

        except Exception as e:
            logger.error(f"Error parsing categories from prompt: {e}")
            logger.debug(f"Attempted to parse from message: {message[:200]}...")
            return {}

    def _parse_categories_from_prompt(self, message: str) -> Dict[str, Dict]:
        """Extract categories from the prompt message."""
        try:
            # Find categories list in the prompt
            start_marker = "Categories:"
            start_idx = message.find(start_marker)
            if start_idx == -1:
                return {}

            # Find the section between Categories: and the next prompt section
            text_marker = "Guidelines:"
            end_idx = message.find(text_marker, start_idx)
            if end_idx == -1:
                end_idx = len(message)

            categories_text = message[
                start_idx + len(start_marker) : end_idx
            ].strip()

            # Parse the list of category dictionaries
            categories = []
            try:
                categories = json.loads(categories_text)
            except json.JSONDecodeError:
                # Try cleaning the text and parsing again
                cleaned_text = categories_text.replace("'", '"').replace(
                    "\n", " "
                )
                categories = json.loads(cleaned_text)

            # Convert to expected format
            return {
                cat["name"]: {
                    "description": cat.get("description", ""),
                    "keywords": cat.get("keywords", []),
                    "threshold": cat.get("threshold", 0.7),
                }
                for cat in categories
                if isinstance(cat, dict) and "name" in cat
            }

        except Exception as e:
            logger.error(f"Error parsing categories from prompt: {e}")
            return {}

    def _get_english_response(self, content_type: str) -> str:
        """Get English category response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "categories": [
                        {
                            "category": "Machine Learning",
                            "confidence": 0.9,
                            "explanation": "Machine learning and AI content",
                            "evidence": [
                                {
                                    "text": "Machine learning models are trained using large datasets",
                                    "relevance": 0.9,
                                }
                            ],
                            "themes": ["AI Technology", "Neural Networks"],
                        },
                        {
                            "category": "Data Science",
                            "confidence": 0.85,
                            "explanation": "Data processing and analysis",
                            "evidence": [
                                {
                                    "text": "Data preprocessing and feature engineering",
                                    "relevance": 0.8,
                                }
                            ],
                            "themes": [
                                "Data Processing",
                                "Feature Engineering",
                            ],
                        },
                    ],
                    "success": True,
                    "language": "en",
                }
            )
        else:  # business content
            return json.dumps(
                {
                    "categories": [
                        {
                            "category": "Financial Analysis",  # Note different category
                            "confidence": 0.9,
                            "explanation": "Financial metrics and performance analysis",
                            "evidence": [
                                {
                                    "text": "Q3 financial results show 15% revenue growth",
                                    "relevance": 0.9,
                                }
                            ],
                            "themes": [
                                "Financial Performance",
                                "Growth Metrics",
                            ],
                        },
                        {
                            "category": "Market Strategy",  # Business focused category
                            "confidence": 0.85,
                            "explanation": "Market and business strategy analysis",
                            "evidence": [
                                {
                                    "text": "Market expansion strategy focuses on emerging sectors",
                                    "relevance": 0.85,
                                }
                            ],
                            "themes": ["Market Expansion", "Business Strategy"],
                        },
                    ],
                    "success": True,
                    "language": "en",
                }
            )

    def _get_finnish_response(self, content_type: str) -> str:
        """Get Finnish category response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "categories": [
                        {
                            "category": "Machine Learning",
                            "confidence": 0.9,
                            "explanation": "Koneoppimisen ja tekoälyn teknologia",
                            "evidence": [
                                {
                                    "text": "Koneoppimismalleja koulutetaan suurilla datajoukolla",
                                    "relevance": 0.9,
                                }
                            ],
                            "themes": ["Tekoäly", "Neuroverkot"],
                        },
                        {
                            "category": "Data Science",
                            "confidence": 0.85,
                            "explanation": "Datan käsittely ja analysointi",
                            "evidence": [
                                {
                                    "text": "Datan esikäsittely ja piirteiden suunnittelu",
                                    "relevance": 0.8,
                                }
                            ],
                            "themes": [
                                "Datan käsittely",
                                "Piirteiden suunnittelu",
                            ],
                        },
                    ],
                    "success": True,
                    "language": "fi",
                }
            )
        else:  # business content
            return json.dumps(
                {
                    "categories": [
                        {
                            "category": "Financial Analysis",
                            "confidence": 0.9,
                            "explanation": "Taloudelliset tulokset ja mittarit",
                            "evidence": [
                                {
                                    "text": "Q3 taloudelliset tulokset osoittavat liikevaihdon kasvun",
                                    "relevance": 0.9,
                                }
                            ],
                            "themes": ["Taloudellinen Suorituskyky", "Kasvu"],
                        },
                        {
                            "category": "Market Strategy",
                            "confidence": 0.85,
                            "explanation": "Markkinoiden ja liiketoiminnan strategia",
                            "evidence": [
                                {
                                    "text": "Markkinalaajennusstrategia keskittyy uusiin sektoreihin",
                                    "relevance": 0.85,
                                }
                            ],
                            "themes": ["Markkinakehitys", "Strategia"],
                        },
                    ],
                    "success": True,
                    "language": "fi",
                }
            )

    def get_calls(self) -> List[str]:
        """Get list of recorded calls."""
        return self._call_history

    def get_last_call(self) -> Optional[str]:
        """Get the last recorded call."""
        return self._call_history[-1] if self._call_history else None

    def reset_calls(self) -> None:
        """Reset call history."""
        self._call_history = []
