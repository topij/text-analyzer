# tests/helpers/mock_llms/category_mock.py

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class CategoryMockLLM(BaseMockLLM):
    """Mock LLM for category analysis testing."""

    def __init__(self):
        """Initialize with call tracking."""
        super().__init__()
        self._call_history = []

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get category-specific mock response with call tracking."""
        last_message = messages[-1].content if messages else ""
        self._call_history.append(last_message)

        if not last_message:
            return json.dumps(
                {
                    "categories": [],
                    "success": False,
                    "error": "Empty input text",
                    "language": "en",
                }
            )

        language, content_type = self._detect_content_type(last_message)

        # Get base text for content type detection
        text_marker = "Text:"
        start_idx = last_message.find(text_marker)
        content = (
            last_message[start_idx + len(text_marker) :].strip()
            if start_idx != -1
            else last_message
        )

        # Detect content type from actual text
        is_business = any(
            term in content.lower()
            for term in [
                "financial",
                "revenue",
                "profit",
                "market",
                "liikevaihto",
                "tulos",
                "markkina",
            ]
        )

        # Override content type based on actual content
        if is_business:
            content_type = "business"

        # Get appropriate response based on language and content
        return (
            self._get_finnish_response(content_type)
            if language == "fi"
            else self._get_english_response(content_type)
        )

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
                                },
                                {
                                    "text": "Neural network architecture includes multiple layers",
                                    "relevance": 0.85,
                                },
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
        else:  # business
            return json.dumps(
                {
                    "categories": [
                        {
                            "category": "Financial Analysis",
                            "confidence": 0.9,
                            "explanation": "Financial metrics and performance",
                            "evidence": [
                                {
                                    "text": "Q3 financial results show 15% revenue growth",
                                    "relevance": 0.9,
                                },
                                {
                                    "text": "Improved profit margins and market performance",
                                    "relevance": 0.85,
                                },
                            ],
                            "themes": [
                                "Financial Performance",
                                "Growth Metrics",
                            ],
                        },
                        {
                            "category": "Market Strategy",
                            "confidence": 0.85,
                            "explanation": "Market and business strategy",
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
                            "category": "Koneoppiminen",
                            "confidence": 0.9,
                            "explanation": "Koneoppimisen ja tekoälyn teknologia",
                            "evidence": [
                                {
                                    "text": "Koneoppimismalleja koulutetaan suurilla datajoukolla",
                                    "relevance": 0.9,
                                },
                                {
                                    "text": "Neuroverkon arkkitehtuuri sisältää useita kerroksia",
                                    "relevance": 0.85,
                                },
                            ],
                            "themes": ["Tekoäly", "Neuroverkot"],
                        },
                        {
                            "category": "Data-analyysi",
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
        else:  # business
            return json.dumps(
                {
                    "categories": [
                        {
                            "category": "Taloudellinen Analyysi",
                            "confidence": 0.9,
                            "explanation": "Taloudelliset tulokset ja mittarit",
                            "evidence": [
                                {
                                    "text": "Q3 taloudelliset tulokset osoittavat 15% kasvun",
                                    "relevance": 0.9,
                                }
                            ],
                            "themes": ["Taloudellinen Suorituskyky", "Kasvu"],
                        },
                        {
                            "category": "Markkinastrategia",
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

    def _parse_categories_from_prompt(self, message: str) -> Dict[str, Dict]:
        """Extract categories configuration from the prompt message."""
        try:
            # Find categories section in prompt
            marker = "Categories:"
            start_idx = message.find(marker)
            if start_idx == -1:
                logger.debug("No categories marker found in prompt")
                return {}

            # Extract text after "Categories:" and before "Text:"
            text_marker = "Text:"
            text_idx = message.find(text_marker)
            categories_text = (
                message[start_idx + len(marker) :]
                if text_idx == -1
                else message[start_idx + len(marker) : text_idx]
            )

            # Clean up the text and parse JSON
            json_text = categories_text.strip()
            categories = json.loads(json_text)

            return {
                cat.get("name"): {
                    "description": cat.get("description", ""),
                    "keywords": cat.get("keywords", []),
                    "threshold": cat.get("threshold", 0.7),
                }
                for cat in categories
                if "name" in cat
            }

        except Exception as e:
            logger.error(f"Error parsing categories from prompt: {e}")
            # Log the problematic text
            logger.debug(f"Attempted to parse: {message[:500]}...")
            return {}

    def get_calls(self) -> List[str]:
        """Get list of recorded calls."""
        return self._call_history

    def get_last_call(self) -> Optional[str]:
        """Get the last recorded call."""
        return self._call_history[-1] if self._call_history else None

    def reset_calls(self) -> None:
        """Reset call history."""
        self._call_history = []
