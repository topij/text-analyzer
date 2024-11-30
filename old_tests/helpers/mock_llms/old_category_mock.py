# tests/helpers/mock_llms/category_mock.py

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from .base import BaseMessage, BaseMockLLM

logger = logging.getLogger(__name__)


class CategoryMockLLM(BaseMockLLM):
    """Mock LLM for category analysis testing."""

    def _handle_empty_input(self, language: str = "en") -> str:
        """Handle empty input case."""
        return json.dumps(
            {
                "categories": [],
                "success": False,
                "error": "Empty input text",
                "language": language,
            }
        )

    def _process_mock_response(self, categories: List[Dict]) -> List[Dict]:
        """Process mock response to match category config."""
        processed = []
        for cat in categories:
            category = {
                "name": cat.get("name", ""),
                "confidence": cat.get("confidence", 0.0),
                "description": cat.get("description", ""),
                "evidence": [
                    {
                        "text": ev.get("text", ""),
                        "relevance": ev.get("relevance", 0.8),
                    }
                    for ev in cat.get("evidence", [])
                ],
                "themes": cat.get("themes", []),
            }
            processed.append(category)
        return processed

    def _parse_categories_from_prompt(self, message: str) -> Dict[str, Dict]:
        """Extract categories from the prompt message."""
        try:
            # Print full message for debugging
            logger.debug(f"Parsing categories from message: {message[:500]}...")

            # Find all text between first '[' and last ']' that appears before "Text:"
            text_marker = "Text:"
            text_pos = message.find(text_marker)
            if text_pos == -1:
                text_pos = len(message)

            search_text = message[:text_pos]
            start = search_text.find("[")
            end = search_text.rfind("]")

            if start == -1 or end == -1:
                logger.error("Could not find category JSON markers")
                return {}

            # Extract and clean the JSON text
            json_text = search_text[start : end + 1]

            # Debug the extracted JSON
            logger.debug(f"Extracted JSON text: {json_text}")

            # Parse the JSON
            categories = json.loads(json_text)

            # Convert to dictionary format
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
            # Return default categories as fallback
            return {
                "Machine Learning": {
                    "description": "Machine learning and AI content",
                    "keywords": ["machine learning", "neural network", "data"],
                    "threshold": 0.7,
                },
                "Data Science": {
                    "description": "Data processing content",
                    "keywords": ["data", "processing", "analysis"],
                    "threshold": 0.7,
                },
                "Financial Analysis": {
                    "description": "Financial metrics content",
                    "keywords": ["revenue", "financial", "profit"],
                    "threshold": 0.7,
                },
                "Market Strategy": {
                    "description": "Market strategy content",
                    "keywords": ["market", "strategy", "growth"],
                    "threshold": 0.7,
                },
            }

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get mock response based on configured categories."""
        last_message = messages[-1].content if messages else ""

        if not last_message:
            return json.dumps(
                {
                    "categories": [],
                    "success": False,
                    "error": "Empty input text",
                    "language": "en",
                }
            )

        try:
            categories = self._parse_categories_from_prompt(last_message)
            language, content_type = self._detect_content_type(last_message)

            # Find text after "Text:" marker
            text_start = last_message.find("Text:") + 5
            text = (
                last_message[text_start:].strip()
                if text_start > 5
                else last_message
            )

            # Match categories based on content
            matching_categories = []
            for name, config in categories.items():
                keywords = config["keywords"]
                matches = sum(
                    1 for kw in keywords if kw.lower() in text.lower()
                )

                if matches > 0:
                    matching_categories.append(
                        {
                            "name": name,
                            "confidence": min(0.9, 0.7 + (matches * 0.1)),
                            "description": config["description"],
                            "evidence": self._create_evidence(text, keywords),
                            "themes": [kw.title() for kw in keywords[:2]],
                        }
                    )

            # Sort by confidence and limit to top 2
            matching_categories.sort(
                key=lambda x: x["confidence"], reverse=True
            )
            matching_categories = matching_categories[:2]

            return json.dumps(
                {
                    "categories": matching_categories,
                    "success": True,
                    "language": language,
                }
            )

        except Exception as e:
            logger.error(f"Error in mock LLM: {e}")
            return json.dumps(
                {
                    "categories": [],
                    "success": False,
                    "error": str(e),
                    "language": language,
                }
            )

    def _detect_content_type(self, text: str) -> Tuple[str, str]:
        """Enhanced content type detection."""
        text = text.lower()

        # Find actual content after "Text:" marker
        content_start = text.find("text:") + 5
        content = text[content_start:] if content_start > 5 else text

        # Language detection
        is_finnish = any(
            term in content
            for term in [
                "koneoppimis",
                "neuroverko",
                "datan",
                "piirteiden",
                "taloudellinen",
                "liikevaihto",
            ]
        )

        # Content type detection
        is_technical = any(
            term in content
            for term in [
                "machine learning",
                "neural",
                "data process",
                "koneoppimis",
                "neuroverko",
                "datan",
            ]
        )

        is_business = (
            any(
                term in content
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
            and not is_technical
        )

        language = "fi" if is_finnish else "en"
        content_type = (
            "technical"
            if is_technical
            else "business" if is_business else "unknown"
        )

        logger.debug(
            f"Detected language: {language}, content type: {content_type}"
        )
        return language, content_type

    def _get_english_response(self, content_type: str) -> str:
        """Get English category response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "categories": [
                        {
                            "name": "Machine Learning",
                            "confidence": 0.9,
                            "description": "Machine learning and AI technology content",
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
                            "name": "Data Science",
                            "confidence": 0.85,
                            "description": "Data processing and analysis content",
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
                            "name": "Financial Analysis",
                            "confidence": 0.9,
                            "description": "Financial performance and metrics content",
                            "evidence": [
                                {
                                    "text": "Q3 financial results show 15% revenue growth",
                                    "relevance": 0.9,
                                },
                                {
                                    "text": "Improved profit margins",
                                    "relevance": 0.85,
                                },
                            ],
                            "themes": [
                                "Financial Performance",
                                "Growth Metrics",
                            ],
                        },
                        {
                            "name": "Market Strategy",
                            "confidence": 0.85,
                            "description": "Market and business strategy content",
                            "evidence": [
                                {
                                    "text": "Market expansion strategy focuses on emerging sectors",
                                    "relevance": 0.85,
                                }
                            ],
                            "themes": ["Market Expansion", "Growth"],
                        },
                    ],
                    "success": True,
                    "language": "en",
                }
            )

    def _create_evidence(
        self, text: str, keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Create evidence entries from text based on keywords."""
        evidence = []
        sentences = re.split(r"[.!?]+\s*", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    evidence.append({"text": sentence, "relevance": 0.8})
                    break

            if len(evidence) >= 2:  # Limit to 2 pieces of evidence
                break

        return evidence

    def _get_finnish_response(self, content_type: str) -> str:
        """Get Finnish category response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "categories": [
                        {
                            "name": "Koneoppiminen",
                            "confidence": 0.9,
                            "description": "Koneoppimisen ja tekoälyn teknologia",
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
                            "name": "Data-analyysi",
                            "confidence": 0.85,
                            "description": "Datan käsittely ja analysointi",
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
                            "name": "Taloudellinen Analyysi",
                            "confidence": 0.9,
                            "description": "Taloudelliset tulokset ja mittarit",
                            "evidence": [
                                {
                                    "text": "Q3 taloudelliset tulokset osoittavat kasvua",
                                    "relevance": 0.9,
                                }
                            ],
                            "themes": ["Taloudellinen Suorituskyky"],
                        },
                        {
                            "name": "Markkinastrategia",
                            "confidence": 0.85,
                            "description": "Markkinoiden ja liiketoiminnan kehitys",
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
