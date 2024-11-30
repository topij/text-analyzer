# tests/helpers/mock_llms/keyword_mock.py

import json
from typing import List

from .base import BaseMessage, BaseMockLLM


class KeywordMockLLM(BaseMockLLM):
    """Mock LLM for keyword analysis testing."""

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get keyword-specific mock response."""
        last_message = messages[-1].content if messages else ""

        if not last_message:
            return json.dumps(
                {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "success": False,
                    "error": "Empty input text",
                    "language": "en",
                }
            )

        language, content_type = self._detect_content_type(last_message)

        # Return language and content type specific responses
        if language == "fi":
            return self._get_finnish_response(content_type)
        return self._get_english_response(content_type)

    def _get_english_response(self, content_type: str) -> str:
        """Get English keyword response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "machine learning",
                            "score": 0.9,
                            "domain": "technical",
                            "compound_parts": ["machine", "learning"],
                        },
                        {
                            "keyword": "neural network",
                            "score": 0.85,
                            "domain": "technical",
                            "compound_parts": ["neural", "network"],
                        },
                        {
                            "keyword": "data",
                            "score": 0.8,
                            "domain": "technical",
                        },
                    ],
                    "compound_words": ["machine learning", "neural network"],
                    "domain_keywords": {
                        "technical": [
                            "machine learning",
                            "neural network",
                            "data",
                        ]
                    },
                    "success": True,
                    "language": "en",
                }
            )
        else:  # business
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "revenue growth",
                            "score": 0.9,
                            "domain": "business",
                            "compound_parts": ["revenue", "growth"],
                        },
                        {
                            "keyword": "market",
                            "score": 0.85,
                            "domain": "business",
                        },
                        {
                            "keyword": "profit margins",
                            "score": 0.8,
                            "domain": "business",
                            "compound_parts": ["profit", "margins"],
                        },
                    ],
                    "compound_words": ["revenue growth", "profit margins"],
                    "domain_keywords": {
                        "business": [
                            "revenue growth",
                            "market",
                            "profit margins",
                        ]
                    },
                    "success": True,
                    "language": "en",
                }
            )

    def _get_finnish_response(self, content_type: str) -> str:
        """Get Finnish keyword response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "koneoppimismalli",
                            "score": 0.9,
                            "domain": "technical",
                            "compound_parts": ["kone", "oppimis", "malli"],
                        },
                        {
                            "keyword": "neuroverkko",
                            "score": 0.85,
                            "domain": "technical",
                            "compound_parts": ["neuro", "verkko"],
                        },
                    ],
                    "compound_words": ["koneoppimismalli", "neuroverkko"],
                    "domain_keywords": {
                        "technical": ["koneoppimismalli", "neuroverkko", "data"]
                    },
                    "success": True,
                    "language": "fi",
                }
            )
        else:  # business
            return json.dumps(
                {
                    "keywords": [
                        {
                            "keyword": "liikevaihdon kasvu",
                            "score": 0.9,
                            "domain": "business",
                            "compound_parts": ["liikevaihto", "kasvu"],
                        },
                        {
                            "keyword": "markkinalaajennusstrategia",
                            "score": 0.85,
                            "domain": "business",
                            "compound_parts": [
                                "markkina",
                                "laajennus",
                                "strategia",
                            ],
                        },
                    ],
                    "compound_words": [
                        "liikevaihdon kasvu",
                        "markkinalaajennusstrategia",
                    ],
                    "domain_keywords": {
                        "business": [
                            "liikevaihdon kasvu",
                            "markkinalaajennusstrategia",
                        ]
                    },
                    "success": True,
                    "language": "fi",
                }
            )
