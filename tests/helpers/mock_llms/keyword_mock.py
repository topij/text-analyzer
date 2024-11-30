# tests/helpers/mock_llms/keyword_mock.py

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class KeywordMockLLM(BaseMockLLM):
    """Mock LLM for keyword analysis testing."""

    def __init__(self):
        """Initialize with call tracking."""
        super().__init__()
        self._call_history = []

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get keyword-specific mock response with call tracking."""
        last_message = messages[-1].content if messages else ""
        self._call_history.append(last_message)

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
        return (
            self._get_finnish_response(content_type)
            if language == "fi"
            else self._get_english_response(content_type)
        )

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
                            "keyword": "market share",
                            "score": 0.85,
                            "domain": "business",
                            "compound_parts": ["market", "share"],
                        },
                    ],
                    "compound_words": ["revenue growth", "market share"],
                    "domain_keywords": {
                        "business": ["revenue growth", "market share"]
                    },
                    "success": True,
                    "language": "en",
                }
            )

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get keyword-specific mock response with call tracking."""
        last_message = messages[-1].content if messages else ""
        self._call_history.append(last_message)

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

        if language == "fi":
            if (
                "liikevaihdo" in last_message.lower()
                or "markkina" in last_message.lower()
            ):
                return json.dumps(
                    {
                        "keywords": [
                            {
                                "keyword": "liikevaihdon kasvu",
                                "score": 0.9,
                                "domain": "business",
                                "compound_parts": ["liike", "vaihto", "kasvu"],
                            },
                            {
                                "keyword": "markkinaosuus",
                                "score": 0.85,
                                "domain": "business",
                                "compound_parts": ["markkina", "osuus"],
                            },
                        ],
                        "compound_words": ["liikevaihto", "markkinaosuus"],
                        "domain_keywords": {
                            "business": ["liikevaihdon kasvu", "markkinaosuus"]
                        },
                        "success": True,
                        "language": "fi",
                    }
                )
            elif "järjestelmä" in last_message.lower():
                return json.dumps(
                    {
                        "keywords": [
                            {
                                "keyword": "tietokantajärjestelmä",
                                "score": 0.9,
                                "domain": "technical",
                                "compound_parts": [
                                    "tieto",
                                    "kanta",
                                    "järjestelmä",
                                ],
                            },
                            {
                                "keyword": "koneoppimisalgoritmi",
                                "score": 0.85,
                                "domain": "technical",
                                "compound_parts": [
                                    "kone",
                                    "oppia",
                                    "algoritmi",
                                ],
                            },
                        ],
                        "compound_words": [
                            "tietokantajärjestelmä",
                            "koneoppimisalgoritmi",
                        ],
                        "domain_keywords": {
                            "technical": [
                                "tietokantajärjestelmä",
                                "koneoppimisalgoritmi",
                            ]
                        },
                        "success": True,
                        "language": "fi",
                    }
                )
            else:
                # Default Finnish technical response
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
                            "technical": ["koneoppimismalli", "neuroverkko"]
                        },
                        "success": True,
                        "language": "fi",
                    }
                )

        return self._get_english_response(content_type)

    def get_calls(self) -> List[str]:
        """Get list of recorded calls."""
        return self._call_history

    def get_last_call(self) -> Optional[str]:
        """Get the last recorded call."""
        return self._call_history[-1] if self._call_history else None

    def reset_calls(self) -> None:
        """Reset call history."""
        self._call_history = []
