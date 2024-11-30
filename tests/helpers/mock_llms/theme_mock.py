# tests/helpers/mock_llms/theme_mock.py

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class ThemeMockLLM(BaseMockLLM):
    """Mock LLM for theme analysis testing."""

    def __init__(self):
        """Initialize with call tracking."""
        super().__init__()
        self._call_history = []

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get theme-specific mock response with call tracking."""
        last_message = messages[-1].content if messages else ""
        self._call_history.append(last_message)

        if not last_message:
            return json.dumps(
                {
                    "themes": [],
                    "theme_hierarchy": {},
                    "evidence": {},
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
        """Get English theme response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "themes": [
                        {
                            "name": "Machine Learning",
                            "description": "Application of machine learning and neural networks",
                            "confidence": 0.9,
                            "keywords": [
                                "machine learning",
                                "neural network",
                                "data",
                            ],
                            "parent_theme": None,
                        },
                        {
                            "name": "Data Processing",
                            "description": "Data preprocessing and feature engineering",
                            "confidence": 0.85,
                            "keywords": [
                                "data",
                                "preprocessing",
                                "feature engineering",
                            ],
                            "parent_theme": "Machine Learning",
                        },
                    ],
                    "theme_hierarchy": {
                        "Machine Learning": ["Data Processing"]
                    },
                    "evidence": {
                        "Machine Learning": [
                            {
                                "text": "Machine learning models are trained using large datasets",
                                "relevance": 0.9,
                                "keywords": [
                                    "machine learning",
                                    "models",
                                    "datasets",
                                ],
                            }
                        ],
                        "Data Processing": [
                            {
                                "text": "Data preprocessing and feature engineering are crucial",
                                "relevance": 0.85,
                                "keywords": [
                                    "preprocessing",
                                    "feature engineering",
                                ],
                            }
                        ],
                    },
                    "success": True,
                    "language": "en",
                }
            )
        else:  # business
            return json.dumps(
                {
                    "themes": [
                        {
                            "name": "Financial Performance",
                            "description": "Company financial results and metrics",
                            "confidence": 0.9,
                            "keywords": [
                                "revenue",
                                "growth",
                                "profit",
                                "margins",
                            ],
                            "parent_theme": None,
                        },
                        {
                            "name": "Market Strategy",
                            "description": "Market expansion and strategic growth",
                            "confidence": 0.85,
                            "keywords": ["market", "expansion", "strategy"],
                            "parent_theme": "Financial Performance",
                        },
                    ],
                    "theme_hierarchy": {
                        "Financial Performance": ["Market Strategy"]
                    },
                    "evidence": {
                        "Financial Performance": [
                            {
                                "text": "Q3 financial results show 15% revenue growth",
                                "relevance": 0.9,
                                "keywords": ["revenue", "growth"],
                            }
                        ],
                        "Market Strategy": [
                            {
                                "text": "Market expansion strategy focuses on emerging sectors",
                                "relevance": 0.85,
                                "keywords": ["market", "expansion", "strategy"],
                            }
                        ],
                    },
                    "success": True,
                    "language": "en",
                }
            )

    def _get_finnish_response(self, content_type: str) -> str:
        """Get Finnish theme response."""
        if content_type == "technical":
            return json.dumps(
                {
                    "themes": [
                        {
                            "name": "Koneoppiminen",
                            "description": "Tekoälyn ja koneoppimisen soveltaminen",
                            "confidence": 0.9,
                            "keywords": [
                                "koneoppimismalli",
                                "neuroverkko",
                                "data",
                            ],
                            "parent_theme": None,
                        },
                        {
                            "name": "Data-analyysi",
                            "description": "Datan käsittely ja analysointi",
                            "confidence": 0.85,
                            "keywords": ["data", "esikäsittely", "piirteet"],
                            "parent_theme": "Koneoppiminen",
                        },
                    ],
                    "theme_hierarchy": {"Koneoppiminen": ["Data-analyysi"]},
                    "evidence": {
                        "Koneoppiminen": [
                            {
                                "text": "Koneoppimismalleja koulutetaan suurilla datajoukolla",
                                "relevance": 0.9,
                                "keywords": ["koneoppimismalli", "data"],
                            }
                        ],
                        "Data-analyysi": [
                            {
                                "text": "Datan esikäsittely ja piirteiden suunnittelu",
                                "relevance": 0.85,
                                "keywords": [
                                    "data",
                                    "esikäsittely",
                                    "piirteet",
                                ],
                            }
                        ],
                    },
                    "success": True,
                    "language": "fi",
                }
            )
        else:  # business
            return json.dumps(
                {
                    "themes": [
                        {
                            "name": "Taloudellinen Suorituskyky",
                            "description": "Yrityksen taloudelliset tulokset ja kasvu",
                            "confidence": 0.9,
                            "keywords": ["liikevaihto", "kasvu", "tulos"],
                            "parent_theme": None,
                        },
                        {
                            "name": "Markkinakehitys",
                            "description": "Markkinoiden ja liiketoiminnan kehitys",
                            "confidence": 0.85,
                            "keywords": ["markkina", "strategia", "kasvu"],
                            "parent_theme": "Taloudellinen Suorituskyky",
                        },
                    ],
                    "theme_hierarchy": {
                        "Taloudellinen Suorituskyky": ["Markkinakehitys"]
                    },
                    "evidence": {
                        "Taloudellinen Suorituskyky": [
                            {
                                "text": "Q3 taloudelliset tulokset osoittavat kasvua",
                                "relevance": 0.9,
                                "keywords": ["tulos", "kasvu"],
                            }
                        ],
                        "Markkinakehitys": [
                            {
                                "text": "Markkinalaajennusstrategia keskittyy uusiin sektoreihin",
                                "relevance": 0.85,
                                "keywords": ["markkina", "strategia"],
                            }
                        ],
                    },
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
