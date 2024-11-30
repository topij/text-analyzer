# tests/helpers/mock_llms/theme_mock.py

import json
from typing import List

from .base import BaseMessage, BaseMockLLM


class ThemeMockLLM(BaseMockLLM):
    """Mock LLM for theme analysis testing."""

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get theme-specific mock response."""
        last_message = messages[-1].content if messages else ""

        if not last_message:
            return json.dumps(
                {
                    "themes": [],
                    "evidence": {},
                    "relationships": {},
                    "success": False,
                    "error": "Empty input text",
                    "language": "en",
                }
            )

        language, content_type = self._detect_content_type(last_message)

        if language == "fi":
            return self._get_finnish_response(content_type)
        return self._get_english_response(content_type)

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
                            "domain": "technical",
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
                            "domain": "technical",
                            "parent_theme": "Machine Learning",
                        },
                    ],
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
                                "text": "Data preprocessing and feature engineering are crucial steps",
                                "relevance": 0.85,
                                "keywords": [
                                    "preprocessing",
                                    "feature engineering",
                                ],
                            }
                        ],
                    },
                    "relationships": {"Machine Learning": ["Data Processing"]},
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
                            "description": "Company financial results and growth metrics",
                            "confidence": 0.9,
                            "keywords": [
                                "revenue",
                                "growth",
                                "profit",
                                "margins",
                            ],
                            "domain": "business",
                            "parent_theme": None,
                        },
                        {
                            "name": "Market Strategy",
                            "description": "Market expansion and customer acquisition",
                            "confidence": 0.85,
                            "keywords": [
                                "market",
                                "customer",
                                "acquisition",
                                "retention",
                            ],
                            "domain": "business",
                            "parent_theme": "Financial Performance",
                        },
                    ],
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
                    "relationships": {
                        "Financial Performance": ["Market Strategy"]
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
                            "domain": "technical",
                            "parent_theme": None,
                        },
                        {
                            "name": "Data-analyysi",
                            "description": "Datan käsittely ja analysointi",
                            "confidence": 0.85,
                            "keywords": ["data", "esikäsittely", "piirteet"],
                            "domain": "technical",
                            "parent_theme": "Koneoppiminen",
                        },
                    ],
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
                    "relationships": {"Koneoppiminen": ["Data-analyysi"]},
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
                            "keywords": [
                                "liikevaihto",
                                "kasvu",
                                "tulos",
                                "katteet",
                            ],
                            "domain": "business",
                            "parent_theme": None,
                        },
                        {
                            "name": "Markkinakehitys",
                            "description": "Markkinoiden ja liiketoiminnan kehitys",
                            "confidence": 0.85,
                            "keywords": ["markkina", "strategia", "kasvu"],
                            "domain": "business",
                            "parent_theme": "Taloudellinen Suorituskyky",
                        },
                    ],
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
                    "relationships": {
                        "Taloudellinen Suorituskyky": ["Markkinakehitys"]
                    },
                    "success": True,
                    "language": "fi",
                }
            )
