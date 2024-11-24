# tests/helpers/mock_llm.py

from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import json
import logging

logger = logging.getLogger(__name__)


class MockLLM(BaseChatModel):
    """Mock LLM for testing with support for both themes and keywords."""

    def _get_mock_response(self, messages: List[BaseMessage]) -> str:
        """Get appropriate mock response based on input."""
        last_message = messages[-1].content if messages else ""

        # Handle empty/None input
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

        # Handle very short input
        text_content = last_message.strip()
        if len(text_content) < 10:
            return json.dumps(
                {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "success": False,
                    "error": "Input text too short for meaningful analysis",
                    "language": "en",
                }
            )

        try:
            # Detect content type and language
            language, content_type = self._detect_content_type(text_content)
            print(
                f"\nDetected language: {language}, content type: {content_type}"
            )

            # Check if this is a theme analysis request
            is_theme_request = "identify themes" in last_message.lower()

            if is_theme_request:
                # Return theme-specific responses
                if language == "fi":
                    return (
                        self._get_finnish_technical_themes()
                        if content_type == "technical"
                        else self._get_finnish_business_themes()
                    )
                else:
                    return (
                        self._get_english_technical_themes()
                        if content_type == "technical"
                        else self._get_english_business_themes()
                    )
            else:
                # Return keyword-specific responses
                if language == "fi":
                    return self._get_finnish_keyword_response(content_type)
                else:
                    return self._get_english_keyword_response(content_type)

        except Exception as e:
            logger.error(f"Error in mock LLM: {e}")
            return json.dumps(
                {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "success": False,
                    "error": str(e),
                    "language": language if "language" in locals() else "en",
                }
            )

    def _get_english_keyword_response(self, content_type: str) -> str:
        """Get English keyword response based on content type."""
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

    def _get_finnish_keyword_response(self, content_type: str) -> str:
        """Get Finnish keyword response based on content type."""
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
                        {
                            "keyword": "data",
                            "score": 0.8,
                            "domain": "technical",
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

    def _get_theme_response(self, language: str, content_type: str) -> Dict:
        """Get mock theme response."""
        if language == "fi":
            if content_type == "technical":
                return {
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
                    "success": True,
                    "language": "fi",
                }
            elif content_type == "business":
                return {
                    "themes": [
                        {
                            "name": "Taloudellinen suorituskyky",
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
                            "parent_theme": "Taloudellinen suorituskyky",
                        },
                    ],
                    "theme_hierarchy": {
                        "Taloudellinen suorituskyky": ["Markkinakehitys"]
                    },
                    "success": True,
                    "language": "fi",
                }
        else:  # English
            if content_type == "technical":
                return {
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
                    "success": True,
                    "language": "en",
                }
            elif content_type == "business":
                return {
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
                            "parent_theme": "Financial Performance",
                        },
                    ],
                    "theme_hierarchy": {
                        "Financial Performance": ["Market Strategy"]
                    },
                    "success": True,
                    "language": "en",
                }

        # Return empty but valid theme response
        return {
            "themes": [],
            "theme_hierarchy": {},
            "success": True,
            "language": language,
        }

    def _get_keyword_response(self, language: str, content_type: str) -> Dict:
        """Get mock keyword response."""
        if language == "fi":
            if content_type == "technical":
                return {
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
                        {
                            "keyword": "data",
                            "score": 0.8,
                            "domain": "technical",
                        },
                    ],
                    "compound_words": ["koneoppimismalli", "neuroverkko"],
                    "domain_keywords": {
                        "technical": ["koneoppimismalli", "neuroverkko", "data"]
                    },
                    "success": True,
                    "language": "fi",
                }
            elif content_type == "business":
                return {
                    "keywords": [
                        {
                            "keyword": "liikevaihto",
                            "score": 0.9,
                            "domain": "business",
                        },
                        {
                            "keyword": "kasvu",
                            "score": 0.85,
                            "domain": "business",
                        },
                        {
                            "keyword": "tulos",
                            "score": 0.8,
                            "domain": "business",
                        },
                    ],
                    "compound_words": [],
                    "domain_keywords": {
                        "business": ["liikevaihto", "kasvu", "tulos"]
                    },
                    "success": True,
                    "language": "fi",
                }
        else:  # English
            if content_type == "technical":
                return {
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
            elif content_type == "business":
                return {
                    "keywords": [
                        {
                            "keyword": "revenue growth",
                            "score": 0.9,
                            "domain": "business",
                            "compound_parts": ["revenue", "growth"],
                        },
                        {
                            "keyword": "profit margins",
                            "score": 0.85,
                            "domain": "business",
                            "compound_parts": ["profit", "margins"],
                        },
                        {
                            "keyword": "market",
                            "score": 0.8,
                            "domain": "business",
                        },
                    ],
                    "compound_words": ["revenue growth", "profit margins"],
                    "domain_keywords": {
                        "business": [
                            "revenue growth",
                            "profit margins",
                            "market",
                        ]
                    },
                    "success": True,
                    "language": "en",
                }

        # Return empty but valid keyword response
        return {
            "keywords": [],
            "compound_words": [],
            "domain_keywords": {},
            "success": True,
            "language": language,
        }

    def _get_english_technical_themes(self) -> str:
        """Get mock theme response for English technical content."""
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

    def _get_english_business_themes(self) -> str:
        """Get mock theme response for English business content."""
        return json.dumps(
            {
                "themes": [
                    {
                        "name": "Financial Performance",
                        "description": "Company financial results and growth metrics",
                        "confidence": 0.9,
                        "keywords": ["revenue", "growth", "profit", "margins"],
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
                "relationships": {"Financial Performance": ["Market Strategy"]},
                "success": True,
                "language": "en",
            }
        )

    def _get_finnish_technical_themes(self) -> str:
        """Get mock theme response for Finnish technical content."""
        return json.dumps(
            {
                "themes": [
                    {
                        "name": "Koneoppiminen",
                        "description": "Tekoälyn ja koneoppimisen soveltaminen",
                        "confidence": 0.9,
                        "keywords": ["koneoppimismalli", "neuroverkko", "data"],
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
                            "keywords": ["data", "esikäsittely", "piirteet"],
                        }
                    ],
                },
                "relationships": {"Koneoppiminen": ["Data-analyysi"]},
                "success": True,
                "language": "fi",
            }
        )

    def _validate_input(self, text: str) -> Optional[str]:
        """Validate input text."""
        if text is None:
            raise ValueError("Input text cannot be None")

        if not isinstance(text, str):
            raise TypeError(
                f"Invalid input type: expected str, got {type(text)}"
            )

        text = text.strip()
        if not text:
            return "Empty input text"

        if len(text) < 10:  # Minimum length for meaningful analysis
            return "Input text too short for meaningful analysis"

        return None

    def _detect_content_type(self, message: str) -> Tuple[str, str]:
        """Detect language and content type from message."""
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
            ]
        )

        # Content type detection
        is_technical = any(
            term in message
            for term in [
                # English technical terms
                "machine learning",
                "neural",
                "data",
                # Finnish technical terms
                "koneoppimis",
                "neuroverko",
                "datan",
            ]
        )

        is_business = any(
            term in message
            for term in [
                # English business terms
                "financial",
                "revenue",
                "market",
                "profit",
                "q3",
                # Finnish business terms
                "liikevaihto",
                "tulos",
                "markkina",
            ]
        )

        language = "fi" if is_finnish else "en"
        content_type = (
            "technical"
            if is_technical
            else "business" if is_business else "unknown"
        )

        print(f"\nDetected language: {language}, content type: {content_type}")
        return language, content_type

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate mock response."""
        try:
            print(
                f"\nMockLLM received message: {messages[-1].content[:100]}..."
            )
            content = self._get_mock_response(messages)
            if isinstance(content, str):
                print(
                    "\nMockLLM returning content:",
                    content[:100] if len(content) > 100 else content,
                )
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Error in mock LLM: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {"mock_param": True}
