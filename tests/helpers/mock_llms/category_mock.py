# tests/helpers/mock_llms/category_mock.py

import logging
from typing import Any, List, Dict, Tuple

from langchain_core.messages import BaseMessage
from src.schemas import CategoryMatch, CategoryOutput, Evidence
from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class CategoryMockLLM(BaseMockLLM):
    """Mock LLM for category analysis testing."""

    def _get_mock_response(self, messages: List[BaseMessage]) -> CategoryOutput:
        """Get category-specific mock response with structured output."""
        last_message = messages[-1].content if messages else ""
        logger.debug(f"CategoryMockLLM: Processing message: {last_message}")

        if not last_message:
            return CategoryOutput(
                categories=[],
                language="en",
                success=False,
                error="Empty input text",
            )

        language, content_type = self._detect_content_type(last_message)
        logger.debug(
            f"CategoryMockLLM: Detected language={language}, content_type={content_type}"
        )

        if language == "fi":
            return self._get_finnish_response(content_type)
        return self._get_english_response(content_type)

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
                        confidence=0.9,
                        description="Technical content with software and API focus",
                        evidence=[
                            Evidence(
                                text="Machine learning models are trained using large datasets",
                                relevance=0.9,
                            ),
                        ],
                        themes=["Software Development"],
                    ),
                ],
                language="en",
                success=True,
            )
        else:  # business content
            logger.debug("CategoryMockLLM: Generating business response")
            return CategoryOutput(
                categories=[
                    CategoryMatch(
                        name="Business",
                        confidence=0.95,
                        description="Business metrics and financial analysis",
                        evidence=[
                            Evidence(
                                text="Q3 financial results show 15% revenue growth",
                                relevance=0.95,
                            ),
                        ],
                        themes=["Financial Performance"],
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
                        confidence=0.9,
                        description="Tekninen sisältö ja ohjelmistokehitys",
                        evidence=[
                            Evidence(
                                text="Koneoppimismallit analysoivat dataa",
                                relevance=0.9,
                            ),
                        ],
                        themes=["Ohjelmistokehitys"],
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
                        confidence=0.95,
                        description="Liiketoiminnan mittarit ja analyysi",
                        evidence=[
                            Evidence(
                                text="Liikevaihto kasvoi 15% kolmannella neljänneksellä",
                                relevance=0.95,
                            ),
                        ],
                        themes=["Taloudellinen Suorituskyky"],
                    ),
                ],
                language="fi",
                success=True,
            )

    def _detect_content_type(self, message: str) -> Tuple[str, str]:
        """Detect language and content type from message."""
        message = message.lower()

        # Business content detection - make this more explicit
        is_business = any(
            term in message
            for term in [
                "financial results",
                "revenue growth",
                "market expansion",
                "customer acquisition",
                "metrics",
                "q3",
                "quarterly",
            ]
        )

        # Technical content detection
        is_technical = any(
            term in message
            for term in ["machine learning", "neural", "data", "api", "code"]
        )

        # Language detection
        is_finnish = any(
            term in message
            for term in ["koneoppimis", "neuroverko", "liikevaihto", "tulos"]
        )

        language = "fi" if is_finnish else "en"
        content_type = "business" if is_business else "technical"

        logger.debug(
            f"CategoryMockLLM: Content detection - language={language}, is_business={is_business}, is_technical={is_technical}"
        )
        return language, content_type
