# tests/helpers/mock_llms/theme_mock.py

import logging
from typing import Any, List

from langchain_core.messages import BaseMessage

from src.schemas import ThemeInfo, ThemeOutput
from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class ThemeMockLLM(BaseMockLLM):
    """Mock LLM for theme analysis testing."""

    def _get_mock_response(self, messages: List[BaseMessage]) -> ThemeOutput:
        """Get theme-specific mock response with structured output."""
        last_message = messages[-1].content if messages else ""

        if not last_message:
            return ThemeOutput(
                themes=[],
                theme_hierarchy={},
                language="en",
                success=False,
                error="Empty input text",
            )

        language, content_type = self._detect_content_type(last_message)
        return self._get_language_specific_response(language, content_type)

    def _get_language_specific_response(
        self, language: str, content_type: str
    ) -> ThemeOutput:
        """Get language-specific response."""
        if language == "fi":
            return self._get_finnish_response(content_type)
        return self._get_english_response(content_type)

    def _get_english_response(self, content_type: str) -> ThemeOutput:
        """Get English theme response."""
        if content_type == "technical":
            return ThemeOutput(
                themes=[
                    ThemeInfo(
                        name="Machine Learning",
                        description="Application of machine learning and neural networks",
                        confidence=0.9,
                        keywords=["machine learning", "neural network", "data"],
                        parent_theme=None,
                    ),
                    ThemeInfo(
                        name="Data Processing",
                        description="Data preprocessing and feature engineering",
                        confidence=0.85,
                        keywords=[
                            "data",
                            "preprocessing",
                            "feature engineering",
                        ],
                        parent_theme="Machine Learning",
                    ),
                ],
                theme_hierarchy={"Machine Learning": ["Data Processing"]},
                language="en",
                success=True,
            )
        else:  # business content
            return ThemeOutput(
                themes=[
                    ThemeInfo(
                        name="Financial Performance",
                        description="Company financial results and metrics",
                        confidence=0.9,
                        keywords=["revenue", "growth", "profit", "margins"],
                        parent_theme=None,
                    ),
                    ThemeInfo(
                        name="Market Strategy",
                        description="Market expansion and strategic growth",
                        confidence=0.85,
                        keywords=["market", "expansion", "strategy"],
                        parent_theme="Financial Performance",
                    ),
                ],
                theme_hierarchy={"Financial Performance": ["Market Strategy"]},
                language="en",
                success=True,
            )

    def _get_finnish_response(self, content_type: str) -> ThemeOutput:
        """Get Finnish theme response."""
        if content_type == "technical":
            return ThemeOutput(
                themes=[
                    ThemeInfo(
                        name="Koneoppiminen",
                        description="Tekoälyn ja koneoppimisen soveltaminen",
                        confidence=0.9,
                        keywords=["koneoppimismalli", "neuroverkko", "data"],
                        parent_theme=None,
                    ),
                    ThemeInfo(
                        name="Data-analyysi",
                        description="Datan käsittely ja analysointi",
                        confidence=0.85,
                        keywords=["data", "esikäsittely", "piirteet"],
                        parent_theme="Koneoppiminen",
                    ),
                ],
                theme_hierarchy={"Koneoppiminen": ["Data-analyysi"]},
                language="fi",
                success=True,
            )
        else:  # business content
            return ThemeOutput(
                themes=[
                    ThemeInfo(
                        name="Taloudellinen Suorituskyky",
                        description="Yrityksen taloudelliset tulokset ja kasvu",
                        confidence=0.9,
                        keywords=["liikevaihto", "kasvu", "tulos"],
                        parent_theme=None,
                    ),
                    ThemeInfo(
                        name="Markkinakehitys",
                        description="Markkinoiden ja liiketoiminnan kehitys",
                        confidence=0.85,
                        keywords=["markkina", "strategia", "kasvu"],
                        parent_theme="Taloudellinen Suorituskyky",
                    ),
                ],
                theme_hierarchy={
                    "Taloudellinen Suorituskyky": ["Markkinakehitys"]
                },
                language="fi",
                success=True,
            )
