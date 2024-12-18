# tests/helpers/mock_llms/keyword_mock.py

import logging
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.schemas import KeywordInfo, KeywordOutput
from .base import BaseMockLLM

logger = logging.getLogger(__name__)


class KeywordMockLLM(BaseMockLLM):
    """Mock LLM for keyword analysis testing."""

    def _get_mock_response(
        self, messages: List[BaseMessage]
    ) -> str:  # Change return type to str
        """Get keyword-specific mock response as JSON string."""
        last_message = messages[-1].content if messages else ""

        if not last_message:
            # Return directly as JSON string
            return json.dumps(
                {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "language": "en",
                    "success": False,
                    "error": "Empty input text",
                }
            )

        language, content_type = self._detect_content_type(last_message)

        # Get appropriate response
        response = (
            self._get_finnish_response(content_type)
            if language == "fi"
            else self._get_english_response(content_type)
        )

        # Return JSON string
        return (
            response.model_dump_json()
            if isinstance(response, BaseModel)
            else json.dumps(response)
        )

    def _get_english_response(self, content_type: str) -> KeywordOutput:
        """Get English keyword response."""
        if content_type == "technical":
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="machine learning",
                        score=0.9,
                        domain="technical",
                        compound_parts=["machine", "learning"],
                    ),
                    KeywordInfo(
                        keyword="neural network",
                        score=0.85,
                        domain="technical",
                        compound_parts=["neural", "network"],
                    ),
                    KeywordInfo(keyword="data", score=0.8, domain="technical"),
                ],
                compound_words=["machine learning", "neural network"],
                domain_keywords={
                    "technical": ["machine learning", "neural network", "data"]
                },
                language="en",
                success=True,
            )
        else:  # business content
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="revenue growth",
                        score=0.9,
                        domain="business",
                        compound_parts=["revenue", "growth"],
                    ),
                    KeywordInfo(
                        keyword="market share",
                        score=0.85,
                        domain="business",
                        compound_parts=["market", "share"],
                    ),
                ],
                compound_words=["revenue growth", "market share"],
                domain_keywords={
                    "business": ["revenue growth", "market share"]
                },
                language="en",
                success=True,
            )

    def _get_finnish_response(self, content_type: str) -> KeywordOutput:
        """Get Finnish keyword response."""
        if content_type == "technical":
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="koneoppimismalli",
                        score=0.9,
                        domain="technical",
                        compound_parts=["kone", "oppimis", "malli"],
                    ),
                    KeywordInfo(
                        keyword="neuroverkko",
                        score=0.85,
                        domain="technical",
                        compound_parts=["neuro", "verkko"],
                    ),
                ],
                compound_words=["koneoppimismalli", "neuroverkko"],
                domain_keywords={
                    "technical": ["koneoppimismalli", "neuroverkko"]
                },
                language="fi",
                success=True,
            )
        else:  # business content
            return KeywordOutput(
                keywords=[
                    KeywordInfo(
                        keyword="liikevaihdon kasvu",
                        score=0.9,
                        domain="business",
                        compound_parts=["liike", "vaihto", "kasvu"],
                    ),
                    KeywordInfo(
                        keyword="markkinaosuus",
                        score=0.85,
                        domain="business",
                        compound_parts=["markkina", "osuus"],
                    ),
                ],
                compound_words=["liikevaihdon kasvu", "markkinaosuus"],
                domain_keywords={
                    "business": ["liikevaihdon kasvu", "markkinaosuus"]
                },
                language="fi",
                success=True,
            )
