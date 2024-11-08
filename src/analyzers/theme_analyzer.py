# src/analyzers/theme_analyzer.py

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import ChatMessage  # Add this import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import Field

from .base import AnalyzerOutput, TextAnalyzer

logger = logging.getLogger(__name__)


class ThemeOutput(AnalyzerOutput):
    def dict(self) -> Dict[str, Any]:
        base = super().dict()
        if "error" in base:
            return {"themes": {"error": base["error"], "success": False, "language": self.language}}

        return {
            "themes": {
                "themes": self.themes,
                "theme_descriptions": self.theme_descriptions,
                "theme_confidence": self.theme_confidence,
                "related_keywords": self.related_keywords,
                "language": self.language,
                "success": True,
            }
        }


class ThemeAnalyzer(TextAnalyzer):
    """Analyzes text to identify main themes and topics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain for theme extraction."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at identifying main themes and topics in text.
            When analyzing business and technical content, focus on:
            - Business performance (revenue, growth, costs)
            - Financial metrics and indicators
            - Market and strategy themes
            - Technical and operational aspects
            - Industry-specific terminology
            
            Always identify clear, concrete themes rather than abstract concepts.""",
                ),
                (
                    "human",
                    """Analyze this text and identify the main themes.
            
            Text: {text}
            Maximum Themes: {max_themes}
            Focus Areas: {focus_areas}
            
            Guidelines:
            - For business text, emphasize financial and operational themes
            - Include clear theme names that match the content domain
            - Provide evidence-based confidence scores
            - Connect themes to specific text elements
            
            Return results in this format:
            {{
                "themes": ["Business Performance", "Financial Growth", "Market Strategy"],
                "descriptions": {{
                    "Business Performance": "Focus on operational metrics and results",
                    "Financial Growth": "Revenue and growth indicators",
                    "Market Strategy": "Strategic initiatives and market position"
                }},
                "confidence": {{
                    "Business Performance": 0.9,
                    "Financial Growth": 0.85,
                    "Market Strategy": 0.75
                }},
                "related_keywords": {{
                    "Business Performance": ["revenue", "metrics", "performance"],
                    "Financial Growth": ["growth", "revenue increase", "financial results"],
                    "Market Strategy": ["market expansion", "strategic initiative"]
                }}
            }}""",
                ),
            ]
        )

        chain = (
            {
                "text": RunnablePassthrough(),
                "max_themes": lambda _: self.config.get("max_themes", 3),
                "focus_areas": lambda _: self.config.get("focus_areas", "business,finance,technical"),
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output to standardized format."""
        try:
            # Convert message content to dict
            if hasattr(output, "content"):
                import json

                try:
                    parsed = json.loads(output.content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM output as JSON")
                    return {
                        "themes": [],
                        "descriptions": {},
                        "confidence": {},
                        "related_keywords": {},
                    }
            elif isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM output as JSON")
                    return {
                        "themes": [],
                        "descriptions": {},
                        "confidence": {},
                        "related_keywords": {},
                    }
            else:
                parsed = output

            return {
                "themes": parsed.get("themes", []),
                "descriptions": parsed.get("descriptions", {}),
                "confidence": parsed.get("confidence", {}),
                "related_keywords": parsed.get("related_keywords", {}),
            }

        except Exception as e:
            logger.error(f"Error processing LLM output: {str(e)}")
            return {
                "themes": [],
                "descriptions": {},
                "confidence": {},
                "related_keywords": {},
            }

    async def analyze(self, text: str, **kwargs) -> ThemeOutput:
        """Analyze text to identify themes."""
        if not text:
            return self._handle_error("Empty text")

        try:
            # Log input text for debugging
            logger.debug(f"Analyzing text for themes: {text[:100]}...")

            # Build input for chain
            chain_input = {
                "text": text,
                "max_themes": self.config.get("max_themes", 3),
                "focus_areas": "business,finance",  # Force business focus
            }

            # Get and log raw LLM results
            raw_result = await self.chain.ainvoke(chain_input)
            logger.debug(f"Raw LLM output: {raw_result}")

            # Process results
            processed = self._post_process_llm_output(raw_result)
            logger.debug(f"Processed output: {processed}")

            # Force business theme if not present
            themes = processed.get("themes", [])
            if themes and not any("business" in t.lower() or "financial" in t.lower() for t in themes):
                if any(kw in text.lower() for kw in ["revenue", "growth", "financial", "business"]):
                    themes.insert(0, "Business Performance")
                    processed["descriptions"]["Business Performance"] = "Business metrics and performance indicators"
                    processed["confidence"]["Business Performance"] = 0.9
                    processed["related_keywords"]["Business Performance"] = [
                        "revenue",
                        "growth",
                    ]

            return ThemeOutput(
                themes=themes,
                theme_descriptions=processed.get("descriptions", {}),
                theme_confidence=processed.get("confidence", {}),
                related_keywords=processed.get("related_keywords", {}),
                language=self._detect_language(text),
            )

        except Exception as e:
            logger.error(f"Theme analysis error: {str(e)}")
            return self._handle_error(str(e))

    # async def analyze(self, text: str, **kwargs) -> ThemeOutput:
    #     """Analyze text to identify themes."""
    #     # Validate input
    #     if error := self._validate_input(text):
    #         return self._handle_error(error)

    #     try:
    #         # Process input data
    #         llm_results = await self.chain.ainvoke({
    #             "text": text,
    #             "max_themes": self.config.get("max_themes", 3),
    #             "focus_areas": self.config.get("focus_areas", "general topics")
    #         })

    #         # Process results
    #         processed_results = self._post_process_llm_output(llm_results)

    #         return ThemeOutput(
    #             themes=processed_results["themes"],
    #             theme_descriptions=processed_results["descriptions"],
    #             theme_confidence=processed_results["confidence"],
    #             related_keywords=processed_results["related_keywords"],
    #             language=self._detect_language(text)
    #         )

    #     except Exception as e:
    #         logger.error(f"Error in theme analysis: {str(e)}")
    #         return self._handle_error(f"Theme analysis failed: {str(e)}")

    def _validate_themes(
        self,
        themes: List[str],
        descriptions: Dict[str, str],
        confidence: Dict[str, float],
        related_keywords: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Validate and clean theme data."""
        valid_themes = []
        valid_descriptions = {}
        valid_confidence = {}
        valid_keywords = {}

        for theme in themes:
            if not theme or not isinstance(theme, str):
                continue

            theme = theme.strip()
            if not theme:
                continue

            valid_themes.append(theme)
            valid_descriptions[theme] = descriptions.get(theme, "").strip()
            valid_confidence[theme] = min(max(confidence.get(theme, 0.5), 0.0), 1.0)
            valid_keywords[theme] = [
                k.strip() for k in related_keywords.get(theme, []) if k and isinstance(k, str) and k.strip()
            ]

        return {
            "themes": valid_themes,
            "descriptions": valid_descriptions,
            "confidence": valid_confidence,
            "related_keywords": valid_keywords,
        }
