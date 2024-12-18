# src/analyzers/category_analyzer.py

import json
import logging
from typing import Any, Dict, List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

from src.config.manager import ConfigManager
from src.core.config import AnalyzerConfig

from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel
from src.core.language_processing.base import BaseTextProcessor
from src.loaders.models import CategoryConfig
from src.schemas import CategoryMatch, Evidence, CategoryOutput
from src.analyzers.base import TextAnalyzer

logger = logging.getLogger(__name__)


class CategoryAnalyzer(TextAnalyzer):
    """Analyzes text to identify predefined categories."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None,
    ):
        """Initialize analyzer with categories and structured output."""
        # Initialize analyzer config if not provided
        if llm is None:
            analyzer_config = AnalyzerConfig()
            llm = create_llm(config=analyzer_config)

            # Merge analyzer config with provided config
            if config is None:
                config = {}
            config = {**analyzer_config.config.get("analysis", {}), **config}

        super().__init__(llm, config)
        self.language_processor = language_processor
        self.categories = categories or {}

        # Initialize configuration values
        self.min_confidence = config.get("min_confidence", 0.3)

        # Create chain with structured output
        self.chain = self._create_chain()

    # temporary fix for the test failure
    async def analyze(self, text: str) -> CategoryOutput:
        """Analyze text with AIMessage handling."""

        if not self.categories:
            logger.warning("No categories available for analysis")
            return CategoryOutput(
                categories=[],
                language=self._get_language(),
                success=False,
                error="No categories configured",
            )
        if text is None:
            raise ValueError("Input text cannot be None")

        if not text:
            return CategoryOutput(
                categories=[],
                language=self._get_language(),
                success=False,
                error="Empty input text",
            )

        try:
            logger.debug("CategoryAnalyzer.analyze: Starting analysis")
            logger.debug(
                f"Input text: {text[:100]}..."
            )  # Log first 100 chars of input
            logger.debug(f"Available categories: {self.categories}")

            result = await self.chain.ainvoke(text)

            logger.debug(
                f"CategoryAnalyzer.analyze: Chain result type: {type(result)}"
            )
            logger.debug(f"CategoryAnalyzer.analyze: Chain result: {result}")

            # Handle AIMessage from mock LLMs
            if hasattr(result, "content"):
                try:
                    data = json.loads(result.content)
                    logger.debug(
                        f"CategoryAnalyzer.analyze: Parsed JSON data: {data}"
                    )
                    result = CategoryOutput(**data)
                except Exception as e:
                    logger.error(f"Error parsing AIMessage content: {e}")
                    return CategoryOutput(
                        categories=[],
                        language=self._get_language(),
                        success=False,
                        error=f"Error parsing response: {str(e)}",
                    )

            # Filter categories by confidence
            if getattr(result, "categories", None):
                result.categories = [
                    cat
                    for cat in result.categories
                    if cat.confidence >= self.min_confidence
                    and cat.name in self.categories
                ]

            return result

        except Exception as e:
            logger.error(f"CategoryAnalyzer.analyze: Exception occurred: {e}")
            return CategoryOutput(
                categories=[],
                language=self._get_language(),
                success=False,
                error=str(e),
            )

    # don't remove, this is the working production version
    # async def analyze(self, text: str) -> CategoryOutput:
    #     """Analyze text with structured output validation."""
    #     if text is None:
    #         raise ValueError("Input text cannot be None")

    #     if not text:
    #         return CategoryOutput(
    #             categories=[],
    #             language=self._get_language(),
    #             success=False,
    #             error="Empty input text",
    #         )

    #     try:
    #         result = await self.chain.ainvoke(text)

    #         # Filter categories by confidence and match against predefined categories
    #         result.categories = [
    #             cat
    #             for cat in result.categories
    #             if cat.confidence >= self.min_confidence
    #             and cat.name in self.categories
    #         ]

    #         return result

    #     except Exception as e:
    #         logger.error(f"Category analysis failed: {str(e)}")
    #         return CategoryOutput(
    #             categories=[],
    #             language=self._get_language(),
    #             success=False,
    #             error=str(e),
    #         )

    def _create_error_output(self) -> CategoryOutput:
        """Create error output."""
        return CategoryOutput(
            categories=[],
            language=self._get_language(),
            success=False,
            error="Analysis failed",
        )

    def _create_chain(self) -> RunnableSequence:
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a categorization expert. For the given text:
            1. Match text against the provided categories using direct evidence
            2. Calculate confidence scores based on:
               - Presence of category keywords
               - Contextual relevance
               - Overall theme alignment
            3. Provide specific evidence from the text for each category match
            
            Only match categories when there is clear supporting evidence in the text.
            Each match must include specific quotes or references from the text.""",
                ),
                (
                    "human",
                    """Analyze this text against the following categories:
            {categories_json}
            
            Text: {text}
            Language: {language}
            
            Required: For each matching category, provide:
            - Confidence score based on evidence strength
            - Specific supporting quotes from the text
            - Related themes found in the text
            
            Only include categories with clear supporting evidence.""",
                ),
            ]
        )

        return (
            {
                "text": RunnablePassthrough(),
                "language": lambda _: self._get_language(),
                "categories_json": self._format_categories_json,
            }
            | template
            | self.llm.with_structured_output(CategoryOutput)
        )

    # def _create_chain(self) -> RunnableSequence:
    #     """Create enhanced LangChain processing chain."""
    #     template = ChatPromptTemplate.from_messages(
    #         [
    #             (
    #                 "system",
    #                 """You are a JSON-focused classification expert. Your responses must be ONLY valid JSON, no additional text.
    #             Analyze text and classify it into predefined categories.""",
    #             ),
    #             (
    #                 "human",
    #                 """Categories for classification:
    #             {categories_json}

    #             Text to analyze (Language: {language}):
    #             {text}

    #             Return ONLY a JSON object with this exact structure (no other text):
    #             {{
    #                 "categories": [
    #                     {{
    #                         "name": "exact_category_name",
    #                         "confidence": 0.95,
    #                         "explanation": "why this category applies",
    #                         "evidence": [
    #                             {{
    #                                 "text": "relevant quote",
    #                                 "relevance": 0.9
    #                             }}
    #                         ]
    #                     }}
    #                 ]
    #             }}""",
    #             ),
    #         ]
    #     )

    #     return (
    #         {
    #             "text": RunnablePassthrough(),
    #             "categories_json": self._format_categories_json,
    #             "language": lambda _: self._get_language(),
    #         }
    #         | template
    #         | self.llm
    #         | self._post_process_llm_output
    #     )

    def _format_categories_json(self, _: Any) -> str:
        """Format categories as detailed JSON for prompt."""
        logger.debug(
            f"Categories before formatting: {self.categories}"
        )  # Add debug logging

        categories_list = [
            {
                "name": name,
                "description": config.description,
                "keywords": config.keywords,
                "threshold": config.threshold,
            }
            for name, config in self.categories.items()
        ]

        logger.debug(
            f"Formatted categories: {categories_list}"
        )  # Add debug logging
        return json.dumps(categories_list, ensure_ascii=False, indent=2)

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract content from responses
            if hasattr(response, "content"):
                data = json.loads(response.content)
            elif isinstance(response, str):
                data = json.loads(response)
            elif isinstance(response, dict):
                data = response
            else:
                return {
                    "categories": [],
                    "success": True,
                    "language": self._get_language(),
                }

            # Transform category format to match Pydantic model
            transformed_categories = []
            for cat in data.get("categories", []):
                # Map 'category' field to 'name'
                transformed = {
                    "name": cat.get("category"),
                    "confidence": cat.get("confidence", 0.5),
                    "description": cat.get("explanation", ""),
                    "evidence": [
                        {
                            "text": e.get("text", ""),
                            "relevance": e.get("relevance", 0.5),
                        }
                        for e in cat.get("evidence", [])
                    ],
                    "themes": cat.get("themes", []),
                }
                transformed_categories.append(transformed)

            return {
                "categories": transformed_categories,
                "language": data.get("language", self._get_language()),
                "success": data.get("success", True),
            }

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "categories": [],
                "success": False,
                "language": self._get_language(),
                "error": str(e),
            }

    def _validate_category(self, category: Dict[str, Any]) -> bool:
        """Validate category data."""
        if not isinstance(category, dict):
            return False

        required_fields = {"name", "confidence"}
        if not all(field in category for field in required_fields):
            return False

        try:
            confidence = float(category["confidence"])
            return 0 <= confidence <= 1.0
        except (TypeError, ValueError):
            return False

    def _get_language(self) -> str:
        """Get current language."""
        return (
            self.language_processor.language
            if self.language_processor
            else "en"
        )
