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
        """Initialize analyzer with new configuration system.

        Args:
            llm: Optional LLM instance
            config: Optional configuration dictionary
            language_processor: Optional language processor instance
            categories: Dictionary mapping category names to configurations
        """
        # Initialize analyzer config if not provided
        if llm is None:
            analyzer_config = AnalyzerConfig()
            llm = create_llm(config=analyzer_config)

            # Merge analyzer config with provided config
            if config is None:
                config = {}
            config = {**analyzer_config.config.get("analysis", {}), **config}

        # Call parent init
        super().__init__(llm, config)

        # Set up components
        self.language_processor = language_processor
        self.categories = categories or {}
        self.min_confidence = config.get("min_confidence", 0.3)

        # Create chain
        self.chain = self._create_chain()
        logger.debug(
            f"CategoryAnalyzer initialized with {len(self.categories)} categories"
        )

    # def __init__(
    #     self,
    #     llm: Optional[BaseChatModel] = None,
    #     config: Optional[Dict[str, Any]] = None,
    #     language_processor: Optional[BaseTextProcessor] = None,
    #     categories: Optional[Dict[str, CategoryConfig]] = None,
    # ):
    #     """Initialize analyzer with categories and configuration.

    #     Args:
    #         llm: Optional LLM instance (will create default if None)
    #         config: Optional configuration dictionary
    #         language_processor: Optional language processor instance
    #         categories: Dictionary mapping category names to their configurations
    #     """
    #     # Initialize base analyzer first
    #     super().__init__(llm, config)

    #     # Store language processor
    #     self.language_processor = language_processor

    #     # Store categories and ensure proper defaults
    #     self.categories = categories or {}

    #     # Set confidence threshold from config or default
    #     self.min_confidence = self.config.get("min_confidence", 0.3)

    #     # Create chain after all initialization
    #     self.chain = self._create_chain()

    async def analyze(self, text: str) -> CategoryOutput:
        """Analyze text to identify categories with proper error handling."""
        if text is None:
            raise ValueError("Input text cannot be None")

        if not text:
            return CategoryOutput(
                categories=[],
                error="Empty input text",
                success=False,
                language=self._get_language(),
            )

        try:
            # Get LLM analysis
            response = await self.chain.ainvoke(text)
            logger.debug("Got LLM response for category analysis")

            if not response or "categories" not in response:
                return CategoryOutput(
                    categories=[],
                    error="Invalid response from LLM",
                    success=False,
                    language=self._get_language(),
                )

            # Filter and validate categories
            categories = [
                CategoryMatch(
                    name=cat["name"],
                    confidence=float(cat["confidence"]),
                    description=cat.get("explanation", ""),
                    evidence=cat.get("evidence", []),
                    themes=cat.get("themes", []),
                )
                for cat in response["categories"]
                if cat.get("confidence", 0) >= self.min_confidence
                and cat["name"] in self.categories
            ]

            return CategoryOutput(
                categories=categories,
                language=response.get("language", self._get_language()),
                success=True,
            )

        except Exception as e:
            logger.error(f"Category analysis failed: {str(e)}")
            return CategoryOutput(
                categories=[],
                error=str(e),
                success=False,
                language=self._get_language(),
            )

    # async def analyze(self, text: str) -> CategoryOutput:
    #     """Analyze text using predefined or default categories."""
    #     if text is None:
    #         raise ValueError("Input text cannot be None")

    #     if not text:
    #         return CategoryOutput(
    #             categories=[],
    #             error="Empty input text",
    #             success=False,
    #             language=self._get_language(),
    #         )

    #     try:
    #         # Get response from LLM
    #         response = await self.chain.ainvoke(text)
    #         logger.debug(f"Got LLM response: {response}")

    #         # Process response
    #         categories = []
    #         language = (
    #             self.language_processor.language
    #             if self.language_processor
    #             else "en"
    #         )

    #         if isinstance(response, dict):
    #             categories = response.get("categories", [])
    #             # Use response's language if provided
    #             if "language" in response:
    #                 language = response["language"]

    #         # Always create a successful output unless there's an error
    #         return CategoryOutput(
    #             categories=categories,
    #             language=language,
    #             success=True,  # Default to True
    #             error="",  # Empty error string for success
    #         )

    #     except Exception as e:
    #         logger.error(f"Analysis failed: {str(e)}")
    #         return CategoryOutput(
    #             categories=[],
    #             error=str(e),
    #             success=False,
    #             language=language if "language" in locals() else "en",
    #         )

    def _create_chain(self) -> RunnableSequence:
        """Create enhanced LangChain processing chain."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a classification expert. Analyze text and classify it into 
            predefined categories considering:
            1. Category descriptions and keywords
            2. Supporting evidence
            3. Confidence levels
            4. Hierarchical relationships""",
                ),
                (
                    "human",
                    """Analyze this text and classify into categories:
            Text: {text}
            
            Categories:
            {categories_json}
            
            Guidelines:
            - Language: {language}
            - Minimum confidence: {min_confidence}
            - Include evidence for each category
            
            Return in this format:
            {{
                "categories": [
                    {{
                        "name": "category_name",
                        "confidence": 0.95,
                        "explanation": "Detailed explanation",
                        "evidence": [
                            {{
                                "text": "relevant text",
                                "relevance": 0.9
                            }}
                        ],
                        "themes": ["theme1", "theme2"]
                    }}
                ],
                "success": true,
                "language": "en"
            }}""",
                ),
            ]
        )

        return (
            {
                "text": RunnablePassthrough(),
                "categories_json": self._format_categories_json,
                "language": lambda _: (
                    self.language_processor.language
                    if self.language_processor
                    else "en"
                ),
                "min_confidence": lambda _: self.min_confidence,
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

    def _format_categories_json(self, _: Any) -> str:
        """Format categories as JSON for prompt."""
        categories_list = [
            {
                "name": name,
                "description": config.description,
                "keywords": config.keywords,
                "threshold": config.threshold,
                "parent": config.parent,
            }
            for name, config in self.categories.items()
        ]
        return str(categories_list)

    # def _create_chain(self) -> RunnableSequence:
    #     """Create LangChain processing chain."""
    #     template = ChatPromptTemplate.from_messages(
    #         [
    #             (
    #                 "system",
    #                 """You are a classification expert focusing on content analysis.
    #             Analyze text and classify it into the provided categories only.
    #             Consider keyword matches, context, and evidence.
    #             Return ONLY valid JSON format.""",
    #             ),
    #             (
    #                 "human",
    #                 """Analyze this text and classify into categories:
    #             Text: {text}

    #             Categories: {categories_json}

    #             Language: {language}
    #             Guidelines:
    #             - Minimum confidence: {min_confidence}
    #             - Use ONLY the provided categories
    #             - Include evidence for each category

    #             Return in this exact format:
    #             {{
    #                 "categories": [
    #                     {{
    #                         "category": "category_name",
    #                         "confidence": 0.95,
    #                         "explanation": "Detailed explanation",
    #                         "evidence": [
    #                             {{
    #                                 "text": "relevant text",
    #                                 "relevance": 0.9
    #                             }}
    #                         ],
    #                         "themes": ["related_theme1", "related_theme2"]
    #                     }}
    #                 ],
    #                 "success": true,
    #                 "language": "en"
    #             }}""",
    #             ),
    #         ]
    #     )

    #     def get_categories_json(input_data: Dict) -> str:
    #         """Format categories as JSON string."""
    #         categories_list = [
    #             {
    #                 "name": name,
    #                 "description": config.description,
    #                 "keywords": config.keywords,
    #                 "threshold": config.threshold,
    #             }
    #             for name, config in self.categories.items()
    #         ]
    #         return json.dumps(categories_list)

    #     def get_language(input_data: Dict) -> str:
    #         """Get language for prompt."""
    #         return (
    #             self.language_processor.language
    #             if self.language_processor
    #             else "en"
    #         )

    #     def get_min_confidence(input_data: Dict) -> float:
    #         """Get minimum confidence threshold."""
    #         return self.min_confidence

    #     return (
    #         {
    #             "text": RunnablePassthrough(),
    #             "categories_json": get_categories_json,
    #             "language": get_language,
    #             "min_confidence": get_min_confidence,
    #         }
    #         | template
    #         | self.llm
    #         | self._parse_response
    #     )

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

    # def _get_language(self) -> str:
    #     """Get current language."""
    #     return (
    #         self.language_processor.language
    #         if self.language_processor
    #         else "en"
    #     )
