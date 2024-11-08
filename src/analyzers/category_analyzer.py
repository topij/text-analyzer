# src/analyzers/category_analyzer.py

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

from .base import AnalyzerOutput, TextAnalyzer

logger = logging.getLogger(__name__)


# src/analyzers/category_analyzer.py


class CategoryOutput(AnalyzerOutput):
    """Output model for category analysis."""

    categories: List[Dict[str, Any]] = Field(default_factory=list)
    explanations: Dict[str, str] = Field(default_factory=dict)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)
    success: bool = Field(default=True)
    language: str = Field(default="unknown")
    error: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dict with proper structure."""
        if self.error:
            return {"categories": {"error": self.error, "success": False, "language": self.language}}

        return {
            "categories": {
                "categories": self.categories,
                "explanations": self.explanations,
                "evidence": self.evidence,
                "success": self.success,
                "language": self.language,
            }
        }


class CategoryAnalyzer(TextAnalyzer):
    def __init__(
        self,
        categories: Dict[str, Any],
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor=None,  # Add language processor
    ):
        """Initialize category analyzer.

        Args:
            categories: Dictionary of categories to detect
            llm: Optional language model override
            config: Optional configuration override
            language_processor: Optional language processor
        """
        super().__init__(llm, config)
        self.categories = categories
        self.language_processor = language_processor

    def _create_chain(self) -> RunnableSequence:
        """Create analysis chain."""
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a text classification expert. Analyze text and classify it into relevant categories.
            Return results in JSON format with these exact fields:
            {{
                "categories": [
                    {{
                        "name": "category_name",
                        "confidence": 0.95,
                        "explanation": "Explanation for classification",
                        "evidence": ["evidence text 1", "evidence text 2"]
                    }}
                ]
            }}""",
                ),
                (
                    "human",
                    """Analyze this text and classify it into these categories:
            {categories_json}
            
            Text: {text}
            
            Guidelines:
            - Return confidence scores between 0.0 and 1.0
            - Include explanations and evidence
            - Only return categories that clearly match""",
                ),
            ]
        )

        # Create chain
        chain = (
            {"text": RunnablePassthrough(), "categories_json": lambda _: self._format_categories()}
            | template
            | self.llm
            | self._process_llm_output
        )

        return chain

    def _format_categories(self) -> str:
        """Format categories for prompt."""
        formatted = []
        for name, config in self.categories.items():
            cat_info = {
                "name": name,
                "description": config.description,
                "keywords": config.keywords,
                "threshold": config.threshold,
            }
            formatted.append(cat_info)
        return json.dumps(formatted, indent=2)

    def _process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process raw LLM output."""
        try:
            # Handle different output types
            if hasattr(output, "content"):
                content = output.content
            elif isinstance(output, str):
                content = output
            elif isinstance(output, dict):
                return output
            else:
                return self._create_empty_output()

            # Parse JSON content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return self._create_empty_output()

        except Exception as e:
            self.logger.error(f"Error processing output: {e}")
            return self._create_empty_output()

    def _create_empty_output(self) -> Dict[str, Any]:
        """Create empty output structure."""
        return {"categories": []}

    async def analyze(self, text: str, **kwargs) -> CategoryOutput:
        """Analyze text and categorize it."""
        try:
            if kwargs:
                self.config.update(kwargs)

            # Get LLM analysis
            llm_results = await self.chain.ainvoke(text)

            if "error" in llm_results:
                return self._create_error_output()

            # Get language if possible
            language = self.language_processor.language if self.language_processor else "unknown"

            # Process categories from LLM results
            categories = []
            explanations = {}
            evidence = {}

            for cat in llm_results.get("categories", []):
                name = cat.get("name", "")
                confidence = float(cat.get("confidence", 0))

                if name in self.categories and confidence >= self.categories[name].threshold:
                    category_info = {"name": name, "confidence": confidence}
                    categories.append(category_info)
                    explanations[name] = cat.get("explanation", "")
                    evidence[name] = cat.get("evidence", [])

            # Return properly structured output
            return CategoryOutput(
                categories=categories, explanations=explanations, evidence=evidence, language=language, success=True
            )

        except Exception as e:
            return self._create_error_output(str(e))

    def _create_error_output(self, error: Optional[str] = None) -> CategoryOutput:
        """Create error output."""
        return CategoryOutput(
            error=error or "Category analysis failed",
            success=False,
            language=self.language_processor.language if self.language_processor else "unknown",
        )

    def _update_config(self, kwargs: Dict[str, Any]) -> None:
        """Update configuration with runtime parameters."""
        if "min_confidence" in kwargs:
            self.min_confidence = kwargs["min_confidence"]
        if "max_categories" in kwargs:
            self.max_categories = kwargs["max_categories"]
        if "require_evidence" in kwargs:
            self.require_evidence = kwargs["require_evidence"]
        if "categories" in kwargs:
            self.categories.update(kwargs["categories"])

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output into standardized format.
        Clean and validate LLM output.

        Args:
            output: Raw LLM output

        Returns:
            Dict[str, Any]: Cleaned and validated output
        """
        try:
            # First use base class processing to handle JSON parsing
            data = super()._post_process_llm_output(output)

            if "error" in data:
                return data

            return {
                "categories": data.get("categories", {}),
                "explanations": data.get("explanations", {}),
                "evidence": data.get("evidence", {}),
            }
        except Exception as e:
            self.logger.error(f"Error processing category output: {e}")
            return {"error": str(e)}

    def add_category(self, name: str, description: str) -> None:
        """Add a new category to the analyzer.

        Args:
            name: Category name
            description: Category description
        """
        self.categories[name] = description
        logger.info(f"Added new category: {name}")

    def remove_category(self, name: str) -> None:
        """Remove a category from the analyzer.

        Args:
            name: Category name to remove
        """
        if name in self.categories:
            del self.categories[name]
            logger.info(f"Removed category: {name}")

    def update_category(self, name: str, description: str) -> None:
        """Update a category's description.

        Args:
            name: Category name
            description: New category description
        """
        if name in self.categories:
            self.categories[name] = description
            logger.info(f"Updated category: {name}")
        else:
            logger.warning(f"Category not found: {name}")
