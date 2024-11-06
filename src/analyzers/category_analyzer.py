# src/analyzers/category_analyzer.py

# from typing import List, Dict, Any, Optional, Tuple
#
# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableSequence, RunnablePassthrough

# from .base import TextAnalyzer, AnalyzerOutput


import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field

from .base import AnalyzerOutput, TextAnalyzer

logger = logging.getLogger(__name__)


class CategoryOutput(AnalyzerOutput):
    """Output model for category analysis."""

    categories: Dict[str, float] = Field(default_factory=dict, description="Category names and their confidence scores")
    category_explanations: Dict[str, str] = Field(
        default_factory=dict, description="Explanations for category assignments"
    )
    top_categories: List[Tuple[str, float]] = Field(
        default_factory=list, description="Top categories sorted by confidence"
    )
    supporting_evidence: Dict[str, List[str]] = Field(
        default_factory=dict, description="Supporting evidence for each category"
    )


class CategoryAnalyzer(TextAnalyzer):
    """Analyzes text to classify it into predefined categories."""

    def __init__(
        self,
        categories: Dict[str, str],
        llm=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize category analyzer.

        Args:
            categories: Dictionary of category names and their descriptions
            llm: Language model to use
            config: Configuration parameters
        """
        super().__init__(llm, config)
        self.categories = categories
        self.min_confidence = config.get("min_confidence", 0.3)
        self.max_categories = config.get("max_categories", 3)
        self.require_evidence = config.get("require_evidence", True)

    def _create_chain(self) -> RunnableSequence:
        template = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a text classification expert..."""),
                (
                    "human",
                    """Analyze this text for the following categories:
            
            Categories:
            {category_definitions}

            Text: {text}
            
            Guidelines:
            - Minimum confidence: {min_confidence}
            - Maximum categories: {max_categories}
            - Must provide evidence: {require_evidence}""",
                ),
            ]
        )

        # Create processing chain
        chain = (
            {
                "text": RunnablePassthrough(),
                "category_definitions": lambda x: self._format_categories(),  # Remove '_' parameter
                "min_confidence": lambda x: self.min_confidence,
                "max_categories": lambda x: self.max_categories,
                "require_evidence": lambda x: self.require_evidence,
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    def _format_categories(self) -> str:
        """Format category definitions."""
        return "\n".join(f"- {name}: {description}" for name, description in self.categories.items())

    async def analyze(self, text: str, **kwargs) -> CategoryOutput:
        """Analyze text to classify it into categories.

        Args:
            text: Input text to analyze
            **kwargs: Additional parameters

        Returns:
            CategoryOutput: Analysis results

        The analysis includes:
        - Category assignments with confidence scores
        - Explanations for each assignment
        - Supporting evidence for each category
        - Ranked list of top categories
        """
        # Validate input
        if error := self._validate_input(text):
            return self._handle_error(error)

        if not self.categories:
            return self._handle_error("No categories defined for classification")

        try:
            # Update config with any runtime parameters
            self._update_config(kwargs)

            # Get LLM analysis
            results = await self.chain.ainvoke(text)

            # Filter and sort categories by confidence
            categories = results.get("categories", {})
            filtered_categories = {cat: score for cat, score in categories.items() if score >= self.min_confidence}

            # Sort categories by confidence
            top_categories = sorted(filtered_categories.items(), key=lambda x: x[1], reverse=True)[
                : self.max_categories
            ]

            # Build response
            return CategoryOutput(
                categories=filtered_categories,
                category_explanations=results.get("explanations", {}),
                top_categories=top_categories,
                supporting_evidence=results.get("evidence", {}),
                language=self._detect_language(text),
            )

        except Exception as e:
            logger.error(f"Error in category analysis: {str(e)}", exc_info=True)
            return self._handle_error(f"Category analysis failed: {str(e)}")

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

    # def _post_process_llm_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
    #     """Clean and validate LLM output.

    #     Args:
    #         output: Raw LLM output

    #     Returns:
    #         Dict[str, Any]: Cleaned and validated output
    #     """
    # if isinstance(output, str):
    #     import json
    #     try:
    #         output = json.loads(output)
    #     except json.JSONDecodeError:
    #         logger.error("Failed to parse LLM output as JSON")
    #         return {
    #             "categories": {},
    #             "explanations": {},
    #             "evidence": {}
    #         }

    # # Ensure we only have valid categories
    # categories = {
    #     cat: score
    #     for cat, score in output.get("categories", {}).items()
    #     if cat in self.categories and isinstance(score, (int, float))
    # }

    # # Normalize confidence scores
    # max_score = max(categories.values()) if categories else 1.0
    # categories = {
    #     cat: score / max_score
    #     for cat, score in categories.items()
    # }

    # # Ensure explanations and evidence for each category
    # explanations = output.get("explanations", {})
    # evidence = output.get("evidence", {})

    # for category in categories:
    #     if category not in explanations:
    #         explanations[category] = f"Category: {category}"
    #     if category not in evidence:
    #         evidence[category] = []

    # return {
    #     "categories": categories,
    #     "explanations": explanations,
    #     "evidence": evidence
    # }

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
