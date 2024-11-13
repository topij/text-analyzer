# src/analyzers/category_analyzer.py

from typing import Any, Dict, List, Optional
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import TextAnalyzer, AnalyzerOutput
from src.core.language_processing.base import BaseTextProcessor
from src.loaders.models import CategoryConfig

class CategoryInfo(BaseModel):
    """Information about a category match."""
    name: str = Field(..., description="Category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: str = Field(..., description="Explanation for classification")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    themes: List[str] = Field(default_factory=list, description="Related themes")

class CategoryOutput(AnalyzerOutput):
    """Output model for category analysis."""
    categories: List[CategoryInfo] = Field(default_factory=list)
    explanations: Dict[str, str] = Field(default_factory=dict)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)

class CategoryAnalyzer(TextAnalyzer):
    """Analyzes text to classify it into predefined categories."""
    
    def __init__(
        self,
        categories: Dict[str, CategoryConfig],
        llm=None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None
    ):
        """Initialize category analyzer.
        
        Args:
            categories: Dictionary of category configurations
            llm: Optional language model override
            config: Optional configuration parameters
            language_processor: Optional language processor
        """
        super().__init__(llm, config)
        self.categories = categories
        self.language_processor = language_processor
        self.min_confidence = config.get("min_confidence", 0.3)
        self.chain = self._create_chain()

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain."""
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a text classification expert. Analyze text and classify it into relevant categories.
                For each category:
                1. Determine if the text belongs in that category
                2. Assign a confidence score (0.0 to 1.0)
                3. Provide explanation and supporting evidence

                Return results in JSON format with these exact fields:
                {{
                    "categories": [
                        {{
                            "name": "category_name",
                            "confidence": 0.95,
                            "explanation": "Explanation text",
                            "evidence": ["evidence1", "evidence2"],
                            "themes": ["theme1", "theme2"]
                        }}
                    ]
                }}"""
            ),
            (
                "human",
                """Analyze this text and classify it into these categories:
                {categories_json}
                
                Text: {text}
                
                Guidelines:
                - Only return categories with confidence >= {min_confidence}
                - Include clear explanations and evidence
                - Consider themes and context"""
            )
        ])

        chain = (
            {
                "text": RunnablePassthrough(),
                "categories_json": lambda _: self._format_categories(),
                "min_confidence": lambda _: self.min_confidence
            }
            | template
            | self.llm
            | self._post_process_llm_output
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
                "threshold": config.threshold
            }
            formatted.append(cat_info)
        return json.dumps(formatted, indent=2)

    async def analyze(self, text: str, **kwargs) -> CategoryOutput:
        """Analyze text and categorize it."""
        try:
            # Validate input
            if error := self._validate_input(text):
                return CategoryOutput(
                    error=error,
                    success=False,
                    language="unknown"
                )

            # Get language
            language = (
                self.language_processor.language 
                if self.language_processor 
                else self._detect_language(text)
            )

            # Get LLM analysis
            results = await self.chain.ainvoke(text)

            # Process results
            categories = []
            explanations = {}
            evidence = {}

            for cat in results.get("categories", []):
                name = cat.get("name")
                if name in self.categories:
                    # Check against category threshold
                    confidence = float(cat.get("confidence", 0))
                    if confidence >= self.categories[name].threshold:
                        categories.append(CategoryInfo(
                            name=name,
                            confidence=confidence,
                            explanation=cat.get("explanation", ""),
                            evidence=cat.get("evidence", []),
                            themes=cat.get("themes", [])
                        ))
                        explanations[name] = cat.get("explanation", "")
                        evidence[name] = cat.get("evidence", [])

            return CategoryOutput(
                categories=categories,
                explanations=explanations,
                evidence=evidence,
                language=language,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return CategoryOutput(
                error=str(e),
                success=False,
                language=language if 'language' in locals() else "unknown"
            )

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output to standardized format."""
        try:
            # Get raw content
            if hasattr(output, "content"):
                content = output.content
                return json.loads(content)
            elif isinstance(output, dict):
                return output
            else:
                return {"categories": []}

        except Exception as e:
            self.logger.error(f"Error processing LLM output: {e}")
            return {"categories": []}