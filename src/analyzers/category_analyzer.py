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
from src.schemas import CategoryMatch, Evidence, CategoryOutput, ThemeContext
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

    async def analyze(self, text: str) -> CategoryOutput:
        """Analyze text with structured output handling."""
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
            logger.debug(f"Input text: {text[:100]}...")
            logger.debug(f"Available categories: {list(self.categories.keys())}")

            # Get the result directly from the chain
            result = await self.chain.ainvoke(text)
            logger.debug(f"Raw result from chain: {result}")

            # If result is already a CategoryOutput, process it
            if isinstance(result, CategoryOutput):
                # Filter categories by confidence
                result.categories = [
                    cat for cat in result.categories 
                    if cat.confidence >= self.min_confidence
                ]
                
                # Ensure predefined categories have proper descriptions
                for cat in result.categories:
                    if cat.name in self.categories:
                        cat.description = self.categories[cat.name].description

                return result

            # If we got a dict or other format, convert it
            processed = self._post_process_llm_output(result)
            
            # Create CategoryOutput from processed result
            return CategoryOutput(
                categories=[CategoryMatch(**cat) for cat in processed["categories"]],
                language=processed["language"],
                success=processed["success"],
                error=processed.get("error")
            )

        except Exception as e:
            logger.error(f"CategoryAnalyzer.analyze: Exception occurred: {e}")
            return CategoryOutput(
                categories=[],
                language=self._get_language(),
                success=False,
                error=str(e),
            )

    def _create_error_output(self) -> CategoryOutput:
        """Create error output."""
        return CategoryOutput(
            categories=[],
            language=self._get_language(),
            success=False,
            error="Analysis failed",
        )

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain with theme context support."""
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a category analysis expert. For the given text:
1. Match text content against both predefined categories and discover new relevant categories
2. Consider category hierarchies and relationships
3. Calculate match confidence scores
4. Identify supporting evidence

The predefined categories should be used as guidance for what is particularly interesting,
but you should also identify other significant categories that emerge from the content.

{theme_context_prompt}

Focus on categories that are clearly supported by the text content.
Do not force matches for categories without clear evidence."""
            ),
            (
                "human",
                """Analyze this text for category matches:
{text}

Language: {language}
{theme_context_str}

Predefined categories to consider (use as guidance):
{categories_str}

Required: 
1. Match against predefined categories with confidence scores and evidence
2. Identify additional relevant categories that emerge from the content
3. Provide clear supporting evidence for all category matches
4. Base all matches on actual content and evidence from the text"""
            ),
        ])

        return (
            {
                "text": RunnablePassthrough(),
                "language": lambda _: self._get_language(),
                "theme_context_str": lambda x: self._format_theme_context(x.get("theme_context") if isinstance(x, dict) else None),
                "theme_context_prompt": lambda x: self._get_theme_context_prompt(x.get("theme_context") if isinstance(x, dict) else None),
                "categories_str": lambda _: self._format_categories(),
            }
            | template
            | self.llm.with_structured_output(CategoryOutput)
        )

    def _get_theme_context_prompt(self, theme_context: Optional[ThemeContext]) -> str:
        """Get theme-specific prompt section based on context availability."""
        if theme_context and theme_context.main_themes:
            return """If theme context is provided:
1. Use themes to guide category matching
2. Consider theme hierarchy in scoring
3. Look for thematic alignment with categories
4. Use theme evidence to support category matches"""
        return ""  # Return empty string if no theme context

    def _format_theme_context(self, theme_context: Optional[ThemeContext]) -> str:
        """Format theme context for prompt if available."""
        if not theme_context or not theme_context.main_themes:
            return ""

        context_parts = ["Theme Context:"]
        
        # Add main themes with descriptions
        context_parts.append("\nMain Themes:")
        for theme in theme_context.main_themes:
            desc = theme_context.theme_descriptions.get(theme, "")
            conf = theme_context.theme_confidence.get(theme, 0.0)
            context_parts.append(f"- {theme} (confidence: {conf:.2f}): {desc}")
        
        # Add theme hierarchy if available
        if theme_context.theme_hierarchy:
            context_parts.append("\nTheme Relationships:")
            for parent, children in theme_context.theme_hierarchy.items():
                context_parts.append(f"- {parent} -> {', '.join(children)}")
        
        return "\n".join(context_parts)

    def _format_categories(self) -> str:
        """Format category information for prompt."""
        if not self.categories:
            return "No predefined categories available. Identify relevant categories from the content."

        parts = ["The following categories are of particular interest, but also look for other relevant categories:"]
        for name, config in self.categories.items():
            keywords = ", ".join(config.keywords[:5])  # Show first 5 keywords as examples
            parts.append(f"- {name}:")
            parts.append(f"  Description: {config.description}")
            parts.append(f"  Example keywords: {keywords}")
            
        return "\n".join(parts)

    def _adjust_score_with_themes(
        self,
        category: str,
        base_score: float,
        theme_context: Optional[ThemeContext]
    ) -> float:
        """Adjust category match score based on theme relevance."""
        if not theme_context or not theme_context.main_themes:
            return base_score

        max_theme_relevance = 0.0
        category_lower = category.lower()

        # Check relevance to theme descriptions
        for theme, desc in theme_context.theme_descriptions.items():
            # Calculate semantic similarity between category and theme
            similarity = self._calculate_theme_category_similarity(
                category_lower,
                theme.lower(),
                desc.lower()
            )
            if similarity > 0:
                theme_conf = theme_context.theme_confidence.get(theme, 0.0)
                relevance = similarity * theme_conf
                max_theme_relevance = max(max_theme_relevance, relevance)

        # Apply theme relevance boost (up to 25% boost for highly relevant categories)
        theme_boost = 1.0 + (max_theme_relevance * 0.25)
        return min(base_score * theme_boost, 1.0)

    def _calculate_theme_category_similarity(
        self,
        category: str,
        theme: str,
        theme_desc: str
    ) -> float:
        """Calculate semantic similarity between category and theme."""
        # Direct match in theme name
        if category in theme or theme in category:
            return 0.8
        
        # Match in theme description
        if category in theme_desc:
            return 0.6
        
        # Check category keywords in theme and description
        if category in self.categories:
            category_keywords = self.categories[category].keywords
            theme_matches = sum(1 for kw in category_keywords if kw.lower() in theme)
            desc_matches = sum(1 for kw in category_keywords if kw.lower() in theme_desc)
            
            if theme_matches or desc_matches:
                match_score = (theme_matches * 0.4 + desc_matches * 0.3) / len(category_keywords)
                return min(match_score, 0.7)
        
        return 0.0

    def _calculate_match_score(
        self,
        category: str,
        text: str,
        evidence: List[Dict[str, Any]],
        theme_context: Optional[ThemeContext] = None
    ) -> float:
        """Calculate category match score with flexible scoring for new categories."""
        base_score = 0.0
        
        # For predefined categories, use keyword-based scoring as a base
        if category in self.categories:
            config = self.categories[category]
            keyword_matches = sum(1 for kw in config.keywords if kw.lower() in text.lower())
            base_score = keyword_matches / max(len(config.keywords), 1)
            
            # Apply category weight if defined
            if hasattr(config, 'weight') and config.weight:
                base_score *= config.weight
        else:
            # For new categories, score based on evidence strength
            evidence_scores = [e.get("relevance", 0.0) for e in evidence]
            if evidence_scores:
                base_score = sum(evidence_scores) / len(evidence_scores)
                # Slightly lower confidence for new categories to prioritize predefined ones
                base_score *= 0.9
        
        # Adjust score based on themes if available
        if theme_context and theme_context.main_themes:
            score = self._adjust_score_with_themes(category, base_score, theme_context)
        else:
            score = base_score
        
        return min(max(score, 0.0), 1.0)

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
                # Handle both string and dict formats for category name
                if isinstance(cat, str):
                    transformed = {
                        "name": cat,
                        "confidence": 0.5,
                        "description": "",
                        "evidence": [],
                        "themes": []
                    }
                else:
                    # Map 'category' field to 'name' and handle various field names
                    transformed = {
                        "name": cat.get("category", cat.get("name", "")),
                        "confidence": cat.get("confidence", cat.get("score", 0.5)),
                        "description": cat.get("explanation", cat.get("description", "")),
                        "evidence": [
                            {
                                "text": e.get("text", ""),
                                "relevance": e.get("relevance", 0.5),
                            }
                            for e in cat.get("evidence", [])
                        ] if cat.get("evidence") else [],
                        "themes": cat.get("themes", [])
                    }
                transformed_categories.append(transformed)

            # Ensure we have a valid language
            language = data.get("language", self._get_language())
            if not language or language == "unknown":
                language = self._get_language()

            return {
                "categories": transformed_categories,
                "language": language,
                "success": data.get("success", True)
            }

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "categories": [],
                "success": False,
                "language": self._get_language(),
                "error": str(e)
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

    def _post_process_llm_output(self, response: Any) -> Dict[str, Any]:
        """Process LLM response and apply user-defined category parameters as guidance."""
        try:
            # Handle raw LLM response
            if hasattr(response, "content"):
                response = response.content
            elif isinstance(response, str):
                # Try to extract JSON from the string response
                try:
                    # Find JSON-like content between triple backticks if present
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                    if json_match:
                        response = json_match.group(1)
                    response = json.loads(response)
                except json.JSONDecodeError:
                    # If not valid JSON, create a basic structure
                    response = {
                        "categories": [],
                        "language": self._get_language(),
                        "success": True
                    }
            
            parsed = self._parse_response(response)
            
            # Process all categories, both predefined and new
            processed_categories = []
            for cat in parsed.get("categories", []):
                category_name = cat.get("name")
                if not category_name:  # Skip categories without names
                    continue
                    
                confidence = cat.get("confidence", 0.0)
                evidence = cat.get("evidence", [])
                
                # Calculate score based on category type and evidence
                score = self._calculate_match_score(
                    category_name,
                    str(evidence),  # Convert evidence to string for keyword matching
                    evidence,
                    None  # Theme context would be passed here if available
                )
                
                # Include if meets confidence threshold
                if score >= self.min_confidence:
                    # For predefined categories, use config description
                    if category_name in self.categories:
                        description = self.categories[category_name].description
                        is_predefined = True
                    else:
                        description = cat.get("description", "")
                        is_predefined = False
                        
                    processed_categories.append({
                        "name": category_name,
                        "confidence": score,
                        "description": description,
                        "evidence": evidence,
                        "themes": cat.get("themes", []),
                        "is_predefined": is_predefined
                    })
            
            # Sort categories: predefined first, then by confidence
            processed_categories.sort(key=lambda x: (not x["is_predefined"], -x["confidence"]))
            
            # Remove the temporary is_predefined field before returning
            for cat in processed_categories:
                cat.pop("is_predefined", None)
            
            return {
                "categories": processed_categories,
                "language": parsed.get("language", self._get_language()),
                "success": parsed.get("success", True)
            }
            
        except Exception as e:
            logger.error(f"Error post-processing response: {e}")
            return {
                "categories": [],
                "success": False,
                "language": self._get_language(),
                "error": str(e)
            }
