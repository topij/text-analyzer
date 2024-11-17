# src/nb_helpers/testers.py
import logging
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel

from src.nb_helpers.base import AnalysisTester, DisplayMixin
from src.analyzers import KeywordAnalyzer, ThemeAnalyzer, CategoryAnalyzer
from src.core.language_processing import create_text_processor
from src.core.llm.factory import create_llm
from src.loaders.models import CategoryConfig
from src.schemas import KeywordInfo, ThemeInfo
from src.loaders.parameter_handler import ParameterHandler

logger = logging.getLogger(__name__)


class BaseTester(AnalysisTester, DisplayMixin):
    """Base class for all testers with common parameter handling."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        parameter_file: Optional[str] = None,
    ):
        """Initialize with unified parameter handling."""
        if parameter_file:
            try:
                handler = ParameterHandler(parameter_file)
                params = handler.get_parameters()
                config = config or {}
                config.update(params.model_dump())
            except Exception as e:
                logger.warning(f"Could not load parameters from file: {e}")

        super().__init__(llm, config)


class KeywordTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = KeywordAnalyzer(
            llm=self.llm,
            config=self.config or {"weights": {"statistical": 0.4, "llm": 0.6}},
            language_processor=create_text_processor(),
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        logger.debug("KeywordTester starting analysis")  # Add debug logging
        result = await self.analyzer.analyze(text)
        logger.debug("KeywordTester analysis complete")  # Add debug logging
        return result

    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        if hasattr(results, "keywords"):
            print("\nKeywords Found:")
            for kw in results.keywords:
                bar = self.display_confidence_bar(kw.score)
                print(f"  • {kw.keyword:<20} [{bar}] ({kw.score:.2f})")


class ThemeTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = ThemeAnalyzer(llm=self.llm, config=self.config or {"max_themes": 3, "min_confidence": 0.3})

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        if hasattr(results, "themes"):
            print("\nThemes Found:")
            for theme in results.themes:
                bar = self.display_confidence_bar(theme.confidence)
                print(f"\n  • {theme.name}")
                print(f"    Confidence: [{bar}] ({theme.confidence:.2f})")
                print(f"    {theme.description}")


class CategoryTester(BaseTester):
    def __init__(self, *args, categories: Optional[Dict[str, CategoryConfig]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Get categories from parameters if available
        if not categories and self.config and "categories" in self.config:
            categories = self.config["categories"]

        self.categories = categories or self._get_default_categories()
        self.analyzer = CategoryAnalyzer(
            categories=self.categories, llm=self.llm, config=self.config, language_processor=create_text_processor()
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        if hasattr(results, "categories"):
            print("\nCategories Found:")
            for cat in results.categories:
                bar = self.display_confidence_bar(cat.confidence)
                print(f"\n  • {cat.name}")
                print(f"    Confidence: [{bar}] ({cat.confidence:.2f})")
                print(f"    {cat.explanation}")
                if detailed and cat.evidence:
                    print("    Evidence:")
                    for ev in cat.evidence:
                        print(f"      - {ev}")

    def _get_default_categories(self) -> Dict[str, CategoryConfig]:
        return {
            "technical": CategoryConfig(
                description="Technical content", keywords=["software", "development", "api", "system"], threshold=0.6
            ),
            "business": CategoryConfig(
                description="Business content", keywords=["revenue", "sales", "market", "growth"], threshold=0.6
            ),
        }
