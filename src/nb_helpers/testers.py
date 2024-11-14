# src/nb_helpers/testers.py
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel

from src.nb_helpers.base import AnalysisTester, DisplayMixin
from src.analyzers import KeywordAnalyzer, ThemeAnalyzer, CategoryAnalyzer
from src.core.language_processing import create_text_processor
from src.core.llm.factory import create_llm
from src.loaders.models import CategoryConfig
from src.schemas import KeywordInfo, ThemeInfo

class KeywordTester(AnalysisTester, DisplayMixin):
    def __init__(self, llm: Optional[BaseChatModel] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm, config)
        self.analyzer = KeywordAnalyzer(
            llm=self.llm,
            config=self.config or {"weights": {"statistical": 0.4, "llm": 0.6}},
            language_processor=create_text_processor()
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        if hasattr(results, 'keywords'):
            print("\nKeywords Found:")
            for kw in results.keywords:
                bar = self.display_confidence_bar(kw.score)
                print(f"  • {kw.keyword:<20} [{bar}] ({kw.score:.2f})")

class ThemeTester(AnalysisTester, DisplayMixin):
    def __init__(self, llm: Optional[BaseChatModel] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(llm, config)
        self.analyzer = ThemeAnalyzer(
            llm=self.llm,
            config=self.config or {"max_themes": 3, "min_confidence": 0.3}
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        if hasattr(results, 'themes'):
            print("\nThemes Found:")
            for theme in results.themes:
                bar = self.display_confidence_bar(theme.confidence)
                print(f"\n  • {theme.name}")
                print(f"    Confidence: [{bar}] ({theme.confidence:.2f})")
                print(f"    {theme.description}")

class CategoryTester(AnalysisTester, DisplayMixin):
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None, 
        config: Optional[Dict[str, Any]] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None
    ):
        super().__init__(llm, config)
        self.categories = categories or self._get_default_categories()
        self.analyzer = CategoryAnalyzer(
            categories=self.categories,
            llm=self.llm,
            config=self.config,
            language_processor=create_text_processor()
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(self, results: Dict[str, Any], detailed: bool) -> None:
        if hasattr(results, 'categories'):
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
                description="Technical content",
                keywords=["software", "development", "api", "system"],
                threshold=0.6
            ),
            "business": CategoryConfig(
                description="Business content",
                keywords=["revenue", "sales", "market", "growth"],
                threshold=0.6
            )
        }