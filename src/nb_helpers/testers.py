# src/nb_helpers/testers.py
import logging

from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel

from src.nb_helpers.base import AnalysisTester, DisplayMixin
from src.analyzers import KeywordAnalyzer, ThemeAnalyzer, CategoryAnalyzer
from src.core.language_processing import create_text_processor
from src.core.language_processing.base import BaseTextProcessor
from src.core.language_processing.finnish import FinnishTextProcessor
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
        language_processor: Optional[
            BaseTextProcessor
        ] = None,  # Add language_processor
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
        self.language_processor = language_processor  # Store language processor


class KeywordTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = KeywordAnalyzer(
            llm=self.llm,
            config=self.config or {"weights": {"statistical": 0.4, "llm": 0.6}},
            language_processor=kwargs.get("language_processor")
            or create_text_processor(),
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        logger.debug("KeywordTester starting analysis")
        result = await self.analyzer.analyze(text)
        logger.debug("KeywordTester analysis complete")
        return result

    def analyze_words(self, words: List[str]) -> None:
        """Analyze specific words using the current language processor."""
        if not isinstance(
            self.analyzer.language_processor, FinnishTextProcessor
        ):
            print("Word analysis is only available for Finnish text processor")
            return

    def _display_specific_results(
        self, results: Dict[str, Any], detailed: bool
    ) -> None:
        if hasattr(results, "keywords"):
            print("\nKeywords Found:")
            for kw in results.keywords:
                bar = self.display_confidence_bar(kw.score)
                print(f"  • {kw.keyword:<20} [{bar}] ({kw.score:.2f})")


class ThemeTester(BaseTester):
    def __init__(self, *args, **kwargs):
        # Extract language_processor before calling super
        language_processor = kwargs.pop("language_processor", None)
        super().__init__(*args, **kwargs)

        self.analyzer = ThemeAnalyzer(
            llm=self.llm,
            config=self.config or {"max_themes": 3, "min_confidence": 0.3},
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(
        self, results: Dict[str, Any], detailed: bool
    ) -> None:
        if hasattr(results, "themes"):
            print("\nThemes Found:")
            for theme in results.themes:
                bar = self.display_confidence_bar(theme.confidence)
                print(f"\n  • {theme.name}")
                print(f"    Confidence: [{bar}] ({theme.confidence:.2f})")
                print(f"    {theme.description}")


class CategoryTester(BaseTester):
    def __init__(
        self,
        *args,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        **kwargs,
    ):
        # Extract language_processor before calling super
        language_processor = kwargs.pop("language_processor", None)
        super().__init__(*args, **kwargs)

        # Get categories from parameters if available
        if not categories and self.config and "categories" in self.config:
            categories = self.config["categories"]

        self.categories = categories or self._get_default_categories()
        self.analyzer = CategoryAnalyzer(
            categories=self.categories,
            llm=self.llm,
            config=self.config,
            language_processor=language_processor or create_text_processor(),
        )

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        return await self.analyzer.analyze(text)

    def _display_specific_results(
        self, results: Dict[str, Any], detailed: bool
    ) -> None:
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
                description="Technical content",
                keywords=["software", "development", "api", "system"],
                threshold=0.6,
            ),
            "business": CategoryConfig(
                description="Business content",
                keywords=["revenue", "sales", "market", "growth"],
                threshold=0.6,
            ),
        }


def analyze_problematic_words(
    processor: FinnishTextProcessor, words: List[str]
) -> None:
    """Analyze specific words in detail to debug processing decisions.

    Args:
        processor: FinnishTextProcessor instance
        words: List of words to analyze

    Example:
        >>> processor = create_text_processor(language="fi")
        >>> problematic_words = ["para", "parani", "parantua", "kasvu", "kasvaa"]
        >>> analyze_problematic_words(processor, problematic_words)
    """
    for word in words:
        print(f"\nAnalyzing word: '{word}'")
        print("=" * 50)

        # Get basic info
        base_form = processor.get_base_form(word)
        is_verb = processor.is_verb(word)
        should_keep = processor.should_keep_word(word)

        # Get full Voikko analysis
        analyses = processor.voikko.analyze(word) if processor.voikko else []

        print(f"Base form: {base_form}")
        print(f"Is verb: {is_verb}")
        print(f"Should keep: {should_keep}")

        if analyses:
            print("\nVoikko analysis:")
            for key, value in analyses[0].items():
                print(f"  {key}: {value}")
        else:
            print("No Voikko analysis available")

        print("-" * 50)
