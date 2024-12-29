import pytest
from typing import Dict, Any

from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import ThemeOutput, ThemeInfo
from tests.helpers.mock_llms.theme_mock import ThemeMockLLM
from tests.conftest import test_environment_manager

class TestThemeAnalyzer:
    """Tests for theme analysis functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM instance."""
        return ThemeMockLLM()

    @pytest.fixture
    def analyzer(self, mock_llm, test_environment_manager):
        """Create analyzer with mock LLM and test config."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        
        config = {
            "min_confidence": 0.3,
            "max_themes": 5,
            "language": "en",
            "focus_on": "general content analysis",
        }
        
        return ThemeAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="en", file_utils=file_utils),
        )

    def _validate_theme_result(self, result: ThemeOutput) -> None:
        """Validate theme analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert result.themes, "No themes found in result"

        for theme in result.themes:
            assert isinstance(theme, ThemeInfo)
            assert theme.name, "Theme missing name"
            assert theme.description, "Theme missing description"
            assert (
                0 <= theme.confidence <= 1.0
            ), f"Invalid confidence score: {theme.confidence}"
            assert isinstance(
                theme.keywords, list
            ), "Theme keywords should be a list"

            if theme.parent_theme:
                assert any(
                    t.name == theme.parent_theme for t in result.themes
                ), f"Parent theme {theme.parent_theme} not found"

        # Validate hierarchy if present
        if result.theme_hierarchy:
            for parent, children in result.theme_hierarchy.items():
                assert isinstance(
                    children, list
                ), "Theme hierarchy children should be a list"
                assert all(
                    isinstance(child, str) for child in children
                ), "Theme hierarchy children should be strings"

    @pytest.mark.asyncio
    async def test_technical_theme_analysis(self, analyzer):
        """Test theme analysis of technical content."""
        text = """Machine learning models are trained using large datasets.
                Neural network architecture includes multiple layers.
                Data preprocessing and feature engineering are crucial steps."""

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        # Verify specific themes
        themes = {theme.name.lower() for theme in result.themes}
        assert "machine learning" in themes, "Machine Learning theme not found"
        assert "data processing" in themes, "Data Processing theme not found"

        # Verify theme hierarchy
        assert result.theme_hierarchy, "Theme hierarchy missing"
        assert any(
            "Machine Learning" in parents
            for parents in result.theme_hierarchy.keys()
        ), "Expected main theme not found in hierarchy"

    @pytest.mark.asyncio
    async def test_business_theme_analysis(self, analyzer):
        """Test theme analysis of business content."""
        text = """Q3 financial results show 15% revenue growth.
                Market expansion strategy focuses on emerging sectors.
                Customer acquisition and retention metrics improved."""

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        # Check specific themes
        themes = {theme.name.lower() for theme in result.themes}
        assert "financial performance" in themes
        assert "market strategy" in themes

        # Verify hierarchy relationships
        for theme in result.themes:
            if theme.parent_theme:
                assert theme.parent_theme in [t.name for t in result.themes]

    @pytest.mark.asyncio
    async def test_empty_input(self, analyzer):
        """Test analyzer behavior with empty input."""
        result = await analyzer.analyze("")
        assert not result.success
        assert not result.themes
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_none_input(self, analyzer):
        """Test analyzer behavior with None input."""
        with pytest.raises(ValueError):
            await analyzer.analyze(None)

    @pytest.mark.asyncio
    async def test_finnish_language(
        self, mock_llm, test_environment_manager
    ):
        """Test Finnish language support."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        
        config = {
            "min_confidence": 0.3,
            "max_themes": 5,
            "language": "fi",
            "focus_on": "general content analysis",
        }
        
        analyzer = ThemeAnalyzer(
            llm=mock_llm,
            config=config,
            language_processor=create_text_processor(language="fi", file_utils=file_utils),
        )

        text = """
        Tekoäly ja koneoppiminen ovat tärkeitä teknologioita.
        Ohjelmistokehitys vaatii paljon osaamista.
        """

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        # Verify Finnish themes
        themes = {theme.name.lower() for theme in result.themes}
        assert "tekoäly" in themes, "Expected Finnish theme not found"
        assert "ohjelmistokehitys" in themes, "Expected Finnish theme not found"

    @pytest.mark.asyncio
    async def test_theme_hierarchy(self, analyzer):
        """Test theme hierarchy relationships."""
        text = """Machine learning models perform complex data analysis.
                 Neural networks enable advanced pattern recognition.
                 Data preprocessing improves model accuracy."""

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        # Verify hierarchy structure
        assert result.theme_hierarchy, "Theme hierarchy missing"

        # At least one theme should be a parent
        parent_theme = next(
            theme for theme in result.themes if not theme.parent_theme
        )
        child_themes = [
            theme
            for theme in result.themes
            if theme.parent_theme == parent_theme.name
        ]

        assert child_themes, "No child themes found in hierarchy"
        assert (
            child_themes[0].parent_theme == parent_theme.name
        ), "Parent-child relationship not properly established"
