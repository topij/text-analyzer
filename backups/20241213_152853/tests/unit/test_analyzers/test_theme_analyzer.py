# tests/unit/test_analyzers/test_theme_analyzer.py

import json
from typing import Dict, List

import pytest

from src.core.config import AnalyzerConfig

from src.analyzers.theme_analyzer import ThemeAnalyzer, ThemeOutput
from src.core.language_processing import create_text_processor
from src.schemas import ThemeInfo
from tests.helpers.mock_llms.theme_mock import ThemeMockLLM


class TestThemeAnalyzer:
    """Tests for theme analysis functionality."""

    @pytest.fixture
    def mock_llm(self) -> ThemeMockLLM:
        """Create mock LLM instance."""
        return ThemeMockLLM()

    # @pytest.fixture
    # def analyzer(self, mock_llm: ThemeMockLLM) -> ThemeAnalyzer:
    #     """Create English theme analyzer with mock LLM."""
    #     return ThemeAnalyzer(
    #         llm=mock_llm,
    #         config={
    #             "max_themes": 3,
    #             "min_confidence": 0.3,
    #             "language": "en",
    #             "focus_on": "theme extraction",
    #         },
    #         language_processor=create_text_processor(language="en"),
    #     )

    # @pytest.fixture
    # def fi_analyzer(self, mock_llm: ThemeMockLLM) -> ThemeAnalyzer:
    #     """Create Finnish theme analyzer with mock LLM."""
    #     return ThemeAnalyzer(
    #         llm=mock_llm,
    #         config={
    #             "max_themes": 3,
    #             "min_confidence": 0.3,
    #             "language": "fi",
    #             "focus_on": "theme extraction",
    #         },
    #         language_processor=create_text_processor(language="fi"),
    #     )

    @pytest.fixture
    def analyzer(
        self, mock_llm: ThemeMockLLM, analyzer_config: AnalyzerConfig
    ) -> ThemeAnalyzer:
        """Create English theme analyzer with mock LLM."""
        return ThemeAnalyzer(
            llm=mock_llm,
            config=analyzer_config.config.get("analysis", {}),
            language_processor=create_text_processor(language="en"),
        )

    @pytest.fixture
    def fi_analyzer(
        self, mock_llm: ThemeMockLLM, analyzer_config: AnalyzerConfig
    ) -> ThemeAnalyzer:
        """Create Finnish theme analyzer with mock LLM."""
        return ThemeAnalyzer(
            llm=mock_llm,
            config={
                **analyzer_config.config.get("analysis", {}),
                "language": "fi",
            },
            language_processor=create_text_processor(language="fi"),
        )

    def _validate_theme_result(self, result: ThemeOutput) -> None:
        """Validate theme analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert result.themes, "No themes found in result"

        for theme in result.themes:
            assert isinstance(
                theme, ThemeInfo
            ), f"Invalid theme type: {type(theme)}"
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
    async def test_technical_theme_analysis(self, analyzer: ThemeAnalyzer):
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
        main_theme = "Machine Learning"
        assert any(
            main_theme in parents for parents in result.theme_hierarchy.keys()
        ), "Expected main theme not found in hierarchy"

    @pytest.mark.asyncio
    async def test_business_theme_analysis(self, analyzer: ThemeAnalyzer):
        """Test theme analysis of business content."""
        text = """Q3 financial results show 15% revenue growth.
                Market expansion strategy focuses on emerging sectors.
                Customer acquisition and retention metrics improved."""

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        themes = {theme.name.lower() for theme in result.themes}
        assert (
            "financial performance" in themes
        ), "Financial Performance theme not found"
        assert "market strategy" in themes, "Market Strategy theme not found"

        # Verify evidence matches themes
        for theme in result.themes:
            if theme.name.lower() == "financial performance":
                assert any(
                    "revenue" in kw.lower() for kw in theme.keywords
                ), "Missing revenue-related keywords"

    @pytest.mark.asyncio
    async def test_finnish_technical_analysis(self, fi_analyzer: ThemeAnalyzer):
        """Test theme analysis of Finnish technical content."""
        text = """Koneoppimismalleja koulutetaan suurilla datajoukolla.
                 Neuroverkon arkkitehtuuri sisältää useita kerroksia.
                 Datan esikäsittely ja piirteiden suunnittelu ovat tärkeitä."""

        result = await fi_analyzer.analyze(text)
        self._validate_theme_result(result)

        themes = {theme.name.lower() for theme in result.themes}
        assert "koneoppiminen" in themes, "Koneoppiminen theme not found"
        assert "data-analyysi" in themes, "Data-analyysi theme not found"

    @pytest.mark.asyncio
    async def test_finnish_business_analysis(self, fi_analyzer: ThemeAnalyzer):
        """Test theme analysis of Finnish business content."""
        text = """Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun.
                 Markkinalaajennusstrategia keskittyy uusiin sektoreihin."""

        result = await fi_analyzer.analyze(text)
        self._validate_theme_result(result)

        themes = {theme.name.lower() for theme in result.themes}
        assert (
            "taloudellinen suorituskyky" in themes
        ), "Taloudellinen Suorituskyky theme not found"
        assert "markkinakehitys" in themes, "Markkinakehitys theme not found"

    @pytest.mark.asyncio
    async def test_theme_hierarchy(self, analyzer: ThemeAnalyzer):
        """Test theme hierarchy relationships."""
        text = """Machine learning models perform complex data analysis.
                 Neural networks enable advanced pattern recognition.
                 Data preprocessing improves model accuracy."""

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        # Verify hierarchy structure
        assert result.theme_hierarchy, "Theme hierarchy missing"
        parent_theme = next(
            theme for theme in result.themes if theme.parent_theme is None
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

    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer: ThemeAnalyzer):
        """Test error handling for invalid inputs."""
        # Empty input
        result = await analyzer.analyze("")
        assert not result.success
        assert result.error is not None
        assert "Empty input" in result.error

        # None input
        with pytest.raises(ValueError) as exc_info:
            await analyzer.analyze(None)
        assert "Input text cannot be None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evidence_validation(self, analyzer: ThemeAnalyzer):
        """Test theme evidence validation."""
        text = """Machine learning models are becoming increasingly sophisticated.
                 Neural networks enable complex pattern recognition.
                 Data preprocessing is essential for model accuracy."""

        result = await analyzer.analyze(text)
        self._validate_theme_result(result)

        # Verify keywords presence
        for theme in result.themes:
            assert theme.keywords, f"No keywords found for theme {theme.name}"
            assert all(
                isinstance(kw, str) for kw in theme.keywords
            ), "Invalid keyword type found"

            # Verify keyword relevance to theme
            assert any(
                kw.lower() in theme.description.lower() for kw in theme.keywords
            ), f"No relevant keywords found in description for theme {theme.name}"
