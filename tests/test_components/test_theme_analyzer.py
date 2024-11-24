# tests/test_components/test_theme_analyzer.py

import pytest
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.schemas import ThemeOutput, ThemeInfo
from src.core.language_processing import create_text_processor
from tests.helpers.mock_llms import ThemeMockLLM


@pytest.fixture
def test_content() -> Dict[str, Dict[str, str]]:
    """Provide test content."""
    return {
        "en": {
            "technical": "Machine learning models are trained using large datasets to recognize patterns. "
            "Neural network architecture includes multiple layers for feature extraction. "
            "Data preprocessing and feature engineering are crucial steps.",
            "business": "Q3 financial results show 15% revenue growth and improved profit margins. "
            "Customer acquisition costs decreased while retention rates increased. "
            "Market expansion strategy focuses on emerging technology sectors.",
        },
        "fi": {
            "technical": "Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja. "
            "Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen. "
            "Datan esikäsittely ja piirteiden suunnittelu ovat keskeisiä vaiheita.",
            "business": "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet. "
            "Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani. "
            "Markkinalaajennusstrategia keskittyy nouseviin teknologiasektoreihin.",
        },
    }


class TestThemeAnalyzer:
    @pytest.fixture
    def theme_analyzer_en(self) -> ThemeAnalyzer:
        """Create English theme analyzer with mock LLM."""
        return ThemeAnalyzer(
            llm=ThemeMockLLM(),  # Updated to use specific mock
            config={
                "max_themes": 3,
                "min_confidence": 0.3,
                "language": "en",
                "models": {
                    "default_model": "gpt-4o-mini",
                    "parameters": {"temperature": 0.0, "max_tokens": 1000},
                },
                "focus_on": "theme extraction",
            },
            language_processor=create_text_processor(language="en"),
        )

    @pytest.fixture
    def theme_analyzer_fi(self) -> ThemeAnalyzer:
        """Create Finnish theme analyzer with mock LLM."""
        return ThemeAnalyzer(
            llm=ThemeMockLLM(),  # Updated to use specific mock
            config={
                "max_themes": 3,
                "min_confidence": 0.3,
                "language": "fi",
                "models": {
                    "default_model": "gpt-4o-mini",
                    "parameters": {"temperature": 0.0, "max_tokens": 1000},
                },
                "focus_on": "theme extraction",
            },
            language_processor=create_text_processor(language="fi"),
        )

    def _validate_theme_output(self, result: Union[Dict, BaseModel]) -> None:
        """Validate theme analysis output with debugging."""
        print(f"\nValidating theme output: {result}")  # Debug print

        # Early error check
        if not result.success:
            print(f"Result marked as failed: {result.error}")
            assert result.error is None, f"Analysis failed: {result.error}"
            return

        # Get themes
        themes = result.themes
        print(f"Number of themes found: {len(themes)}")
        assert len(themes) > 0, "No themes found in result"

        # Validate each theme
        for theme in themes:
            assert isinstance(theme, ThemeInfo)
            assert theme.name
            assert theme.description
            assert 0 <= theme.confidence <= 1.0
            if hasattr(theme, "keywords"):
                assert isinstance(theme.keywords, list)
            if theme.parent_theme:
                # Verify parent theme exists
                assert any(t.name == theme.parent_theme for t in themes)

        # Verify hierarchy if present
        if hasattr(result, "theme_hierarchy"):
            hierarchy = result.theme_hierarchy
            print(f"Theme hierarchy: {hierarchy}")  # Debug print
            if hierarchy:
                for parent, children in hierarchy.items():
                    assert isinstance(children, list)
                    assert all(isinstance(child, str) for child in children)

    @pytest.mark.asyncio
    async def test_english_technical_themes(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme extraction from English technical content."""
        result = await theme_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_theme_output(result)

        themes = {theme.name.lower() for theme in result.themes}
        expected_themes = {"machine learning", "data processing"}
        assert any(expected in " ".join(themes) for expected in expected_themes)

    @pytest.mark.asyncio
    async def test_english_business_themes(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme extraction from English business content."""
        result = await theme_analyzer_en.analyze(test_content["en"]["business"])
        self._validate_theme_output(result)

        themes = {theme.name.lower() for theme in result.themes}
        expected_themes = {"financial performance", "market growth"}
        assert any(
            expected in " ".join(themes) for expected in expected_themes
        ), f"Expected themes not found. Got: {themes}"

    @pytest.mark.asyncio
    async def test_finnish_technical_themes(
        self, theme_analyzer_fi: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme extraction from Finnish technical content."""
        result = await theme_analyzer_fi.analyze(
            test_content["fi"]["technical"]
        )
        self._validate_theme_output(result)

        themes = {theme.name.lower() for theme in result.themes}
        expected_themes = {"koneoppiminen", "data-analyysi"}
        assert any(
            expected in " ".join(themes) for expected in expected_themes
        ), f"Expected themes not found. Got: {themes}"

    @pytest.mark.asyncio
    async def test_theme_hierarchy(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme hierarchy relationships."""
        result = await theme_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_theme_output(result)

        print(f"\nTheme hierarchy from result: {result.theme_hierarchy}")
        print(f"\nAll themes: {[t.name for t in result.themes]}")
        print(
            f"\nTheme details: {[{'name': t.name, 'parent': t.parent_theme} for t in result.themes]}"
        )

        # Check hierarchy exists
        assert hasattr(result, "theme_hierarchy")
        hierarchy = result.theme_hierarchy

        # Expected relationship
        expected_parent = "Machine Learning"
        expected_child = "Data Processing"

        # Check hierarchy both ways
        has_hierarchy = False

        # Check direct hierarchy dictionary
        if hierarchy and expected_parent in hierarchy:
            if expected_child in hierarchy[expected_parent]:
                has_hierarchy = True
                print(
                    f"\nFound in hierarchy dict: {expected_parent} -> {expected_child}"
                )

        # Check theme parent references
        if not has_hierarchy:
            for theme in result.themes:
                if (
                    theme.name == expected_child
                    and theme.parent_theme == expected_parent
                ):
                    has_hierarchy = True
                    print(
                        f"\nFound in theme parent reference: {theme.name} -> {theme.parent_theme}"
                    )
                    break

        assert has_hierarchy, (
            f"Expected hierarchy relationship not found between "
            f"{expected_parent} and {expected_child}. "
            f"Hierarchy: {hierarchy}, "
            f"Themes: {[{'name': t.name, 'parent': t.parent_theme} for t in result.themes]}"
        )

    @pytest.mark.asyncio
    async def test_error_handling(
        self, theme_analyzer_en: ThemeAnalyzer
    ) -> None:
        """Test error handling."""
        # Empty input
        empty_result = await theme_analyzer_en.analyze("")
        assert empty_result.success is False
        assert empty_result.error is not None
        assert "Empty input" in empty_result.error

        # None input
        with pytest.raises(ValueError) as exc_info:
            await theme_analyzer_en.analyze(None)
        assert "Input text cannot be None" in str(exc_info.value)

        # Very short input
        short_result = await theme_analyzer_en.analyze("a")
        assert short_result.success is False
        assert short_result.error is not None
        assert (
            "too short" in short_result.error.lower()
            or "minimum" in short_result.error.lower()
        )

        # Invalid type input
        with pytest.raises(TypeError) as exc_info:
            await theme_analyzer_en.analyze(123)  # type: ignore
        assert "Invalid input type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_theme_evidence(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme evidence extraction."""
        result = await theme_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_theme_output(result)

        # Check theme evidence
        for theme in result.themes:
            assert theme.description, "Theme should have a description"
            if hasattr(theme, "keywords"):
                assert isinstance(theme.keywords, list)
                if theme.keywords:  # If keywords are present
                    assert all(isinstance(kw, str) for kw in theme.keywords)
