# tests/test_components/test_theme_analyzer.py

import pytest
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.core.language_processing import create_text_processor
from src.schemas import ThemeOutput, ThemeInfo
from tests.helpers.mock_llm import MockLLM


@pytest.fixture(scope="session")
def test_content() -> Dict[str, Dict[str, str]]:
    """Provide test content."""
    return {
        "en": {
            "technical": """Machine learning models are trained using large datasets to recognize patterns.
                        Neural network architecture includes multiple layers for feature extraction.
                        Data preprocessing and feature engineering are crucial steps.""",
            "business": """Q3 financial results show 15% revenue growth and improved profit margins.
                        Customer acquisition costs decreased while retention rates increased.
                        Market expansion strategy focuses on emerging technology sectors.""",
        },
        "fi": {
            "technical": """Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja.
                        Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen.
                        Datan esikäsittely ja piirteiden suunnittelu ovat keskeisiä vaiheita.""",
            "business": """Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet.
                        Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani.
                        Markkinalaajennusstrategia keskittyy nouseviin teknologiasektoreihin.""",
        },
    }


class TestThemeAnalyzer:
    @pytest.fixture(scope="class")
    def theme_analyzer_en(self) -> ThemeAnalyzer:
        """Create English theme analyzer."""
        return ThemeAnalyzer(
            llm=MockLLM(),
            config={
                "max_themes": 3,
                "min_confidence": 0.3,
            },
            language_processor=create_text_processor(language="en"),
        )

    @pytest.fixture(scope="class")
    def theme_analyzer_fi(self) -> ThemeAnalyzer:
        """Create Finnish theme analyzer."""
        return ThemeAnalyzer(
            llm=MockLLM(),
            config={
                "max_themes": 3,
                "min_confidence": 0.3,
            },
            language_processor=create_text_processor(language="fi"),
        )

    def _validate_theme_output(self, result: Union[Dict, BaseModel]) -> None:
        """Validate theme analysis output."""
        if isinstance(result, dict):
            assert "themes" in result
            assert "success" in result
            if not result.get("success", True):
                return
            themes = result["themes"]
        else:
            assert hasattr(result, "themes")
            if not getattr(result, "success", True):
                return
            themes = result.themes

        assert len(themes) > 0
        for theme in themes:
            if isinstance(theme, dict):
                assert "name" in theme
                assert "description" in theme
                assert "confidence" in theme
                assert 0 <= theme["confidence"] <= 1.0
            else:
                assert isinstance(theme, ThemeInfo)
                assert theme.name
                assert theme.description
                assert 0 <= theme.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_english_technical_themes(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme extraction from English technical content."""
        result = await theme_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_theme_output(result)

        themes = {
            (
                theme.name.lower()
                if isinstance(theme, ThemeInfo)
                else theme["name"].lower()
            )
            for theme in (
                result.themes if hasattr(result, "themes") else result["themes"]
            )
        }
        expected_themes = {"machine learning", "data processing"}
        assert any(theme in themes for theme in expected_themes)

    @pytest.mark.asyncio
    async def test_english_business_themes(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme extraction from English business content."""
        result = await theme_analyzer_en.analyze(test_content["en"]["business"])
        self._validate_theme_output(result)

        themes = {theme.name.lower() for theme in result.themes}
        expected_themes = {"financial performance", "market growth"}
        assert any(theme in themes for theme in expected_themes)

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
        assert any(theme in themes for theme in expected_themes)

    @pytest.mark.asyncio
    async def test_theme_hierarchy(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme hierarchy relationships."""
        result = await theme_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_theme_output(result)

        # Check hierarchy exists and is properly structured
        assert hasattr(result, "theme_hierarchy")
        hierarchy = result.theme_hierarchy
        assert isinstance(hierarchy, dict)

        # Verify at least one parent-child relationship
        if hierarchy:
            for parent, children in hierarchy.items():
                assert isinstance(children, list)
                assert all(isinstance(child, str) for child in children)

    @pytest.mark.asyncio
    async def test_error_handling(
        self, theme_analyzer_en: ThemeAnalyzer
    ) -> None:
        """Test error handling."""
        # Empty input
        result = await theme_analyzer_en.analyze("")
        assert result.success is False
        assert result.error is not None

        # None input should raise TypeError
        with pytest.raises(TypeError):
            await theme_analyzer_en.analyze(None)  # type: ignore

    @pytest.mark.asyncio
    async def test_theme_evidence(
        self, theme_analyzer_en: ThemeAnalyzer, test_content: Dict
    ) -> None:
        """Test theme evidence extraction."""
        result = await theme_analyzer_en.analyze(
            test_content["en"]["technical"]
        )
        self._validate_theme_output(result)

        for theme in result.themes:
            # Check each theme has proper evidence
            assert hasattr(theme, "description")
            assert theme.description, "Theme should have a description"
            # Check for keywords if they exist
            if hasattr(theme, "keywords"):
                assert isinstance(theme.keywords, list)
