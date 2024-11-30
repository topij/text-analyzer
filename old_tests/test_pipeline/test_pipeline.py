# tests/test_pipeline/test_pipeline.py

import pytest

from src.schemas import CompleteAnalysisResult
from src.semantic_analyzer import SemanticAnalyzer


class TestPipeline:
    @pytest.fixture
    def analyzer(self, file_utils):
        return SemanticAnalyzer(parameter_file="parameters_en.xlsx")

    @pytest.mark.asyncio
    async def test_full_pipeline(self, analyzer, test_data):
        """Test complete analysis pipeline."""
        result = await analyzer.analyze(test_data["en_technical"])

        assert isinstance(result, CompleteAnalysisResult)
        assert result.success
        assert result.processing_time > 0

        # Check all components
        assert len(result.keywords.keywords) > 0
        assert len(result.themes.themes) > 0
        assert len(result.categories.matches) > 0

    @pytest.mark.asyncio
    async def test_multilingual_support(self, test_data, file_utils):
        """Test pipeline with different languages."""
        en_analyzer = SemanticAnalyzer(parameter_file="parameters_en.xlsx")
        fi_analyzer = SemanticAnalyzer(parameter_file="parameters_fi.xlsx")

        en_result = await en_analyzer.analyze(test_data["en_technical"])
        fi_result = await fi_analyzer.analyze(test_data["fi_technical"])

        assert en_result.language == "en"
        assert fi_result.language == "fi"
        assert en_result.success and fi_result.success
