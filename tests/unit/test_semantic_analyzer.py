# tests/unit/test_semantic_analyzer.py

"""Test cases for semantic analyzer."""

import asyncio
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.loaders.parameter_handler import ParameterHandler, ParameterSheets
from src.schemas import CompleteAnalysisResult
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.FileUtils.file_utils import FileUtils, OutputFileType
from tests.helpers.mock_llms.category_mock import CategoryMockLLM
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM
from tests.helpers.mock_llms.theme_mock import ThemeMockLLM


@pytest.fixture
def test_parameters() -> Dict[str, Any]:
    """Provide test parameter configurations."""
    return {
        "general": {
            "max_keywords": 8,
            "min_keyword_length": 3,
            "language": "en",
            "focus_on": "technical content",
            "include_compounds": True,
            "max_themes": 3,
            "min_confidence": 0.3,
            "column_name_to_analyze": "content",
        },
        "categories": {
            "technical": {
                "description": "Technical content",
                "keywords": ["software", "api", "data"],
                "threshold": 0.6,
            },
            "business": {
                "description": "Business content",
                "keywords": ["revenue", "growth", "market"],
                "threshold": 0.6,
            },
        },
        "analysis_settings": {
            "theme_analysis": {
                "enabled": True,
                "min_confidence": 0.5,
            },
            "weights": {
                "statistical": 0.4,
                "llm": 0.6,
            },
        },
    }


class TestSemanticAnalyzer:
    """Tests for semantic analysis functionality."""

    @pytest.fixture
    def keyword_mock(self):
        """Create KeywordMockLLM instance."""
        return KeywordMockLLM()

    @pytest.fixture
    def theme_mock(self):
        """Create ThemeMockLLM instance."""
        return ThemeMockLLM()

    @pytest.fixture
    def category_mock(self):
        """Create CategoryMockLLM instance."""
        return CategoryMockLLM()

    def _save_parameter_file(
        self,
        file_utils: FileUtils,
        sheet_data: Dict[str, Any],
        file_name: str,
    ) -> Path:
        """Helper to save parameter files using FileUtils."""
        dataframes = {}
        for sheet_name, params in sheet_data.items():
            # Get language from sheet name
            is_finnish = sheet_name == ParameterSheets.get_sheet_name(
                "general", "fi"
            )
            language = "fi" if is_finnish else "en"

            # Get correct column names and parameter mappings
            column_names = ParameterSheets.get_column_names("general", language)
            param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
                "parameters"
            ][language]

            rows = []
            for key, value in params.items():
                # Map internal parameter names to language-specific names
                # Find the Excel name that maps to this internal key
                excel_name = next(
                    (
                        excel_key
                        for excel_key, internal_key in param_mappings.items()
                        if internal_key == key
                    ),
                    key,  # fallback to original key if no mapping found
                )

                rows.append(
                    {
                        column_names["parameter"]: excel_name,
                        column_names["value"]: value,
                        column_names[
                            "description"
                        ]: f"Description for {excel_name}",
                    }
                )
            dataframes[sheet_name] = pd.DataFrame(rows)

        saved_files, _ = file_utils.save_data_to_disk(
            data=dataframes,
            output_filetype=OutputFileType.XLSX,
            output_type="parameters",
            file_name=file_name,
            include_timestamp=False,
        )

        return Path(next(iter(saved_files.values())))

    @pytest.mark.asyncio
    async def test_complete_analysis(
        self,
        file_utils: FileUtils,
        keyword_mock: KeywordMockLLM,
        theme_mock: ThemeMockLLM,
        category_mock: CategoryMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test complete analysis pipeline with all mock types."""
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={"General Parameters": test_parameters["general"]},
            file_name="test_params",
        )

        # Create analyzers with different mocks
        keyword_analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
        )

        theme_analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=theme_mock
        )

        category_analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=category_mock
        )

        text = """Machine learning models are trained using large datasets.
                Neural networks enable complex pattern recognition.
                The system's performance metrics show significant improvements."""

        # Run analyses with specific types
        keyword_result = await keyword_analyzer.analyze(
            text, analysis_types=["keywords"]
        )
        assert keyword_result.keywords.success
        assert len(keyword_result.keywords.keywords) > 0
        assert any(
            "machine learning" in kw.keyword.lower()
            for kw in keyword_result.keywords.keywords
        )

        theme_result = await theme_analyzer.analyze(
            text, analysis_types=["themes"]
        )
        assert theme_result.themes.success
        assert len(theme_result.themes.themes) > 0
        assert any(
            "learning" in theme.name.lower()
            for theme in theme_result.themes.themes
        )

        category_result = await category_analyzer.analyze(
            text, analysis_types=["categories"]
        )
        assert category_result.categories.success
        assert len(category_result.categories.matches) > 0
        # Update expectation to match what the mock actually returns
        assert any(
            "learning" in cat.name.lower()
            for cat in category_result.categories.matches
        )

    @pytest.mark.asyncio
    async def test_specific_analysis_types(
        self,
        file_utils: FileUtils,
        keyword_mock: KeywordMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test keyword analysis type specifically."""
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={"General Parameters": test_parameters["general"]},
            file_name="test_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            file_utils=file_utils,
            llm=keyword_mock,  # Use keyword mock for keyword analysis
        )

        text = "Machine learning models process data efficiently."
        result = await analyzer.analyze(text, analysis_types=["keywords"])

        # Only keywords should be analyzed
        assert result.keywords.success
        assert len(result.keywords.keywords) > 0
        assert not result.themes.themes
        assert not result.categories.matches

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        file_utils: FileUtils,
        keyword_mock: KeywordMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test error handling in analysis."""
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={"General Parameters": test_parameters["general"]},
            file_name="test_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
        )

        # Test with empty input
        result = await analyzer.analyze("")
        assert not result.success
        assert result.keywords.error is not None
        assert "Empty input text" in result.keywords.error
        assert "Empty input text" in result.themes.error
        assert "Empty input text" in result.categories.error

        # Test with invalid analysis type - check result instead of expecting exception
        result = await analyzer.analyze("test", analysis_types=["invalid"])
        assert not result.success
        assert "Invalid analysis types" in result.error

    @pytest.mark.asyncio
    async def test_finnish_analysis(
        self,
        file_utils: FileUtils,
        keyword_mock: KeywordMockLLM,
        theme_mock: ThemeMockLLM,
        category_mock: CategoryMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test Finnish language analysis with all mock types."""
        fi_params = test_parameters.copy()
        fi_params["general"]["language"] = "fi"

        # Get correct Finnish sheet name
        sheet_name = ParameterSheets.get_sheet_name("general", "fi")

        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: fi_params["general"]},
            file_name="test_params_fi",
        )

        # Create analyzers with different mocks
        keyword_analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
        )

        theme_analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=theme_mock
        )

        category_analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=category_mock
        )

        text = """Koneoppimismallit analysoivat dataa tehokkaasti.
                 Neuroverkon arkkitehtuuri mahdollistaa monimutkaisen hahmontunnistuksen."""

        # Run and validate each type of analysis
        keyword_result = await keyword_analyzer.analyze(
            text, analysis_types=["keywords"]
        )
        assert keyword_result.keywords.success
        assert any(
            "koneoppimis" in kw.keyword.lower()
            for kw in keyword_result.keywords.keywords
        )

        theme_result = await theme_analyzer.analyze(
            text, analysis_types=["themes"]
        )
        assert theme_result.themes.success
        assert any(
            "koneoppiminen" in theme.name.lower()
            for theme in theme_result.themes.themes
        )

        category_result = await category_analyzer.analyze(
            text, analysis_types=["categories"]
        )
        assert category_result.categories.success
        # Update expectation to match the mock's actual Finnish output
        assert any(
            "oppiminen" in cat.name.lower()
            for cat in category_result.categories.matches
        )

    @pytest.mark.asyncio
    async def test_batch_analysis(
        self,
        file_utils: FileUtils,
        keyword_mock: KeywordMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test batch analysis functionality with keyword mock."""
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={"General Parameters": test_parameters["general"]},
            file_name="test_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
        )

        texts = [
            "Machine learning models analyze data.",
            "Revenue growth exceeded expectations.",
            "System performance metrics improved.",
        ]

        results = await analyzer.analyze_batch(texts, batch_size=2)
        assert len(results) == len(texts)
        assert all(isinstance(r, CompleteAnalysisResult) for r in results)

    def test_result_saving(
        self,
        file_utils: FileUtils,
        keyword_mock: KeywordMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test saving analysis results."""
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={"General Parameters": test_parameters["general"]},
            file_name="test_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
        )

        result = CompleteAnalysisResult(
            keywords=analyzer._create_error_result_by_type("keywords"),
            themes=analyzer._create_error_result_by_type("themes"),
            categories=analyzer._create_error_result_by_type("categories"),
            language="en",
            success=True,
            processing_time=1.0,
        )

        output_path = analyzer.save_results(
            results=result, output_file="test_results", output_type="processed"
        )

        assert output_path.exists()
        assert output_path.suffix == ".xlsx"

    def _validate_analysis_result(self, result: CompleteAnalysisResult) -> None:
        """Validate complete analysis result structure."""
        assert result.success, f"Analysis failed: {result.error}"
        assert result.processing_time > 0

        # Validate keywords
        assert result.keywords.success
        assert len(result.keywords.keywords) > 0
        for kw in result.keywords.keywords:
            assert kw.keyword
            assert 0 <= kw.score <= 1.0

        # Validate themes
        assert result.themes.success
        assert len(result.themes.themes) > 0
        for theme in result.themes.themes:
            assert theme.name
            assert theme.description
            assert 0 <= theme.confidence <= 1.0

        # Validate categories
        assert result.categories.success
        assert len(result.categories.matches) > 0
        for cat in result.categories.matches:
            assert cat.name
            assert 0 <= cat.confidence <= 1.0
            assert cat.evidence
