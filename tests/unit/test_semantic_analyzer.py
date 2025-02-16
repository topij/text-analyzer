# tests/unit/test_semantic_analyzer.py

"""Test cases for semantic analyzer."""

import pytest
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import asyncio
from langchain_core.language_models import BaseChatModel

from src.semantic_analyzer import SemanticAnalyzer
from src.loaders.models import CategoryConfig
from src.schemas import (
    CompleteAnalysisResult,
    KeywordOutput,
    ThemeOutput,
    CategoryOutput,
    KeywordAnalysisResult,
    ThemeAnalysisResult,
    CategoryAnalysisResult,
)
from src.config.manager import ConfigManager
from FileUtils import FileUtils
from tests.helpers.mock_llms.category_mock import CategoryMockLLM
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM
from tests.helpers.mock_llms.theme_mock import ThemeMockLLM
from tests.conftest import test_environment_manager

import logging

logger = logging.getLogger(__name__)


class TestSemanticAnalyzer:
    """Tests for semantic analysis functionality."""

    @pytest.mark.asyncio
    async def test_complete_analysis(
        self,
        test_environment_manager,
        test_parameters: Dict[str, Any],
    ):
        """Test complete analysis pipeline with all mock types."""
        # Get components from environment manager
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        config_manager = components["config_manager"]

        # Create parameter file
        sheet_name, param_df = self._create_parameter_df(test_parameters)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params",
        )

        # Create test categories
        categories = {
            "Technical": CategoryConfig(
                description="Technical content",
                keywords=["software", "api", "data"],
                threshold=0.6,
            ),
            "Business": CategoryConfig(
                description="Business content",
                keywords=["revenue", "growth", "market"],
                threshold=0.6,
            ),
        }

        # Create analyzers with different mocks
        analyzers = {
            "keywords": SemanticAnalyzer(
                parameter_file=file_path,
                environment_manager=test_environment_manager,
                llm=KeywordMockLLM().with_structured_output(KeywordOutput),
                config_manager=config_manager,
            ),
            "themes": SemanticAnalyzer(
                parameter_file=file_path,
                environment_manager=test_environment_manager,
                llm=ThemeMockLLM().with_structured_output(ThemeOutput),
                config_manager=config_manager,
            ),
            "categories": SemanticAnalyzer(
                parameter_file=file_path,
                environment_manager=test_environment_manager,
                llm=CategoryMockLLM().with_structured_output(CategoryOutput),
                categories=categories,
                config_manager=config_manager,
            ),
        }

        text = """Machine learning models are trained using large datasets.
                Neural networks enable complex pattern recognition.
                The system's performance metrics show significant improvements."""

        # Test each analyzer
        for analysis_type, analyzer in analyzers.items():
            result = await analyzer.analyze(text, analysis_types=[analysis_type])
            
            assert result.success, f"{analysis_type} analysis failed"
            
            if analysis_type == "keywords":
                assert result.keywords is not None
                assert result.keywords.success
                assert len(result.keywords.keywords) > 0
                assert any(
                    "machine learning" in kw.keyword.lower()
                    for kw in result.keywords.keywords
                )
                # Additional assertions for keyword structure
                for kw in result.keywords.keywords:
                    assert hasattr(kw, "score")
                    assert 0 <= kw.score <= 1.0
                    
            elif analysis_type == "themes":
                assert result.themes is not None
                assert result.themes.success
                assert len(result.themes.themes) > 0
                assert any(
                    "learning" in theme.name.lower()
                    for theme in result.themes.themes
                )
                # Additional assertions for theme structure
                for theme in result.themes.themes:
                    assert hasattr(theme, "confidence")
                    assert 0 <= theme.confidence <= 1.0
                    
            elif analysis_type == "categories":
                assert result.categories is not None
                assert result.categories.success
                assert len(result.categories.matches) > 0
                assert any(
                    cat.name == "Technical"
                    for cat in result.categories.matches
                )
                # Additional assertions for category structure
                for cat in result.categories.matches:
                    assert hasattr(cat, "confidence")
                    assert 0 <= cat.confidence <= 1.0
                    assert hasattr(cat, "evidence")

    def _create_parameter_df(
        self, test_parameters: Dict[str, Any], language: str = "en"
    ) -> Tuple[str, pd.DataFrame]:
        """Helper method to create parameter DataFrame with proper mappings."""
        from src.loaders.parameter_config import ParameterSheets

        # Get proper sheet name and mappings
        general_sheet_name = ParameterSheets.get_sheet_name("general", language)
        column_names = ParameterSheets.get_column_names("general", language)
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ][language]

        # Create DataFrame rows
        data_rows = []
        for internal_name, value in test_parameters["general"].items():
            excel_name = next(
                (
                    excel
                    for excel, internal in param_mappings.items()
                    if internal == internal_name
                ),
                internal_name,
            )
            data_rows.append(
                {
                    column_names["parameter"]: excel_name,
                    column_names["value"]: value,
                    column_names[
                        "description"
                    ]: f"Description for {excel_name}",
                }
            )

        return general_sheet_name, pd.DataFrame(data_rows)

    def _save_parameter_file(
        self,
        file_utils: FileUtils,
        sheet_data: Dict[str, pd.DataFrame],
        file_name: str,
    ) -> Path:
        """Helper to save parameter files using FileUtils."""
        try:
            saved_files, _ = file_utils.save_data_to_storage(
                data=sheet_data,
                output_filetype="xlsx",
                output_type="parameters",
                file_name=file_name,
                include_timestamp=False,
                engine="openpyxl",
            )

            return Path(next(iter(saved_files.values())))

        except Exception as e:
            print(f"Error saving parameter file: {e}")
            raise

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

    @pytest.fixture
    def test_parameters(self) -> Dict[str, Any]:
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

    @pytest.mark.asyncio
    async def test_specific_analysis_types(
        self,
        test_environment_manager,
        keyword_mock: KeywordMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test keyword analysis type specifically."""
        # Get components from environment manager
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        config_manager = components["config_manager"]

        # Create parameter file
        sheet_name, param_df = self._create_parameter_df(test_parameters)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params",
        )

        # Create test categories
        categories = {
            "Technical": CategoryConfig(
                description="Technical content",
                keywords=["software", "api", "data"],
                threshold=0.6,
            ),
            "Business": CategoryConfig(
                description="Business content",
                keywords=["revenue", "growth", "market"],
                threshold=0.6,
            ),
        }

        # Test keyword analysis
        keyword_analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=KeywordMockLLM().with_structured_output(KeywordOutput),  # Configure for structured output
            categories=categories,
            config_manager=config_manager,
        )

        text = "Machine learning models process data efficiently."
        result = await keyword_analyzer.analyze(text, analysis_types=["keywords"])

        # Only keywords should be analyzed
        assert result.keywords is not None
        assert result.keywords.success
        assert len(result.keywords.keywords) > 0
        assert any(
            "machine learning" in kw.keyword.lower()
            for kw in result.keywords.keywords
        )

        # Test theme analysis
        theme_analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=ThemeMockLLM().with_structured_output(ThemeOutput),  # Configure for structured output
            categories=categories,
            config_manager=config_manager,
        )

        theme_result = await theme_analyzer.analyze(
            text, analysis_types=["themes"]
        )
        assert theme_result.themes is not None
        assert theme_result.themes.success
        assert len(theme_result.themes.themes) > 0
        assert any(
            "learning" in theme.name.lower()
            for theme in theme_result.themes.themes
        )

        # Test category analysis
        category_analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=CategoryMockLLM().with_structured_output(CategoryOutput),  # Configure for structured output
            categories=categories,
            config_manager=config_manager,
        )

        category_result = await category_analyzer.analyze(
            text, analysis_types=["categories"]
        )
        assert category_result.categories is not None
        assert category_result.categories.success
        assert len(category_result.categories.matches) > 0
        assert any(
            cat.name == "Technical"  # Match the category name we defined
            for cat in category_result.categories.matches
        )

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        test_environment_manager,
        keyword_mock: KeywordMockLLM,
        test_parameters: Dict[str, Any],
    ):
        """Test error handling in analysis."""

        # Get components from environment manager
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        config_manager = components["config_manager"]

        # Create parameter file
        sheet_name, param_df = self._create_parameter_df(test_parameters)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params",
        )

        # Create test categories
        categories = {
            "Technical": CategoryConfig(
                description="Technical content",
                keywords=["software", "api", "data"],
                threshold=0.6,
            ),
            "Business": CategoryConfig(
                description="Business content",
                keywords=["revenue", "growth", "market"],
                threshold=0.6,
            ),
        }

        # Test empty input with keyword analysis
        keyword_analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=KeywordMockLLM().with_structured_output(KeywordOutput),
            categories=categories,
            config_manager=config_manager,
        )
        result = await keyword_analyzer.analyze("", analysis_types=["keywords"])
        assert not result.success
        assert result.keywords is not None
        assert not result.keywords.success
        assert result.keywords.error is not None
        assert "Empty input text" in result.keywords.error

        # Test empty input with theme analysis
        theme_analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=ThemeMockLLM().with_structured_output(ThemeOutput),
            categories=categories,
            config_manager=config_manager,
        )
        result = await theme_analyzer.analyze("", analysis_types=["themes"])
        assert not result.success
        assert result.themes is not None
        assert not result.themes.success
        assert result.themes.error is not None
        assert "Empty input text" in result.themes.error

        # Test empty input with category analysis
        category_analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=CategoryMockLLM().with_structured_output(CategoryOutput),
            categories=categories,
            config_manager=config_manager,
        )
        result = await category_analyzer.analyze("", analysis_types=["categories"])
        assert not result.success
        assert result.categories is not None
        assert not result.categories.success
        assert result.categories.error is not None
        assert "Empty input text" in result.categories.error

        # Test invalid analysis type
        result = await keyword_analyzer.analyze("test", analysis_types=["invalid"])
        assert not result.success
        assert "Invalid analysis types" in result.error

    @pytest.mark.asyncio
    async def test_finnish_analysis(
        self,
        test_environment_manager,
        test_parameters: Dict[str, Any],
    ):
        """Test Finnish language analysis."""
        # Get components from environment manager
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        config_manager = components["config_manager"]

        # Update test parameters for Finnish
        fi_params = test_parameters.copy()
        fi_params["general"]["language"] = "fi"

        # Create parameter file
        sheet_name, param_df = self._create_parameter_df(
            fi_params, language="fi"
        )
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params_fi",
        )

        # Create test categories with Finnish translations
        categories = {
            "Technical": CategoryConfig(
                description="Tekninen sisältö",
                keywords=["ohjelmisto", "rajapinta", "data"],
                threshold=0.6,
            ),
            "Business": CategoryConfig(
                description="Liiketoimintasisältö",
                keywords=["liikevaihto", "kasvu", "markkina"],
                threshold=0.6,
            ),
        }

        # Create analyzers
        analyzers = {
            "keywords": SemanticAnalyzer(
                parameter_file=file_path,
                environment_manager=test_environment_manager,
                llm=KeywordMockLLM().with_structured_output(KeywordOutput),
                config_manager=config_manager,
            ),
            "themes": SemanticAnalyzer(
                parameter_file=file_path,
                environment_manager=test_environment_manager,
                llm=ThemeMockLLM().with_structured_output(ThemeOutput),
                config_manager=config_manager,
            ),
            "categories": SemanticAnalyzer(
                parameter_file=file_path,
                environment_manager=test_environment_manager,
                llm=CategoryMockLLM().with_structured_output(CategoryOutput),
                categories=categories,
                config_manager=config_manager,
            ),
        }

        text = """Koneoppimismallit analysoivat dataa tehokkaasti.
                Neuroverkon arkkitehtuuri mahdollistaa monimutkaisen hahmontunnistuksen."""

        # Test each analyzer
        for analysis_type, analyzer in analyzers.items():
            result = await analyzer.analyze(
                text, analysis_types=[analysis_type]
            )
            assert result.success
            if analysis_type == "keywords":
                assert result.keywords is not None
                assert result.keywords.success
                assert any(
                    "koneoppiminen" in kw.keyword.lower()
                    for kw in result.keywords.keywords
                )
            elif analysis_type == "themes":
                assert result.themes is not None
                assert result.themes.success
                assert any(
                    "oppiminen" in theme.name.lower()
                    for theme in result.themes.themes
                )
            elif analysis_type == "categories":
                assert result.categories is not None
                assert result.categories.success
                assert any(
                    cat.name == "Technical"
                    for cat in result.categories.matches
                )

    @pytest.mark.asyncio
    async def test_batch_analysis(
        self,
        test_environment_manager,
        test_parameters: Dict[str, Any],
    ):
        """Test batch analysis functionality with keyword mock."""
        # Get components from environment manager
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        config_manager = components["config_manager"]

        # Create parameter file
        sheet_name, param_df = self._create_parameter_df(test_parameters)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=KeywordMockLLM(),
            config_manager=config_manager,  # Add config_manager
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
        test_environment_manager,
        test_parameters: Dict[str, Any],
    ):
        """Test saving analysis results."""
        # Get components from environment manager
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]
        config_manager = components["config_manager"]

        # Create parameter file
        sheet_name, param_df = self._create_parameter_df(test_parameters)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=file_path,
            environment_manager=test_environment_manager,
            llm=KeywordMockLLM(),
            config_manager=config_manager,  # Add config_manager
        )

        # Create a complete test result with all required components
        result = CompleteAnalysisResult(
            keywords=analyzer._create_error_result_by_type("keywords"),
            themes=analyzer._create_error_result_by_type("themes"),
            categories=analyzer._create_error_result_by_type("categories"),
            language="en",
            success=True,
            processing_time=1.0,
        )

        # Test saving
        output_path = analyzer.save_results(
            results=result, output_file="test_results", output_type="processed"
        )

        assert output_path.exists()
        assert output_path.suffix == ".xlsx"
