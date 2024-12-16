# tests/unit/test_semantic_analyzer.py

"""Test cases for semantic analyzer."""

import pytest
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path

from src.semantic_analyzer import SemanticAnalyzer
from src.loaders.models import CategoryConfig
from src.schemas import CompleteAnalysisResult
from FileUtils import FileUtils
from tests.helpers.mock_llms.category_mock import CategoryMockLLM
from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM
from tests.helpers.mock_llms.theme_mock import ThemeMockLLM


class TestSemanticAnalyzer:
    """Tests for semantic analysis functionality."""

    # In test_semantic_analyzer.py
    @pytest.mark.asyncio
    async def test_complete_analysis(
        self,
        file_utils: FileUtils,
        test_config_manager,
        test_parameters: Dict[str, Any],
    ):
        """Test complete analysis pipeline with all mock types."""
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

        # Create separate mock LLMs
        category_mock = CategoryMockLLM()
        theme_mock = ThemeMockLLM()
        keyword_mock = KeywordMockLLM()

        # Create analyzers with different mocks
        analyzers = {
            "keywords": SemanticAnalyzer(
                parameter_file=file_path,
                file_utils=file_utils,
                llm=keyword_mock,
            ),
            "themes": SemanticAnalyzer(
                parameter_file=file_path,
                file_utils=file_utils,
                llm=theme_mock,
            ),
            "categories": SemanticAnalyzer(
                parameter_file=file_path,
                file_utils=file_utils,
                llm=category_mock,
                categories=categories,  # Pass categories here
            ),
        }

        text = """Machine learning models are trained using large datasets.
                Neural networks enable complex pattern recognition.
                The system's performance metrics show significant improvements."""

        # Run analyses with specific types
        for analysis_type, analyzer in analyzers.items():
            result = await analyzer.analyze(
                text, analysis_types=[analysis_type]
            )
            assert result.success, f"Analysis failed for {analysis_type}"

            if analysis_type == "keywords":
                assert len(result.keywords.keywords) > 0
                assert any(
                    "machine learning" in kw.keyword.lower()
                    for kw in result.keywords.keywords
                )
            elif analysis_type == "themes":
                assert len(result.themes.themes) > 0
                assert any(
                    "learning" in theme.name.lower()
                    for theme in result.themes.themes
                )
            elif analysis_type == "categories":
                assert len(result.categories.matches) > 0
                assert any(
                    cat.name == "Technical" for cat in result.categories.matches
                )

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


# import pytest
# from typing import Any, Dict, List, Optional, Tuple
# from pathlib import Path
# import pandas as pd

# from langchain_core.language_models import BaseChatModel
# from FileUtils import FileUtils, OutputFileType

# from src.config.manager import ConfigManager
# from src.core.language_processing import create_text_processor

# # from src.core.llm.factory import create_llm

# from src.loaders.parameter_handler import ParameterHandler, ParameterSheets
# from src.loaders.models import CategoryConfig
# from src.schemas import CompleteAnalysisResult
# from src.semantic_analyzer import SemanticAnalyzer
# from tests.helpers.mock_llms.category_mock import CategoryMockLLM
# from tests.helpers.mock_llms.keyword_mock import KeywordMockLLM
# from tests.helpers.mock_llms.theme_mock import ThemeMockLLM


# @pytest.fixture
# def test_parameters() -> Dict[str, Any]:
#     """Provide test parameter configurations."""
#     return {
#         "general": {
#             "max_keywords": 8,
#             "min_keyword_length": 3,
#             "language": "en",
#             "focus_on": "technical content",
#             "include_compounds": True,
#             "max_themes": 3,
#             "min_confidence": 0.3,
#             "column_name_to_analyze": "content",
#         },
#         "categories": {
#             "technical": {
#                 "description": "Technical content",
#                 "keywords": ["software", "api", "data"],
#                 "threshold": 0.6,
#             },
#             "business": {
#                 "description": "Business content",
#                 "keywords": ["revenue", "growth", "market"],
#                 "threshold": 0.6,
#             },
#         },
#         "analysis_settings": {
#             "theme_analysis": {
#                 "enabled": True,
#                 "min_confidence": 0.5,
#             },
#             "weights": {
#                 "statistical": 0.4,
#                 "llm": 0.6,
#             },
#         },
#     }


# class TestSemanticAnalyzer:
#     """Tests for semantic analysis functionality."""

#     @pytest.fixture
#     def keyword_mock(self):
#         """Create KeywordMockLLM instance."""
#         return KeywordMockLLM()

#     @pytest.fixture
#     def theme_mock(self):
#         """Create ThemeMockLLM instance."""
#         return ThemeMockLLM()

#     @pytest.fixture
#     def category_mock(self):
#         """Create CategoryMockLLM instance."""
#         return CategoryMockLLM()

#     def _create_parameter_df(
#         self, test_parameters: Dict[str, Any], language: str = "en"
#     ) -> Tuple[str, pd.DataFrame]:
#         """Helper method to create parameter DataFrame with proper mappings."""
#         # Get proper names and mappings
#         general_sheet_name = ParameterSheets.get_sheet_name("general", language)
#         column_names = ParameterSheets.get_column_names("general", language)
#         param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
#             "parameters"
#         ][language]

#         # Create DataFrame rows
#         data_rows = []
#         for internal_name, value in test_parameters["general"].items():
#             excel_name = next(
#                 (
#                     excel
#                     for excel, internal in param_mappings.items()
#                     if internal == internal_name
#                 ),
#                 internal_name,
#             )
#             data_rows.append(
#                 {
#                     column_names["parameter"]: excel_name,
#                     column_names["value"]: value,
#                     column_names[
#                         "description"
#                     ]: f"Description for {excel_name}",
#                 }
#             )

#         # Create and return sheet name and DataFrame
#         return general_sheet_name, pd.DataFrame(data_rows)

#     def _save_parameter_file(
#         self,
#         file_utils: FileUtils,
#         sheet_data: Dict[str, pd.DataFrame],
#         file_name: str,
#     ) -> Path:
#         """Helper to save parameter files using FileUtils."""

#         try:
#             # Save data
#             saved_files, _ = file_utils.save_data_to_storage(
#                 data=sheet_data,
#                 output_filetype=OutputFileType.XLSX,
#                 output_type="parameters",
#                 file_name=file_name,
#                 include_timestamp=False,
#                 engine="openpyxl",  # Add this explicitly
#             )

#             return Path(next(iter(saved_files.values())))

#         except Exception as e:
#             print(f"Error saving parameter file: {e}")
#             print(f"Sheet data keys: {list(sheet_data.keys())}")
#             raise

#     @pytest.fixture
#     def mock_analyzer(
#         self,
#         file_utils: FileUtils,
#         config_manager: ConfigManager,  # Updated fixture name and type
#         mock_llm: BaseChatModel,
#     ):
#         return SemanticAnalyzer(file_utils=file_utils, llm=mock_llm)

#     @pytest.mark.asyncio
#     async def test_specific_analysis_types(
#         self,
#         file_utils: FileUtils,
#         keyword_mock: KeywordMockLLM,
#         test_parameters: Dict[str, Any],
#     ):
#         """Test keyword analysis type specifically."""
#         # file_path = self._save_parameter_file(
#         #     file_utils=file_utils,
#         #     sheet_data={"General Parameters": test_parameters["general"]},
#         #     file_name="test_params",
#         # )

#         # Create parameter file
#         sheet_name, param_df = self._create_parameter_df(test_parameters)
#         file_path = self._save_parameter_file(
#             file_utils=file_utils,
#             sheet_data={sheet_name: param_df},
#             file_name="test_params",
#         )

#         analyzer = SemanticAnalyzer(
#             parameter_file=file_path,
#             file_utils=file_utils,
#             llm=keyword_mock,  # Use keyword mock for keyword analysis
#         )

#         text = "Machine learning models process data efficiently."
#         result = await analyzer.analyze(text, analysis_types=["keywords"])

#         # Only keywords should be analyzed
#         assert result.keywords.success
#         assert len(result.keywords.keywords) > 0
#         assert not result.themes.themes
#         assert not result.categories.matches

#     @pytest.mark.asyncio
#     async def test_error_handling(
#         self,
#         file_utils: FileUtils,
#         keyword_mock: KeywordMockLLM,
#         test_parameters: Dict[str, Any],
#     ):
#         """Test error handling in analysis."""

#         # Create parameter file
#         sheet_name, param_df = self._create_parameter_df(test_parameters)
#         file_path = self._save_parameter_file(
#             file_utils=file_utils,
#             sheet_data={sheet_name: param_df},
#             file_name="test_params",
#         )

#         analyzer = SemanticAnalyzer(
#             parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
#         )

#         # Test with empty input
#         result = await analyzer.analyze("")
#         assert not result.success
#         assert result.keywords.error is not None
#         assert "Empty input text" in result.keywords.error
#         assert "Empty input text" in result.themes.error
#         assert "Empty input text" in result.categories.error

#         # Test with invalid analysis type - check result instead of expecting exception
#         result = await analyzer.analyze("test", analysis_types=["invalid"])
#         assert not result.success
#         assert "Invalid analysis types" in result.error

#     # tests/unit/test_semantic_analyzer.py

#     @pytest.mark.asyncio
#     async def test_complete_analysis(
#         self,
#         file_utils: FileUtils,
#         config_manager: ConfigManager,
#         test_parameters: Dict[str, Any],
#     ):
#         """Test complete analysis pipeline with all mock types."""
#         # Create parameter file
#         sheet_name, param_df = self._create_parameter_df(test_parameters)
#         file_path = self._save_parameter_file(
#             file_utils=file_utils,
#             sheet_data={sheet_name: param_df},
#             file_name="test_params",
#         )

#         # Create test categories
#         categories = {
#             "Technical": CategoryConfig(
#                 description="Technical content",
#                 keywords=["software", "api", "data"],
#                 threshold=0.6,
#             ),
#             "Business": CategoryConfig(
#                 description="Business content",
#                 keywords=["revenue", "growth", "market"],
#                 threshold=0.6,
#             ),
#         }

#         # Create analyzers with different mocks
#         keyword_analyzer = SemanticAnalyzer(
#             parameter_file=file_path,
#             file_utils=file_utils,
#             llm=KeywordMockLLM(),
#         )

#         theme_analyzer = SemanticAnalyzer(
#             parameter_file=file_path, file_utils=file_utils, llm=ThemeMockLLM()
#         )

#         category_analyzer = SemanticAnalyzer(
#             parameter_file=file_path,
#             file_utils=file_utils,
#             llm=CategoryMockLLM(),
#             categories=categories,  # Pass test categories
#         )

#         text = """Machine learning models are trained using large datasets.
#                 Neural networks enable complex pattern recognition.
#                 The system's performance metrics show significant improvements."""

#         # Run analyses with specific types
#         keyword_result = await keyword_analyzer.analyze(
#             text, analysis_types=["keywords"]
#         )
#         assert keyword_result.keywords.success
#         assert len(keyword_result.keywords.keywords) > 0
#         assert any(
#             "machine learning" in kw.keyword.lower()
#             for kw in keyword_result.keywords.keywords
#         )

#         theme_result = await theme_analyzer.analyze(
#             text, analysis_types=["themes"]
#         )
#         assert theme_result.themes.success
#         assert len(theme_result.themes.themes) > 0
#         assert any(
#             "learning" in theme.name.lower()
#             for theme in theme_result.themes.themes
#         )

#         category_result = await category_analyzer.analyze(
#             text, analysis_types=["categories"]
#         )
#         assert category_result.categories.success
#         assert len(category_result.categories.matches) > 0
#         assert any(
#             cat.name == "Machine Learning"
#             for cat in category_result.categories.matches
#         )

#     @pytest.mark.asyncio
#     async def test_finnish_analysis(
#         self,
#         file_utils: FileUtils,
#         config_manager: ConfigManager,
#         test_parameters: Dict[str, Any],
#     ):
#         """Test Finnish language analysis."""
#         # Update test parameters for Finnish
#         fi_params = test_parameters.copy()
#         fi_params["general"]["language"] = "fi"

#         # Create parameter file
#         sheet_name, param_df = self._create_parameter_df(
#             fi_params, language="fi"
#         )
#         file_path = self._save_parameter_file(
#             file_utils=file_utils,
#             sheet_data={sheet_name: param_df},
#             file_name="test_params_fi",
#         )

#         # Create test categories with Finnish translations
#         categories = {
#             "Technical": CategoryConfig(
#                 description="Tekninen sisältö",
#                 keywords=["ohjelmisto", "rajapinta", "data"],
#                 threshold=0.6,
#             ),
#             "Business": CategoryConfig(
#                 description="Liiketoimintasisältö",
#                 keywords=["liikevaihto", "kasvu", "markkina"],
#                 threshold=0.6,
#             ),
#         }

#         # Create analyzers
#         analyzers = {
#             "keywords": SemanticAnalyzer(
#                 parameter_file=file_path,
#                 file_utils=file_utils,
#                 llm=KeywordMockLLM(),
#             ),
#             "themes": SemanticAnalyzer(
#                 parameter_file=file_path,
#                 file_utils=file_utils,
#                 llm=ThemeMockLLM(),
#             ),
#             "categories": SemanticAnalyzer(
#                 parameter_file=file_path,
#                 file_utils=file_utils,
#                 llm=CategoryMockLLM(),
#                 categories=categories,
#             ),
#         }

#         text = """Koneoppimismallit analysoivat dataa tehokkaasti.
#                  Neuroverkon arkkitehtuuri mahdollistaa monimutkaisen hahmontunnistuksen."""

#         # Test each analyzer
#         for analysis_type, analyzer in analyzers.items():
#             result = await analyzer.analyze(
#                 text, analysis_types=[analysis_type]
#             )
#             assert result.success
#             if analysis_type == "keywords":
#                 assert any(
#                     "koneoppimis" in kw.keyword.lower()
#                     for kw in result.keywords.keywords
#                 )
#             elif analysis_type == "themes":
#                 assert any(
#                     "oppiminen" in theme.name.lower()
#                     for theme in result.themes.themes
#                 )
#             elif analysis_type == "categories":
#                 print(f"analysis_type: {analysis_type}")
#                 print(f"result.categories.matches {result.categories.matches}")
#                 assert any(
#                     cat.name == "Machine Learning"
#                     for cat in result.categories.matches
#                 )

#     @pytest.mark.asyncio
#     async def test_batch_analysis(
#         self,
#         file_utils: FileUtils,
#         keyword_mock: KeywordMockLLM,
#         test_parameters: Dict[str, Any],
#     ):
#         """Test batch analysis functionality with keyword mock."""

#         # Create parameter file
#         sheet_name, param_df = self._create_parameter_df(test_parameters)
#         file_path = self._save_parameter_file(
#             file_utils=file_utils,
#             sheet_data={sheet_name: param_df},
#             file_name="test_params",
#         )
#         analyzer = SemanticAnalyzer(
#             parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
#         )

#         texts = [
#             "Machine learning models analyze data.",
#             "Revenue growth exceeded expectations.",
#             "System performance metrics improved.",
#         ]

#         results = await analyzer.analyze_batch(texts, batch_size=2)
#         assert len(results) == len(texts)
#         assert all(isinstance(r, CompleteAnalysisResult) for r in results)

#     def test_result_saving(
#         self,
#         file_utils: FileUtils,
#         keyword_mock: KeywordMockLLM,
#         test_parameters: Dict[str, Any],
#     ):
#         """Test saving analysis results."""
#         # Get proper sheet name from ParameterSheets
#         general_sheet_name = ParameterSheets.get_sheet_name("general", "en")

#         # Create DataFrame with proper structure
#         column_names = ParameterSheets.get_column_names("general", "en")
#         param_df = pd.DataFrame(
#             [
#                 {
#                     column_names["parameter"]: "max_keywords",
#                     column_names["value"]: test_parameters["general"][
#                         "max_keywords"
#                     ],
#                     column_names["description"]: "Maximum keywords to extract",
#                 },
#                 {
#                     column_names["parameter"]: "language",
#                     column_names["value"]: "en",
#                     column_names["description"]: "Content language",
#                 },
#                 {
#                     column_names["parameter"]: "focus_on",
#                     column_names["value"]: test_parameters["general"][
#                         "focus_on"
#                     ],
#                     column_names["description"]: "Analysis focus",
#                 },
#                 {
#                     column_names["parameter"]: "column_name_to_analyze",
#                     column_names["value"]: "content",
#                     column_names["description"]: "Content column name",
#                 },
#             ]
#         )

#         # Save parameter file with correct sheet name
#         file_path = self._save_parameter_file(
#             file_utils=file_utils,
#             sheet_data={general_sheet_name: param_df},
#             file_name="test_params",
#         )

#         # Create analyzer with parameters
#         analyzer = SemanticAnalyzer(
#             parameter_file=file_path, file_utils=file_utils, llm=keyword_mock
#         )

#         # Create a complete test result with all required components
#         result = CompleteAnalysisResult(
#             keywords=analyzer._create_error_result_by_type("keywords"),
#             themes=analyzer._create_error_result_by_type("themes"),
#             categories=analyzer._create_error_result_by_type("categories"),
#             language="en",
#             success=True,
#             processing_time=1.0,
#         )

#         # Test saving
#         output_path = analyzer.save_results(
#             results=result, output_file="test_results", output_type="processed"
#         )

#         assert output_path.exists()
#         assert output_path.suffix == ".xlsx"

#     def _validate_analysis_result(self, result: CompleteAnalysisResult) -> None:
#         """Validate complete analysis result structure."""
#         assert result.success, f"Analysis failed: {result.error}"
#         assert result.processing_time > 0

#         # Validate keywords
#         assert result.keywords.success
#         assert len(result.keywords.keywords) > 0
#         for kw in result.keywords.keywords:
#             assert kw.keyword
#             assert 0 <= kw.score <= 1.0

#         # Validate themes
#         assert result.themes.success
#         assert len(result.themes.themes) > 0
#         for theme in result.themes.themes:
#             assert theme.name
#             assert theme.description
#             assert 0 <= theme.confidence <= 1.0

#         # Validate categories
#         assert result.categories.success
#         assert len(result.categories.matches) > 0
#         for cat in result.categories.matches:
#             assert cat.name
#             assert 0 <= cat.confidence <= 1.0
#             assert cat.evidence
