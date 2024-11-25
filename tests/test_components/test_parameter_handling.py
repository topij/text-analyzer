import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.loaders.parameter_handler import ParameterHandler
from src.loaders.models import ParameterSet, GeneralParameters, CategoryConfig
from src.loaders.parameter_validation import ParameterValidation
from src.loaders.parameter_config import ParameterSheets
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.FileUtils.file_utils import FileUtils
from tests.helpers.mock_llms import KeywordMockLLM


@pytest.fixture
def parameter_handler():
    """Create ParameterHandler with validator."""
    handler = ParameterHandler()
    handler.validator = ParameterValidation()
    return handler


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


class TestParameterHandling:
    """Test parameter handling and integration."""

    def test_parameter_loading(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter loading and validation."""
        # Get correct sheet name and mappings for English
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["en"]
        column_names = ParameterSheets.get_column_names("general", "en")

        # Create parameter DataFrame with the correct format
        # Need to find the Excel names that map to our internal parameter names
        param_data = []
        for key, value in test_parameters["general"].items():
            # Find the Excel name that maps to our internal name
            excel_name = None
            for excel_param, internal_param in param_mappings.items():
                if internal_param == key:
                    excel_name = excel_param
                    break

            if excel_name:
                param_data.append(
                    {
                        column_names[
                            "parameter"
                        ]: excel_name,  # Use Excel parameter name
                        column_names["value"]: value,
                    }
                )

        param_df = pd.DataFrame(param_data)

        print("\nParameter Mappings (Excel -> Internal):")
        for excel_key, internal_key in param_mappings.items():
            print(f"{excel_key} -> {internal_key}")

        print("\nFinal DataFrame (Using Excel Names):")
        print(param_df)

        # Save test parameters to Excel
        result = file_utils.save_data_to_disk(
            data={general_sheet_name: param_df},
            output_filetype="xlsx",
            file_name="test_params",
            output_type="parameters",
            include_timestamp=False,
        )

        # Get the actual file path from result and load params
        param_file = Path(list(result[0].values())[0])
        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        # Debug info
        print("\nLoaded Parameters:")
        print(f"Language: {params.general.language}")
        print(f"Focus On: {params.general.focus_on}")
        print(f"Max Keywords: {params.general.max_keywords}")

        # Verify values
        assert params.general.language == test_parameters["general"]["language"]
        assert params.general.focus_on == test_parameters["general"]["focus_on"]
        assert (
            params.general.max_keywords
            == test_parameters["general"]["max_keywords"]
        )

    def test_language_specific_parameters(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test language-specific parameter handling."""
        # Get correct sheet name and mappings for Finnish
        general_sheet_name = ParameterSheets.get_sheet_name("general", "fi")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["fi"]
        column_names = ParameterSheets.get_column_names("general", "fi")

        # Create Finnish parameters
        fi_params = test_parameters.copy()
        fi_params["general"]["language"] = "fi"
        fi_params["general"]["focus_on"] = "tekninen sisältö"

        # Create parameter DataFrame with mapped names
        param_df = pd.DataFrame(
            [
                {
                    column_names["parameter"]: param_mappings[
                        key
                    ],  # Map parameter names
                    column_names["value"]: value,
                }
                for key, value in fi_params["general"].items()
                if key in param_mappings  # Only include mapped parameters
            ]
        )

        # Save Finnish parameters to Excel
        result = file_utils.save_data_to_disk(
            data={general_sheet_name: param_df},
            output_filetype="xlsx",
            file_name="fi_params",
            output_type="parameters",
            include_timestamp=False,
        )

        # Get the actual file path
        param_file = Path(list(result[0].values())[0])

        # Load Finnish parameters
        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        # Verify values
        assert params.general.language == "fi"
        assert params.general.focus_on == "tekninen sisältö"
        assert params.general.max_keywords == 8

    def test_parameter_validation(
        self,
        parameter_handler: ParameterHandler,
        test_parameters: Dict[str, Any],
    ) -> None:
        """Test parameter validation rules."""
        # Test valid parameters
        is_valid, warnings, errors = (
            parameter_handler.validator.validate_parameters(test_parameters)
        )
        assert is_valid, f"Valid parameters failed validation: {errors}"

        # Test invalid max_keywords
        invalid_params = test_parameters.copy()
        invalid_params["general"]["max_keywords"] = -1

        is_valid, _, errors = parameter_handler.validator.validate_parameters(
            invalid_params
        )
        assert not is_valid
        assert any("max_keywords" in error for error in errors)

    @pytest.mark.asyncio
    async def test_analyzer_parameter_integration(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter integration with analyzers."""
        # Get correct sheet name
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")

        # Create parameter DataFrame
        param_df = pd.DataFrame(
            [
                {
                    column_names["parameter"]: key,
                    column_names["value"]: value,
                }
                for key, value in test_parameters["general"].items()
            ]
        )

        # Save test parameters
        result = file_utils.save_data_to_disk(
            data={general_sheet_name: param_df},
            output_filetype="xlsx",
            file_name="analyzer_params",
            output_type="parameters",
            include_timestamp=False,
        )

        # Get the actual file path
        param_file = Path(list(result[0].values())[0])

        # Create analyzer with parameters and mock LLM
        analyzer = SemanticAnalyzer(
            parameter_file=param_file, llm=KeywordMockLLM()
        )

        # Test text for analysis
        test_text = "Machine learning models process data efficiently."

        # Verify parameters affect analysis
        result = await analyzer.analyze(test_text)

        # Check keyword limit
        assert (
            len(result.keywords.keywords)
            <= test_parameters["general"]["max_keywords"]
        )
