import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.loaders.parameter_handler import ParameterHandler
from src.loaders.models import ParameterSet, GeneralParameters, CategoryConfig
from src.loaders.parameter_validation import ParameterValidation
from src.loaders.parameter_config import ParameterSheets
from src.semantic_analyzer import SemanticAnalyzer
from src.utils.FileUtils.file_utils import FileUtils, OutputFileType
from tests.helpers.mock_llms import KeywordMockLLM

import logging

logging.basicConfig(level=logging.DEBUG)


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

    def _save_parameter_file(
        self,
        file_utils: FileUtils,
        sheet_data: Dict[str, pd.DataFrame],
        file_name: str,
    ) -> Path:
        """Helper to consistently save parameter files."""
        saved_files, _ = file_utils.save_data_to_disk(
            data=sheet_data,
            output_filetype=OutputFileType.XLSX,
            file_name=file_name,
            output_type="parameters",
            include_timestamp=False,
        )
        return Path(next(iter(saved_files.values())))

    def test_parameter_loading(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter loading with strict sheet naming."""
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["en"]
        column_names = ParameterSheets.get_column_names("general", "en")

        # Create DataFrame with proper structure
        df = pd.DataFrame(
            {
                column_names["parameter"]: [],
                column_names["value"]: [],
                column_names["description"]: [],
            }
        )

        for key, value in test_parameters["general"].items():
            for excel_name, internal_name in param_mappings.items():
                if internal_name == key:
                    df.loc[len(df)] = {
                        column_names["parameter"]: excel_name,
                        column_names["value"]: value,
                        column_names[
                            "description"
                        ]: f"Description for {excel_name}",
                    }

        # Save using FileUtils
        sheet_data = {general_sheet_name: df}
        saved_files, _ = file_utils.save_data_to_disk(
            data=sheet_data,
            output_filetype=OutputFileType.XLSX,
            output_type="parameters",
            file_name="test_params",
            include_timestamp=False,
        )

        param_file = Path(next(iter(saved_files.values())))
        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        assert params.general.language == test_parameters["general"]["language"]
        assert params.general.focus_on == test_parameters["general"]["focus_on"]

    def test_language_specific_parameters(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test language-specific parameter handling."""
        general_sheet_name = ParameterSheets.get_sheet_name("general", "fi")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["fi"]
        column_names = ParameterSheets.get_column_names("general", "fi")

        fi_params = test_parameters.copy()
        fi_params["general"]["language"] = "fi"
        fi_params["general"]["focus_on"] = "tekninen sisältö"

        # Create mapping of internal names to Excel names
        excel_mappings = {
            internal: excel for excel, internal in param_mappings.items()
        }

        param_data = []
        for key, value in fi_params["general"].items():
            if key in excel_mappings:
                param_data.append(
                    {
                        column_names["parameter"]: excel_mappings[key],
                        column_names["value"]: value,
                    }
                )

        param_df = pd.DataFrame(param_data)
        sheet_data = {general_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils, sheet_data=sheet_data, file_name="fi_params"
        )

        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        assert params.general.language == "fi"
        assert params.general.focus_on == "tekninen sisältö"

    def test_invalid_parameter_file(self, file_utils: FileUtils) -> None:
        """Test handling of invalid parameter files."""
        wrong_sheet_name = "Sheet1"
        param_df = pd.DataFrame({"parameter": ["language"], "value": ["en"]})
        sheet_data = {wrong_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="invalid_params",
        )

        with pytest.raises(ValueError) as exc_info:
            handler = ParameterHandler(param_file)
            handler.get_parameters()

        expected_sheet = ParameterSheets.get_sheet_name("general", "en")
        assert f"Required sheet '{expected_sheet}' not found" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_analyzer_parameter_integration(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter integration with analyzers."""
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")

        param_data = [
            {
                column_names["parameter"]: key,
                column_names["value"]: value,
            }
            for key, value in test_parameters["general"].items()
        ]

        param_df = pd.DataFrame(param_data)
        sheet_data = {general_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="analyzer_params",
        )

        analyzer = SemanticAnalyzer(
            parameter_file=param_file, llm=KeywordMockLLM()
        )

        test_text = "Machine learning models process data efficiently."
        result = await analyzer.analyze(test_text)

        assert (
            len(result.keywords.keywords)
            <= test_parameters["general"]["max_keywords"]
        )

    def test_empty_parameter_file(self, file_utils: FileUtils) -> None:
        """Test handling of empty parameter files."""
        sheet_name = ParameterSheets.get_sheet_name("general", "en")
        sheet_data = {sheet_name: pd.DataFrame()}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="empty_params",
        )

        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        assert params.general.language == "en"
        assert params.general.focus_on == "general content analysis"
        assert params.general.max_keywords == 10
