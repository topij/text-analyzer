# tests/unit/test_loaders/test_parameter_handler.py

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.loaders.models import CategoryConfig, GeneralParameters, ParameterSet
from src.loaders.parameter_config import ParameterSheets
from src.loaders.parameter_handler import ParameterHandler
from src.loaders.parameter_validation import ParameterValidation
from FileUtils import FileUtils, OutputFileType

logger = logging.getLogger(__name__)


@pytest.fixture
def file_utils() -> FileUtils:
    """Create FileUtils instance."""
    return FileUtils()


@pytest.fixture
def parameter_handler() -> ParameterHandler:
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
    """Test parameter handling and validation."""

    def _save_parameter_file(
        self,
        file_utils: FileUtils,
        sheet_data: Dict[str, pd.DataFrame],
        file_name: str,
    ) -> Path:
        """Helper to save parameter files using FileUtils."""
        try:
            # Verify input is correct format
            if not all(
                isinstance(df, pd.DataFrame) for df in sheet_data.values()
            ):
                raise ValueError(
                    "All values in sheet_data must be pandas DataFrames"
                )

            saved_files, _ = file_utils.save_data_to_storage(
                data=sheet_data,
                output_filetype=OutputFileType.XLSX,
                output_type="parameters",
                file_name=file_name,
                include_timestamp=False,
                engine="openpyxl",
            )

            return Path(next(iter(saved_files.values())))
        except Exception as e:
            print(f"Error saving parameter file: {e}")
            print(f"Sheet data: {sheet_data}")
            raise

    def test_parameter_loading(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter loading with proper sheet structure."""
        # Get correct sheet name and column names
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["en"]

        # Create DataFrame with proper structure
        data_rows = []
        for internal_name, value in test_parameters["general"].items():
            # Find the Excel parameter name that maps to this internal name
            excel_name = next(
                (
                    excel
                    for excel, internal in param_mappings.items()
                    if internal == internal_name
                ),
                internal_name,  # Fallback to internal name if no mapping found
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

        # Create DataFrame
        param_df = pd.DataFrame(data_rows)

        # Create the dictionary of DataFrames with proper sheet name
        sheet_data = {
            general_sheet_name: param_df
        }  # Now this is Dict[str, pd.DataFrame]

        # Save file
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="test_params",
        )

        # Test parameter loading
        handler = ParameterHandler(file_path)
        params = handler.get_parameters()

        assert isinstance(params, ParameterSet)
        assert params.general.language == test_parameters["general"]["language"]
        assert params.general.focus_on == test_parameters["general"]["focus_on"]
        assert (
            params.general.max_keywords
            == test_parameters["general"]["max_keywords"]
        )

    def test_finnish_parameters(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test Finnish parameter handling."""
        # Get Finnish sheet and column names
        general_sheet_name = ParameterSheets.get_sheet_name("general", "fi")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["fi"]
        column_names = ParameterSheets.get_column_names("general", "fi")

        # Modify for Finnish
        fi_params = test_parameters.copy()
        fi_params["general"]["language"] = "fi"
        fi_params["general"]["focus_on"] = "tekninen sisältö"

        # Create Excel mappings and DataFrame rows
        data_rows = []
        for internal_name, value in fi_params["general"].items():
            # Find the Excel parameter name that maps to this internal name
            excel_name = next(
                (
                    excel
                    for excel, internal in param_mappings.items()
                    if internal == internal_name
                ),
                internal_name,  # Fallback to internal name if no mapping found
            )
            data_rows.append(
                {
                    column_names["parameter"]: excel_name,
                    column_names["value"]: value,
                    column_names["description"]: f"Kuvaus: {excel_name}",
                }
            )

        # Create DataFrame and sheet data
        param_df = pd.DataFrame(data_rows)
        sheet_data = {general_sheet_name: param_df}

        # Save and test
        param_file = self._save_parameter_file(
            file_utils=file_utils, sheet_data=sheet_data, file_name="fi_params"
        )

        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        assert params.general.language == "fi"
        assert params.general.focus_on == "tekninen sisältö"

    def test_parameter_validation(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter validation rules."""
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")

        # Create invalid parameters
        invalid_params = test_parameters.copy()
        invalid_params["general"]["max_keywords"] = 50  # Above maximum
        invalid_params["general"]["min_confidence"] = 2.0  # Above 1.0

        # Create DataFrame rows
        data_rows = []
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["en"]
        for internal_name, value in invalid_params["general"].items():
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

        # Create DataFrame and sheet data
        param_df = pd.DataFrame(data_rows)
        sheet_data = {general_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="invalid_params",
        )

        # Expect ValueError with validation errors
        with pytest.raises(ValueError) as exc_info:
            handler = ParameterHandler(param_file)

        # Verify specific validation error messages
        error_msg = str(exc_info.value)
        assert "max_keywords" in error_msg
        assert "less than or equal to 20" in error_msg
        assert "min_confidence" in error_msg
        assert "less than or equal to 1" in error_msg

    def test_missing_mandatory_fields(self, file_utils: FileUtils) -> None:
        """Test validation of mandatory fields."""
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")

        # Create minimal valid parameter set
        param_data = [
            {
                column_names["parameter"]: "language",
                column_names["value"]: "en",
            },
            {
                column_names["parameter"]: "max_keywords",
                column_names["value"]: 8,
            },
            {
                column_names["parameter"]: "focus_on",
                column_names["value"]: "test content",
            },
            {
                column_names["parameter"]: "column_name_to_analyze",
                column_names["value"]: "text",
            },
        ]

        param_df = pd.DataFrame(param_data)
        sheet_data = {general_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="incomplete_params",
        )

        # Test with incomplete but valid parameters
        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        # Should fill in missing non-mandatory fields with defaults
        assert params.general.language == "en"
        assert params.general.min_keyword_length == 3  # default value
        assert params.general.include_compounds is True  # default value

        # Now test with missing mandatory field
        param_data.pop()  # Remove column_name_to_analyze
        param_df = pd.DataFrame(param_data)
        sheet_data = {general_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="missing_mandatory",
        )

        # Should raise ValueError for missing mandatory field
        with pytest.raises(ValueError) as exc_info:
            handler = ParameterHandler(param_file)

        assert "Missing mandatory parameters" in str(exc_info.value)
        assert "column_name_to_analyze" in str(exc_info.value)

    def test_empty_parameter_file(self, file_utils: FileUtils) -> None:
        """Test handling of empty parameter files."""
        sheet_name = ParameterSheets.get_sheet_name("general", "en")

        # Create DataFrame with required fields but empty values
        column_names = ParameterSheets.get_column_names("general", "en")
        param_data = [
            {
                column_names["parameter"]: "focus_on",
                column_names["value"]: "general content analysis",
                column_names["description"]: "Default focus area",
            },
            {
                column_names["parameter"]: "max_keywords",
                column_names["value"]: 10,
                column_names["description"]: "Default max keywords",
            },
            {
                column_names["parameter"]: "column_name_to_analyze",
                column_names["value"]: "text",
                column_names["description"]: "Default column name",
            },
        ]

        df = pd.DataFrame(param_data)
        sheet_data = {sheet_name: df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="empty_params",
        )

        handler = ParameterHandler(param_file)
        params = handler.get_parameters()

        # Should use defaults while preserving required fields
        assert params.general.language == "en"
        assert params.general.max_keywords == 10
        assert params.general.focus_on == "general content analysis"
        assert params.general.column_name_to_analyze == "text"
        assert params.general.min_keyword_length == 3  # default
        assert params.general.include_compounds is True  # default

    def test_parameter_updates(
        self, file_utils: FileUtils, test_parameters: Dict[str, Any]
    ) -> None:
        """Test parameter update functionality."""
        general_sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ]["en"]

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

        # Create DataFrame and sheet data
        param_df = pd.DataFrame(data_rows)
        sheet_data = {general_sheet_name: param_df}

        param_file = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data=sheet_data,
            file_name="update_params",
        )

        handler = ParameterHandler(param_file)

        # Update parameters
        updated = handler.update_parameters(
            general={"max_keywords": 15, "focus_on": "updated focus"}
        )

        assert updated.general.max_keywords == 15
        assert updated.general.focus_on == "updated focus"
        # Original parameters should be unchanged
        assert test_parameters["general"]["max_keywords"] == 8
