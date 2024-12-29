# tests/unit/test_loaders/test_parameter_handler.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator
import tempfile
from contextlib import contextmanager

import pandas as pd
import pytest
from src.core.managers.environment_manager import EnvironmentManager
from src.config.manager import ConfigManager
from src.core.managers import EnvironmentConfig
from src.loaders.models import CategoryConfig, GeneralParameters, ParameterSet
from src.loaders.parameter_config import ParameterSheets
from src.loaders.parameter_handler import ParameterHandler
from src.loaders.parameter_validation import ParameterValidation
from FileUtils import FileUtils, OutputFileType

logger = logging.getLogger(__name__)

TEST_DIRECTORY_STRUCTURE = {
    "data": ["raw", "processed", "config", "parameters"],
    "logs": [],
    "reports": [],
    "models": [],
}


@pytest.fixture
def test_root(tmp_path_factory) -> Path:
    """Create a test root directory."""
    root = tmp_path_factory.mktemp("test_data")
    # Create data directory structure
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["raw", "processed", "config", "parameters"]:
        subdir_path = data_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {subdir_path}")
    return root


@pytest.fixture
def file_utils(test_root) -> FileUtils:
    """Create FileUtils instance."""
    logger.debug(f"Initializing FileUtils with project root: {test_root}")
    return FileUtils(
        project_root=test_root,
        directory_structure=TEST_DIRECTORY_STRUCTURE,
        create_directories=True,
    )


@pytest.fixture
def test_environment_manager(test_root, file_utils: FileUtils) -> Generator[EnvironmentManager, None, None]:
    """Create EnvironmentManager instance."""
    logger.debug(f"Setting up test environment in: {test_root}")
    
    # Create test config.yaml FIRST
    config_content = """
environment: test
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  disable_existing_loggers: false

models:
  default_provider: mock
  default_model: mock-model
  parameters:
    temperature: 0.7
    max_tokens: 1000
  providers:
    mock:
      api_key: mock-key
      models:
        - mock-model
        - mock-model-large

languages:
  default_language: en
  languages:
    en:
      name: English
      code: en
    fi:
      name: Finnish
      code: fi

features:
  use_cache: true
  batch_processing: true
"""
    # Create config file in data/config directory
    config_file = test_root / "data" / "config" / "config.yaml"
    config_file.write_text(config_content)
    logger.debug(f"Created config file at: {config_file}")
    
    # Verify config file exists
    if not config_file.exists():
        logger.error(f"Failed to create config file at: {config_file}")
        raise RuntimeError(f"Failed to create config file at: {config_file}")
    else:
        logger.debug(f"Config file exists with size: {config_file.stat().st_size} bytes")
    
    # THEN initialize config manager with file_utils and the temp directory
    config_manager = ConfigManager(
        config_dir="config",  # This will be relative to data directory
        file_utils=file_utils,
        project_root=test_root
    )
    
    # Create test parameter file
    param_content = """
general:
  max_keywords: 10
  min_keyword_length: 3
  language: en
  focus_on: general content analysis
  include_compounds: true
  max_themes: 3
  min_confidence: 0.3
  column_name_to_analyze: text

categories:
  technical:
    description: Technical content
    keywords: [programming, software, algorithm]
    threshold: 0.6
  business:
    description: Business content
    keywords: [strategy, market, revenue]
    threshold: 0.5

predefined_keywords:
  technical: [python, java, cloud]
  business: [sales, marketing, finance]

excluded_keywords:
  - the
  - and
  - or

analysis_settings:
  theme_analysis:
    enabled: true
    min_confidence: 0.5
  weights:
    statistical: 0.4
    llm: 0.6

domain_context:
  technical:
    focus: technology and software development
  business:
    focus: business strategy and operations
"""
    param_file = test_root / "data" / "parameters" / "parameters.yaml"
    param_file.write_text(param_content)
    logger.debug(f"Created parameter file at: {param_file}")
    
    # Initialize environment manager with test components
    config = EnvironmentConfig(
        project_root=test_root,
        log_level="DEBUG",
        config_dir="config",  # This will be relative to data directory
    )
    
    # Reset the singleton state to ensure clean initialization
    EnvironmentManager._instance = None
    EnvironmentManager._initialized = False
    
    # Create new instance with config
    env_manager = EnvironmentManager(config=config)
    
    # Set up test components
    env_manager.file_utils = file_utils
    env_manager.config_manager = config_manager
    env_manager._components = {
        "file_utils": file_utils,
        "config_manager": config_manager,
    }
    env_manager._initialized = True
    
    yield env_manager


@pytest.fixture
def parameter_handler(test_environment_manager) -> ParameterHandler:
    """Create ParameterHandler with validator."""
    components = test_environment_manager.get_components()
    file_utils = components["file_utils"]
    handler = ParameterHandler(file_utils=file_utils)
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
        self, test_environment_manager, test_parameters
    ) -> None:
        """Test loading parameters from Excel file."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]

        # Get sheet name and column names
        sheet_name = ParameterSheets.get_sheet_name("general", "en")
        column_names = ParameterSheets.get_column_names("general", "en")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"]["parameters"]["en"]

        # Create DataFrame rows
        data_rows = []
        for internal_name, value in test_parameters["general"].items():
            excel_name = next(
                (excel for excel, internal in param_mappings.items() if internal == internal_name),
                internal_name
            )
            data_rows.append({
                column_names["parameter"]: excel_name,
                column_names["value"]: value,
                column_names["description"]: f"Description for {excel_name}",
            })

        # Create DataFrame and sheet data
        param_df = pd.DataFrame(data_rows)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params",
        )

        # Test loading
        handler = ParameterHandler(file_path=file_path, file_utils=file_utils)
        params = handler.get_parameters()

        # Verify loaded parameters
        assert isinstance(params, ParameterSet)
        assert params.general.max_keywords == test_parameters["general"]["max_keywords"]
        assert params.general.focus_on == test_parameters["general"]["focus_on"]
        assert params.general.column_name_to_analyze == test_parameters["general"]["column_name_to_analyze"]

    def test_finnish_parameters(
        self, test_environment_manager, test_parameters
    ) -> None:
        """Test loading Finnish language parameters."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]

        # Update test parameters for Finnish
        fi_params = test_parameters.copy()
        fi_params["general"]["language"] = "fi"

        # Get sheet name and column names for Finnish
        sheet_name = ParameterSheets.get_sheet_name("general", "fi")
        column_names = ParameterSheets.get_column_names("general", "fi")
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"]["parameters"]["fi"]

        # Create DataFrame rows
        data_rows = []
        for internal_name, value in fi_params["general"].items():
            excel_name = next(
                (excel for excel, internal in param_mappings.items() if internal == internal_name),
                internal_name
            )
            data_rows.append({
                column_names["parameter"]: excel_name,
                column_names["value"]: value,
                column_names["description"]: f"Kuvaus: {excel_name}",
            })

        # Create DataFrame and sheet data
        param_df = pd.DataFrame(data_rows)
        file_path = self._save_parameter_file(
            file_utils=file_utils,
            sheet_data={sheet_name: param_df},
            file_name="test_params_fi",
        )

        # Test loading
        handler = ParameterHandler(file_path=file_path, file_utils=file_utils)
        params = handler.get_parameters()

        # Verify Finnish settings
        assert params.general.language == "fi"
        assert params.general.max_keywords == fi_params["general"]["max_keywords"]
        assert params.general.focus_on == fi_params["general"]["focus_on"]
        assert params.general.column_name_to_analyze == fi_params["general"]["column_name_to_analyze"]

    def test_parameter_validation(
        self, test_environment_manager, test_parameters
    ) -> None:
        """Test parameter validation rules."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]

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
            handler = ParameterHandler(file_path=param_file, file_utils=file_utils)
            handler.get_parameters()

        # Verify specific validation error messages
        error_msg = str(exc_info.value)
        assert "max_keywords" in error_msg
        assert "less than or equal to 20" in error_msg
        assert "min_confidence" in error_msg
        assert "less than or equal to 1" in error_msg

    def test_missing_mandatory_fields(
        self, test_environment_manager
    ) -> None:
        """Test validation of mandatory fields."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]

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
        handler = ParameterHandler(file_path=param_file, file_utils=file_utils)
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
            handler = ParameterHandler(file_path=param_file, file_utils=file_utils)
            handler.get_parameters()

        assert "Missing mandatory parameters" in str(exc_info.value)
        assert "column_name_to_analyze" in str(exc_info.value)

    def test_empty_parameter_file(
        self, test_environment_manager
    ) -> None:
        """Test handling of empty parameter files."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]

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

        handler = ParameterHandler(file_path=param_file, file_utils=file_utils)
        params = handler.get_parameters()

        # Should use defaults while preserving required fields
        assert params.general.language == "en"
        assert params.general.max_keywords == 10
        assert params.general.focus_on == "general content analysis"
        assert params.general.column_name_to_analyze == "text"
        assert params.general.min_keyword_length == 3  # default
        assert params.general.include_compounds is True  # default

    def test_parameter_updates(
        self, test_environment_manager, test_parameters
    ) -> None:
        """Test parameter update functionality."""
        components = test_environment_manager.get_components()
        file_utils = components["file_utils"]

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

        handler = ParameterHandler(file_path=param_file, file_utils=file_utils)
        params = handler.get_parameters()

        # Update parameters
        updated = handler.update_parameters(
            general={"max_keywords": 15, "focus_on": "updated focus"}
        )

        assert updated.general.max_keywords == 15
        assert updated.general.focus_on == "updated focus"
        # Original parameters should be unchanged
        assert test_parameters["general"]["max_keywords"] == 8
