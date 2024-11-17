# tests/test_parameter_loading.py

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

# old parameter model
from src.loaders.models import CategoryConfig, GeneralParameters, ParameterSet
from src.loaders.parameter_adapter import ParameterAdapter
from src.loaders.parameter_validator import ParameterValidator
from src.models.parameter_models import CategoryEntry, ExtractionParameters, GeneralParameters, KeywordEntry


def create_test_excel(path: Path, language: str = "en") -> None:
    """Create test parameter Excel file."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # General Parameters
        pd.DataFrame(
            {
                "parameter": ["max_kws", "language", "column_name_to_analyze", "focus_on"],
                "value": [
                    8,
                    language,
                    "text" if language == "en" else "keskustelu",
                    "education and career-related topics",
                ],
                "description": ["Maximum keywords to extract", "Content language", "Content column name", "Focus area"],
            }
        ).to_excel(writer, sheet_name="General Parameters", index=False)

        # Categories
        pd.DataFrame(
            {
                "category": ["education_type"],
                "description": ["Type of education"],
                "keywords": ["online,remote"],
                "threshold": [0.5],
                "parent": ["education"],
            }
        ).to_excel(writer, sheet_name="Categories", index=False)

        # Predefined Keywords
        pd.DataFrame(
            {"keyword": ["programming"], "domain": ["tech"], "importance": [1.0], "compound_parts": [""]}
        ).to_excel(writer, sheet_name="Predefined Keywords", index=False)

        # Excluded Keywords
        pd.DataFrame({"keyword": ["education"], "reason": ["too general"]}).to_excel(
            writer, sheet_name="Excluded Keywords", index=False
        )

        # Analysis Settings
        pd.DataFrame(
            {
                "setting": [
                    "theme_analysis.enabled",
                    "theme_analysis.min_confidence",
                    "weights.statistical",
                    "weights.llm",
                ],
                "value": [True, 0.5, 0.4, 0.6],
                "description": [
                    "Enable theme analysis",
                    "Theme confidence threshold",
                    "Weight for statistical analysis",
                    "Weight for LLM analysis",
                ],
            }
        ).to_excel(writer, sheet_name="Analysis Settings", index=False)


@pytest.fixture
def sample_excel_path(tmp_path) -> Path:
    """Create sample English parameter file."""
    file_path = tmp_path / "parameters.xlsx"
    create_test_excel(file_path, "en")
    return file_path


@pytest.fixture
def sample_finnish_excel_path(tmp_path) -> Path:
    """Create sample Finnish parameter file."""
    file_path = tmp_path / "parametrit.xlsx"
    create_test_excel(file_path, "fi")
    return file_path


# tests/test_parameter_loading.py


def test_parameter_loading_english(sample_excel_path):
    """Test loading English parameters."""
    adapter = ParameterAdapter(sample_excel_path)
    params = adapter.load_and_convert()

    assert params.general.language == "en"
    assert params.general.max_keywords == 8  # Updated to match Excel
    assert params.general.column_name_to_analyze == "text"

    # Check categories
    assert "education_type" in params.categories
    assert params.categories["education_type"].threshold == 0.5


def test_parameter_loading_finnish(sample_finnish_excel_path):
    """Test loading Finnish parameters."""
    adapter = ParameterAdapter(sample_finnish_excel_path)
    params = adapter.load_and_convert()

    assert params.general.language == "fi"  # Fixed expectation
    assert params.general.column_name_to_analyze == "keskustelu"


def test_parameter_validation():
    """Test parameter validation."""
    validator = ParameterValidator()

    # Create valid general parameters
    params = {
        "general": {
            "max_keywords": 15,  # Valid value
            "language": "en",
            "focus_on": "test",
            "min_keyword_length": 3,
            "include_compounds": True,
        }
    }

    # Test validation
    is_valid, warnings, errors = validator.validate(params)
    assert is_valid
    assert len(errors) == 0

    # Test with invalid parameters
    invalid_params = {"general": {"max_keywords": 25, "language": "en", "min_keyword_length": 1}}  # Too high  # Too low

    is_valid, warnings, errors = validator.validate(invalid_params)
    assert not is_valid
    assert len(errors) > 0


def test_backward_compatibility():
    """Test compatibility with old parameter format."""
    # Create old-style parameter file
    old_params = pd.DataFrame({"parameter": ["max_kws", "focus_on", "language"], "value": [5, "test", "en"]})

    # Test conversion
    adapter = ParameterAdapter(file_path=None)
    params_dict = adapter._convert_general_params(old_params)
    params = GeneralParameters(**params_dict)
    assert params.max_keywords == 5


@pytest.fixture
def parameter_files(tmp_path) -> Dict[str, Path]:
    """Create parameter files for testing."""
    # Create English parameters
    en_path = tmp_path / "parameters.xlsx"
    create_test_excel(en_path, "en")

    # Create Finnish parameters
    fi_path = tmp_path / "parametrit.xlsx"
    create_test_excel(fi_path, "fi")

    return {"en": en_path, "fi": fi_path}


def test_parameter_integration(sample_excel_path):
    """Test integration with analyzer."""
    from src.semantic_analyzer.analyzer import SemanticAnalyzer

    analyzer = SemanticAnalyzer(parameter_file=sample_excel_path)

    # Check if parameters were properly loaded
    assert analyzer.language == "en"
    assert analyzer.keyword_analyzer.config["max_keywords"] == 8
    assert "education" in analyzer.keyword_analyzer.config["excluded_keywords"]
