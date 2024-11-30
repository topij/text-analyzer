# tests/test_basic.py

from pathlib import Path

import pytest

from src.utils.FileUtils.file_utils import FileUtils
from tests.helpers.mock_llm import MockLLM


def test_project_setup(project_path: Path) -> None:
    """Test basic project setup."""
    assert isinstance(project_path, Path)
    assert project_path.exists()
    assert (project_path / "src").exists()
    assert (project_path / "tests").exists()


def test_file_utils(file_utils: FileUtils) -> None:
    """Test FileUtils functionality."""
    assert isinstance(file_utils, FileUtils)
    assert file_utils.project_root.exists()


def test_test_content(test_content: dict) -> None:
    """Test content fixture."""
    assert test_content is not None
    assert "en" in test_content
    assert "technical" in test_content["en"]
    assert "business" in test_content["en"]
    assert isinstance(test_content["en"]["technical"], str)


@pytest.fixture(scope="session")
def llm():
    """Provide mock LLM instance."""
    return MockLLM()
