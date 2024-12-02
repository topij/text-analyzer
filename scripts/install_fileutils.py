"""Setup FileUtils in project."""

import os
from pathlib import Path
import shutil
import sys


def setup_fileutils():
    """Set up FileUtils in project."""
    # Create the directory structure if it doesn't exist
    src_utils_path = Path("src/utils/FileUtils")
    src_utils_path.mkdir(parents=True, exist_ok=True)

    # Create compatibility __init__.py
    init_content = '''"""Compatibility layer for existing imports."""
from FileUtils import FileUtils, OutputFileType

import warnings
warnings.warn(
    "Importing from src.utils.FileUtils is deprecated. Use 'from FileUtils import FileUtils' instead.",
    DeprecationWarning
)

__all__ = ['FileUtils', 'OutputFileType']
'''

    with open(src_utils_path / "__init__.py", "w") as f:
        f.write(init_content)

    print("FileUtils compatibility layer installed.")
    print("You can now use either:")
    print("  from src.utils.FileUtils.file_utils import FileUtils  # Old way")
    print("  from FileUtils import FileUtils  # New way")


if __name__ == "__main__":
    setup_fileutils()
