# src/excel_analysis/factory.py

from pathlib import Path
from typing import Optional, Type, Union

from langchain_core.language_models import BaseChatModel

from src.analyzers.excel_support import (
    ExcelKeywordAnalyzer,
    ExcelThemeAnalyzer,
    ExcelCategoryAnalyzer,
)
from src.core.llm.factory import create_llm
from FileUtils import FileUtils


class ExcelAnalyzerFactory:
    """Factory for creating Excel-aware analyzers."""

    ANALYZER_TYPES = {
        "keywords": ExcelKeywordAnalyzer,
        "themes": ExcelThemeAnalyzer,
        "categories": ExcelCategoryAnalyzer,
    }

    @classmethod
    def create_analyzer(
        cls,
        analyzer_type: str,
        content_file: Union[str, Path],
        parameter_file: Union[str, Path],
        llm: Optional[BaseChatModel] = None,
        file_utils: Optional[FileUtils] = None,
        **kwargs,
    ) -> Union[ExcelKeywordAnalyzer, ExcelThemeAnalyzer, ExcelCategoryAnalyzer]:
        """Create Excel-aware analyzer instance.

        Args:
            analyzer_type: Type of analyzer to create
            content_file: Path to content Excel file
            parameter_file: Path to parameter Excel file
            llm: Optional LLM instance
            file_utils: Optional FileUtils instance
            **kwargs: Additional configuration options

        Returns:
            Configured analyzer instance

        Raises:
            ValueError: If analyzer type is invalid
        """
        if analyzer_type not in cls.ANALYZER_TYPES:
            raise ValueError(
                f"Invalid analyzer type: {analyzer_type}. "
                f"Must be one of: {list(cls.ANALYZER_TYPES.keys())}"
            )

        # Get analyzer class
        analyzer_class = cls.ANALYZER_TYPES[analyzer_type]

        # Create LLM if not provided
        if llm is None:
            llm = create_llm()

        # Create analyzer
        return analyzer_class(
            content_file=content_file,
            parameter_file=parameter_file,
            llm=llm,
            file_utils=file_utils,
            **kwargs,
        )
