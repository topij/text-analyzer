# src/excel_analysis/formatters.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from src.utils.output_formatter import (
    BaseFormatter,
    ExcelFormatter,
    ExcelOutputConfig,
    OutputDetail,
)
from FileUtils import FileUtils, OutputFileType

logger = logging.getLogger(__name__)


class ExcelAnalysisFormatter(ExcelFormatter):
    """Enhanced Excel formatter with proper detail level support."""

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config: Optional[ExcelOutputConfig] = None,
        use_tqdm: bool = True,
    ):
        """Initialize formatter with optional progress bars."""
        super().__init__(file_utils, config)
        self.use_tqdm = use_tqdm

    def format_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        original_content: pd.DataFrame,
        content_column: str,
    ) -> pd.DataFrame:
        """Format complete analysis results into a DataFrame.

        Args:
            analysis_results: Results from different analyzers
            original_content: Original content DataFrame
            content_column: Name of content column

        Returns:
            DataFrame with formatted results
        """
        # Start with original content
        results_df = original_content.copy()

        # Format each analysis type
        for analysis_type, result in analysis_results.items():
            if result is None or (
                isinstance(result, pd.DataFrame) and result.empty
            ):
                logger.warning(f"No results for {analysis_type}")
                continue

            try:
                # Extract relevant data from the result
                formatted_data = self._format_analysis_type(
                    analysis_type, result
                )

                # Add formatted columns to results DataFrame
                for col_name, values in formatted_data.items():
                    final_col_name = (
                        f"{analysis_type}_{col_name}"
                        if not col_name.startswith(analysis_type)
                        else col_name
                    )
                    results_df[final_col_name] = values

            except Exception as e:
                logger.error(f"Error formatting {analysis_type} results: {e}")
                continue

        return results_df

    def _format_analysis_type(
        self, analysis_type: str, result: Any
    ) -> Dict[str, Any]:
        """Format specific analysis type results with detail level consideration."""
        if isinstance(result, pd.DataFrame):
            # Apply detail level filtering to DataFrame columns
            return self._filter_columns_by_detail(result)

        # Handle other result types based on analysis type and detail level
        if analysis_type == "keywords":
            return self._format_keyword_results(result)
        elif analysis_type == "themes":
            return self._format_theme_results(result)
        elif analysis_type == "categories":
            return self._format_category_results(result)
        else:
            logger.warning(f"Unknown analysis type: {analysis_type}")
            return {}

    def _filter_columns_by_detail(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Filter DataFrame columns based on detail level."""
        if self.config.detail_level == OutputDetail.MINIMAL:
            # Keep only essential columns
            essential_cols = [
                col
                for col in df.columns
                if any(
                    term in col.lower()
                    for term in ["id", "name", "score", "confidence"]
                )
            ]
            return {col: df[col] for col in essential_cols}

        elif self.config.detail_level == OutputDetail.SUMMARY:
            # Exclude detailed analysis columns
            exclude_terms = ["description", "evidence", "metadata"]
            cols = [
                col
                for col in df.columns
                if not any(term in col.lower() for term in exclude_terms)
            ]
            return {col: df[col] for col in cols}

        else:  # DETAILED or DEBUG
            return {col: df[col] for col in df.columns}

    def _format_keyword_results(self, result: Any) -> Dict[str, Any]:
        """Format keyword results based on detail level."""
        formatted = {}

        if hasattr(result, "keywords"):
            if self.config.detail_level == OutputDetail.MINIMAL:
                # Only keywords and scores
                formatted["keywords"] = [
                    f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords
                ]

            elif self.config.detail_level == OutputDetail.SUMMARY:
                # Include domains
                formatted["keywords"] = [
                    f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords
                ]
                formatted["keyword_domains"] = [
                    kw.domain for kw in result.keywords if kw.domain
                ]

            else:  # DETAILED or DEBUG
                # Include all information
                formatted["keywords"] = [
                    f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords
                ]
                formatted["keyword_scores"] = [
                    kw.score for kw in result.keywords
                ]
                formatted["keyword_domains"] = [
                    kw.domain for kw in result.keywords if kw.domain
                ]
                if hasattr(result, "compound_words"):
                    formatted["compound_words"] = result.compound_words

        return formatted

    def _format_theme_results(self, result: Any) -> Dict[str, Any]:
        """Format theme results based on detail level."""
        formatted = {}

        if hasattr(result, "themes"):
            if self.config.detail_level == OutputDetail.MINIMAL:
                # Only theme names
                formatted["themes"] = [theme.name for theme in result.themes]

            elif self.config.detail_level == OutputDetail.SUMMARY:
                # Names and confidence scores
                formatted["themes"] = [
                    f"{theme.name} ({theme.confidence:.2f})"
                    for theme in result.themes
                ]

            else:  # DETAILED or DEBUG
                # All theme information
                formatted["themes"] = [theme.name for theme in result.themes]
                formatted["theme_descriptions"] = [
                    theme.description for theme in result.themes
                ]
                formatted["theme_confidence"] = [
                    theme.confidence for theme in result.themes
                ]
                if any(theme.parent_theme for theme in result.themes):
                    formatted["theme_hierarchy"] = [
                        theme.parent_theme for theme in result.themes
                    ]

        return formatted

    def display_results_summary(self, results_df: pd.DataFrame) -> None:
        """Display results summary with detail level consideration."""
        print("\nAnalysis Results Summary")
        print("=" * 50)

        # Basic stats (always shown)
        total_records = len(results_df) if not results_df.empty else 0
        print(f"Total records processed: {total_records}")

        if "language" in results_df.columns and not results_df.empty:
            print(f"Language: {results_df['language'].iloc[0]}")

        # Detail level specific output
        if self.config.detail_level == OutputDetail.MINIMAL:
            self._display_minimal_summary(results_df)
        elif self.config.detail_level == OutputDetail.SUMMARY:
            self._display_summary(results_df)
        else:  # DETAILED or DEBUG
            self._display_detailed_summary(results_df)

    def _display_minimal_summary(self, results_df: pd.DataFrame) -> None:
        """Display minimal summary with only essential information."""
        for analysis_type in ["keywords", "themes", "categories"]:
            cols = [
                col
                for col in results_df.columns
                if col.startswith(f"{analysis_type}_")
            ]
            if cols:
                print(f"\n{analysis_type.title()}:")
                # Show only counts
                result_count = sum(results_df[cols].notna().any(axis=1))
                print(f"Found in {result_count} records")

    def _display_summary(self, results_df: pd.DataFrame) -> None:
        """Display summary with moderate detail."""
        for analysis_type in ["keywords", "themes", "categories"]:
            cols = [
                col
                for col in results_df.columns
                if col.startswith(f"{analysis_type}_")
            ]
            if cols:
                print(f"\n{analysis_type.title()} Results:")
                print("-" * 30)

                # Show first row as example
                if not results_df.empty:
                    row = results_df[cols].iloc[0]
                    for col in cols:
                        if pd.notna(row[col]):
                            display_name = (
                                col.split("_", 1)[1] if "_" in col else col
                            )
                            print(f"  {display_name}: {row[col]}")

    def _display_detailed_summary(self, results_df: pd.DataFrame) -> None:
        """Display detailed summary with all available information."""
        for analysis_type in ["keywords", "themes", "categories"]:
            cols = [
                col
                for col in results_df.columns
                if col.startswith(f"{analysis_type}_")
            ]
            if cols:
                print(f"\n{analysis_type.title()} Analysis Results:")
                print("-" * 30)

                # Show first 3 rows with all details
                sample = results_df[cols].head(3)
                for idx, row in sample.iterrows():
                    print(f"\nRecord {idx + 1}:")
                    for col in cols:
                        if pd.notna(row[col]):
                            display_name = (
                                col.split("_", 1)[1] if "_" in col else col
                            )
                            print(f"  {display_name}: {row[col]}")

                # Add statistics
                print("\nStatistics:")
                for col in cols:
                    non_null = results_df[col].notna().sum()
                    print(f"  {col}: {non_null} values")

    def save_excel_results(
        self,
        results_df: pd.DataFrame,
        output_file: Union[str, Path],
        include_summary: bool = True,
    ) -> Path:
        """Save analysis results to Excel with proper formatting."""
        sheets = {"Analysis Results": results_df}

        if include_summary:
            summary_df = self._create_summary_sheet(results_df)
            sheets["Summary"] = summary_df

        # Save using FileUtils
        saved_files, _ = self.file_utils.save_data_to_storage(
            data=sheets,
            output_type="processed",
            file_name=output_file,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=True,
        )

        saved_path = Path(next(iter(saved_files.values())))
        logger.info(f"Saved formatted results to: {saved_path}")
        return saved_path

    def _create_summary_sheet(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary sheet with analysis statistics."""
        summary_data = {"Metric": [], "Value": []}

        # Basic stats
        total_records = len(results_df) if not results_df.empty else 0
        summary_data["Metric"].append("Total Records")
        summary_data["Value"].append(total_records)

        # Results by type
        for analysis_type in ["keywords", "themes", "categories"]:
            cols = [
                col
                for col in results_df.columns
                if col.startswith(f"{analysis_type}_")
            ]
            if cols:
                summary_data["Metric"].append(
                    f"{analysis_type.title()} Results"
                )
                summary_data["Value"].append(len(cols))

        # Add metadata if available
        if "language" in results_df.columns and not results_df.empty:
            summary_data["Metric"].append("Language")
            summary_data["Value"].append(str(results_df["language"].iloc[0]))

        if "processing_time" in results_df.columns and not results_df.empty:
            avg_time = results_df["processing_time"].mean()
            summary_data["Metric"].append("Average Processing Time (s)")
            summary_data["Value"].append(f"{avg_time:.2f}")

        return pd.DataFrame(summary_data)


class SimpleOutputConfig(ExcelOutputConfig):
    """Configuration for simplified output format."""

    show_scores: bool = Field(
        default=False, description="Whether to show confidence scores"
    )
    separator: str = Field(
        default=", ", description="Separator for items in a column"
    )
    max_items: Optional[int] = Field(
        default=None, description="Maximum items to show per column"
    )


class SimplifiedExcelFormatter(ExcelAnalysisFormatter):
    """Formatter for simplified tabular output."""

    def format_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        original_content: pd.DataFrame,
        content_column: str,
    ) -> pd.DataFrame:
        """Format results into a simple tabular format.

        Args:
            analysis_results: Results from different analyzers
            original_content: Original content DataFrame
            content_column: Name of content column

        Returns:
            DataFrame with simplified columns
        """
        # Start with content column
        results_df = pd.DataFrame({"content": original_content[content_column]})

        # Format keywords if present
        if "keywords" in analysis_results:
            results_df["keywords"] = self._format_simple_keywords(
                analysis_results["keywords"]
            )

        # Format themes if present
        if "themes" in analysis_results:
            results_df["themes"] = self._format_simple_themes(
                analysis_results["themes"]
            )

        return results_df

    def _format_simple_keywords(self, keywords_result: Any) -> pd.Series:
        """Format keywords into a simple string format."""
        if not hasattr(keywords_result, "keywords"):
            return pd.Series([""] * len(keywords_result))

        show_scores = (
            self.config.show_scores
            if hasattr(self.config, "show_scores")
            else False
        )
        max_items = getattr(self.config, "max_items", None)
        separator = getattr(self.config, "separator", ", ")

        def format_row(row):
            keywords = (
                row["keywords"]
                if isinstance(row, pd.Series)
                else keywords_result.keywords
            )

            formatted_keywords = []
            for kw in keywords[:max_items] if max_items else keywords:
                if show_scores:
                    formatted_keywords.append(f"{kw.keyword} ({kw.score:.2f})")
                else:
                    formatted_keywords.append(kw.keyword)

            return separator.join(formatted_keywords)

        if isinstance(keywords_result, pd.DataFrame):
            return keywords_result.apply(format_row, axis=1)
        else:
            return pd.Series([format_row(keywords_result)])

    def _format_simple_themes(self, themes_result: Any) -> pd.Series:
        """Format themes into a simple string format."""
        if not hasattr(themes_result, "themes"):
            return pd.Series([""] * len(themes_result))

        show_scores = (
            self.config.show_scores
            if hasattr(self.config, "show_scores")
            else False
        )
        max_items = getattr(self.config, "max_items", None)
        separator = getattr(self.config, "separator", ", ")

        def format_row(row):
            themes = (
                row["themes"]
                if isinstance(row, pd.Series)
                else themes_result.themes
            )

            formatted_themes = []
            for theme in themes[:max_items] if max_items else themes:
                if show_scores:
                    formatted_themes.append(
                        f"{theme.name} ({theme.confidence:.2f})"
                    )
                else:
                    formatted_themes.append(theme.name)

            return separator.join(formatted_themes)

        if isinstance(themes_result, pd.DataFrame):
            return themes_result.apply(format_row, axis=1)
        else:
            return pd.Series([format_row(themes_result)])


# class ExcelAnalysisFormatter(ExcelFormatter):
#     """Enhanced Excel formatter with support for Excel analysis results."""

#     def __init__(
#         self,
#         file_utils: Optional[FileUtils] = None,
#         config: Optional[ExcelOutputConfig] = None,
#         use_tqdm: bool = True,
#     ):
#         """Initialize formatter with optional progress bars."""
#         super().__init__(file_utils, config)
#         self.use_tqdm = use_tqdm

#

#     def _format_analysis_type(
#         self, analysis_type: str, result: Any
#     ) -> Dict[str, Any]:
#         """Format specific analysis type results.

#         Args:
#             analysis_type: Type of analysis ('keywords', 'themes', 'categories')
#             result: Analysis results to format

#         Returns:
#             Dict of column names and their values
#         """
#         if isinstance(result, pd.DataFrame):
#             # Result is already a DataFrame, return column data
#             return {col: result[col] for col in result.columns}

#         # Handle other result types based on analysis type
#         if analysis_type == "keywords":
#             return self._format_keyword_results(result)
#         elif analysis_type == "themes":
#             return self._format_theme_results(result)
#         elif analysis_type == "categories":
#             return self._format_category_results(result)
#         else:
#             logger.warning(f"Unknown analysis type: {analysis_type}")
#             return {}

#     def _format_keyword_results(self, result: Any) -> Dict[str, List[str]]:
#         """Format keyword analysis results."""
#         formatted = {}

#         if hasattr(result, "keywords"):
#             # Format keywords with scores
#             keywords_with_scores = [
#                 f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords
#             ]
#             formatted["keywords"] = keywords_with_scores

#             # Add domain information if available
#             if hasattr(result, "domain_keywords"):
#                 for domain, keywords in result.domain_keywords.items():
#                     formatted[f"domain_{domain}"] = keywords

#         return formatted

#     def _format_theme_results(self, result: Any) -> Dict[str, List[str]]:
#         """Format theme analysis results."""
#         formatted = {}

#         if hasattr(result, "themes"):
#             # Format themes with confidence
#             themes_with_conf = [
#                 f"{theme.name} ({theme.confidence:.2f})"
#                 for theme in result.themes
#             ]
#             formatted["themes"] = themes_with_conf

#             # Add descriptions if in detailed mode
#             if self.config.detail_level >= OutputDetail.DETAILED:
#                 formatted["descriptions"] = [
#                     theme.description for theme in result.themes
#                 ]

#         return formatted

#     def _format_category_results(self, result: Any) -> Dict[str, List[str]]:
#         """Format category analysis results."""
#         formatted = {}

#         if hasattr(result, "categories"):
#             # Format categories with confidence
#             categories_with_conf = [
#                 f"{cat.name} ({cat.confidence:.2f})"
#                 for cat in result.categories
#             ]
#             formatted["categories"] = categories_with_conf

#             # Add evidence if in detailed mode
#             if self.config.detail_level >= OutputDetail.DETAILED:
#                 formatted["evidence"] = [
#                     "\n".join(e.text for e in cat.evidence)
#                     for cat in result.categories
#                 ]

#         return formatted

#     def save_excel_results(
#         self,
#         results_df: pd.DataFrame,
#         output_file: Union[str, Path],
#         include_summary: bool = True,
#     ) -> Path:
#         """Save analysis results to Excel with proper formatting."""
#         sheets = {"Analysis Results": results_df}

#         if include_summary:
#             summary_df = self._create_summary_sheet(results_df)
#             sheets["Summary"] = summary_df

#         # Save using FileUtils
#         saved_files, _ = self.file_utils.save_data_to_storage(
#             data=sheets,
#             output_type="processed",
#             file_name=output_file,
#             output_filetype=OutputFileType.XLSX,
#             include_timestamp=True,
#         )

#         saved_path = Path(next(iter(saved_files.values())))
#         logger.info(f"Saved formatted results to: {saved_path}")
#         return saved_path

#     def _create_summary_sheet(self, results_df: pd.DataFrame) -> pd.DataFrame:
#         """Create summary sheet with analysis statistics."""
#         summary_data = {"Metric": [], "Value": []}

#         # Basic stats
#         total_records = len(results_df) if not results_df.empty else 0
#         summary_data["Metric"].append("Total Records")
#         summary_data["Value"].append(total_records)

#         # Results by type
#         for analysis_type in ["keywords", "themes", "categories"]:
#             cols = [
#                 col
#                 for col in results_df.columns
#                 if col.startswith(f"{analysis_type}_")
#             ]
#             if cols:
#                 summary_data["Metric"].append(
#                     f"{analysis_type.title()} Results"
#                 )
#                 summary_data["Value"].append(len(cols))

#         # Add metadata if available
#         if "language" in results_df.columns and not results_df.empty:
#             summary_data["Metric"].append("Language")
#             summary_data["Value"].append(str(results_df["language"].iloc[0]))

#         if "processing_time" in results_df.columns and not results_df.empty:
#             avg_time = results_df["processing_time"].mean()
#             summary_data["Metric"].append("Average Processing Time (s)")
#             summary_data["Value"].append(f"{avg_time:.2f}")

#         return pd.DataFrame(summary_data)

#     def display_results_summary(self, results_df: pd.DataFrame) -> None:
#         """Display formatted analysis summary.

#         Args:
#             results_df: DataFrame with analysis results
#         """
#         print("\nAnalysis Results Summary")
#         print("=" * 50)

#         # Display basic stats
#         total_records = len(results_df) if not results_df.empty else 0
#         print(f"Total records processed: {total_records}")

#         if "language" in results_df.columns and not results_df.empty:
#             print(f"Language: {results_df['language'].iloc[0]}")

#         if "processing_time" in results_df.columns and not results_df.empty:
#             avg_time = results_df["processing_time"].mean()
#             print(f"Average processing time: {avg_time:.2f}s")

#         # Results by type
#         for analysis_type in ["keywords", "themes", "categories"]:
#             cols = [
#                 col
#                 for col in results_df.columns
#                 if col.startswith(f"{analysis_type}_")
#             ]
#             if cols:
#                 print(f"\n{analysis_type.title()} Analysis Results:")
#                 print("-" * 30)

#                 # Show sample results (first 3 rows)
#                 sample = results_df[cols].head(3)
#                 for idx, row in sample.iterrows():
#                     print(f"\nRecord {idx + 1}:")
#                     for col in cols:
#                         if pd.notna(row[col]):
#                             # Clean up column name for display
#                             display_name = (
#                                 col.split("_", 1)[1] if "_" in col else col
#                             )
#                             print(f"  {display_name}: {row[col]}")

#         # Show additional stats if available
#         if self.config.detail_level >= OutputDetail.DETAILED:
#             print("\nDetailed Statistics:")
#             print("-" * 30)
#             for analysis_type in ["keywords", "themes", "categories"]:
#                 cols = [
#                     col
#                     for col in results_df.columns
#                     if col.startswith(f"{analysis_type}_")
#                 ]
#                 if cols:
#                     result_count = sum(results_df[cols].notna().any(axis=1))
#                     print(
#                         f"{analysis_type.title()} found in {result_count} records"
#                     )

#         print("\n" + "=" * 50)
