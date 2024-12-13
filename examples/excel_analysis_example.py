# examples/excel_analysis_example.py

import asyncio
from pathlib import Path
import os
import logging
import sys
from typing import List, Optional

# Add project root to Python path if needed
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from tqdm import tqdm
import pandas as pd
from FileUtils import FileUtils, OutputFileType

from src.semantic_analyzer import SemanticAnalyzer
from src.utils.output_formatter import (
    OutputDetail,
    ExcelOutputConfig,
)

from src.excel_analysis.formatters import (
    SimpleOutputConfig,
    SimplifiedExcelFormatter,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Setup environment for analysis."""
    try:
        # Add project root to path
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        # Load environment from the most specific to most general
        from dotenv import load_dotenv

        env_paths = [
            project_root / ".env",
            project_root / ".env.local",
            Path.home() / ".env",
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
                break

        # Configure logging
        import logging

        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Verify environment
        required_vars = ["OPENAI_API_KEY"]  # Add other required variables
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"Missing required environment variables: {missing}")
            return False

        return True

    except Exception as e:
        print(f"Environment setup failed: {e}")
        return False


def display_analysis_summary(results: pd.DataFrame) -> None:
    """Display formatted analysis summary."""
    print("\nAnalysis Results Summary")
    print("=" * 50)

    # Basic stats
    print(f"Total rows processed: {len(results)}")
    if "language" in results.columns:
        print(f"Language: {results['language'].iloc[0]}")
    if "processing_time" in results.columns:
        print(
            f"Average processing time: {results['processing_time'].mean():.2f}s"
        )

    # Results by type
    result_sections = {
        "Keywords": [
            col for col in results.columns if col.startswith("keywords_")
        ],
        "Themes": [col for col in results.columns if col.startswith("themes_")],
        "Categories": [
            col for col in results.columns if col.startswith("categories_")
        ],
    }

    for section_name, columns in result_sections.items():
        if columns:
            print(f"\n{section_name} Results")
            print("-" * 50)

            # Display first 3 rows for each result type
            sample_results = results[columns].head(3)

            # Format each cell for display
            for idx, row in sample_results.iterrows():
                print(f"\nRow {idx + 1}:")
                for col in columns:
                    value = row[col]
                    if pd.notna(value):
                        print(f"  {col.split('_', 1)[1]}: {value}")


async def analyze_excel_content_detailed(
    content_file: str = "test_content_en.xlsx",
    parameter_file: str = "parameters_en.xlsx",
    content_column: str = "content",
    analysis_types: Optional[List[str]] = None,
    batch_size: int = 10,
    output_detail: OutputDetail = OutputDetail.SUMMARY,
) -> None:
    """Run Excel-based analysis with enhanced formatting."""
    try:
        # Initialize FileUtils
        file_utils = FileUtils()

        # Create output configuration
        format_config = ExcelOutputConfig(
            detail_level=output_detail,
            include_confidence=True,
            batch_size=batch_size,
        )

        # Create and configure analyzer
        analyzer = SemanticAnalyzer.from_excel(
            content_file=content_file,
            parameter_file=parameter_file,
            content_column=content_column,
            file_utils=file_utils,
            output_config=format_config,
        )

        # Run analysis with enhanced formatting
        results = await analyzer.analyze_excel(
            analysis_types=analysis_types,
            batch_size=batch_size,
            save_results=True,
            output_file="analysis_results",
            show_progress=True,
            format_config=format_config,
        )

        print("\nAnalysis complete!")
        print(f"Results saved with {output_detail.value} detail level")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)


# examples/excel_analysis_example.py


async def analyze_excel_content(
    content_file: str = "test_content_en.xlsx",
    parameter_file: str = "parameters_en.xlsx",
    content_column: str = "content",
    analysis_types: Optional[List[str]] = None,
    show_scores: bool = True,
) -> None:
    """Run Excel analysis with simplified output."""
    try:
        # Initialize components
        file_utils = FileUtils()

        # Create analyzer
        analyzer = SemanticAnalyzer.from_excel(
            content_file=content_file,
            parameter_file=parameter_file,
            content_column=content_column,
            file_utils=file_utils,
        )

        # Run analysis
        results = await analyzer.analyze_excel(
            analysis_types=analysis_types, batch_size=10, save_results=False
        )

        # Debug: Print available columns
        print("\nAvailable columns:", results.columns.tolist())

        # Extract and format the columns we want
        simplified_df = pd.DataFrame()

        # Add content column
        simplified_df["content"] = results[content_column]

        # Format keywords
        if "keywords" in results.columns:
            if show_scores:
                # Look for keywords and scores in results
                keywords = results["keywords"].copy()
                scores = results.get(
                    "keywords_keyword_scores", pd.Series([""] * len(results))
                )

                # Format with scores if available
                simplified_df["keywords"] = [
                    (
                        ", ".join(
                            f"{kw} ({score})"
                            for kw, score in zip(
                                str(k).split(", "), str(s).split(", ")
                            )
                        )
                        if pd.notna(k) and pd.notna(s)
                        else ""
                    )
                    for k, s in zip(keywords, scores)
                ]
            else:
                simplified_df["keywords"] = results["keywords"]

        # Format themes
        if "themes" in results.columns:
            if show_scores:
                # Look for themes and confidence scores in results
                themes = results["themes"].copy()
                confidence = results.get(
                    "themes_theme_confidence", pd.Series([""] * len(results))
                )

                # Format with scores if available
                simplified_df["themes"] = [
                    (
                        ", ".join(
                            f"{th} ({conf})"
                            for th, conf in zip(
                                str(t).split(", "), str(c).split(", ")
                            )
                        )
                        if pd.notna(t) and pd.notna(c)
                        else ""
                    )
                    for t, c in zip(themes, confidence)
                ]
            else:
                simplified_df["themes"] = results["themes"]

        # Format categories
        if "categories" in results.columns:
            if show_scores:
                # Categories are already formatted with scores from the analyzer
                simplified_df["categories"] = results["categories"]
            else:
                # Remove scores if show_scores is False
                simplified_df["categories"] = results["categories"].apply(
                    lambda x: (
                        ", ".join(
                            cat.split("(")[0].strip()
                            for cat in str(x).split(",")
                        )
                        if pd.notna(x)
                        else ""
                    )
                )

        # Update column spacing for categories
        col_space = {
            "content": 45,
            "keywords": 45,
            "themes": 45,
            "categories": 45,
        }

        # Display results with proper formatting
        print("\nSimplified Analysis Results:")
        print("-" * 120)

        # Set display options
        pd.set_option("display.max_colwidth", 40)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 120)

        # Create a formatted string representation
        formatted_df = simplified_df.copy()
        formatted_df["content"] = (
            formatted_df["content"].str.slice(0, 40) + "..."
        )

        # Ensure consistent column widths
        col_space = {"content": 45, "keywords": 45, "themes": 45}

        # Only keep existing columns in col_space
        col_space = {
            col: width
            for col, width in col_space.items()
            if col in formatted_df.columns
        }

        # Print the formatted DataFrame
        print(
            formatted_df.to_string(
                index=False, justify="left", col_space=col_space
            )
        )
        print("-" * 120)

        # Save simplified results
        saved_files, _ = file_utils.save_data_to_storage(
            data={"Analysis Results": simplified_df},
            output_type="processed",
            file_name="simplified_analysis_results",
            output_filetype="xlsx",
            include_timestamp=True,
        )

        # Print save location
        if saved_files:
            print(f"\nResults saved to: {next(iter(saved_files.values()))}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)


# Example usage


if __name__ == "__main__":
    # Setup environment first
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not setup_environment():
        print("Environment setup failed")
        sys.exit(1)

    # print("Running Excel-based analysis...")
    # asyncio.run(
    #     analyze_excel_content(
    #         content_file="test_content_en.xlsx",
    #         parameter_file="parameters_en.xlsx",
    #         analysis_types=["keywords", "themes"],
    #     )
    # )

    # Run analysis with different detail levels
    # asyncio.run(
    #     analyze_excel_content(
    #         content_file="test_content_en.xlsx",
    #         parameter_file="parameters_en.xlsx",
    #         analysis_types=["keywords", "themes"],
    #         # output_detail=OutputDetail.DETAILED,  # Try different detail levels
    #         output_detail=OutputDetail.SUMMARY,
    #     )
    # )

    # Without scores
    # asyncio.run(
    #     analyze_excel_content_simple(
    #         content_file="test_content_en.xlsx",
    #         parameter_file="parameters_en.xlsx",
    #         analysis_types=["keywords", "themes"],
    #         show_scores=False,
    #     )
    # )

    print("\nRunning analysis with scores...")
    asyncio.run(
        analyze_excel_content(
            content_file="test_content_en.xlsx",
            parameter_file="parameters_en.xlsx",
            analysis_types=["keywords", "themes", "categories"],
            show_scores=True,
        )
    )

    # async def analyze_excel_content(
    #     content_file: str = "test_content_en.xlsx",
    #     parameter_file: str = "parameters_en.xlsx",
    #     content_column: str = "content",
    #     analysis_types: Optional[List[str]] = None,
    #     show_scores: bool = True,
    # ) -> None:

    # # With scores
    # asyncio.run(
    #     analyze_excel_content(
    #         content_file="test_content_en.xlsx",
    #         parameter_file="parameters_en.xlsx",
    #         analysis_types=["keywords", "themes"],
    #         show_scores=True,
    #     )
    # )

    # With limited items
    # asyncio.run(
    #     analyze_excel_content_simple(
    #         content_file="test_content_en.xlsx",
    #         parameter_file="parameters_en.xlsx",
    #         analysis_types=["keywords", "themes"],
    #         show_scores=True,
    #         max_items=3,  # Only show top 3 items per column
    #     )
    # )
