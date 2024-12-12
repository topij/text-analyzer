# examples/excel_analysis_example.py

import asyncio
from pathlib import Path
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


async def analyze_excel_content(
    content_file: str = "test_content_en.xlsx",
    parameter_file: str = "parameters_en.xlsx",
    content_column: str = "content",
    analysis_types: Optional[List[str]] = None,
    batch_size: int = 10,
) -> None:
    """Run Excel-based analysis with progress reporting."""
    try:
        # Initialize FileUtils
        file_utils = FileUtils()

        # Verify paths with progress
        with tqdm(total=2, desc="Checking files") as pbar:
            content_path = file_utils.get_data_path("raw") / content_file
            pbar.update(1)

            param_path = file_utils.get_data_path("parameters") / parameter_file
            pbar.update(1)

            if not content_path.exists():
                raise FileNotFoundError(
                    f"Content file not found: {content_path}"
                )
            if not param_path.exists():
                raise FileNotFoundError(
                    f"Parameter file not found: {param_path}"
                )

        print("\nStarting analysis:")
        print(f"Content file: {content_path}")
        print(f"Parameter file: {param_path}")
        print(f"Analysis types: {analysis_types or 'all'}")

        # Create and run analyzer
        analyzer = SemanticAnalyzer.from_excel(
            content_file=content_file,
            parameter_file=parameter_file,
            content_column=content_column,
            file_utils=file_utils,
        )

        # Run analysis with progress
        results = await analyzer.analyze_excel(
            analysis_types=analysis_types,
            batch_size=batch_size,
            save_results=True,
            output_file="analysis_results",
            show_progress=True,
        )

        # Display formatted results
        display_analysis_summary(results)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)


if __name__ == "__main__":
    print("Running Excel-based analysis...")

    asyncio.run(
        analyze_excel_content(
            content_file="test_content_en.xlsx",
            parameter_file="parameters_en.xlsx",
            analysis_types=["keywords", "themes"],
        )
    )
