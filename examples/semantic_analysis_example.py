# examples/semantic_analysis_example.py

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

# Add project root to Python path if needed
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.semantic_analyzer import SemanticAnalyzer
from src.utils.formatting_config import OutputDetail, ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from FileUtils import FileUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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


async def run_analysis(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    parameter_file: Optional[Union[str, Path]] = None,
    content_column: str = "content",
    detail_level: OutputDetail = OutputDetail.MINIMAL,
    show_scores: bool = True,
    file_utils: Optional[FileUtils] = None,
) -> None:
    """Run semantic analysis on Excel file."""
    try:
        file_utils = file_utils or FileUtils()

        # Initialize components
        config = ExcelOutputConfig(
            detail_level=detail_level, include_confidence=show_scores
        )
        formatter = ExcelAnalysisFormatter(file_utils, config)
        analyzer = SemanticAnalyzer(
            parameter_file=parameter_file, file_utils=file_utils
        )

        # Load input file
        logger.info(f"Loading input file: {input_file}")
        input_df = file_utils.load_single_file(input_file, input_type="raw")

        if content_column not in input_df.columns:
            raise ValueError(f"Content column '{content_column}' not found")

        # Process rows
        results = []
        for idx, row in tqdm(
            input_df.iterrows(), total=len(input_df), desc="Analyzing"
        ):
            # Run analysis
            analysis_result = await analyzer.analyze(
                str(row[content_column]),
                analysis_types=["keywords", "themes", "categories"],
            )

            # Format results
            formatted = formatter.format_output(
                {
                    "keywords": analysis_result.keywords,
                    "themes": analysis_result.themes,
                    "categories": analysis_result.categories,
                },
                ["keywords", "themes", "categories"],
            )
            formatted[content_column] = row[content_column]
            results.append(formatted)

            # Print to console
            print(f"\nResult {idx + 1}:")
            print("-" * 80)
            print(f"Content: {row[content_column][:100]}...")
            for key, value in formatted.items():
                if key != content_column:
                    print(f"{key.title()}: {value}")
            print("-" * 80)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save to Excel if output file specified
        if output_file:
            logger.info(f"Saving results to: {output_file}")
            formatter.save_excel_results(
                results_df, output_file, include_summary=True
            )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


async def main():
    """Run example analysis."""
    await run_analysis(
        input_file="test_content_fi.xlsx",
        output_file="analysis_results_fi.xlsx",
        parameter_file="parameters_fi.xlsx",
        # detail_level=OutputDetail.SUMMARY,
        detail_level=OutputDetail.MINIMAL,
        show_scores=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
