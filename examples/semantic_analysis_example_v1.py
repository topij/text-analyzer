# examples/semantic_analysis_example.py

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to Python path if needed
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.semantic_analyzer import SemanticAnalyzer
from src.utils.formatting_config import OutputDetail, ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from src.config import ConfigManager
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


def setup_environment() -> Dict[str, Any]:
    """Setup environment for analysis."""
    try:
        # Get project root
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        # Initialize FileUtils with project root
        file_utils = FileUtils(project_root=project_root)

        # Initialize ConfigManager with FileUtils
        config_manager = ConfigManager(
            file_utils=file_utils, project_root=project_root
        )

        # Get logging config from config manager
        config = config_manager.get_config()
        log_level = config.logging.level

        # Configure logging using config
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=config.logging.format,
            force=True,  # This ensures we override any existing configuration
        )

        # Load environment variables
        env_paths = [
            project_root / ".env",
            project_root / ".env.local",
            Path.home() / ".env",
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from: {env_path}")
                break

        return {
            "file_utils": file_utils,
            "config_manager": config_manager,
            "project_root": project_root,
        }

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        raise


async def run_analysis(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    parameter_file: Optional[Union[str, Path]] = None,
    content_column: str = "content",
    detail_level: OutputDetail = OutputDetail.MINIMAL,
    show_scores: bool = True,
    initialized_components: Optional[Dict[str, Any]] = None,
) -> None:
    """Run semantic analysis on Excel file."""
    try:
        # Use initialized components or create new ones
        if initialized_components:
            file_utils = initialized_components["file_utils"]
            config_manager = initialized_components["config_manager"]
        else:
            # Setup environment if not provided
            initialized_components = setup_environment()
            file_utils = initialized_components["file_utils"]
            config_manager = initialized_components["config_manager"]

        # Load input file using the shared file_utils instance
        logger.info(f"Loading input file: {input_file}")
        input_df = file_utils.load_single_file(input_file, input_type="raw")

        if content_column not in input_df.columns:
            raise ValueError(f"Content column '{content_column}' not found")

        # Initialize components with shared instances
        config = ExcelOutputConfig(
            detail_level=detail_level, include_confidence=show_scores
        )
        formatter = ExcelAnalysisFormatter(file_utils, config)

        # Initialize analyzer with shared instances
        analyzer = SemanticAnalyzer(
            parameter_file=parameter_file,
            file_utils=file_utils,  # Pass the shared instance
            config_manager=config_manager,
        )

        # Initialize components with shared instances
        config = ExcelOutputConfig(
            detail_level=detail_level, include_confidence=show_scores
        )
        formatter = ExcelAnalysisFormatter(file_utils, config)

        # Load input file
        logger.info(f"Loading input file: {input_file}")
        input_df = file_utils.load_single_file(input_file, input_type="raw")

        if content_column not in input_df.columns:
            raise ValueError(f"Content column '{content_column}' not found")

        # Process rows with progress bar
        results = []
        for idx, row in tqdm(
            input_df.iterrows(), total=len(input_df), desc="Analyzing"
        ):
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

            # Print progress
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
    # Setup environment once
    initialized_components = setup_environment()

    # Use components for analysis
    await run_analysis(
        input_file="test_content_fi.xlsx",
        output_file="analysis_results_fi.xlsx",
        parameter_file="parameters_fi.xlsx",
        detail_level=OutputDetail.MINIMAL,
        show_scores=True,
        initialized_components=initialized_components,  # Pass initialized components
    )


if __name__ == "__main__":
    asyncio.run(main())
