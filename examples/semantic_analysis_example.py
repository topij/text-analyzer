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
from src.core.managers import EnvironmentManager, EnvironmentConfig

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

# Configure handler if none exists
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(handler)

logger = logging.getLogger(__name__)

# Configure environment
env_config = EnvironmentConfig(
    project_root=Path().resolve(),
    log_level="ERROR"
)

# Initialize environment once
environment = EnvironmentManager(env_config)

# Get shared components
components = environment.get_components()


class SemanticAnalysisRunner:
    """Runner for semantic analysis tasks."""

    def __init__(self):
        """Initialize runner with shared components from environment."""
        # Get components from environment manager
        components = EnvironmentManager.get_instance().get_components()
        self.file_utils = components["file_utils"]
        self.config_manager = components["config_manager"]

    async def run_analysis(
        self,
        input_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        parameter_file: Optional[Union[str, Path]] = None,
        content_column: str = "content",
        detail_level: OutputDetail = OutputDetail.MINIMAL,
        show_scores: bool = True,
        batch_size: int = 3,
    ) -> None:
        """Run semantic analysis on Excel file.

        Args:
            input_file: Path to input Excel file
            output_file: Optional path to output file
            parameter_file: Optional path to parameter file
            content_column: Name of content column
            detail_level: Level of detail in output
            show_scores: Whether to show scores
            batch_size: Size of batches for processing
        """
        try:
            # Create analyzer with shared components
            analyzer = SemanticAnalyzer(
                parameter_file=parameter_file,
                file_utils=self.file_utils,
            )

            # Configure output formatting
            output_config = ExcelOutputConfig(
                detail_level=detail_level,
                show_scores=show_scores,
            )

            # Run analysis
            await analyzer.analyze_excel(
                content_file=input_file,
                content_column=content_column,
                output_file=output_file,
                batch_size=batch_size,
                format_config=output_config,
                show_progress=True,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


async def main():
    """Run example analysis."""
    try:
        # Create runner
        runner = SemanticAnalysisRunner()

        # Set up file paths relative to project root
        data_dir = Path().resolve() / "data"
        input_file = data_dir / "raw" / "test_content_en.xlsx"
        output_file = data_dir / "processed" / "analysis_results_en.xlsx"
        parameter_file = data_dir / "parameters" / "parameters_en.xlsx"

        # Run analysis
        await runner.run_analysis(
            input_file=input_file,
            output_file=output_file,
            parameter_file=parameter_file,
            content_column="content",
            detail_level=OutputDetail.MINIMAL,
            show_scores=True,
            batch_size=3,
        )

    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
