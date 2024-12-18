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

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)


class AnalysisEnvironment:
    """Environment manager for semantic analysis."""

    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize analysis environment."""
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self.project_root = project_root or Path(__file__).resolve().parents[1]

        # Initialize FileUtils first
        self.file_utils = FileUtils(project_root=self.project_root)

        # Initialize ConfigManager with shared FileUtils instance
        self.config_manager = ConfigManager(
            file_utils=self.file_utils, project_root=self.project_root
        )

        # Mark as initialized
        self._initialized = True

    def get_components(self) -> Dict[str, Any]:
        """Get initialized components for reuse."""
        return {
            "file_utils": self.file_utils,
            "config_manager": self.config_manager,
            "project_root": self.project_root,
        }

    def _setup_logging(self) -> None:
        """Configure logging using config manager settings."""
        config = self.config_manager.get_config()
        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format=config.logging.format,
            force=True,
        )

    def _load_environment(self) -> None:
        """Load environment variables from available .env files."""
        env_paths = [
            self.project_root / ".env",
            self.project_root / ".env.local",
            Path.home() / ".env",
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from: {env_path}")
                break

    def get_components(self) -> Dict[str, Any]:
        """Get initialized components for reuse."""
        return {
            "file_utils": self.file_utils,
            "config_manager": self.config_manager,
            "project_root": self.project_root,
        }


class SemanticAnalysisRunner:
    """Runner for semantic analysis tasks."""

    def __init__(self, components: Dict[str, Any]):
        """Initialize runner with shared components."""
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
    ) -> None:
        """Run semantic analysis on Excel file."""
        try:
            # Load input file using shared FileUtils
            logger.info(f"Loading input file: {input_file}")
            input_df = self.file_utils.load_single_file(
                input_file, input_type="raw"
            )

            if content_column not in input_df.columns:
                raise ValueError(f"Content column '{content_column}' not found")

            # Initialize components with shared instances
            config = ExcelOutputConfig(
                detail_level=detail_level, include_confidence=show_scores
            )
            formatter = ExcelAnalysisFormatter(self.file_utils, config)

            # Initialize analyzer with shared instances
            analyzer = SemanticAnalyzer(
                parameter_file=parameter_file,
                file_utils=self.file_utils,
                config_manager=self.config_manager,
            )

            # Process rows with progress bar
            results = []
            for idx, row in tqdm(
                input_df.iterrows(), total=len(input_df), desc="Analyzing"
            ):
                # Analyze text
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
    try:
        # Initialize environment once
        env = AnalysisEnvironment()
        components = env.get_components()

        # Create runner with shared components
        runner = SemanticAnalysisRunner(components)

        # Run analysis
        await runner.run_analysis(
            input_file="test_content_short_en.xlsx",
            output_file="analysis_results_en.xlsx",
            parameter_file="parameters_en.xlsx",
            detail_level=OutputDetail.MINIMAL,
            show_scores=True,
        )

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
