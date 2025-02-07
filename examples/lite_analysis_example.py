# examples/lite_analysis_example.py

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to Python path if needed
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.analyzers.lite_analyzer import LiteSemanticAnalyzer
from src.utils.formatting_config import OutputDetail, ExcelOutputConfig
from src.excel_analysis.formatters import ExcelAnalysisFormatter
from src.config import ConfigManager
from src.core.managers import EnvironmentManager, EnvironmentConfig
from src.core.llm.factory import create_llm
from src.core.config import AnalyzerConfig
from src.loaders.parameter_handler import ParameterHandler

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


class LiteAnalysisRunner:
    """Runner for lite semantic analysis tasks."""

    def __init__(self):
        """Initialize runner with shared components from environment."""
        # Get components from environment manager
        components = EnvironmentManager.get_instance().get_components()
        self.file_utils = components["file_utils"]
        self.config_manager = components["config_manager"]
        
        # Get configurations
        base_config = self.config_manager.get_analyzer_config("base")
        config = self.config_manager.get_config()
        
        # Get default provider and model from config
        model_config = config.get("models", {})
        provider = model_config.get("default_provider", "openai")
        model = model_config.get("default_model", "gpt-3.5-turbo")
        
        # Create analyzer config
        base_config.update({
            "models": {
                "default_provider": provider,
                "default_model": model,
            }
        })
        analyzer_config = AnalyzerConfig(base_config)
        
        # Create LLM instance
        self.llm = create_llm(
            analyzer_config=analyzer_config,
            provider=provider,
            model=model
        )

    def _get_available_categories(self, parameter_file: Optional[Union[str, Path]]) -> Set[str]:
        """Get available categories from parameter file."""
        if not parameter_file:
            return set()
            
        try:
            parameter_handler = ParameterHandler(parameter_file)
            parameters = parameter_handler.get_parameters()
            return {cat for cat in parameters.categories.keys()} if parameters.categories else set()
        except Exception as e:
            logger.warning(f"Failed to load categories from parameter file: {e}")
            return set()

    async def analyze_single_text(
        self,
        text: str,
        language: str = "en",
        analysis_types: Optional[List[str]] = None,
        parameter_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Run analysis on a single text."""
        try:
            # Get available categories from parameter file
            available_categories = self._get_available_categories(parameter_file)
            
            # Create analyzer
            analyzer = LiteSemanticAnalyzer(
                llm=self.llm,
                parameter_file=parameter_file,
                file_utils=self.file_utils,
                language=language,
                available_categories=available_categories
            )

            # Run analysis
            result = await analyzer.analyze(
                text=text,
                analysis_types=analysis_types
            )

            # Print results
            print(f"\nAnalysis Results for text in {language}:")
            
            if result.keywords and result.keywords.success:
                print("\nKeywords:")
                keywords = [f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords.keywords]
                # Print keywords in a wrapped format
                print(self._wrap_text(", ".join(keywords), width=100))
            
            if result.themes and result.themes.success:
                print("\nThemes:")
                themes = [f"{theme.name} ({theme.confidence:.2f})" for theme in result.themes.themes]
                print(self._wrap_text(", ".join(themes), width=100))
                
                if result.themes.theme_hierarchy:
                    print("\nTheme Hierarchy:")
                    for main_theme, sub_themes in result.themes.theme_hierarchy.items():
                        sub_theme_text = self._wrap_text(", ".join(sub_themes), width=80, subsequent_indent="    ")
                        print(f"- {main_theme}:\n    {sub_theme_text}")
            
            if result.categories and result.categories.success and result.categories.matches:
                print("\nCategories:")
                categories = [f"{cat.name} ({cat.confidence:.2f})" for cat in result.categories.matches]
                print(self._wrap_text(", ".join(categories), width=100))

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _wrap_text(self, text: str, width: int = 100, subsequent_indent: str = "") -> str:
        """Wrap text to specified width."""
        import textwrap
        return textwrap.fill(text, width=width, subsequent_indent=subsequent_indent)

    async def analyze_excel(
        self,
        input_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        parameter_file: Optional[Union[str, Path]] = None,
        content_column: str = "content",
        language: str = "en",
        analysis_types: Optional[List[str]] = None,
        batch_size: int = 3,
    ) -> None:
        """Run analysis on Excel file."""
        try:
            # Load input data
            df = self.file_utils.load_single_file(input_file, input_type="raw")
            
            # Get available categories from parameter file
            available_categories = self._get_available_categories(parameter_file)
            
            # Create analyzer
            analyzer = LiteSemanticAnalyzer(
                llm=self.llm,
                parameter_file=parameter_file,
                file_utils=self.file_utils,
                language=language,
                available_categories=available_categories
            )

            results = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts", ncols=100):
                text = str(row[content_column])
                result = await analyzer.analyze(
                    text=text,
                    analysis_types=analysis_types
                )
                
                # Only print analysis results if in debug mode
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    await self.analyze_single_text(text, language, analysis_types, parameter_file)
                
                # Format result for DataFrame
                result_dict = {
                    content_column: text,
                    "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Processing Time (s)": f"{result.processing_time:.2f}",
                    "Language": result.language,
                    "Success": result.success,
                    
                    # Keywords section
                    "Keywords": self._wrap_text(", ".join(f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords.keywords)),
                    
                    # Themes section
                    "Main Themes": self._wrap_text(", ".join(f"{theme.name} ({theme.confidence:.2f})" for theme in result.themes.themes)),
                    "Theme Hierarchy": self._format_theme_hierarchy(result.themes.theme_hierarchy),
                    
                    # Categories section
                    "Categories": self._wrap_text(", ".join(f"{cat.name} ({cat.confidence:.2f})" for cat in result.categories.matches)),
                    "Category Evidence": self._format_category_evidence(result.categories.matches)
                }
                results.append(result_dict)

            # Create results DataFrame
            results_df = pd.DataFrame(results)

            # Save results if output file specified
            if output_file:
                output_file_str = str(output_file) if isinstance(output_file, Path) else output_file
                
                # Create summary sheet
                summary_data = {
                    "Metric": [
                        "Total Records",
                        "Records with Keywords",
                        "Records with Themes",
                        "Records with Categories",
                        "Average Processing Time (s)",
                        "Success Rate (%)",
                        "Analysis Timestamp",
                        "Language"
                    ],
                    "Value": [
                        len(results_df),
                        results_df["Keywords"].notna().sum(),
                        results_df["Main Themes"].notna().sum(),
                        results_df["Categories"].notna().sum(),
                        f"{results_df['Processing Time (s)'].astype(float).mean():.2f}",
                        f"{(results_df['Success'] == True).mean() * 100:.1f}",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        language
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                
                # Save both sheets
                self.file_utils.save_data_to_storage(
                    data={
                        "Analysis Results": results_df,
                        "Summary": summary_df
                    },
                    output_type="processed",
                    file_name=output_file_str,
                    output_filetype="xlsx",
                    include_timestamp=True
                )
                logger.info(f"Results saved to {output_file}")
                
                # Print summary
                print("\nAnalysis Summary:")
                for _, row in summary_df.iterrows():
                    print(f"{row['Metric']}: {row['Value']}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _format_theme_hierarchy(self, hierarchy: Dict[str, List[str]]) -> str:
        """Format theme hierarchy as a readable string."""
        if not hierarchy:
            return ""
        
        lines = []
        for main_theme, sub_themes in hierarchy.items():
            sub_theme_text = ", ".join(sub_themes) if sub_themes else "-"
            lines.append(f"{main_theme}: {sub_theme_text}")
        return "\n".join(lines)

    def _format_category_evidence(self, categories: List[Any]) -> str:
        """Format category evidence as a readable string."""
        if not categories:
            return ""
        
        lines = []
        for cat in categories:
            evidence = cat.evidence if hasattr(cat, 'evidence') and cat.evidence else []
            themes = cat.themes if hasattr(cat, 'themes') and cat.themes else []
            
            evidence_str = f"Evidence: {', '.join(evidence)}" if evidence else ""
            themes_str = f"Related Themes: {', '.join(themes)}" if themes else ""
            
            if evidence_str and themes_str:
                lines.append(f"{cat.name} - {evidence_str}; {themes_str}")
            elif evidence_str:
                lines.append(f"{cat.name} - {evidence_str}")
            elif themes_str:
                lines.append(f"{cat.name} - {themes_str}")
            else:
                lines.append(cat.name)
                
        return "\n".join(lines)


async def main():
    """Run example analysis."""
    try:
        # Create runner
        runner = LiteAnalysisRunner()

        # Example 1: Single text analysis
        print("\n=== Single Text Analysis ===")
        text = "Artificial intelligence and machine learning are transforming the technology landscape. Cloud computing and data analytics enable businesses to make data-driven decisions. Cybersecurity remains a critical concern for organizations worldwide."
        
        await runner.analyze_single_text(
            text=text,
            language="en",
            analysis_types=["keywords", "themes", "categories"]
        )

        # Example 2: Excel file analysis
        print("\n=== Excel File Analysis ===")
        
        # Set up file paths relative to project root
        data_dir = Path().resolve() / "data"
        input_file = data_dir / "raw" / "test_content_en.xlsx"
        output_file = data_dir / "processed" / "lite_analysis_results_en.xlsx"
        parameter_file = data_dir / "parameters" / "parameters_en.xlsx"

        await runner.analyze_excel(
            input_file=input_file,
            output_file=output_file,
            parameter_file=parameter_file,
            content_column="content",
            language="en",
            analysis_types=["keywords", "themes", "categories"],
            batch_size=3
        )

    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 