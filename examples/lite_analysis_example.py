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
import nltk

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
root_logger.setLevel(logging.WARNING)

# Configure handler if none exists
if not root_logger.handlers:
    # Console handler with WARNING level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
    # File handler with DEBUG level
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "lite_analysis.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")

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
            # Convert parameter file to Path if it's a string
            param_path = Path(parameter_file) if isinstance(parameter_file, str) else parameter_file
            
            # Use FileUtils to get the full path if needed
            if not param_path.is_absolute():
                param_path = self.file_utils.get_data_path("parameters") / param_path.name
                
            parameter_handler = ParameterHandler(param_path)
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
            # Convert paths to Path objects and resolve them using FileUtils
            input_path = self.file_utils.get_data_path("raw") / Path(input_file).name
            
            if output_file:
                output_path = self.file_utils.get_data_path("processed") / Path(output_file).name
            else:
                # Generate default output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.file_utils.get_data_path("processed") / f"lite_analysis_results_{language}_{timestamp}.xlsx"
            
            if parameter_file:
                param_path = self.file_utils.get_data_path("parameters") / Path(parameter_file).name
            else:
                param_path = self.file_utils.get_data_path("parameters") / f"parameters_{language}.xlsx"
            
            # Load input data using FileUtils
            df = self.file_utils.load_single_file(input_path, input_type="raw")
            
            # Get available categories from parameter file
            available_categories = self._get_available_categories(param_path)
            
            # Create analyzer
            analyzer = LiteSemanticAnalyzer(
                llm=self.llm,
                parameter_file=param_path,
                file_utils=self.file_utils,
                language=language,
                available_categories=available_categories
            )

            results = []
            failed_records = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts", ncols=100):
                try:
                    text = str(row[content_column])
                    result = await analyzer.analyze(
                        text=text,
                        analysis_types=analysis_types
                    )
                    
                    # Only print analysis results if in debug mode
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        await self.analyze_single_text(text, language, analysis_types, param_path)
                    
                    # Format result for DataFrame
                    result_dict = {
                        content_column: text,
                        "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Processing Time (s)": f"{result.processing_time:.2f}",
                        "Language": result.language,
                        "Success": result.success,
                        
                        # Keywords section
                        "Keywords": self._wrap_text(", ".join(f"{kw.keyword} ({kw.score:.2f})" for kw in result.keywords.keywords)) if result.keywords and result.keywords.success else "",
                        
                        # Themes section
                        "Main Themes": self._wrap_text(", ".join(f"{theme.name} ({theme.confidence:.2f})" for theme in result.themes.themes)) if result.themes and result.themes.success else "",
                        "Theme Hierarchy": self._format_theme_hierarchy(result.themes.theme_hierarchy) if result.themes and result.themes.success else "",
                        
                        # Categories section
                        "Categories": self._wrap_text(", ".join(f"{cat.name} ({cat.confidence:.2f})" for cat in result.categories.matches)) if result.categories and result.categories.success else "",
                        "Category Evidence": self._format_category_evidence(result.categories.matches) if result.categories and result.categories.success else ""
                    }
                    results.append(result_dict)
                except Exception as e:
                    logger.error(f"Failed to process record {idx}: {str(e)}")
                    failed_records.append({
                        content_column: text,
                        "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Error": str(e),
                        "Success": False
                    })
                    continue

            # Create results DataFrame
            results_df = pd.DataFrame(results)

            # Save results if output file specified
            if output_file:
                output_file_str = str(output_file) if isinstance(output_file, Path) else output_file
                
                # Create summary sheet
                summary_data = {
                    "Metric": [
                        "Total Records",
                        "Successfully Processed Records",
                        "Failed Records",
                        "Records with Keywords",
                        "Records with Themes",
                        "Records with Categories",
                        "Average Processing Time (s)",
                        "Success Rate (%)",
                        "Analysis Timestamp",
                        "Language"
                    ],
                    "Value": [
                        len(df),
                        len(results),
                        len(failed_records),
                        results_df["Keywords"].notna().sum(),
                        results_df["Main Themes"].notna().sum(),
                        results_df["Categories"].notna().sum(),
                        f"{results_df['Processing Time (s)'].astype(float).mean():.2f}",
                        f"{(len(results) / len(df) * 100):.1f}",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        language
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                
                # Save results and failures to separate sheets
                output_data = {
                    "Analysis Results": results_df,
                    "Failed Records": pd.DataFrame(failed_records) if failed_records else pd.DataFrame(),
                    "Summary": summary_df
                }
                
                self.file_utils.save_data_to_storage(
                    data=output_data,
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
            try:
                evidence = cat.evidence if hasattr(cat, 'evidence') and cat.evidence else []
                themes = cat.themes if hasattr(cat, 'themes') and cat.themes else []
                
                # Extract text from Evidence objects with error handling
                evidence_texts = []
                for e in evidence:
                    try:
                        if hasattr(e, 'text'):
                            evidence_texts.append(e.text)
                        elif isinstance(e, str):
                            evidence_texts.append(e)
                        else:
                            logger.warning(f"Unexpected evidence type: {type(e)}")
                    except Exception as e:
                        logger.warning(f"Error processing evidence: {str(e)}")
                        continue
                
                evidence_str = f"Evidence: {', '.join(evidence_texts)}" if evidence_texts else ""
                themes_str = f"Related Themes: {', '.join(themes)}" if themes else ""
                
                if evidence_str and themes_str:
                    lines.append(f"{cat.name} - {evidence_str}; {themes_str}")
                elif evidence_str:
                    lines.append(f"{cat.name} - {evidence_str}")
                elif themes_str:
                    lines.append(f"{cat.name} - {themes_str}")
                else:
                    lines.append(cat.name)
                    
            except Exception as e:
                logger.error(f"Error formatting category {getattr(cat, 'name', 'unknown')}: {str(e)}")
                lines.append(getattr(cat, 'name', 'unknown category'))
                
        return "\n".join(lines)


async def main():
    """Run example analysis."""
    try:
        # Create runner
        runner = LiteAnalysisRunner()

        # Example 1: Business Content Analysis
        print("\n=== Business Content Analysis (English) ===")
        text_business_en = "Q3 financial results show 15% revenue growth. Market expansion strategy focuses on emerging sectors."
        await runner.analyze_single_text(
            text=text_business_en,
            language="en",
            analysis_types=["keywords", "themes", "categories"],
            parameter_file="business_parameters_en.xlsx"
        )

        # Example 2: Tech Support Analysis
        print("\n=== Tech Support Analysis (English) ===")
        text_support_en = "I'm having trouble logging into the admin dashboard. The system keeps showing 'Invalid credentials' even though I'm sure the password is correct."
        await runner.analyze_single_text(
            text=text_support_en,
            language="en",
            analysis_types=["keywords", "themes", "categories"],
            parameter_file="support_parameters_en.xlsx"
        )

        # Example 3: Training Services Analysis
        print("\n=== Training Services Analysis (English) ===")
        text_training_en = "Hi, I'm interested in the Advanced Data Science course. What are the prerequisites and when does the next cohort start?"
        await runner.analyze_single_text(
            text=text_training_en,
            language="en",
            analysis_types=["keywords", "themes", "categories"],
            parameter_file="training_parameters_en.xlsx"
        )

        # Example 4: Business Content Analysis (Finnish)
        print("\n=== Business Content Analysis (Finnish) ===")
        text_business_fi = "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun. Markkinalaajennusstrategia keskittyy uusiin sektoreihin."
        await runner.analyze_single_text(
            text=text_business_fi,
            language="fi",
            analysis_types=["keywords", "themes", "categories"],
            parameter_file="business_parameters_fi.xlsx"
        )

        # Example 5: Tech Support Analysis (Finnish)
        print("\n=== Tech Support Analysis (Finnish) ===")
        text_support_fi = "Minulla on ongelmia kirjautua hallintapaneeliin. Järjestelmä näyttää 'Virheelliset tunnukset' vaikka salasana on varmasti oikein."
        await runner.analyze_single_text(
            text=text_support_fi,
            language="fi",
            analysis_types=["keywords", "themes", "categories"],
            parameter_file="support_parameters_fi.xlsx"
        )

        # Example 6: Training Services Analysis (Finnish)
        print("\n=== Training Services Analysis (Finnish) ===")
        text_training_fi = "Hei, olen kiinnostunut Edistynyt Data Science -kurssista. Mitkä ovat esitietovaatimukset ja milloin seuraava ryhmä alkaa?"
        await runner.analyze_single_text(
            text=text_training_fi,
            language="fi",
            analysis_types=["keywords", "themes", "categories"],
            parameter_file="training_parameters_fi.xlsx"
        )

        # Example 7: Excel file analysis for each topic
        topics = ["business", "support", "training"]
        languages = ["en", "fi"]
        
        for topic in topics:
            for lang in languages:
                print(f"\n=== Excel File Analysis ({topic.title()}, {lang.upper()}) ===")
                await runner.analyze_excel(
                    input_file=f"{topic}_test_content_{lang}.xlsx",
                    output_file=f"lite_analysis_results_{topic}_{lang}.xlsx",
                    parameter_file=f"{topic}_parameters_{lang}.xlsx",
                    content_column="content",
                    language=lang,
                    analysis_types=["keywords", "themes", "categories"],
                    batch_size=3
                )

    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 