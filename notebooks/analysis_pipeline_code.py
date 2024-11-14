# code copied from: notebooks/analysis_pipeline_notebook.ipynb

import os
import sys
from pathlib import Path
import asyncio
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pprint import pprint

import pandas as pd

# Add project root to Python path
project_root = str(Path().resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")

# Core components
from src.semantic_analyzer.analyzer import SemanticAnalyzer
from src.utils.FileUtils.file_utils import FileUtils
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.analyzers.category_analyzer import CategoryAnalyzer, CategoryOutput
from src.core.language_processing import create_text_processor
from src.loaders.parameter_adapter import ParameterAdapter
from src.loaders.models import CategoryConfig
from src.schemas import KeywordAnalysisResult


# Initialize FileUtils and set up logging
file_utils = FileUtils()
# logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
## Tester classes
# revise to have the option to set loading files to False (when custom texts are used)
class BaseTester:
    """Base class for analysis testing."""
    
    def __init__(self):
        from src.core.llm.factory import create_llm
        self.file_utils = FileUtils()
        self.llm = create_llm()  # Create LLM instance
        self.test_texts = self._load_test_texts()
        
    def _load_test_texts(self) -> Dict[str, str]:
        """Load test texts from files."""
        try:
            # Try to load from existing files
            texts = {}
            for lang in ["en", "fi"]:
                df = self.file_utils.load_single_file(
                    f"test_content_{lang}.xlsx",
                    input_type="raw"
                )
                if df is not None:
                    for _, row in df.iterrows():
                        key = f"{lang}_{row['type']}"
                        texts[key] = row['content']
            return texts
                        
        except Exception as e:
            logger.warning(f"Could not load test texts: {e}. Using defaults.")
            return self._create_default_texts()
    
    def _create_default_texts(self) -> Dict[str, str]:
        """Create default test texts."""
        return {
            "en_technical": """
                Machine learning models are trained using large datasets.
                Neural networks extract features through multiple layers.
                Data preprocessing improves model performance.
            """,
            "en_business": """
                Q3 financial results show 15% revenue growth.
                Customer acquisition costs decreased while retention improved.
                Market expansion strategy targets emerging sectors.
            """,
            "fi_technical": """
                Ohjelmistokehittäjä työskentelee asiakasprojektissa.
                Tekninen toteutus vaatii erityistä huomiota.
                Tietoturva on keskeinen osa kehitystä.
            """
        }

    def save_test_texts(self) -> None:
        """Save test texts using FileUtils."""
        df = pd.DataFrame([
            {
                "language": key.split("_")[0],
                "type": key.split("_")[1],
                "content": content.strip()
            }
            for key, content in self.test_texts.items()
        ])
        
        self.file_utils.save_data_to_disk(
            data={"texts": df},
            output_type="raw",
            file_name="test_texts",
            output_filetype="xlsx",
            include_timestamp=False
        )

    async def analyze_text(self, text: str, language: str, analyzer: Any) -> Dict[str, Any]:
        """Base method for text analysis."""
        try:
            return await analyzer.analyze(text)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {"error": str(e)}
class KeywordTester(BaseTester):
    """Helper class for testing keyword analysis."""
    
    async def test_statistical_analysis(self, text: str, language: str = None) -> Dict[str, Any]:
        """Test statistical keyword extraction."""
        if language is None:
            from langdetect import detect
            try:
                language = detect(text)
            except:
                language = "en"
        
        # Create processor and analyzer
        processor = create_text_processor(language=language)
        analyzer = KeywordAnalyzer(
            llm=self.llm,  # Pass LLM instance
            config={"weights": {"statistical": 1.0, "llm": 0.0}},  # Statistical only
            language_processor=processor
        )
        
        return await self.analyze_text(text, language, analyzer)

    async def test_llm_analysis(self, text: str, language: str = None) -> Dict[str, Any]:
        """Test LLM-based keyword extraction."""
        if language is None:
            from langdetect import detect
            try:
                language = detect(text)
            except:
                language = "en"
        
        analyzer = KeywordAnalyzer(
            llm=self.llm,  # Pass LLM instance
            config={"weights": {"statistical": 0.0, "llm": 1.0}},  # LLM only
            language_processor=create_text_processor(language=language)
        )
        
        return await self.analyze_text(text, language, analyzer)

    async def test_combined_analysis(self, text: str, language: str = None) -> Dict[str, Any]:
        """Test combined statistical and LLM analysis."""
        if language is None:
            from langdetect import detect
            try:
                language = detect(text)
            except:
                language = "en"
        
        analyzer = KeywordAnalyzer(
            llm=self.llm,  # Pass LLM instance
            config={
                "weights": {"statistical": 0.4, "llm": 0.6},
                "max_keywords": 8,
                "min_confidence": 0.3
            },
            language_processor=create_text_processor(language=language)
        )
        
        return await self.analyze_text(text, language, analyzer)

    def display_keyword_results(self, results: Dict[str, Any]) -> None:
        """Display keyword analysis results."""
        print("\nKeyword Analysis Results:")
        print("-" * 50)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
            
        if "keywords" in results:
            print("\nKeywords:", results["keywords"])
            
        if "domain_keywords" in results:
            print("\nDomain Keywords:")
            for domain, keywords in results["domain_keywords"].items():
                print(f"{domain}: {keywords}")
class ThemeTester(BaseTester):
    """Helper class for testing theme analysis."""
    
    async def test_theme_analysis(self, text: str, language: str = None) -> Dict[str, Any]:
        """Test theme analysis on text."""
        if language is None:
            from langdetect import detect
            try:
                language = detect(text)
            except:
                language = "en"
        
        analyzer = ThemeAnalyzer(
            llm=self.llm,
            config={
                "max_themes": 3,
                "min_confidence": 0.3,
                "focus_areas": "business,technical"
            }
        )
        
        return await self.analyze_text(text, language, analyzer)
        
    def display_theme_results(self, results: Any) -> None:
        """Display theme analysis results.
        
        Args:
            results: Either a dict or ThemeOutput model
        """
        print("\nTheme Analysis Results:")
        print("-" * 50)
        
        # Convert to dict if it's a pydantic model
        if hasattr(results, "model_dump"):
            results = results.model_dump()
        elif hasattr(results, "dict"):
            results = results.dict()
            
        # Handle error case
        if isinstance(results, dict) and "error" in results:
            print(f"Error: {results['error']}")
            return

        # Access theme data
        themes_data = results.get("themes", {})
        if isinstance(themes_data, dict):
            themes = themes_data.get("themes", [])
            descriptions = themes_data.get("theme_descriptions", {})
            confidence = themes_data.get("theme_confidence", {})
            keywords = themes_data.get("related_keywords", {})
        else:
            themes = []
            descriptions = {}
            confidence = {}
            keywords = {}
            
        # Display themes
        if not themes:
            print("No themes found.")
            return
            
        for theme in themes:
            print(f"\nTheme: {theme}")
            print(f"Description: {descriptions.get(theme, 'No description available')}")
            print(f"Confidence: {confidence.get(theme, 0):.2f}")
            theme_keywords = keywords.get(theme, [])
            if theme_keywords:
                print(f"Keywords: {', '.join(theme_keywords)}")


class CategoryTester(BaseTester):
    """Helper class for testing category analysis."""
    
    def __init__(self, parameter_file: Optional[str] = None):
        """Initialize with optional parameter file path."""
        super().__init__()
        
        # Load categories from parameters if file provided
        if parameter_file:
            parameter_adapter = ParameterAdapter(parameter_file)
            params = parameter_adapter.load_and_convert()
            self.categories = params.categories
        else:
            # Default categories for testing
            self.categories = {
                "technical_content": CategoryConfig(
                    description="Technical and software development content",
                    keywords=["software", "development", "api", "programming", "technical"],
                    threshold=0.6
                ),
                "business_content": CategoryConfig(
                    description="Business and financial content",
                    keywords=["revenue", "sales", "market", "growth", "business"],
                    threshold=0.6
                ),
                "educational_content": CategoryConfig(
                    description="Educational and learning content",
                    keywords=["learning", "education", "training", "teaching"],
                    threshold=0.5
                )
            }
    
    async def test_category_analysis(
        self, 
        text: str, 
        language: str = None,
        min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """Test category analysis on text."""
        if language is None:
            from langdetect import detect
            try:
                language = detect(text)
            except:
                language = "en"
        
        # Create processor for language
        processor = create_text_processor(language=language)
        
        analyzer = CategoryAnalyzer(
            categories=self.categories,
            llm=self.llm,
            config={
                "min_confidence": min_confidence
            },
            language_processor=processor
        )
        
        return await self.analyze_text(text, language, analyzer)
        
    def display_category_results(self, results: Any) -> None:
        """Display category analysis results.
        
        Args:
            results: Either a dict or CategoryOutput model
        """
        print("\nCategory Analysis Results:")
        print("-" * 50)
        
        # Convert to dict if needed
        if hasattr(results, "model_dump"):
            results = results.model_dump()
        elif hasattr(results, "dict"):
            results = results.dict()
            
        # Handle error case
        if isinstance(results, dict) and "error" in results:
            print(f"Error: {results['error']}")
            return

        # Access category data
        categories = results.get("categories", [])
        explanations = results.get("explanations", {})
        evidence = results.get("evidence", {})
        
        if not categories:
            print("No matching categories found.")
            return
            
        # Display results
        for category in categories:
            name = category.get("name", "")
            confidence = category.get("confidence", 0.0)
            print(f"\nCategory: {name}")
            print(f"Confidence: {confidence:.2f}")
            
            # Show explanation
            if name in explanations:
                print(f"Explanation: {explanations[name]}")
                
            # Show evidence
            if name in evidence:
                print("Evidence:")
                for item in evidence[name]:
                    print(f"- {item}")
## Pipeline class
class AnalysisPipeline:
    """Complete analysis pipeline for testing multiple analyzers."""
    
    def __init__(self, parameter_file: Optional[str] = None):
        self.file_utils = FileUtils()
        self.keyword_tester = KeywordTester()
        self.theme_tester = ThemeTester()
        self.category_tester = CategoryTester(parameter_file)
        
    async def analyze_text(self, text: str, language: str = None) -> Dict[str, Any]:
        """Run complete analysis pipeline on text."""
        if language is None:
            from langdetect import detect
            try:
                language = detect(text)
            except:
                language = "en"
        
        # Run analyses
        keyword_results = await self.keyword_tester.test_combined_analysis(
            text, language=language
        )
        theme_results = await self.theme_tester.test_theme_analysis(
            text, language=language
        )
        category_results = await self.category_tester.test_category_analysis(
            text, language=language
        )
        
        return {
            "keywords": keyword_results,
            "themes": theme_results,
            "categories": category_results,
            "language": language
        }
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display complete analysis results."""
        print("\nComplete Analysis Results")
        print("=" * 50)
        
        # Display keyword results
        print("\nKeyword Analysis:")
        print("-" * 20)
        self.keyword_tester.display_keyword_results(results.get("keywords", {}))
        
        # Display theme results
        print("\nTheme Analysis:")
        print("-" * 20)
        self.theme_tester.display_theme_results(results.get("themes", {}))
        
        # Display category results
        print("\nCategory Analysis:")  
        print("-" * 20)
        self.category_tester.display_category_results(results.get("categories", {}))
## Analysis functions
async def run_analysis_examples():
    """Run example analyses on different content types."""
    # Configure root logger to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    # Example texts
    example_texts = {
        "Business Analysis": """
            Q3 revenue increased by 15% with strong growth in enterprise sales.
            Customer retention improved while acquisition costs decreased.
            New market expansion initiatives are showing positive early results.
        """,
        
        "Technical Content": """
            The application uses microservices architecture with containerized deployments.
            Data processing pipeline incorporates machine learning models for prediction.
            System monitoring ensures high availability and performance metrics.
        """,
        
        "Mixed Content": """
            The IT department's cloud migration project reduced infrastructure costs by 25%.
            DevOps implementation improved deployment frequency while maintaining quality.
            Monthly recurring revenue from SaaS products grew steadily.
        """
    }
    
    # Initialize pipeline with error handling for each analyzer
    pipeline = AnalysisPipeline()
    
    for title, text in example_texts.items():
        print(f"\nAnalyzing {title}")
        print("=" * 50)
        
        try:
            # Create analyzers with proper error handling
            processor = create_text_processor()
            
            # 1. Keyword Analysis
            print("\n1. Keyword Analysis")
            try:
                keyword_analyzer = KeywordAnalyzer(
                    llm=pipeline.keyword_tester.llm,
                    config={
                        "weights": {"statistical": 0.4, "llm": 0.6},
                        "max_keywords": 8,
                        "min_confidence": 0.3
                    },
                    language_processor=processor
                )
                keyword_results = await keyword_analyzer.analyze(text)
                print("✓ Complete")
                
                if hasattr(keyword_results, 'keywords') and keyword_results.keywords:
                    print("\nKeywords Found:")
                    for kw in keyword_results.keywords:
                        bar = "█" * int(kw.score * 20) + "░" * (20 - int(kw.score * 20))
                        print(f"  • {kw.keyword:<20} [{bar}] ({kw.score:.2f})")
                else:
                    print("No keywords found")
                    
            except Exception as e:
                print(f"✗ Keyword analysis failed: {str(e)}")
                keyword_results = None
            
            # 2. Theme Analysis
            print("\n2. Theme Analysis")
            try:
                theme_results = await pipeline.theme_tester.test_theme_analysis(text)
                print("✓ Complete")
                
                if hasattr(theme_results, 'themes') and theme_results.themes:
                    print("\nThemes Found:")
                    for theme in theme_results.themes:
                        print(f"\n  • {theme.name} ({theme.confidence:.2f})")
                        print(f"    {theme.description}")
                        if theme.keywords:
                            print(f"    Keywords: {', '.join(theme.keywords)}")
                else:
                    print("No themes found")
                    
            except Exception as e:
                print(f"✗ Theme analysis failed: {str(e)}")
                theme_results = None
            
            # 3. Category Analysis
            print("\n3. Category Analysis")
            try:
                category_results = await pipeline.category_tester.test_category_analysis(text)
                print("✓ Complete")
                
                if hasattr(category_results, 'categories') and category_results.categories:
                    print("\nCategories Found:")
                    for cat in category_results.categories:
                        bar = "█" * int(cat.confidence * 20) + "░" * (20 - int(cat.confidence * 20))
                        print(f"\n  • {cat.name}")
                        print(f"    Confidence: [{bar}] ({cat.confidence:.2f})")
                        print(f"    {cat.explanation}")
                        if cat.evidence:
                            print("    Evidence:")
                            for ev in cat.evidence:
                                print(f"      - {ev}")
                else:
                    print("No categories found")
                    
            except Exception as e:
                print(f"✗ Category analysis failed: {str(e)}")
                category_results = None
            
            print("\n" + "-" * 80 + "\n")
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}\n")
            continue

# Usage:
# await run_analysis_examples()
async def analyze_custom_text(
    text: str, 
    parameter_file: Optional[str] = None,
    detailed_output: bool = True,
    timing_info: bool = True
) -> Dict[str, Any]:
    """Analyze custom text with all analyzers.
    
    Args:
        text: Text to analyze
        parameter_file: Optional path to parameter file
        detailed_output: Whether to show detailed analysis output
        timing_info: Whether to show timing information
    """
    # Configure root logger to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    print("\nComplete Analysis Pipeline")
    print("=" * 50)
    
    print("\nInput Text:")
    print("-" * 20)
    print(text.strip())
    
    try:
        pipeline = AnalysisPipeline(parameter_file)
        
        print("\nRunning Analysis...")
        print("-" * 20)
        
        # Track timing for each step
        import time
        total_start = time.time()
        results = {}
        timings = {}
        
        # Keyword Analysis
        print("\n1. Keyword Analysis")
        start = time.time()
        try:
            processor = create_text_processor()
            analyzer = KeywordAnalyzer(
                llm=pipeline.keyword_tester.llm,
                config={
                    "weights": {"statistical": 0.4, "llm": 0.6},
                    "max_keywords": 8,
                    "min_confidence": 0.3
                },
                language_processor=processor
            )
            keyword_results = await analyzer.analyze(text)
            print("✓ Complete")
            
            print("\nKeywords Found:")
            if keyword_results.keywords:
                for kw in keyword_results.keywords:
                    bar = "█" * int(kw.score * 20) + "░" * (20 - int(kw.score * 20))
                    print(f"  • {kw.keyword:<20} [{bar}] ({kw.score:.2f})")
            if keyword_results.domain_keywords:
                print("\nDomain Keywords:")
                for domain, keywords in keyword_results.domain_keywords.items():
                    print(f"  {domain}: {', '.join(keywords)}")
                    
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            keyword_results = KeywordAnalysisResult(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language="unknown",
                success=False,
                error=str(e)
            )
        
        # Theme Analysis
        print("\n2. Theme Analysis")
        start = time.time()
        try:
            theme_results = await pipeline.theme_tester.test_theme_analysis(text)
            print("✓ Complete")
            
            if detailed_output and hasattr(theme_results, 'themes'):
                confidence_scores = []
                print("\nThemes Found:")
                for theme in theme_results.themes:
                    confidence_scores.append(theme.confidence)
                    print(f"\n  • {theme.name} ({theme.confidence:.2f})")
                    print(f"    {theme.description}")
                    if theme.keywords:
                        print(f"    Keywords: {', '.join(theme.keywords)}")
                if confidence_scores:
                    print(f"\n  Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
                        
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            theme_results = {"error": str(e)}
        timings['themes'] = time.time() - start
        if timing_info:
            print(f"Time: {timings['themes']:.2f}s")
        
        # Category Analysis
        print("\n3. Category Analysis")
        start = time.time()
        try:
            category_results = await pipeline.category_tester.test_category_analysis(text)
            print("✓ Complete")
            
            if detailed_output and hasattr(category_results, 'categories'):
                confidence_scores = []
                print("\nCategories Found:")
                for cat in category_results.categories:
                    confidence_scores.append(cat.confidence)
                    bar = "█" * int(cat.confidence * 20) + "░" * (20 - int(cat.confidence * 20))
                    print(f"\n  • {cat.name}")
                    print(f"    Confidence: [{bar}] ({cat.confidence:.2f})")
                    print(f"    {cat.explanation}")
                    if cat.evidence:
                        print("    Evidence:")
                        for ev in cat.evidence:
                            print(f"      - {ev}")
                if confidence_scores:
                    print(f"\n  Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
                        
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            category_results = {"error": str(e)}
        timings['categories'] = time.time() - start
        if timing_info:
            print(f"Time: {timings['categories']:.2f}s")
        
        total_time = time.time() - total_start
        results = {
            "keywords": keyword_results,
            "themes": theme_results,
            "categories": category_results,
            "language": "en",
            "timings": timings,
            "total_time": total_time
        }
        
        print("\nAnalysis Complete")
        print("=" * 50)
        if timing_info:
            print(f"Total time: {total_time:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"\nPipeline Error: {str(e)}")
        return {
            "error": str(e),
            "keywords": {"error": str(e)},
            "themes": {"error": str(e)},
            "categories": {"error": str(e)}
        }
async def analyze_excel_content(
    input_file: str,
    output_file: str,
    content_column: str = "content",
    parameter_file: Optional[str] = None
) -> None:
    """Analyze content from Excel file and save results to new Excel file.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to save results
        content_column: Name of the column containing text to analyze
        parameter_file: Optional path to parameter file
    """
    # Configure logging
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Initialize components
        file_utils = FileUtils()
        pipeline = AnalysisPipeline(parameter_file)
        
        print(f"\nAnalyzing content from: {input_file}")
        print("=" * 50)
        
        # Load input data
        df = file_utils.load_single_file(input_file)
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in input file")
            
        # Prepare results storage
        results = []
        total = len(df)
        
        # Process each text
        for idx, row in df.iterrows():
            text = str(row[content_column])
            print(f"\nProcessing text {idx + 1}/{total}")
            
            try:
                # Run analysis
                analysis = await pipeline.analyze_text(text)
                
                # Extract keywords
                keywords = []
                if "keywords" in analysis and hasattr(analysis["keywords"], "keywords"):
                    keywords = [k.keyword for k in analysis["keywords"].keywords]
                
                # Extract categories
                categories = []
                if "categories" in analysis and hasattr(analysis["categories"], "categories"):
                    categories = [c.name for c in analysis["categories"].categories]
                
                # Extract themes
                themes = []
                if "themes" in analysis and hasattr(analysis["themes"], "themes"):
                    themes = [t.name for t in analysis["themes"].themes]
                
                # Store results
                results.append({
                    "content": text,
                    "keywords": ", ".join(keywords) if keywords else "",
                    "categories": ", ".join(categories) if categories else "",
                    "themes": ", ".join(themes) if themes else ""
                })
                
                print("✓ Analysis complete")
                
            except Exception as e:
                print(f"✗ Analysis failed: {str(e)}")
                results.append({
                    "content": text,
                    "keywords": "",
                    "categories": "",
                    "themes": ""
                })
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Save to Excel
        file_utils.save_data_to_disk(
            data={"Analysis Results": output_df},
            output_filetype="xlsx",
            file_name=output_file
        )
        
        print("\nAnalysis Complete")
        print(f"Results saved to: {output_file}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Process failed: {str(e)}")

# Example usage:
# await analyze_excel_content(
#     input_file="input_texts.xlsx",
#     output_file="analysis_results",
#     content_column="content"
# )
### Debug functions
async def debug_theme_analysis(text: str):
    """Debug theme analysis with detailed output."""
    print("\nDebug Theme Analysis")
    print("=" * 50)
    
    print("\nInput Text:")
    print("-" * 20)
    print(text.strip())
    
    # Configure logging
    logger = logging.getLogger("src.analyzers.theme_analyzer")
    logger.setLevel(logging.DEBUG)
    
    # Add handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    print("\nRunning Analysis...")
    print("-" * 20)
    
    # Run analysis
    theme_tester = ThemeTester()
    results = await theme_tester.test_theme_analysis(text)
    
    # Display results
    theme_tester.display_theme_results(results)
    
    if logger.isEnabledFor(logging.DEBUG):
        print("\nDebug Information:")
        print("-" * 20)
        if hasattr(results, "model_dump"):
            print(json.dumps(results.model_dump(), indent=2))
        else:
            print(json.dumps(results, indent=2))
    
    return results


async def debug_category_analysis(text: str, parameter_file: Optional[str] = None):
    """Debug category analysis with detailed output and visualizations."""
    print("\nDebug Category Analysis")
    print("=" * 50)
    
    # Show input text
    print("\nInput Text:")
    print("-" * 20)
    print(text.strip())
    
    # Configure logging
    logger = logging.getLogger("src.analyzers.category_analyzer")
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    try:
        # Initialize analyzer
        category_tester = CategoryTester(parameter_file)
        
        # Show configured categories with hierarchy
        print("\nConfigured Categories:")
        print("-" * 20)
        for name, config in category_tester.categories.items():
            indent = "  " if config.parent else ""
            print(f"\n{indent}{name}:")
            print(f"{indent}  Description: {config.description}")
            print(f"{indent}  Keywords: {', '.join(config.keywords)}")
            print(f"{indent}  Threshold: {config.threshold}")
            if config.parent:
                print(f"{indent}  Parent: {config.parent}")
        
        print("\nRunning Analysis...")
        print("-" * 20)
        
        # Run analysis
        results = await category_tester.test_category_analysis(text)
        
        # Display formatted results
        print("\nAnalysis Results:")
        print("-" * 20)
        
        if results.error:
            print(f"Error: {results.error}")
            return results
            
        # Show matched categories with details and visualization
        categories = results.categories if isinstance(results.categories, list) else []
        
        if not categories:
            print("No categories matched the confidence threshold.")
        else:
            print(f"\nMatched {len(categories)} categories:")
            
            # Sort categories by confidence
            categories.sort(key=lambda x: x.confidence, reverse=True)
            
            for category in categories:
                # Create confidence bar visualization
                bar_length = 40
                filled = int(category.confidence * bar_length)
                confidence_bar = "█" * filled + "░" * (bar_length - filled)
                
                print(f"\n{category.name}")
                print(f"Confidence: [{confidence_bar}] {category.confidence:.2%}")
                print("  " + "-" * 40)
                print(f"  Explanation: {results.explanations.get(category.name, 'No explanation')}")
                
                if evidence := results.evidence.get(category.name, []):
                    print("\n  Evidence:")
                    for idx, item in enumerate(evidence, 1):
                        print(f"    {idx}. {item}")
                        
                if hasattr(category, 'themes') and category.themes:
                    print("\n  Related Themes:")
                    for theme in category.themes:
                        print(f"    • {theme}")
                        
                # Show keyword matches if available
                if category.name in category_tester.categories:
                    config = category_tester.categories[category.name]
                    matched_keywords = [
                        kw for kw in config.keywords 
                        if kw.lower() in text.lower()
                    ]
                    if matched_keywords:
                        print("\n  Matched Keywords:")
                        print(f"    {', '.join(matched_keywords)}")
        
        # Add confidence threshold summary
        print("\nConfidence Summary:")
        print("-" * 20)
        thresholds = {
            "High (>0.8)": len([c for c in categories if c.confidence > 0.8]),
            "Medium (0.6-0.8)": len([c for c in categories if 0.6 <= c.confidence <= 0.8]),
            "Low (0.3-0.6)": len([c for c in categories if 0.3 <= c.confidence < 0.6])
        }
        for level, count in thresholds.items():
            print(f"{level}: {count} categories")
        
        # Show complete debug output
        print("\nRaw Analysis Data:")
        print("-" * 20)
        if hasattr(results, "model_dump"):
            print(json.dumps(results.model_dump(), indent=2))
        else:
            print(json.dumps(results, indent=2))
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return CategoryOutput(
            error=str(e),
            success=False,
            language="unknown",
            categories=[],
            explanations={},
            evidence={}
        )


async def test_category_analysis():
    """Test category analysis with educational content."""
    text = """
    The online learning platform features interactive modules and self-paced progress tracking.
    Virtual classrooms enable real-time collaboration between students and instructors.
    Digital assessment tools provide immediate feedback on learning outcomes.
    """
    
    # Create test categories
    categories = {
        "online_learning": CategoryConfig(
            description="Online and e-learning content",
            keywords=["online", "virtual", "digital", "platform", "e-learning"],
            threshold=0.5
        ),
        "in_person_learning": CategoryConfig(
            description="Traditional classroom learning",
            keywords=["classroom", "face-to-face", "physical", "workshop"],
            threshold=0.5
        ),
        "assessment": CategoryConfig(
            description="Learning assessment and feedback",
            keywords=["assessment", "feedback", "tracking", "progress", "outcomes"],
            threshold=0.5
        )
    }
    
    # Run analysis with debug output
    results = await debug_category_analysis(text)
    return results
# # implement custom text analysis
# async def analyze_custom_text(text: str, language: str = None):
#     pass
## Setup and verify environment
def verify_environment() -> bool:
    """Verify notebook environment setup."""
    from dotenv import load_dotenv
    
    # Load environment variables
    env_path = Path(project_root) / ".env"
    env_loaded = load_dotenv(env_path)
    
    # Required variables
    required_env_vars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
    ]
    
    # Basic checks
    basic_checks = {
        "Project root in path": project_root in sys.path,
        "Can import src": "src" in sys.modules,
        "FileUtils initialized": hasattr(file_utils, "project_root"),
        ".env file loaded": env_loaded,
    }
    
    # Environment variable checks
    env_var_checks = {
        f"{var} set": os.getenv(var) is not None
        for var in required_env_vars
    }
    
    # Path checks
    expected_paths = {
        "Raw data": file_utils.get_data_path("raw"),
        "Processed data": file_utils.get_data_path("processed"),
        "Configuration": file_utils.get_data_path("configurations"),
        "Main config.yaml": Path(project_root) / "config.yaml"
    }
    
    path_checks = {
        f"{name} exists": path.exists()
        for name, path in expected_paths.items()
    }
    
    # Combine all checks
    all_checks = {
        **basic_checks,
        **env_var_checks,
        **path_checks
    }
    
    print("Environment Check Results:")
    print("=" * 50)
    print()
    
    # Print Basic Setup section
    print("Basic Setup:")
    print("-" * 11)
    for check, result in basic_checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")
    
    # Print Environment Variables section
    print("\nEnvironment Variables:")
    print("-" * 21)
    for check, result in env_var_checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")
    
    # Print Project Structure section
    print("\nProject Structure:")
    print("-" * 17)
    for check, result in path_checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")
    
    # Overall status
    all_passed = all(all_checks.values())
    print("\n" + "=" * 50)
    print("Environment Status:", "Ready ✓" if all_passed else "Setup needed ✗")
    
    # Print setup instructions if needed
    if not all_passed:
        print("\nSetup Instructions:")
        if not env_loaded:
            print("- Create a .env file in the project root with required API keys")
        for var in required_env_vars:
            if not os.getenv(var):
                print(f"- Add {var} to your .env file")
        for name, path in expected_paths.items():
            if not path.exists():
                print(f"- Create {name} directory at {path}")
    
    return all_passed
## Run pipeline
# First cell: Verify environment
verify_environment()
### Run single analysis with debug

text = """
Q3 revenue increased by 15% with strong growth in enterprise sales.
Customer retention improved while acquisition costs decreased.
New market expansion initiatives are showing positive early results.
"""
# analyze with debug output
#### uncomment the line below to run the test
await debug_theme_analysis(text)

# analyze specific text
# await analyze_custom_text(text) # implement this function
text = """
The online learning platform features interactive modules and self-paced progress tracking.
Virtual classrooms enable real-time collaboration between students and instructors.
Digital assessment tools provide immediate feedback on learning outcomes.
"""

# Debug category analysis
# results = await debug_category_analysis(text)

# Usage example:
#### uncomment the line below to run the test
# await test_category_analysis()
### Run complete analysis examples

await run_analysis_examples()
# Or analyze with complete pipeline
results = await analyze_custom_text(text)
# Or analyze specific text:
# text = """Your text here..."""
# await analyze_custom_text(text)
# Process your Excel file

await analyze_excel_content(
    input_file="test_content_en.xlsx",
    output_file="analysis_results",
    content_column="content"  # Change this to match your column name
)