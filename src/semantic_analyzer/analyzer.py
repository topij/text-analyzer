# src/semantic_analyzer/analyzer.py

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging
import asyncio

from src.utils.FileUtils.file_utils import FileUtils
from src.core.config import AnalyzerConfig
from src.core.language_processing.factory import create_text_processor
from src.core.llm.factory import create_llm
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.analyzers.category_analyzer import CategoryAnalyzer
from src.loaders.models import (
    ParameterSet,
    GeneralParameters,
    CategoryConfig
)
from src.loaders.parameter_adapter import ParameterAdapter

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Main interface for semantic text analysis."""
    
    # Class-level constants
    VALID_ANALYSIS_TYPES = {"keywords", "themes", "categories"}
    
    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        file_utils: Optional[FileUtils] = None,
        llm=None,
        language: Optional[str] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        parameter_file: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """Initialize semantic analyzer."""
        # Load parameters first
        self.parameter_adapter = ParameterAdapter(parameter_file)
        self.parameters = self.parameter_adapter.load_and_convert()
        
        # Initialize core components
        self.file_utils = file_utils or FileUtils()
        self.config = config or AnalyzerConfig(self.file_utils)
        
        # Set up language processing
        self.language = language or self.parameters.general.language
        self.text_processor = create_text_processor(
            language=self.language,
            config={
                "predefined_keywords": self.parameters.predefined_keywords,
                "excluded_keywords": self.parameters.excluded_keywords
            }
        )
        
        # Initialize LLM if not provided
        self.llm = llm if llm is not None else create_llm()
        
        # Initialize analyzers with parameters
        # self._init_analyzers(categories)

        self._init_analyzers({
            "education_type": CategoryConfig(
                description="Type of education (online, remote, etc.)",
                keywords=["online", "remote", "classroom"],
                threshold=0.5
            )
        })
        
        logger.info(f"Initialized SemanticAnalyzer with language: {self.language}, "
                   f"using parameter file: {parameter_file}")
        
        # Initialize LLM if not provided
        self.llm = llm if llm is not None else create_llm()
        
        logger.info(f"Initialized SemanticAnalyzer with language: {self.language}, "
                   f"using parameter file: {parameter_file}")
    
    def _init_analyzers(self, categories: Optional[Dict[str, CategoryConfig]] = None) -> None:
        """Initialize analysis components."""
        general = self.parameters.general
        settings = self.parameters.analysis_settings
        
        self.keyword_analyzer = KeywordAnalyzer(
            llm=self.llm,
            config={
                "max_keywords": general.max_keywords,
                "min_keyword_length": general.min_keyword_length,
                "include_compounds": general.include_compounds,
                "excluded_keywords": self.parameters.excluded_keywords,
                "predefined_keywords": self.parameters.predefined_keywords,
                "min_confidence": general.min_confidence,
                "weights": settings.weights.model_dump()
            },
            language_processor=self.text_processor
        )
        
        self.theme_analyzer = ThemeAnalyzer(
            llm=self.llm,
            config={
                "max_themes": general.max_themes,
                "min_confidence": settings.theme_analysis.min_confidence,
                "enabled": settings.theme_analysis.enabled,
                "focus_on": general.focus_on
            }
        )
        
        # Use either provided categories or loaded ones
        # Initialize category analyzer with language processor
        category_config = categories or self.parameters.categories
        if not category_config:
            category_config = {
                "education_type": CategoryConfig(
                    description="Type of education (online, remote, etc.)",
                    keywords=["online", "remote", "classroom"],
                    threshold=0.5
                )
            }
            
        self.category_analyzer = CategoryAnalyzer(
            categories=category_config,
            llm=self.llm,
            config={
                "min_confidence": self.parameters.general.min_confidence,
                "focus_on": self.parameters.general.focus_on
            },
            language_processor=self.text_processor  # Pass language processor
        )
    
    def _validate_analysis_types(self, types: Optional[List[str]] = None) -> List[str]:
        """Validate requested analysis types."""
        if not types:
            return list(self.VALID_ANALYSIS_TYPES)
            
        invalid = set(types) - self.VALID_ANALYSIS_TYPES
        if invalid:
            raise ValueError(f"Invalid analysis types: {invalid}")
            
        return types
    
    def _handle_analysis_error(self, error: Exception) -> Dict[str, Any]:
        """Create error response."""
        logger.error(f"Analysis failed: {str(error)}", exc_info=True)
        return {
            "error": str(error),
            "success": False,
            "language": self.language
        }
    
    async def analyze(self, text: str, analysis_types: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Analyze text with proper output structure."""
        if not text:
            return self._create_error_response("Empty input text")

        try:
            analysis_types = analysis_types or ["keywords", "themes", "categories"]
            analysis_types = self._validate_analysis_types(analysis_types)
            combined_results = {}

            # Process each analysis type
            for analysis_type in analysis_types:
                try:
                    if analysis_type == "keywords":
                        analyzer = self.keyword_analyzer
                        if keyword_params := kwargs.get("keyword_params"):
                            analyzer.config.update(keyword_params)
                    elif analysis_type == "themes":
                        analyzer = self.theme_analyzer
                        if theme_params := kwargs.get("theme_params"):
                            analyzer.config.update(theme_params)
                    elif analysis_type == "categories":
                        analyzer = self.category_analyzer
                        if category_params := kwargs.get("category_params"):
                            analyzer.config.update(category_params)
                    else:
                        continue

                    result = await asyncio.wait_for(
                        analyzer.analyze(text),
                        timeout=10.0
                    )
                    combined_results[analysis_type] = result.dict()[analysis_type]

                except asyncio.TimeoutError:
                    combined_results[analysis_type] = {
                        "error": "Analysis timed out",
                        "success": False,
                        "language": self.language
                    }
                except Exception as e:
                    combined_results[analysis_type] = {
                        "error": str(e),
                        "success": False,
                        "language": self.language
                    }

            return combined_results

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return self._create_error_response(str(e))
    
    async def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 3,
        timeout: float = 30.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Analyze multiple texts with controlled concurrency."""
        results = []
        
        async def process_batch(batch_texts: List[str]) -> List[Dict[str, Any]]:
            tasks = [
                asyncio.create_task(self.analyze(text, **kwargs))
                for text in batch_texts
            ]
            
            batch_results = []
            for task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=timeout)
                    batch_results.append(result)
                except asyncio.TimeoutError:
                    batch_results.append(self._create_error_response("Analysis timed out"))
                except Exception as e:
                    batch_results.append(self._create_error_response(str(e)))
                finally:
                    if not task.done():
                        task.cancel()
            
            return batch_results

        # Process in smaller batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_results = await asyncio.wait_for(
                    process_batch(batch),
                    timeout=timeout * len(batch)
                )
                results.extend(batch_results)
            except asyncio.TimeoutError:
                results.extend([
                    self._create_error_response("Batch processing timed out")
                    for _ in batch
                ])
            except Exception as e:
                results.extend([
                    self._create_error_response(str(e))
                    for _ in batch
                ])

        return results

    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create consistent error response with proper structure."""
        error_data = {
            "error": str(error),
            "success": False,
            "language": self.language,
            "keywords": [],
            "keyword_scores": {},
            "compound_words": [],
            "domain_keywords": {}
        }
        return {
            "keywords": error_data,
            "themes": error_data.copy(),
            "categories": error_data.copy()
        }
    
    def update_parameters(self, **kwargs) -> GeneralParameters:
        """Update analysis parameters."""
        return self.parameter_adapter.update_parameters(**kwargs)
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str,
        output_type: str = "processed"
    ) -> str:
        """Save analysis results."""
        return self.config.save_results(
            results,
            filename,
            output_type=output_type
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Clean up resources
        if hasattr(self, 'llm') and hasattr(self.llm, 'aclose'):
            await self.llm.aclose()