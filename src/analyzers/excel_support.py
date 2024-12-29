# src/analyzers/excel_support.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.analyzers.theme_analyzer import ThemeAnalyzer
from src.analyzers.category_analyzer import CategoryAnalyzer
from src.excel_analysis.base import ExcelAnalysisBase
from src.core.language_processing import create_text_processor
from src.schemas import (
    KeywordAnalysisResult,
    ThemeAnalysisResult,
    CategoryAnalysisResult,
)
import asyncio
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class ExcelKeywordAnalyzer(KeywordAnalyzer, ExcelAnalysisBase):
    """Keyword analyzer with Excel support."""

    def __init__(
        self,
        content_file: Union[str, Path, pd.DataFrame],
        parameter_file: Union[str, Path],
        llm: Optional[BaseChatModel] = None,
        content_column: str = "content",
        file_utils: Optional[FileUtils] = None,
        **kwargs,
    ):
        """Initialize with Excel support."""
        # Initialize Excel base first
        ExcelAnalysisBase.__init__(
            self,
            content_file=content_file,
            parameter_file=parameter_file,
            content_column=content_column,
            file_utils=file_utils,
            **kwargs,
        )

        # Get analyzer config from parameters
        config = self.parameters.get_analyzer_config("keywords")

        # Create language processor
        language_processor = create_text_processor(
            language=config["language"], config=config
        )

        # Initialize keyword analyzer with config
        KeywordAnalyzer.__init__(
            self, llm=llm, config=config, language_processor=language_processor
        )

    async def analyze_excel(
        self, batch_size: int = 10, **kwargs
    ) -> pd.DataFrame:
        """Analyze Excel content with keyword extraction.

        Args:
            batch_size: Size of processing batches
            **kwargs: Additional analysis parameters

        Returns:
            DataFrame with analysis results
        """
        logger.info(f"Starting keyword analysis on {len(self.content)} rows")

        # Get texts from content
        texts = self._get_texts()
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # Process batch
                batch_results = await asyncio.gather(
                    *[self.analyze(text, **kwargs) for text in batch]
                )
                results.extend(batch_results)
                logger.debug(f"Processed batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                results.extend([None] * len(batch))

        # Create results DataFrame
        result_df = self.content.copy()

        # Add keyword columns
        result_df["keywords"] = [
            (
                ", ".join([kw.keyword for kw in r.keywords])
                if r and r.keywords
                else ""
            )
            for r in results
        ]
        result_df["keyword_scores"] = [
            (
                ", ".join([f"{kw.score:.2f}" for kw in r.keywords])
                if r and r.keywords
                else ""
            )
            for r in results
        ]
        result_df["keyword_domains"] = [
            (
                ", ".join([kw.domain for kw in r.keywords if kw.domain])
                if r and r.keywords
                else ""
            )
            for r in results
        ]

        logger.info("Keyword analysis complete")
        return result_df

    async def _process_batch(
        self, batch: List[str], **kwargs
    ) -> List[Any]:
        """Process batch of texts with keyword analysis.

        Args:
            batch: List of texts to analyze
            **kwargs: Additional analysis parameters

        Returns:
            List of keyword analysis results
        """
        # Filter out unsupported kwargs
        analysis_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['show_progress', 'batch_size']}
        
        return await asyncio.gather(
            *[self.analyze(text, **analysis_kwargs) for text in batch]
        )


class ExcelThemeAnalyzer(ThemeAnalyzer, ExcelAnalysisBase):
    """Theme analyzer with Excel support."""

    def __init__(
        self,
        content_file: Union[str, Path, pd.DataFrame],
        parameter_file: Union[str, Path],
        llm: Optional[BaseChatModel] = None,
        content_column: str = "content",
        file_utils: Optional[FileUtils] = None,
        **kwargs,
    ):
        """Initialize with Excel support."""
        # Initialize Excel base
        ExcelAnalysisBase.__init__(
            self,
            content_file=content_file,
            parameter_file=parameter_file,
            content_column=content_column,
            file_utils=file_utils,
            **kwargs,
        )

        # Get analyzer config
        config = self.parameters.get_analyzer_config("themes")

        # Create language processor
        language_processor = create_text_processor(
            language=config["language"], config=config
        )

        # Initialize theme analyzer
        ThemeAnalyzer.__init__(
            self, llm=llm, config=config, language_processor=language_processor
        )

    async def analyze_excel(
        self, batch_size: int = 10, **kwargs
    ) -> pd.DataFrame:
        """Analyze Excel content for themes.

        Args:
            batch_size: Size of processing batches
            **kwargs: Additional analysis parameters

        Returns:
            DataFrame with theme analysis results
        """
        logger.info(f"Starting theme analysis on {len(self.content)} rows")

        # Get texts from content
        texts = self._get_texts()
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # Process batch
                batch_results = await asyncio.gather(
                    *[self.analyze(text, **kwargs) for text in batch]
                )
                results.extend(batch_results)
                logger.debug(f"Processed batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                results.extend([None] * len(batch))

        # Create results DataFrame
        result_df = self.content.copy()

        # Add theme columns
        result_df["themes"] = [
            (
                ", ".join([theme.name for theme in r.themes])
                if r and r.themes
                else ""
            )
            for r in results
        ]
        result_df["theme_descriptions"] = [
            (
                "; ".join([theme.description for theme in r.themes])
                if r and r.themes
                else ""
            )
            for r in results
        ]
        result_df["theme_confidence"] = [
            (
                ", ".join([f"{theme.confidence:.2f}" for theme in r.themes])
                if r and r.themes
                else ""
            )
            for r in results
        ]

        logger.info("Theme analysis complete")
        return result_df

    async def _process_batch(
        self, batch: List[str], **kwargs
    ) -> List[Any]:
        """Process batch of texts with theme analysis.

        Args:
            batch: List of texts to analyze
            **kwargs: Additional analysis parameters

        Returns:
            List of theme analysis results
        """
        # Filter out unsupported kwargs
        analysis_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['show_progress', 'batch_size']}
        
        return await asyncio.gather(
            *[self.analyze(text, **analysis_kwargs) for text in batch]
        )


class ExcelCategoryAnalyzer(CategoryAnalyzer, ExcelAnalysisBase):
    """Category analyzer with Excel support."""

    def __init__(
        self,
        content_file: Union[str, Path, pd.DataFrame],
        parameter_file: Union[str, Path],
        llm: Optional[BaseChatModel] = None,
        content_column: str = "content",
        file_utils: Optional[FileUtils] = None,
        **kwargs,
    ):
        """Initialize with Excel support."""
        # Initialize Excel base
        ExcelAnalysisBase.__init__(
            self,
            content_file=content_file,
            parameter_file=parameter_file,
            content_column=content_column,
            file_utils=file_utils,
            **kwargs,
        )

        # Get analyzer config
        config = self.parameters.get_analyzer_config("categories")

        # Create language processor
        language_processor = create_text_processor(
            language=config["language"], config=config
        )

        # Initialize category analyzer with categories from config
        CategoryAnalyzer.__init__(
            self,
            llm=llm,
            config=config,
            categories=config.get("categories", {}),
            language_processor=language_processor,
        )

    async def analyze_excel(
        self,
        batch_size: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Run category analysis on Excel content."""
        try:
            # Process all content
            results = []
            texts = self.content[self.content_column].tolist()

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_results = []

                for text in batch_texts:
                    try:
                        result = await self.analyze(text, **kwargs)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error analyzing text: {e}")
                        batch_results.append(None)

                results.extend(batch_results)

            # Create results DataFrame
            result_df = self.content.copy()

            # Format categories with scores
            result_df["categories"] = [
                (
                    ", ".join(
                        f"{cat.name} ({cat.confidence:.2f})"
                        for cat in r.categories
                    )  # Change from r.matches to r.categories
                    if r and hasattr(r, "categories") and r.categories
                    else ""
                )
                for r in results
            ]

            return result_df

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    async def _process_batch(
        self, batch: List[str], **kwargs
    ) -> List[Any]:
        """Process batch of texts with category analysis.

        Args:
            batch: List of texts to analyze
            **kwargs: Additional analysis parameters

        Returns:
            List of category analysis results
        """
        # Filter out unsupported kwargs
        analysis_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['show_progress', 'batch_size']}
        
        return await asyncio.gather(
            *[self.analyze(text, **analysis_kwargs) for text in batch]
        )
