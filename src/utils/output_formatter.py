# src/utils/output_formatter.py

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import logging
import pandas as pd
from pydantic import BaseModel, Field

from FileUtils import FileUtils, OutputFileType

logger = logging.getLogger(__name__)


class OutputDetail(str, Enum):
    """Enum for output detail levels."""

    MINIMAL = "minimal"  # Just core results
    SUMMARY = "summary"  # Results with key metadata
    DETAILED = "detailed"  # Full results with evidence
    DEBUG = "debug"  # Everything including internal states


class BaseColumnFormat(BaseModel):
    """Base configuration for output column formatting."""

    column_name: str
    format_template: str
    included_fields: List[str]
    confidence_threshold: Optional[float] = None
    max_items: Optional[int] = None


class BaseOutputConfig(BaseModel):
    """Base configuration for output formatting."""

    detail_level: OutputDetail = OutputDetail.SUMMARY
    batch_size: int = 10
    include_confidence: bool = True
    max_length: Optional[int] = None


class ExcelOutputConfig(BaseOutputConfig):
    """Configuration for Excel output formatting."""

    keywords_format: BaseColumnFormat = Field(
        default_factory=lambda: BaseColumnFormat(
            column_name="keywords",
            format_template="{keywords} ({confidence})",
            included_fields=["keyword", "confidence"],
            confidence_threshold=0.3,
            max_items=5,
        )
    )
    themes_format: BaseColumnFormat = Field(
        default_factory=lambda: BaseColumnFormat(
            column_name="themes",
            format_template="{name}: {description}",
            included_fields=["name", "description"],
            confidence_threshold=0.5,
            max_items=3,
        )
    )
    categories_format: BaseColumnFormat = Field(
        default_factory=lambda: BaseColumnFormat(
            column_name="categories",
            format_template="{name} ({confidence})",
            included_fields=["name", "confidence"],
            confidence_threshold=0.3,
            max_items=3,
        )
    )


class BaseFormatter(ABC):
    """Abstract base class for output formatters."""

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config: Optional[BaseOutputConfig] = None,
    ):
        """Initialize formatter with FileUtils and config."""
        self.file_utils = file_utils or FileUtils()
        self.config = config or BaseOutputConfig()

    @abstractmethod
    def format_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Any:
        """Format analysis results."""
        pass

    def _truncate_text(
        self, text: str, max_length: Optional[int] = None
    ) -> str:
        """Truncate text with ellipsis if needed."""
        max_len = max_length or self.config.max_length
        if max_len and len(text) > max_len:
            return f"{text[:max_len]}..."
        return text


class ResultFormatter(BaseFormatter):
    """Base formatter for individual result types."""

    def format_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Dict[str, str]:
        """Implement abstract method from BaseFormatter."""
        if not results or not analysis_types:
            return {}

        # Take the first analysis type as this is a single-type formatter
        analysis_type = analysis_types[0]
        if analysis_type not in results:
            return {}

        result = results[analysis_type]
        format_config = getattr(self.config, f"{analysis_type}_format", None)
        if not format_config:
            return {}

        formatted = self.format_result(
            result, format_config, self.config.detail_level
        )
        return {analysis_type: formatted}

    def format_result(
        self,
        result: Any,
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> str:
        """Format a single analysis result."""
        if not result or not getattr(result, "success", True):
            return self._format_error(result)

        items = self._get_items(result)
        items = self._filter_items(items, format_config)
        formatted_items = self._format_items(items, format_config, detail_level)

        return "; ".join(formatted_items)

    @abstractmethod
    def _get_items(self, result: Any) -> List[Any]:
        """Get items from result."""
        pass

    def _filter_items(
        self, items: List[Any], format_config: BaseColumnFormat
    ) -> List[Any]:
        """Filter items based on confidence and max items."""
        if format_config.confidence_threshold:
            items = [
                item
                for item in items
                if getattr(item, "confidence", 0)
                >= format_config.confidence_threshold
            ]

        items = sorted(
            items, key=lambda x: getattr(x, "confidence", 0), reverse=True
        )

        if format_config.max_items:
            items = items[: format_config.max_items]

        return items

    @abstractmethod
    def _format_items(
        self,
        items: List[Any],
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> List[str]:
        """Format filtered items."""
        pass

    def _format_error(self, result: Any) -> str:
        """Format error message."""
        if hasattr(result, "error") and result.error:
            return f"Error: {result.error}"
        return "Analysis failed"


class KeywordResultFormatter(ResultFormatter):
    """Formatter for keyword analysis results."""

    def _get_items(self, result: Any) -> List[Any]:
        """Get items from keyword result."""
        # Add debug logging
        logger.debug(f"Getting items from keyword result type: {type(result)}")
        logger.debug(f"Result attributes: {dir(result)}")

        items = []
        if hasattr(result, "keywords"):
            items.extend(result.keywords)
        elif isinstance(result, dict) and "keywords" in result:
            items.extend(result["keywords"])
        elif hasattr(result, "matches"):
            items.extend(result.matches)

        logger.debug(f"Found {len(items)} keyword items")
        return items

    def _format_items(
        self,
        items: List[Any],
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> List[str]:
        """Format keyword items with enhanced output."""
        formatted = []
        for item in items:
            try:
                if not item:
                    continue

                # Extract keyword and score
                keyword = getattr(item, "keyword", None) or item.get("keyword")
                score = getattr(item, "score", None) or item.get("score", 0.0)
                domain = getattr(item, "domain", None) or item.get("domain", "")

                if not keyword:
                    continue

                # Format based on detail level
                if detail_level == OutputDetail.MINIMAL:
                    formatted.append(f"{keyword}")
                else:
                    base_info = f"{keyword} ({score:.2f})"
                    if domain:
                        base_info += f" [{domain}]"
                    formatted.append(base_info)

                    # Add compound parts for detailed view
                    if (
                        detail_level == OutputDetail.DETAILED
                        and hasattr(item, "compound_parts")
                        and item.compound_parts
                    ):
                        formatted.append(
                            f"    Parts: {', '.join(item.compound_parts)}"
                        )

            except Exception as e:
                logger.error(f"Error formatting keyword item: {e}")
                continue

        return formatted


class ThemeResultFormatter(ResultFormatter):
    """Formatter for theme analysis results."""

    def _get_items(self, result: Any) -> List[Any]:
        """Get items from theme result."""
        return result.themes if hasattr(result, "themes") else []

    def _format_items(
        self,
        items: List[Any],
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> List[str]:
        formatted = []
        for theme in items:
            base_info = f"{theme.name} ({theme.confidence:.2f})"
            if detail_level >= OutputDetail.SUMMARY:
                base_info += f"\n    Description: {theme.description}"
                if hasattr(theme, "keywords") and theme.keywords:
                    base_info += f"\n    Keywords: {', '.join(theme.keywords)}"
                if hasattr(theme, "domain") and theme.domain:
                    base_info += f"\n    Domain: {theme.domain}"
            formatted.append(base_info)
        return formatted


class CategoryResultFormatter(ResultFormatter):
    """Formatter for category analysis results."""

    def _get_items(self, result: Any) -> List[Any]:
        """Get items from category result."""
        return result.matches if hasattr(result, "matches") else []

    def _format_items(
        self,
        items: List[Any],
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> List[str]:
        formatted = []
        for cat in items:
            if detail_level == OutputDetail.MINIMAL:
                formatted.append(f"{cat.name} ({cat.confidence:.2f})")
            else:
                base_info = [f"{cat.name} ({cat.confidence:.2f})"]

                if hasattr(cat, "description") and cat.description:
                    base_info.append(f"    Description: {cat.description}")

                if detail_level >= OutputDetail.DETAILED:
                    if hasattr(cat, "evidence") and cat.evidence:
                        evidence_texts = [
                            f"    - {e.text}" for e in cat.evidence
                        ]
                        base_info.append("    Evidence:")
                        base_info.extend(evidence_texts)

                    if hasattr(cat, "themes") and cat.themes:
                        base_info.append(f"    Themes: {', '.join(cat.themes)}")

                formatted.append("\n".join(base_info))

        return formatted


class ExcelFormatter(BaseFormatter):
    """Formatter for Excel output."""

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config: Optional[ExcelOutputConfig] = None,
    ):
        super().__init__(file_utils, config or ExcelOutputConfig())
        self.formatters = {
            "keywords": KeywordResultFormatter(file_utils, config),
            "themes": ThemeResultFormatter(file_utils, config),
            "categories": CategoryResultFormatter(file_utils, config),
        }

    def format_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Dict[str, str]:
        """Format analysis results for Excel output."""
        formatted = {}

        for analysis_type in analysis_types:
            if analysis_type not in results:
                logger.debug(
                    f"Analysis type {analysis_type} not found in results"
                )
                continue

            result = results[analysis_type]
            if not result:
                logger.debug(f"No result for {analysis_type}")
                continue

            formatter = self.formatters.get(analysis_type)
            if not formatter:
                logger.warning(f"No formatter found for {analysis_type}")
                continue

            format_config = getattr(
                self.config, f"{analysis_type}_format", None
            )
            if not format_config:
                logger.warning(f"No format config found for {analysis_type}")
                continue

            try:
                # Get items and format them
                items = formatter._get_items(result)
                if not items:
                    logger.debug(f"No items found for {analysis_type}")
                    formatted[analysis_type] = ""
                    continue

                formatted_items = formatter._format_items(
                    items, format_config, self.config.detail_level
                )
                formatted[analysis_type] = "; ".join(formatted_items)

                logger.debug(
                    f"Formatted {len(formatted_items)} items for {analysis_type}"
                )

            except Exception as e:
                logger.error(f"Error formatting {analysis_type}: {e}")
                formatted[analysis_type] = f"Error: {str(e)}"

        return formatted

    async def process_excel_file(
        self,
        input_file: Union[str, Path],
        text_column: str,
        analyzer: Any,
        analysis_types: List[str],
        output_file: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Process Excel file with semantic analysis."""
        # Load input file using FileUtils
        df = self.file_utils.load_single_file(
            file_path=input_file, input_type="raw"
        )

        if text_column not in df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found in input file"
            )

        # Process in batches
        results = []
        for i in range(0, len(df), self.config.batch_size):
            batch = (
                df[text_column].iloc[i : i + self.config.batch_size].tolist()
            )
            batch_results = await analyzer.analyze_batch(batch)

            # Format results for Excel
            formatted_results = [
                self.format_output(result, analysis_types)
                for result in batch_results
            ]
            results.extend(formatted_results)

        # Add results to DataFrame
        for analysis_type in analysis_types:
            column_name = getattr(
                self.config, f"{analysis_type}_format"
            ).column_name
            df[column_name] = [
                result.get(analysis_type, "") for result in results
            ]

        # Save if output file specified
        if output_file:
            self.file_utils.save_data_to_storage(
                data={"Analysis Results": df},
                output_filetype=OutputFileType.XLSX,
                file_name=output_file,
                output_type="processed",
            )

        return df


class DetailedFormatter(ExcelFormatter):
    """Enhanced formatter for detailed output."""

    def _get_metadata(self, result: Any) -> Dict[str, Any]:
        """Get metadata from analysis result with consistent language handling."""
        # Get language from multiple possible sources
        language = None

        # Try getting language from result
        if hasattr(result, "language"):
            language = result.language
        elif isinstance(result, dict) and "language" in result:
            language = result["language"]

        # If no language found, try getting from analyzer's language processor
        if not language and hasattr(self, "language_processor"):
            language = self.language_processor.language

        metadata = {
            "language": language or "unknown",
            "success": getattr(result, "success", False),
            "processing_time": getattr(result, "processing_time", None),
        }

        # Add compound words count if present
        if hasattr(result, "compound_words"):
            metadata["compound_words_count"] = len(result.compound_words)

        # Initialize domain tracking
        domains = set()
        domain_counts = {"technical": 0, "business": 0}

        # Process based on result type
        if isinstance(result, dict):
            # Handle dictionary format
            items = (
                result.get("keywords", [])
                or result.get("themes", [])
                or result.get("categories", [])
            )
        else:
            # Handle object format
            items = (
                getattr(result, "keywords", [])
                or getattr(result, "themes", [])
                or getattr(result, "categories", [])
                or getattr(result, "matches", [])
            )

        for item in items:
            # Extract domain information
            domain = None
            if hasattr(item, "domain"):
                domain = item.domain
            elif isinstance(item, dict):
                domain = item.get("domain")

            # Infer domain from name/description if not explicitly set
            if not domain:
                name = getattr(item, "name", "") or item.get("name", "")
                desc = getattr(item, "description", "") or item.get(
                    "description", ""
                )
                content = f"{name} {desc}".lower()

                if any(
                    term in content
                    for term in [
                        "ai",
                        "machine learning",
                        "technical",
                        "platform",
                        "data",
                    ]
                ):
                    domain = "technical"
                if any(
                    term in content
                    for term in [
                        "business",
                        "metrics",
                        "performance",
                        "customer",
                    ]
                ):
                    domain = "business"

            # Handle multiple domains
            if domain:
                if "/" in domain:
                    for d in domain.split("/"):
                        domains.add(d)
                        if d in domain_counts:
                            domain_counts[d] += 1
                else:
                    domains.add(domain)
                    if domain in domain_counts:
                        domain_counts[domain] += 1

        # Clean up domain counts
        domain_counts = {k: v for k, v in domain_counts.items() if v > 0}

        metadata["domains"] = sorted(list(domains))
        metadata["domain_counts"] = domain_counts

        return metadata

    def format_detailed_output(
        self, results: Dict[str, Any], analysis_types: List[str]
    ) -> Dict[str, Any]:
        """Format detailed analysis results."""
        formatted = {}

        for analysis_type in analysis_types:
            if analysis_type not in results:
                continue

            result = results[analysis_type]
            if not result:
                continue

            format_config = getattr(
                self.config, f"{analysis_type}_format", None
            )
            if not format_config:
                logger.warning(f"No format config found for {analysis_type}")
                continue

            try:
                formatted[analysis_type] = {
                    "summary": self._format_summary(
                        result, analysis_type, format_config
                    ),
                    "details": self._format_details(
                        result, analysis_type, format_config
                    ),
                    "metadata": self._get_metadata(result),
                }
            except Exception as e:
                logger.error(f"Error formatting {analysis_type}: {e}")
                formatted[analysis_type] = {
                    "summary": f"Error: {str(e)}",
                    "details": "",
                    "metadata": {"error": str(e), "success": False},
                }

        return formatted

    def _format_summary(
        self, result: Any, analysis_type: str, format_config: BaseColumnFormat
    ) -> str:
        """Format summary output."""
        formatter = self.formatters.get(analysis_type)
        if not formatter:
            return ""

        try:
            items = formatter._format_items(
                formatter._get_items(result),
                format_config,
                OutputDetail.SUMMARY,
            )
            return "; ".join(items) if items else ""
        except Exception as e:
            logger.error(f"Error formatting summary for {analysis_type}: {e}")
            return f"Error: {str(e)}"

    def _format_details(
        self, result: Any, analysis_type: str, format_config: BaseColumnFormat
    ) -> str:
        """Format detailed output."""
        formatter = self.formatters.get(analysis_type)
        if not formatter:
            return ""

        try:
            items = formatter._get_items(result)
            formatted_items = formatter._format_items(
                items, format_config, OutputDetail.DETAILED
            )

            # Add type-specific details
            if analysis_type == "keywords":
                if hasattr(result, "compound_words") and result.compound_words:
                    formatted_items.append("\nCompound Words:")
                    formatted_items.extend(
                        [f"    - {word}" for word in result.compound_words]
                    )

                if hasattr(result, "domain_keywords"):
                    formatted_items.append("\nBy Domain:")
                    for domain, keywords in result.domain_keywords.items():
                        formatted_items.append(
                            f"    {domain}: {', '.join(keywords)}"
                        )

            return "\n".join(formatted_items)

        except Exception as e:
            logger.error(f"Error formatting details for {analysis_type}: {e}")
            return f"Error: {str(e)}"
