# src/utils/analysis_formatters.py

from typing import Any, Dict, List
import pandas as pd

from src.utils.output_formatter import (
    BaseFormatter,
    OutputDetail,
    BaseColumnFormat,
    ResultFormatter,
)


class KeywordFormatter(ResultFormatter):
    """Formatter for keyword analysis results."""

    def format_result(
        self,
        result: Any,
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> str:
        """Format keyword results with appropriate detail level."""
        if not result or not getattr(result, "success", True):
            return self._format_error(
                getattr(result, "error", "Analysis failed")
            )

        items = result.keywords if hasattr(result, "keywords") else []

        if detail_level == OutputDetail.MINIMAL:
            return self._format_minimal_keywords(items, format_config)
        elif detail_level == OutputDetail.SUMMARY:
            return self._format_summary_keywords(items, format_config)
        else:
            return self._format_detailed_keywords(result, format_config)

    def _format_minimal_keywords(
        self, keywords: List[Any], format_config: BaseColumnFormat
    ) -> str:
        """Format minimal keyword output."""
        formatted = []
        for kw in keywords[: format_config.max_items]:
            confidence = self._format_confidence(kw.score)
            formatted.append(f"{kw.keyword}{confidence}")
        return ", ".join(formatted)

    def _format_summary_keywords(
        self, keywords: List[Any], format_config: BaseColumnFormat
    ) -> str:
        """Format summary keyword output with domains."""
        formatted = []
        for kw in keywords[: format_config.max_items]:
            confidence = self._format_confidence(kw.score)
            domain = f" [{kw.domain}]" if kw.domain else ""
            formatted.append(f"{kw.keyword}{confidence}{domain}")
        return ", ".join(formatted)

    def _format_detailed_keywords(
        self, result: Any, format_config: BaseColumnFormat
    ) -> Dict[str, Any]:
        """Format detailed keyword output."""
        formatted = {
            "keywords": self._format_summary_keywords(
                result.keywords, format_config
            )
        }

        if hasattr(result, "compound_words") and result.compound_words:
            formatted["compound_words"] = ", ".join(result.compound_words)

        if hasattr(result, "domain_keywords"):
            formatted["by_domain"] = {
                domain: ", ".join(keywords)
                for domain, keywords in result.domain_keywords.items()
            }

        formatted.update(self._get_metadata(result))
        return formatted


class ThemeFormatter(ResultFormatter):
    """Formatter for theme analysis results."""

    def format_result(
        self,
        result: Any,
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> str:
        """Format theme results with appropriate detail level."""
        if not result or not getattr(result, "success", True):
            return self._format_error(
                getattr(result, "error", "Analysis failed")
            )

        items = result.themes if hasattr(result, "themes") else []

        if detail_level == OutputDetail.MINIMAL:
            return self._format_minimal_themes(items, format_config)
        elif detail_level == OutputDetail.SUMMARY:
            return self._format_summary_themes(items, format_config)
        else:
            return self._format_detailed_themes(result, format_config)

    def _format_minimal_themes(
        self, themes: List[Any], format_config: BaseColumnFormat
    ) -> str:
        """Format minimal theme output."""
        formatted = []
        for theme in themes[: format_config.max_items]:
            confidence = self._format_confidence(theme.confidence)
            formatted.append(f"{theme.name}{confidence}")
        return ", ".join(formatted)

    def _format_summary_themes(
        self, themes: List[Any], format_config: BaseColumnFormat
    ) -> str:
        """Format summary theme output with descriptions."""
        formatted = []
        for theme in themes[: format_config.max_items]:
            confidence = self._format_confidence(theme.confidence)
            formatted.append(f"{theme.name}{confidence}: {theme.description}")
        return "; ".join(formatted)


class CategoryFormatter(ResultFormatter):
    """Formatter for category analysis results."""

    def format_result(
        self,
        result: Any,
        format_config: BaseColumnFormat,
        detail_level: OutputDetail,
    ) -> str:
        """Format category results with appropriate detail level."""
        if not result or not getattr(result, "success", True):
            return self._format_error(
                getattr(result, "error", "Analysis failed")
            )

        items = result.matches if hasattr(result, "matches") else []

        if detail_level == OutputDetail.MINIMAL:
            return self._format_minimal_categories(items, format_config)
        elif detail_level == OutputDetail.SUMMARY:
            return self._format_summary_categories(items, format_config)
        else:
            return self._format_detailed_categories(result, format_config)

    def _format_minimal_categories(
        self, categories: List[Any], format_config: BaseColumnFormat
    ) -> str:
        """Format minimal category output."""
        formatted = []
        for cat in categories[: format_config.max_items]:
            confidence = self._format_confidence(cat.confidence)
            formatted.append(f"{cat.name}{confidence}")
        return ", ".join(formatted)

    def _format_summary_categories(
        self, categories: List[Any], format_config: BaseColumnFormat
    ) -> str:
        """Format summary category output with descriptions."""
        formatted = []
        for cat in categories[: format_config.max_items]:
            confidence = self._format_confidence(cat.confidence)
            formatted.append(f"{cat.name}{confidence}: {cat.description}")
        return "; ".join(formatted)
