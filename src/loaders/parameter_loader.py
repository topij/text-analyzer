# src/loaders/parameter_loader.py

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.config_and_logging import file_utils, logger
from src.models.parameter_models import (
    CategoryEntry,
    ExtractionParameters,
    GeneralParameters,
    KeywordEntry,
    ParameterSheets,
    PromptTemplate,
)


class ParameterLoader:
    """Handles loading and validation of extraction parameters."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize with optional config directory."""
        self.config_dir = config_dir or file_utils.get_data_path("configurations")
        self._cached_parameters: Optional[ExtractionParameters] = None

    def load_from_excel(self, file_path: Union[str, Path], cache_results: bool = True) -> ExtractionParameters:
        """Load parameters from Excel file.

        Args:
            file_path: Path to Excel file
            cache_results: Whether to cache loaded parameters

        Returns:
            ExtractionParameters: Validated parameters

        Raises:
            ValueError: If file format is invalid
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Loading parameters from {file_path}")

            # Read Excel sheets
            excel = pd.ExcelFile(file_path)
            sheets = {sheet: excel.parse(sheet) for sheet in excel.sheet_names}

            # Load each section
            params = ExtractionParameters(
                general=self._load_general_params(sheets),
                predefined_keywords=self._load_keywords(sheets),
                excluded_keywords=self._load_excluded(sheets),
                categories=self._load_categories(sheets),
                custom_prompts=self._load_prompts(sheets),
            )

            if cache_results:
                self._cached_parameters = params

            logger.info("Successfully loaded parameters")
            return params

        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            raise ValueError(f"Failed to load parameters: {str(e)}")

    def _load_general_params(self, sheets: Dict[str, pd.DataFrame]) -> GeneralParameters:
        """Load general parameters."""
        if ParameterSheets.GENERAL.value not in sheets:
            logger.warning("No general parameters sheet found, using defaults")
            return GeneralParameters()

        df = sheets[ParameterSheets.GENERAL.value]
        if "parameter" not in df.columns or "value" not in df.columns:
            raise ValueError("Invalid general parameters format")

        params = {}
        for _, row in df.iterrows():
            if pd.isna(row["parameter"]) or pd.isna(row["value"]):
                continue
            params[row["parameter"].strip()] = row["value"]

        return GeneralParameters(**params)

    def _load_keywords(self, sheets: Dict[str, pd.DataFrame]) -> List[KeywordEntry]:
        """Load predefined keywords."""
        if ParameterSheets.KEYWORDS.value not in sheets:
            return []

        df = sheets[ParameterSheets.KEYWORDS.value]
        if "keyword" not in df.columns:
            raise ValueError("Invalid keywords sheet format")

        keywords = []
        for _, row in df.iterrows():
            if pd.isna(row["keyword"]):
                continue

            entry = {
                "keyword": row["keyword"].strip(),
                "importance": row.get("importance", 1.0),
                "domain": row.get("domain"),
                "notes": row.get("notes"),
            }
            keywords.append(KeywordEntry(**entry))

        return keywords

    def _load_excluded(self, sheets: Dict[str, pd.DataFrame]) -> List[str]:
        """Load excluded keywords."""
        if ParameterSheets.EXCLUDED.value not in sheets:
            return []

        df = sheets[ParameterSheets.EXCLUDED.value]
        if "keyword" not in df.columns:
            raise ValueError("Invalid excluded keywords format")

        return [str(k).strip() for k in df["keyword"].dropna() if str(k).strip()]

    def _load_categories(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, CategoryEntry]:
        """Load category definitions."""
        if ParameterSheets.CATEGORIES.value not in sheets:
            return {}

        df = sheets[ParameterSheets.CATEGORIES.value]
        required = {"name", "description"}
        if not all(col in df.columns for col in required):
            raise ValueError("Invalid categories format")

        categories = {}
        for _, row in df.iterrows():
            if pd.isna(row["name"]) or pd.isna(row["description"]):
                continue

            entry = CategoryEntry(
                name=row["name"].strip(),
                description=row["description"].strip(),
                keywords=self._parse_keywords(row.get("keywords", "")),
                threshold=float(row.get("threshold", 0.5)),
            )
            categories[entry.name] = entry

        return categories

    def _load_prompts(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, PromptTemplate]:
        """Load custom prompt templates."""
        if ParameterSheets.PROMPTS.value not in sheets:
            return {}

        df = sheets[ParameterSheets.PROMPTS.value]
        required = {"name", "system_prompt", "user_prompt"}
        if not all(col in df.columns for col in required):
            raise ValueError("Invalid prompts format")

        prompts = {}
        for _, row in df.iterrows():
            if pd.isna(row["name"]):
                continue

            template = PromptTemplate(
                name=row["name"].strip(),
                version=row.get("version", "1.0.0"),
                system_prompt=row["system_prompt"],
                user_prompt=row["user_prompt"],
                language=row.get("language", "en"),
                notes=row.get("notes"),
            )
            prompts[template.name] = template

        return prompts

    @staticmethod
    def _parse_keywords(keywords: str) -> List[str]:
        """Parse comma-separated keywords."""
        if pd.isna(keywords):
            return []
        return [k.strip() for k in str(keywords).split(",") if k.strip()]

    def get_cached_parameters(self) -> Optional[ExtractionParameters]:
        """Get cached parameters if available."""
        return self._cached_parameters

    def clear_cache(self) -> None:
        """Clear parameter cache."""
        self._cached_parameters = None
