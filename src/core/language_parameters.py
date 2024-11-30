# src/core/language_parameters.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from langdetect import detect
from pydantic import BaseModel, Field

from src.loaders.models import ParameterSet
from src.loaders.parameter_handler import ParameterHandler
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class LanguageConfig(BaseModel):
    """Language-specific configuration."""

    language: str = Field(..., description="Language code (e.g., 'en', 'fi')")
    min_word_length: int = Field(default=3)
    max_keywords: int = Field(default=10)
    include_compounds: bool = Field(default=True)
    stopwords_path: Optional[Path] = None

    # Language-specific processing options
    class Config:
        extra = (
            "allow"  # Allow additional fields for language-specific settings
        )


class LanguageParameterManager:
    """Manages language detection and parameter loading."""

    DEFAULT_CONFIGS = {
        "en": {
            "min_word_length": 3,
            "max_keywords": 10,
            "include_compounds": True,
            "use_nltk": True,
            "excluded_patterns": [r"^\d+$", r"^[^a-zA-Z0-9]+$"],
        },
        "fi": {
            "min_word_length": 3,
            "max_keywords": 8,
            "include_compounds": True,
            "use_voikko": True,
            "excluded_patterns": [r"^\d+$", r"^[^a-zA-ZäöåÄÖÅ0-9]+$"],
        },
    }

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the manager.

        Args:
            file_utils: Optional FileUtils instance
            config_path: Optional path to configuration directory
        """
        self.file_utils = file_utils or FileUtils()
        self.config_path = Path(config_path) if config_path else None
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load language configurations from files."""
        self.configs = self.DEFAULT_CONFIGS.copy()

        try:
            # Load main config if exists
            if self.config_path:
                config_file = self.config_path / "language_processing.yaml"
                if config_file.exists():
                    loaded_config = self.file_utils.load_yaml(config_file)
                    if "languages" in loaded_config:
                        for lang, config in loaded_config["languages"].items():
                            if lang in self.configs:
                                self.configs[lang].update(config)

            logger.debug(
                f"Loaded configurations for languages: {list(self.configs.keys())}"
            )

        except Exception as e:
            logger.warning(f"Error loading language configurations: {e}")

    def detect_language(self, text: str) -> str:
        """Detect text language with fallback.

        Args:
            text: Input text

        Returns:
            str: Detected language code
        """
        try:
            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English

    def get_parameters(
        self,
        text: str,
        parameter_file: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        **overrides,
    ) -> ParameterSet:
        """Get parameters for text analysis.

        Args:
            text: Input text
            parameter_file: Optional parameter file path
            language: Optional explicit language code
            **overrides: Parameter overrides

        Returns:
            ParameterSet: Combined parameters
        """
        detected_lang = language or self.detect_language(text)
        logger.debug(f"Using language: {detected_lang}")

        # Start with language defaults
        params = self.configs.get(
            detected_lang, self.DEFAULT_CONFIGS["en"]
        ).copy()

        # Load parameters from file if provided
        if parameter_file:
            try:
                # Use new ParameterHandler instead of ParameterAdapter
                handler = ParameterHandler(parameter_file)
                file_params = handler.get_parameters()
                params.update(file_params.model_dump())
                logger.debug(f"Loaded parameters from {parameter_file}")
            except Exception as e:
                logger.warning(f"Error loading parameter file: {e}")

        # Apply any overrides
        params.update(overrides)

        return ParameterSet(**params)

    def load_excel_parameters(
        self, excel_file: Union[str, Path], sheet_name: Optional[str] = None
    ) -> Dict[str, ParameterSet]:
        """Load parameters from Excel file.

        Args:
            excel_file: Path to Excel file
            sheet_name: Optional sheet name

        Returns:
            Dict[str, ParameterSet]: Parameters by language
        """
        try:
            # Load Excel file
            if sheet_name:
                df = self.file_utils.load_single_file(excel_file)
            else:
                sheets = self.file_utils.load_excel_sheets(excel_file)
                df = next(iter(sheets.values()))  # Use first sheet

            # Process parameters
            params_by_lang = {}
            for _, row in df.iterrows():
                if "language" not in row:
                    continue

                lang = row["language"]
                # Start with language defaults
                lang_params = self.configs.get(
                    lang, self.DEFAULT_CONFIGS["en"]
                ).copy()

                # Update with Excel parameters
                for col in df.columns:
                    if col != "language" and pd.notna(row[col]):
                        lang_params[col] = row[col]

                params_by_lang[lang] = ParameterSet(**lang_params)

            return params_by_lang

        except Exception as e:
            logger.error(f"Error loading Excel parameters: {e}")
            return {}

    async def analyze_with_language(
        self,
        analyzer: Any,  # Type hint for analyzer instance
        text: str,
        parameter_file: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Analyze text with appropriate language parameters.

        Args:
            analyzer: Analyzer instance
            text: Input text
            parameter_file: Optional parameter file
            language: Optional explicit language
            **kwargs: Additional analyzer arguments

        Returns:
            Analysis results
        """
        # Get appropriate parameters
        params = self.get_parameters(text, parameter_file, language)

        # Update analyzer config
        analyzer.config.update(params.dict())

        # Perform analysis
        return await analyzer.analyze(text, **kwargs)
