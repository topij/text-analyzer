# src/core/language_processing/factory.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type

from langdetect import DetectorFactory, detect

from src.core.language_processing.base import BaseTextProcessor
from src.core.language_processing.english import EnglishTextProcessor
from src.core.language_processing.finnish import FinnishTextProcessor
from FileUtils import FileUtils

logger = logging.getLogger(__name__)

# Initialize langdetect with a seed for consistent results
DetectorFactory.seed = 0


class TextProcessorFactory:
    """Factory for creating language-specific text processors."""

    # Registry of available processors
    PROCESSORS: Dict[str, Type[BaseTextProcessor]] = {
        "en": EnglishTextProcessor,
        "fi": FinnishTextProcessor,
    }

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the factory.

        Args:
            config_path: Optional path to configuration directory
        """
        """Initialize the factory."""
        self.config_path = config_path or Path("config")
        self._load_config()

    def _load_config(self) -> None:
        """Load language processing configuration."""
        try:
            config_file = self.config_path / "language_processing.yaml"
            if config_file.exists():
                import yaml

                with open(config_file, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                logger.debug(
                    f"Loaded configuration from {config_file}"
                )  # Changed to DEBUG
            else:
                self.config = self._get_default_config()
                logger.debug("Using default configuration")  # Changed to DEBUG
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "default_language": "en",
            "languages": {
                "en": {
                    "min_word_length": 3,
                    "excluded_patterns": [r"^\d+$", r"^[^a-zA-Z0-9]+$"],
                },
                "fi": {
                    "min_word_length": 3,
                    "excluded_patterns": [r"^\d+$", r"^[^a-zA-ZäöåÄÖÅ0-9]+$"],
                    "voikko_path": None,
                },
            },
        }

    def create_processor(
        self,
        language: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseTextProcessor:
        """Create a text processor with proper language handling."""
        # Normalize language code
        if language:
            language = language.lower()[
                :2
            ]  # Get first two chars: "fin" -> "fi"
        else:
            language = "en"

        logger.debug(f"Creating text processor for language: {language}")

        # Map languages to processors
        processor_map = {"en": EnglishTextProcessor, "fi": FinnishTextProcessor}

        processor_class = processor_map.get(language)
        if not processor_class:
            logger.warning(
                f"Unsupported language: {language}, defaulting to English"
            )
            processor_class = EnglishTextProcessor
            language = "en"

        try:
            logger.debug(f"Creating {language} processor")
            return processor_class(language=language, config=config or {})
        except Exception as e:
            logger.error(f"Error creating processor for {language}: {e}")
            raise

    def detect_language(self, text: str) -> str:
        """Detect the language of the text.

        Args:
            text: Input text

        Returns:
            str: Detected language code
        """
        try:
            # Remove excess whitespace and ensure we have enough text
            text = " ".join(text.split())
            if len(text) < 20:  # Require minimum length for better accuracy
                logger.warning(
                    "Text too short for reliable detection, defaulting to English"
                )
                return "en"

            detected = detect(text)
            # Map common language codes
            lang_map = {"fin": "fi", "eng": "en", "en": "en", "fi": "fi"}
            detected = lang_map.get(detected.lower(), detected.lower()[:2])

            if detected not in self.PROCESSORS:
                logger.warning(
                    f"Detected unsupported language {detected}, defaulting to English"
                )
                return "en"

            logger.debug(f"Detected language: {detected}")
            return detected

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"

    def create_processor_for_text(
        self,
        text: str,
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseTextProcessor:
        """Create appropriate processor for the given text.

        Args:
            text: Input text
            custom_stop_words: Additional stop words
            config: Additional configuration parameters

        Returns:
            BaseTextProcessor: Appropriate text processor instance
        """
        language = self.detect_language(text)
        return self.create_processor(language, custom_stop_words, config)

    @classmethod
    def register_processor(
        cls, language: str, processor_class: Type[BaseTextProcessor]
    ) -> None:
        """Register a new processor class.

        Args:
            language: Language code
            processor_class: Processor class to register
        """
        if not issubclass(processor_class, BaseTextProcessor):
            raise ValueError(
                f"Processor class must inherit from BaseTextProcessor"
            )
        cls.PROCESSORS[language.lower()] = processor_class
        logger.info(f"Registered processor for {language}")

    def clear_cache(self) -> None:
        """Clear the processor cache."""
        self._processor_cache.clear()
        logger.debug("Cleared processor cache")


def create_text_processor(
    language: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    file_utils: Optional[
        FileUtils
    ] = None,  # Remove this parameter as it's not used
) -> BaseTextProcessor:
    """Create text processor instance."""
    factory = TextProcessorFactory(
        config_path=None
    )  # TextProcessorFactory only takes config_path
    return factory.create_processor(language=language, config=config)


# # Convenience function
# def create_text_processor(
#     language: Optional[str] = None,
#     config: Optional[Dict[str, Any]] = None,
#     file_utils: Optional[FileUtils] = None,
# ) -> BaseTextProcessor:
#     """Create text processor instance."""
#     factory = TextProcessorFactory()

#     # Load main config if not provided
#     if config is None and file_utils:
#         try:
#             main_config = file_utils.load_yaml(Path("config.yaml"))
#             lang_config = main_config.get("languages", {}).get(
#                 language or "en", {}
#             )

#             if config:  # Merge if config was provided
#                 lang_config.update(config)
#             config = lang_config
#         except Exception as e:
#             logger.debug(f"Could not load language config: {e}")
#             config = config or {}

#     logger.debug(f"Creating text processor for language: {language}")
#     return factory.create_processor(language=language, config=config)
