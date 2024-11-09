# src/core/language_processing/factory.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type

from src.core.language_processing.base import BaseTextProcessor
from src.core.language_processing.english import EnglishTextProcessor
from src.core.language_processing.finnish import FinnishTextProcessor

logger = logging.getLogger(__name__)


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
                logger.info(f"Loaded configuration from {config_file}")
            else:
                self.config = self._get_default_config()
                logger.info("Using default configuration")
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
        config: Optional[Dict[str, Any]] = None
    ) -> BaseTextProcessor:
        """Create a text processor for the specified language."""
        language = language or self.config.get("default_language", "en")

        if language not in self.PROCESSORS:
            raise ValueError(f"Unsupported language: {language}")

        try:
            # Get base config
            lang_config = self.config.get("languages", {}).get(language, {}).copy()

            # Add any additional config
            if config:
                lang_config.update(config)

            # Create processor
            processor_class = self.PROCESSORS[language]
            return processor_class(
                language=language,
                config=lang_config
            )

        except Exception as e:
            logger.error(f"Error creating processor for {language}: {e}")
            raise

    # def clear_cache(self):
    #     """Clear the processor cache."""
    #     self._processor_cache.clear()

    def detect_language(self, text: str) -> str:
        """Detect the language of the text.

        Args:
            text: Input text

        Returns:
            str: Detected language code
        """
        try:
            from langdetect import detect

            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return self.config.get("default_language", "en")

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
    def register_processor(cls, language: str, processor_class: Type[BaseTextProcessor]) -> None:
        """Register a new processor class.

        Args:
            language: Language code
            processor_class: Processor class to register
        """
        if not issubclass(processor_class, BaseTextProcessor):
            raise ValueError(f"Processor class must inherit from BaseTextProcessor")
        cls.PROCESSORS[language.lower()] = processor_class
        logger.info(f"Registered processor for {language}")

    def clear_cache(self) -> None:
        """Clear the processor cache."""
        self._processor_cache.clear()
        logger.debug("Cleared processor cache")


# Convenience function

def create_text_processor(
    language: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseTextProcessor:
    """Create a text processor instance.

    Args:
        language: Language code
        custom_stop_words: Additional stop words
        config: Additional configuration parameters

    Returns:
        BaseTextProcessor: Appropriate text processor instance
    """
    factory = TextProcessorFactory()
    return factory.create_processor(language=language, config=config)