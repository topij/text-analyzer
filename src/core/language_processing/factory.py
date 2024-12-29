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
    """Factory for creating text processors."""

    PROCESSORS = {
        "en": EnglishTextProcessor,
        "fi": FinnishTextProcessor,
    }

    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize factory.
        
        Args:
            file_utils: FileUtils instance for file operations (required)
        """
        if file_utils is None:
            raise ValueError("FileUtils instance must be provided to TextProcessorFactory")
        self.file_utils = file_utils

    def create_processor(
        self,
        language: Optional[str] = None,
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseTextProcessor:
        """Create a text processor with proper language handling.

        Args:
            language: Language code (e.g. 'en', 'fi')
            custom_stop_words: Optional set of additional stop words
            config: Optional configuration parameters

        Returns:
            Language-specific text processor instance

        Raises:
            ValueError: If language not supported
        """
        if not language:
            language = "en"  # Default to English
            logger.debug("No language specified, using default: en")

        language = language.lower()

        # Validate language
        if language not in self.PROCESSORS:
            raise ValueError(
                f"Language '{language}' not supported. Available languages: {list(self.PROCESSORS.keys())}"
            )

        processor_class = self.PROCESSORS[language]
        logger.debug(f"Creating {language} processor")
        
        try:
            return processor_class(
                language=language,
                config=config or {},
                file_utils=self.file_utils,
                custom_stop_words=custom_stop_words
            )
        except Exception as e:
            logger.error(f"Error creating processor for {language}: {e}")
            raise

    def detect_language(self, text: str) -> str:
        """Detect language of text using langdetect."""
        try:
            detected = detect(text)
            logger.debug(f"Detected language: {detected}")
            return detected
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English

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
        cls, language_code: str, processor_class: Type[BaseTextProcessor]
    ) -> None:
        """Register a new processor class for a language.

        Args:
            language_code: Two-letter language code
            processor_class: Processor class to register
        """
        cls.PROCESSORS[language_code.lower()] = processor_class
        logger.debug(f"Registered processor for {language_code}")

def create_text_processor(
    language: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    file_utils: Optional[FileUtils] = None,
) -> BaseTextProcessor:
    """Create text processor instance.

    Args:
        language: Language code (e.g. 'en', 'fi')
        config: Optional configuration parameters
        file_utils: FileUtils instance for file operations (required)

    Returns:
        Language-specific text processor instance

    Raises:
        ValueError: If language not supported or FileUtils not provided
    """
    if file_utils is None:
        raise ValueError("FileUtils instance must be provided to create_text_processor")

    factory = TextProcessorFactory(file_utils=file_utils)
    return factory.create_processor(language=language, config=config)
