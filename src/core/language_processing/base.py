# src/core/language_processing/base.py

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class BaseTextProcessor(ABC):
    """Abstract base class for language-specific text processors."""

    def __init__(
        self,
        language: str,
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        file_utils: Optional[FileUtils] = None,
    ):
        """Initialize text processor.

        Args:
            language: Language code ('fi' or 'en')
            custom_stop_words: Optional set of additional stop words
            config: Configuration parameters
            file_utils: Optional FileUtils instance
        """
        if not language:
            raise ValueError("Language code must be provided")

        self.language = language.lower()
        self.config = config or {}
        self.file_utils = file_utils or FileUtils()
        self._stop_words = self._load_stop_words(custom_stop_words)
        logger.debug(f"Initialized {self.language} text processor")

    @abstractmethod
    def get_base_form(self, word: str) -> str:
        """Get base form of a word."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        pass

    def _load_stop_words(self, custom_stop_words: Optional[Set[str]] = None) -> Set[str]:
        """Load stop words for the language."""
        try:
            # Use FileUtils to get the config path
            config_dir = self.file_utils.get_data_path("configurations")
            stop_words_path = config_dir / "stop_words" / f"{self.language}.txt"

            if stop_words_path.exists():
                stop_words = set()
                with open(stop_words_path, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip()
                        if word:  # Skip empty lines
                            stop_words.add(word.lower())
                logger.info(f"Loaded {len(stop_words)} stop words from {stop_words_path}")
            else:
                logger.warning(f"No stop words file found at {stop_words_path}")
                stop_words = set()

            # Add custom stop words
            if custom_stop_words:
                stop_words.update(custom_stop_words)
                logger.debug(f"Added {len(custom_stop_words)} custom stop words")

            return stop_words

        except Exception as e:
            logger.error(f"Error loading stop words: {str(e)}")
            return set()

    def is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word."""
        return word.lower() in self._stop_words

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            if isinstance(text, list):
                text = " ".join(str(x) for x in text)  # Convert list to string
            else:
                text = str(text)  # Convert any other type to string

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep language-specific letters
        if self.language == "fi":
            # Keep Finnish/Swedish letters
            text = re.sub(r"[^a-zäöåA-ZÄÖÅ\s]", " ", text)
        else:
            # Default English processing
            text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        logger.debug(f"Preprocessed text: '{text}'")
        return text

    def process_text(self, text: str) -> str:
        """Process text to get preprocessed form.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        return self.preprocess_text(text)

    def should_keep_word(self, word: str, base_form: str) -> bool:
        """Determine if word should be kept in results."""
        # Skip stop words
        if self.is_stop_word(word) or self.is_stop_word(base_form):
            return False

        # Check minimum length
        if len(base_form) < self.config.get("min_word_length", 3):
            return False

        # Check against excluded patterns
        excluded_patterns = self.config.get("excluded_patterns", [])
        if any(re.search(pattern, base_form) for pattern in excluded_patterns):
            return False

        return True
