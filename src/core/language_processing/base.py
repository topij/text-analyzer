# src/core/language_processing/base.py

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from FileUtils import FileUtils

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
            file_utils: FileUtils instance for file operations (required)
        
        Raises:
            ValueError: If file_utils is not provided
        """
        if file_utils is None:
            raise ValueError("FileUtils instance must be provided to BaseTextProcessor")
            
        self.language = language.lower()
        self.config = config or {}
        self.file_utils = file_utils

        # Initialize empty stopwords set - will be populated by derived classes
        self._stop_words = set()

        # Load stopwords
        self._stop_words = self._load_stop_words()

        # Add custom stopwords if provided
        if custom_stop_words:
            self._stop_words.update(custom_stop_words)
            logger.debug(f"Added {len(custom_stop_words)} custom stopwords")

    @abstractmethod
    def _load_stop_words(self) -> Set[str]:
        """Load stop words for the language.

        Returns:
            Set[str]: Set of stopwords
        """
        pass

    @abstractmethod
    def get_base_form(self, word: str) -> str:
        """Get base form of a word."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        pass

    @abstractmethod
    def is_compound_word(self, word: str) -> bool:
        """Check if word is a valid compound word.

        Must be implemented by language-specific classes with appropriate
        logic for compound word detection.

        Args:
            word: Word to check

        Returns:
            bool: True if word is a valid compound
        """
        pass

    @abstractmethod
    def get_compound_parts(self, word: str) -> Optional[List[str]]:
        """Get parts of compound word if applicable.

        Must be implemented by language-specific classes to properly
        decompose compound words according to language rules.

        Args:
            word: Word to analyze

        Returns:
            Optional[List[str]]: List of compound parts or None
        """
        pass

    @abstractmethod
    def get_pos_tag(self, word: str) -> Optional[str]:
        """Get part-of-speech tag for a word.

        Args:
            word: Word to analyze

        Returns:
            Optional[str]: POS tag or None if not determinable
        """
        pass

    def get_data_path(self, data_type: str = "configurations") -> Path:
        """Get path for data files using FileUtils."""
        return self.file_utils.get_data_path(data_type)

    def is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word."""
        return word.lower() in self._stop_words

    def is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation."""
        return bool(re.match(r'^[.,!?;:"\'()[\]{}]+$', token))

    def should_keep_word(self, word: str, base_form: str) -> bool:
        """Determine if word should be kept in results.

        Args:
            word: Original word
            base_form: Base form of word

        Returns:
            bool: True if word should be kept
        """
        # Skip stop words
        if self.is_stop_word(word) or self.is_stop_word(base_form):
            return False

        # Check minimum length
        min_length = self.config.get("min_keyword_length", 3)
        if len(base_form) < min_length:
            return False

        # Check against excluded patterns
        excluded_patterns = self.config.get("excluded_patterns", [])
        if any(re.search(pattern, base_form) for pattern in excluded_patterns):
            return False

        return True

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing with language-specific handling."""
        if not isinstance(text, str):
            text = str(text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove standalone punctuation but keep within words
        text = re.sub(r"\s+([.,!?;:])\s+", " ", text)

        # Handle language-specific characters
        if self.language == "fi":
            # Keep Finnish/Swedish letters
            text = re.sub(r"[^a-zäöåA-ZÄÖÅ0-9\s\'-]", " ", text)
        else:
            # Default English processing
            text = re.sub(r"[^a-zA-Z0-9\s\'-]", " ", text)

        return text.strip()

    def load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration file using FileUtils.

        Args:
            filename: Name of configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            return self.file_utils.load_yaml(
                filename, input_type="configurations"
            )
        except Exception as e:
            logger.error(f"Error loading config file {filename}: {e}")
            return {}
