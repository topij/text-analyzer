# src/core/language_processing/english.py

import logging
from typing import Any, Dict, List, Optional, Set

import nltk
import os

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.FileUtils.file_utils import FileUtils

from .base import BaseTextProcessor

logger = logging.getLogger(__name__)

# Download required NLTK resources
def download_nltk_data():
    """Download required NLTK data."""
    required_packages = [
        'punkt_tab',
        'wordnet',
        'averaged_perceptron_tagger',
        'stopwords'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download NLTK package {package}: {e}")

# Download resources when module is imported
download_nltk_data()


class EnglishTextProcessor(BaseTextProcessor):
    """English text processor using NLTK.

    Handles English-specific text processing including:
    - Word lemmatization
    - POS tagging
    - English-specific tokenization
    """

    def __init__(
        self,
        language: str = "en",
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        file_utils: Optional[FileUtils] = None,
    ):
        """Initialize English text processor.

        Args:
            custom_stop_words: Additional stop words
            config: Configuration parameters
            file_utils: Optional FileUtils instance
        """
        super().__init__(language, custom_stop_words, config, file_utils)

        try:
            self.lemmatizer = WordNetLemmatizer()

            # Load NLTK stop words if available
            try:
                english_stopwords = set(nltk_stopwords.words("english"))
                self._stop_words.update(english_stopwords)
                logger.debug("Added NLTK English stop words")
            except Exception as e:
                logger.warning(f"Could not load NLTK stop words: {e}")
        except Exception as e:
            logger.error(f"Error initializing English text processor: {e}")

    def get_base_form(self, word: str) -> str:
        """Get base form (lemma) of an English word."""
        try:
            # Get POS tag
            pos_tag = self._get_wordnet_pos(word)

            # Lemmatize with POS tag if available
            if pos_tag:
                return self.lemmatizer.lemmatize(word.lower(), pos=pos_tag)

            # Try different POS tags if no specific tag is found
            lemmas = [self.lemmatizer.lemmatize(word.lower(), pos=pos) for pos in ["n", "v", "a", "r"]]

            # Return shortest lemma (usually most basic form)
            return min(lemmas, key=len)

        except Exception as e:
            logger.error(f"Error getting base form for '{word}': {str(e)}")
            return word.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text using NLTK."""
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
                else:
                    text = str(text)

            try:
                # Try NLTK's word_tokenize first
                tokens = word_tokenize(text)
                logger.debug(f"NLTK tokenization successful: {len(tokens)} tokens")
                return tokens
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}, falling back to simple split")
                return text.split()

        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            if isinstance(text, str):
                return text.split()
            return []

    def _get_wordnet_pos(self, word: str) -> Optional[str]:
        """Get WordNet POS tag for a word."""
        try:
            # Get NLTK POS tag
            pos = nltk.pos_tag([word])[0][1]

            # Convert to WordNet POS tag
            tag_map = {
                "JJ": "a",  # Adjective
                "VB": "v",  # Verb
                "NN": "n",  # Noun
                "RB": "r",  # Adverb
            }

            # Get the first letter of the POS tag
            pos_prefix = pos[:2]
            return tag_map.get(pos_prefix)

        except Exception as e:
            logger.error(f"Error getting POS tag for '{word}': {str(e)}")
            return None

    def process_text(self, text: str) -> str:
        """Process English text with POS tagging and lemmatization."""
        if not text:
            return ""

        try:
            # First preprocess the text
            text = self.preprocess_text(text)

            # Return preprocessed text
            return text

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return text if isinstance(text, str) else ""

    def should_keep_word(self, word: str, base_form: str) -> bool:
        """Determine if word should be kept based on English-specific rules."""
        # First check base criteria
        if not super().should_keep_word(word, base_form):
            return False

        # Additional English-specific checks

        # Skip single letters except 'a' and 'i'
        if len(base_form) == 1 and base_form not in ["a", "i"]:
            return False

        # Skip common contractions
        contractions = {"'s", "'t", "'re", "'ve", "'ll", "'d"}
        if any(word.lower().endswith(c) for c in contractions):
            return False

        return True
    

