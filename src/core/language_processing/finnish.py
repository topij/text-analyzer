# src/core/language_processing/finnish.py

import logging
import os
import re  # Add missing import
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from libvoikko import Voikko

from .base import BaseTextProcessor

logger = logging.getLogger(__name__)


class FinnishTextProcessor(BaseTextProcessor):
    """Finnish text processor using Voikko.

    Handles Finnish-specific text processing including:
    - Word base form extraction
    - Compound word handling
    - Finnish-specific token types
    - Special case mappings
    - Finnish-specific stopwords
    """

    # Token type constants
    WORD_TOKEN = 1
    WHITESPACE_TOKEN = 3

    # Word class mappings
    WORD_CLASSES = {
        "nimisana": "noun",
        "teonsana": "verb",
        "laatusana": "adjective",
        "seikkasana": "adverb",
    }

    # Special case word mappings
    WORD_MAPPINGS = {
        "osata": "osaaminen",
        "tilaama": "tilata",
        "pakattu": "pakata",
        "viivästynyt": "viivästyä",
        # Add other special cases
    }

    # Common Finnish compound mappings
    COMPOUND_MAPPINGS = {
        "ohjelmistokehittäjä": ["ohjelmisto", "kehittäjä"],
        "ohjelmistokehitys": ["ohjelmisto", "kehitys"],
        "ohjelmistoprojekti": ["ohjelmisto", "projekti"],
        "ohjelmisto": ["ohjelmisto"],  # Add base word itself
        "verkkokauppa": ["verkko", "kauppa"],
        "asiakasprojekti": ["asiakas", "projekti"],
        "tietoturva": ["tieto", "turva"],
        "käyttöliittymä": ["käyttö", "liittymä"],
        "kehittäjä": ["kehittäjä"],  # Add base word itself
    }

    def __init__(
        self,
        language: str = "fi",
        custom_stop_words: Optional[Set[str]] = None,
        voikko_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Finnish text processor.

        Args:
            # custom_stop_words: Additional stop words
            # voikko_path: Path to Voikko installation
            # config: Configuration parameters
        """
        super().__init__(language, custom_stop_words, config)
        
        # Initialize Voikko
        self.voikko = self._initialize_voikko(voikko_path)

        try:
            # Load Finnish stopwords from file
            self._load_finnish_stopwords()
            
            # Add any custom stopwords
            if custom_stop_words:
                self._stop_words.update(custom_stop_words)
            
            logger.debug(f"Initialized Finnish processor with {len(self._stop_words)} stopwords")

        except Exception as e:
            logger.error(f"Error initializing Finnish text processor: {e}")
            raise

    def _load_finnish_stopwords(self) -> None:
        """Load Finnish stopwords from file."""
        try:
            stopwords_path = self.file_utils.get_data_path("configurations") / "stop_words" / "fi.txt"
            
            if not stopwords_path.exists():
                logger.warning(f"Finnish stopwords file not found at {stopwords_path}")
                self._stop_words = set()
                return

            with open(stopwords_path, "r", encoding="utf-8") as f:
                self._stop_words = {line.strip().lower() for line in f if line.strip()}
                
            logger.info(f"Loaded {len(self._stop_words)} Finnish stopwords from {stopwords_path}")
            
        except Exception as e:
            logger.error(f"Error loading Finnish stopwords: {e}")
            self._stop_words = set()

    def get_base_form(self, word: str) -> str:
        """Get base form with compound word awareness."""
        try:
            word_lower = word.lower()

            # Use Voikko if available
            if self.voikko:
                analyses = self.voikko.analyze(word)
                if analyses:
                    return analyses[0].get("BASEFORM", word).lower()

            return word_lower

        except Exception as e:
            logger.error(f"Error getting base form for {word}: {e}")
            return word.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Finnish text using Voikko if available."""
        if not isinstance(text, str):
            return []

        try:
            if self.voikko:
                tokens = self.voikko.tokens(text)
                words = [t.tokenText.strip() for t in tokens 
                        if hasattr(t, "tokenType") and t.tokenType == 1]
            else:
                # Fallback tokenization
                words = text.split()

            return [w for w in words if w]

        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return text.split()

    def _tokenize_text(self, text: str) -> List[str]:
        """Finnish-specific tokenization using Voikko if available."""
        if self.voikko:
            try:
                # Use Voikko's tokenization
                tokens = self.voikko.tokens(text)
                return [t.tokenText for t in tokens if t.tokenType == Voikko.TOKEN_TYPE.WORD]
            except Exception as e:
                self.logger.error(f"Voikko tokenization failed: {str(e)}")

        # Fall back to base implementation if Voikko fails or is not available
        return super()._tokenize_text(text)

    def _initialize_voikko(self, voikko_path: Optional[str] = None) -> Optional[Voikko]:
        """Initialize Voikko with better path handling."""
        try:
            # Common Voikko installation paths
            default_paths = [
                "/usr/local/lib/voikko",
                "/usr/lib/voikko",
                os.path.expanduser("~/voikko"),
                "C:/Program Files/Voikko",
                "C:/Voikko",
                "C:/scripts/Voikko",
            ]

            paths_to_try = [voikko_path] + default_paths if voikko_path else default_paths

            for path in paths_to_try:
                if not path:
                    continue
                try:
                    if os.name == "nt" and hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(path)
                    voikko = Voikko("fi", path)
                    test_result = voikko.analyze("testi")
                    if test_result:
                        logger.info(f"Successfully initialized Voikko with path: {path}")
                        return voikko
                except Exception as e:
                    logger.debug(f"Failed to initialize Voikko with path {path}: {e}")
                    continue

            logger.warning("Could not initialize Voikko with any available path")
            return None

        except Exception as e:
            logger.error(f"Failed to initialize Voikko: {str(e)}")
            return None

    def _get_word_base_form(self, analyses: List[Dict[str, Any]]) -> str:
        """Get the most appropriate base form from analyses."""
        try:
            best_analysis = self._select_best_analysis(analyses)
            if not best_analysis:
                return analyses[0].get("BASEFORM", "").lower()

            return best_analysis.get("BASEFORM", "").lower()

        except Exception as e:
            logger.error(f"Error getting word base form: {str(e)}")
            return analyses[0].get("BASEFORM", "").lower() if analyses else ""

    def _select_best_analysis(self, analyses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best analysis based on word class and form."""
        for analysis in analyses:
            word_class = analysis.get("CLASS")
            if not word_class:
                continue

            # Prefer nouns
            if word_class == "nimisana":
                return analysis

            # Then verbs in basic form
            if word_class == "teonsana" and not analysis.get("PARTICIPLE"):
                return analysis

        return analyses[0] if analyses else None

    def is_compound_word(self, word: str) -> bool:
        """Check if word is a compound word."""
        if not self.voikko:
            return False

        analyses = self.voikko.analyze(word)
        if not analyses:
            return False

        return "+" in analyses[0].get("WORDBASES", "")

    def get_compound_parts(self, word: str) -> List[str]:
        """Get parts of a compound word."""
        if not self.voikko:
            return [word]

        analyses = self.voikko.analyze(word)
        if not analyses:
            return [word]

        word_bases = analyses[0].get("WORDBASES", "")
        if "+" not in word_bases:
            return [word]

        return [part.split("(")[0] for part in word_bases.split("+") if part and "(" in part]

    def process_text(self, text: str) -> str:
        """Process Finnish text with compound word focus."""
        if not text:
            return ""

        try:
            logger.debug(f"Processing Finnish text: {text}")

            # First look for compound words in original text
            compounds = set()
            text_lower = text.lower()
            for compound, parts in self.COMPOUND_MAPPINGS.items():
                if compound in text_lower:
                    logger.debug(f"Found compound: {compound}")
                    compounds.update(parts)
                    compounds.add(compound)  # Keep the original compound too

            # Preprocess and tokenize
            processed = self.preprocess_text(text)
            tokens = self.tokenize(processed)
            logger.debug(f"Tokens: {tokens}")

            # Process individual tokens
            processed_words = set()
            for token in tokens:
                token_lower = token.lower()

                if self.is_stop_word(token_lower):
                    continue

                # Check for compound words again at token level
                found_compound = False
                for compound, parts in self.COMPOUND_MAPPINGS.items():
                    if compound in token_lower or token_lower in compound:
                        processed_words.update(parts)
                        processed_words.add(compound)
                        found_compound = True
                        logger.debug(f"Found compound in token: {token} -> {parts}")
                        break

                if not found_compound:
                    base = self.get_base_form(token)
                    if base and len(base) >= self.config.get("min_word_length", 3):
                        processed_words.add(base)

            # Combine all found words
            all_words = compounds.union(processed_words)
            logger.debug(f"All processed words: {all_words}")

            return " ".join(all_words)

        except Exception as e:
            logger.error(f"Error processing Finnish text: {e}")
            return text
