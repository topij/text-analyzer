# src/core/language_processing/finnish.py

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from libvoikko import Voikko

from src.core.language_processing.base import BaseTextProcessor
from src.utils.FileUtils.file_utils import FileUtils


logger = logging.getLogger(__name__)


class FinnishTextProcessor(BaseTextProcessor):
    """Finnish text processor using Voikko.

    Handles Finnish-specific text processing including:
    - Word base form extraction using Voikko
    - Compound word handling
    - Finnish-specific stopwords
    - Special case handling for technical terms
    """

    # Common Finnish compound prefixes that should be preserved
    COMPOUND_PREFIXES = {
        "kone",
        "teko",
        "tieto",
        "ohjelmisto",
        "verkko",
        "järjestelmä",
        "käyttö",
        "kehitys",
        "palvelu",
        "asiakas",
        "laatu",
        "turva",
    }

    # Common Finnish compound word parts
    COMPOUND_PARTS = {
        "järjestelmä": ["järjestelmä"],
        "pilvi": ["pilvi"],
        "palvelu": ["palvelu"],
        "pilvipalvelu": ["pilvi", "palvelu"],
        "ohjelmisto": ["ohjelmisto"],
        "kehitys": ["kehitys"],
        "käyttö": ["käyttö"],  # Added
        "prosessi": ["prosessi"],  # Added
        "kustannus": ["kustannus"],  # Added
        "skaalautuvuus": ["skaalautuvuus"],  # Added
    }

    # Platform-specific default paths
    VOIKKO_PATHS = {
        "win32": [
            r"C:\scripts\Voikko",
            r"C:\Program Files\Voikko",
            r"C:\Voikko",
            "~/Voikko",
        ],  # Will be expanded
        "linux": [
            "/usr/lib/voikko",
            "/usr/local/lib/voikko",
            "/usr/share/voikko",
            "~/voikko",
        ],  # Will be expanded
        "darwin": [
            "/usr/local/lib/voikko",
            "/opt/voikko",
            "~/voikko",
        ],  # macOS  # Will be expanded
    }

    def __init__(
        self,
        language: str = "fi",
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        file_utils: Optional[FileUtils] = None,
    ):
        """Initialize Finnish text processor.

        Args:
            language: Language code (should be 'fi')
            custom_stop_words: Optional set of additional stop words
            config: Configuration parameters including Voikko path
            file_utils: Optional FileUtils instance for file operations
        """
        super().__init__(language, custom_stop_words, config, file_utils)

        # Get Voikko path from config
        self.voikko_path = None
        if config and "voikko_path" in config:
            self.voikko_path = config["voikko_path"]
        elif file_utils:
            try:
                main_config = file_utils.load_yaml(Path("config.yaml"))
                self.voikko_path = (
                    main_config.get("languages", {})
                    .get("fi", {})
                    .get("voikko_path")
                )
            except Exception as e:
                logger.warning(
                    f"Could not load Voikko path from config.yaml: {e}"
                )

        # Initialize Voikko
        self.voikko = self._initialize_voikko(self.voikko_path)

        if not self.voikko:
            logger.warning(
                "Voikko initialization failed, using fallback tokenization"
            )

    def _load_stop_words(self) -> Set[str]:
        """Load Finnish stopwords from file and add technical terms."""
        try:
            stop_words = set()

            # Load stopwords from file
            stopwords_path = (
                self.get_data_path("configurations") / "stop_words" / "fi.txt"
            )
            if stopwords_path.exists():
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    stop_words.update(
                        word.strip().lower() for word in f if word.strip()
                    )
                logger.info(
                    f"Loaded {len(stop_words)} stopwords from {stopwords_path}"
                )
            else:
                logger.warning(f"Stopwords file not found: {stopwords_path}")

            # Add common technical terms that should be excluded
            technical_stops = {
                # Common technical words that aren't meaningful alone
                "versio",
                "ohjelma",
                "järjestelmä",
                "sovellus",
                "tiedosto",
                "toiminto",
                "palvelu",
                "käyttö",
                "teksti",
                "tieto",
                # Common verbs in technical context
                "käyttää",
                "toimia",
                "suorittaa",
                "toteuttaa",
                "näyttää",
                "muokata",
                "lisätä",
                "poistaa",
                "hakea",
                "tallentaa",
                # Common adjectives in technical context
                "uusi",
                "vanha",
                "nykyinen",
                "aiempi",
                "seuraava",
                "erilainen",
                "samanlainen",
                "vastaava",
            }

            # Only add technical stopwords that aren't overridden by the file
            new_stops = technical_stops - stop_words
            if new_stops:
                logger.debug(f"Added {len(new_stops)} technical stopwords")
                stop_words.update(new_stops)

            logger.debug(f"Total Finnish stopwords: {len(stop_words)}")
            return stop_words

        except Exception as e:
            logger.error(f"Error loading Finnish stopwords: {e}")
            return set()

    def _initialize_voikko(
        self, voikko_path: Optional[str] = None
    ) -> Optional[Voikko]:
        """Initialize Voikko with cross-platform compatibility."""
        try:
            # Determine platform
            platform = sys.platform
            logger.info(f"Detected platform: {platform}")

            # Get platform-specific search paths
            default_paths = self.VOIKKO_PATHS.get(
                platform, self.VOIKKO_PATHS[platform]
            )

            # Expand user paths
            search_paths = [os.path.expanduser(p) for p in default_paths]

            # Add config path to the beginning if provided
            if voikko_path:
                search_paths.insert(0, os.path.expanduser(voikko_path))
                logger.info(f"Using Voikko path from config: {voikko_path}")

            # Try direct initialization first (system libraries)
            try:
                voikko = Voikko("fi")
                if voikko.analyze("testi"):
                    logger.info(
                        "Successfully initialized Voikko using system libraries"
                    )
                    return voikko
            except Exception as e:
                logger.debug(
                    f"Could not initialize Voikko using system libraries: {e}"
                )

            # Try with explicit paths
            for path in search_paths:
                if not os.path.exists(path):
                    logger.debug(f"Path does not exist: {path}")
                    continue

                try:
                    # Windows-specific DLL handling
                    if platform == "win32":
                        self._add_dll_directory(path)
                        if not self._verify_voikko_installation(path):
                            continue

                    voikko = Voikko("fi", str(path))
                    if voikko.analyze("testi"):
                        logger.info(
                            f"Successfully initialized Voikko with path: {path}"
                        )
                        return voikko
                except Exception as e:
                    logger.debug(
                        f"Failed to initialize Voikko with path {path}: {e}"
                    )

            # Platform-specific guidance
            if platform == "win32":
                logger.warning(
                    "On Windows, ensure Voikko is installed in one of these locations: "
                    + ", ".join(self.VOIKKO_PATHS["win32"])
                )
            else:
                logger.warning(
                    "On Linux/Unix, install Voikko using your package manager, e.g.:\n"
                    + "Ubuntu/Debian: sudo apt-get install libvoikko-dev voikko-fi\n"
                    + "Fedora: sudo dnf install libvoikko voikko-fi\n"
                    + "macOS: brew install voikko"
                )
            return None

        except Exception as e:
            logger.error(f"Failed to initialize Voikko: {str(e)}")
            return None

            # # Default paths
            # default_paths = [
            #     # Linux/WSL paths
            #     "/usr/lib/voikko",
            #     "/usr/local/lib/voikko",
            #     "/usr/share/voikko",
            #     # Home directory
            #     os.path.expanduser("~/voikko"),
            #     # Windows paths
            #     "C:/scripts/Voikko",
            #     "C:/Program Files/Voikko",
            #     "C:/Voikko",
            # ]

    def _add_dll_directory(self, path: str) -> None:
        """Add directory to DLL search path on Windows."""
        if sys.platform == "win32":
            try:
                if hasattr(os, "add_dll_directory"):  # Python 3.8+
                    os.add_dll_directory(path)
                else:
                    if path not in os.environ["PATH"]:
                        os.environ["PATH"] = (
                            path + os.pathsep + os.environ["PATH"]
                        )
                logger.info(f"Added {path} to DLL search path")
            except Exception as e:
                logger.error(f"Error adding DLL directory {path}: {str(e)}")

    def _verify_voikko_installation(self, path: str) -> bool:
        """Verify Voikko installation (Windows-specific)."""
        if sys.platform != "win32":
            return True  # Skip verification on non-Windows platforms

        logger.info("Verifying Voikko installation...")

        # Check DLL
        dll_path = os.path.join(path, "libvoikko-1.dll")
        dll_exists = os.path.exists(dll_path)
        logger.info(f"DLL exists: {dll_exists} ({dll_path})")

        # Check dictionary paths
        found_dict = False
        for version in ["5", "2"]:
            dict_path = os.path.join(path, "voikko", version, "mor-standard")
            if os.path.exists(dict_path):
                logger.info(
                    f"Found dictionary version {version} at: {dict_path}"
                )
                found_dict = True

        if not found_dict:
            logger.error("No dictionary found!")

        return dll_exists and found_dict

    def get_base_form(self, word: str) -> str:
        """Get base form of a Finnish word with compound word handling."""
        try:
            if not self.voikko:
                return word.lower()

            word_lower = word.lower()
            analyses = self.voikko.analyze(word)
            if not analyses:
                return word_lower

            analysis = analyses[0]

            # Special handling for compound words
            if "WORDBASES" in analysis:
                word_bases = analysis["WORDBASES"]
                if "+" in word_bases:
                    parts = word_bases.split("+")
                    first_part = parts[0].split("(")[0].lower()

                    # If it starts with a known prefix, preserve the original form
                    if first_part in self.COMPOUND_PREFIXES:
                        return word_lower

            # For non-compound words, return the base form
            return analysis.get("BASEFORM", word).lower()

        except Exception as e:
            logger.error(f"Error getting base form for {word}: {e}")
            return word.lower()

    def get_pos_tag(self, word: str) -> Optional[str]:
        """Get part-of-speech tag using Voikko.

        Maps Voikko classes to standard POS tags:
        - Nominit -> NN (nouns)
        - Verbit -> VB (verbs)
        - Adjektiivit -> JJ (adjectives)
        - Adverbit -> RB (adverbs)
        """
        try:
            if not self.voikko:
                return None

            analyses = self.voikko.analyze(word)
            if not analyses:
                return None

            # Map Voikko classes to standard POS tags
            class_mapping = {
                "nimisana": "NN",  # Noun
                "laatusana": "JJ",  # Adjective
                "teonsana": "VB",  # Verb
                "seikkasana": "RB",  # Adverb
                "etunimi": "NNP",  # Proper noun
                "sukunimi": "NNP",  # Proper noun
            }

            word_class = analyses[0].get("CLASS", "")
            return class_mapping.get(word_class, "NN")  # Default to noun

        except Exception as e:
            logger.error(f"Error getting POS tag for '{word}': {e}")
            return None

    def preprocess_text(self, text: str) -> str:
        """Preprocess Finnish text with proper encoding."""
        if not isinstance(text, str):
            text = str(text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Handle Finnish/Swedish characters
        text = re.sub(r"[^a-zäöåA-ZÄÖÅ0-9\s\-]", " ", text)

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Finnish text with proper encoding."""
        if not text:
            return []

        try:
            # First preprocess the text
            text = self.preprocess_text(text)

            if self.voikko:
                tokens = self.voikko.tokens(text)
                words = [
                    t.tokenText
                    for t in tokens
                    if hasattr(t, "tokenType") and t.tokenType == 1
                ]  # Word tokens
            else:
                # Fallback tokenization
                words = text.split()

            # Filter and process tokens
            processed = []
            for word in words:
                word = word.strip()
                if word and not self.is_stop_word(word):
                    processed.append(word)

            return processed

        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return []

    def is_compound_word(self, word: str) -> bool:
        """Improved compound word detection for Finnish."""
        if not word:
            return False

        try:
            word_lower = word.lower()

            # Check predefined compounds
            if (
                word_lower in self.COMPOUND_PARTS
                and len(self.COMPOUND_PARTS[word_lower]) > 1
            ):
                return True

            # Check hyphenated words
            if "-" in word:
                return True

            # Use Voikko for more accurate detection
            if self.voikko:
                analyses = self.voikko.analyze(word)
                if analyses:
                    # Check WORDBASES for compound structure
                    word_bases = analyses[0].get("WORDBASES", "")
                    return "+" in word_bases

            return False

        except Exception as e:
            logger.error(f"Error checking compound word {word}: {e}")
            return False

    def get_compound_parts(self, word: str) -> List[str]:
        """Get parts of Finnish compound word."""
        if not word:
            return [word]

        try:
            word_lower = word.lower()

            # Check predefined parts first
            if word_lower in self.COMPOUND_PARTS:
                return self.COMPOUND_PARTS[word_lower]

            # Handle hyphenated words
            if "-" in word:
                return [
                    part.strip() for part in word.split("-") if part.strip()
                ]

            # Use Voikko for analysis
            if self.voikko:
                analyses = self.voikko.analyze(word)
                if analyses:
                    word_bases = analyses[0].get("WORDBASES", "")
                    if "+" in word_bases:
                        # Extract base forms from analysis
                        parts = []
                        for part in word_bases.split("+"):
                            if "(" in part:
                                base = part.split("(")[0].strip()
                                if base:
                                    parts.append(base)
                        return parts if parts else [word]

            return [word]

        except Exception as e:
            logger.error(f"Error getting compound parts for {word}: {e}")
            return [word]

    def should_keep_word(self, word: str) -> bool:
        """Determine if word should be kept based on Finnish-specific rules."""
        # First check base criteria
        if not super().should_keep_word(word):
            return False

        # Get base form for stopword check
        base_form = self.get_base_form(word)

        # Check stopwords against base form
        if self.is_stop_word(base_form):
            return False

        # Keep compound words that start with known prefixes
        if self.is_compound_word(word):
            parts = self.get_compound_parts(word)
            if parts and parts[0] in self.COMPOUND_PREFIXES:
                return True

        return True
