# src/core/language_processing/finnish.py
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import threading

from libvoikko import Voikko

from src.core.language_processing.base import BaseTextProcessor
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class FinnishTextProcessor(BaseTextProcessor):
    """Finnish text processor using Voikko."""

    # Enhance compound prefixes with more business terms
    COMPOUND_PREFIXES = {
        # Technical terms
        "kone",
        "teko",
        "tieto",
        "ohjelmisto",
        "verkko",
        "järjestelmä",
        "käyttö",
        "kehitys",
        "palvelu",
        "turva",
        "data",
        # Business terms
        "liike",
        "markkina",
        "asiakas",
        "myynti",
        "tulos",
        "liiketoiminta",
        "kustannus",
        "tuotto",
        "kilpailu",
        "strategia",
    }

    # Core business/technical vocabulary that should be preserved
    CORE_TERMS = {
        # Business core terms
        "liikevaihto",
        "kustannus",
        "tuotto",
        "markkinaosuus",
        "asiakaspysyvyys",
        "asiakashankinta",
        "vuosineljännes",
        "segmentti",
        "liiketoiminta",
        # Technical core terms
        "pilvipalvelu",
        "järjestelmä",
        "ohjelmisto",
        "infrastruktuuri",
        "teknologia",
    }

    # Keep only the base forms of common verbs
    COMMON_VERBS = {
        # Basic verbs
        "olla",
        "tulla",
        "mennä",
        "tehdä",
        "saada",
        "voida",
        "pitää",
        # Change indicating verbs
        "parantua",
        "parata",
        "kasvaa",
        "vähentyä",
        "lisääntyä",
        "vahvistua",
        "heikentyä",
        "muuttua",
        "kehittyä",
        "toteutua",
        "laskea",
        "nousta",
        "päättyä",
        "jatkua",
        "alkaa",
        # Business/technical context verbs
        "kehittää",
        "toteuttaa",
        "käyttää",
        "suorittaa",
        "analysoida",
        "mitata",
        "arvioida",
        "raportoida",
        "seurata",
        "varmistaa",
    }

    # Generic terms that shouldn't be keywords
    GENERIC_TERMS = {
        # Measurement and quantity terms
        "prosentti",
        "määrä",
        "osuus",
        "osa",
        "vaihe",
        "taso",
        "mittari",
        "yksikkö",
        "lukumäärä",
        "koko",
        "arvo",
        # Time-related terms
        "aika",
        "vuosi",
        "kuukausi",
        "viikko",
        "päivä",
        "hetki",
        # Generic descriptors
        "hyvä",
        "huono",
        "suuri",
        "pieni",
        "uusi",
        "vanha",
        "erilainen",
        "samanlainen",
        "vastaava",
    }

    # Word classes that should typically be excluded from keywords
    EXCLUDED_CLASSES = {
        "seikkasana",  # Adverb
        "asemosana",  # Pronoun
        "suhdesana",  # Preposition/Postposition
        "sidesana",  # Conjunction
        "huudahdussana",  # Interjection
        "lukusana",  # Numeral
        "teonsana",  # Add verbs to excluded classes
    }

    # Enhance business domain terms
    BUSINESS_TERMS = {
        "segmentti",
        "markkina",
        "liikevaihto",
        "kustannus",
        "tuotto",
        "markkinaosuus",
        "asiakaspysyvyys",
        "asiakashankinta",
        "vuosineljännes",
        "liiketoiminta",
        "myynti",
        "strategia",
    }

    # Platform-specific Voikko paths
    # VOIKKO_PATHS = {
    #     "win32": [
    #         r"C:\scripts\Voikko",
    #         r"C:\Program Files\Voikko",
    #         r"C:\Voikko",
    #         "~/Voikko",
    #     ],
    #     "linux": [
    #         "/usr/lib/voikko",
    #         "/usr/local/lib/voikko",
    #         "/usr/share/voikko",
    #         "~/voikko",
    #     ],
    #     "darwin": [
    #         "/usr/local/lib/voikko",
    #         "/opt/voikko",
    #         "~/voikko",
    #     ],
    # }

    # def __init__(
    #     self,
    #     language: str = "fi",
    #     custom_stop_words: Optional[Set[str]] = None,
    #     config: Optional[Dict[str, Any]] = None,
    #     file_utils: Optional[FileUtils] = None,
    # ):
    #     """Initialize Finnish text processor."""
    #     super().__init__(language, custom_stop_words, config, file_utils)

    #     # Get Voikko path from config
    #     self.voikko_path = None
    #     if config and "voikko_path" in config:
    #         self.voikko_path = config["voikko_path"]

    #     # Initialize Voikko
    #     self.voikko = self._initialize_voikko(self.voikko_path)

    #     if not self.voikko:
    #         logger.warning(
    #             "Voikko initialization failed, using fallback tokenization"
    #         )

    def __init__(
        self,
        language: str = "fi",
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(language, custom_stop_words, config)
        self._analysis_cache = {}
        self._failed_words = set()  # Track words that fail analysis

        self.voikko_handler = VoikkoHandler()
        voikko_path = config.get("voikko_path") if config else None
        self.voikko = self.voikko_handler.initialize(voikko_path)

        if not self.voikko:
            logger.warning(
                "Voikko initialization failed. Some Finnish-specific features will be limited."
            )

    # def _initialize_voikko(
    #     self, voikko_path: Optional[str] = None
    # ) -> Optional[Voikko]:
    #     """Initialize Voikko with proper path handling."""
    #     try:
    #         # Determine platform
    #         platform = sys.platform
    #         logger.info(f"Detected platform: {platform}")

    #         # Get platform-specific search paths
    #         default_paths = self.VOIKKO_PATHS.get(
    #             platform, self.VOIKKO_PATHS["linux"]
    #         )
    #         search_paths = [os.path.expanduser(p) for p in default_paths]

    #         # Add config path if provided
    #         if voikko_path:
    #             search_paths.insert(0, os.path.expanduser(voikko_path))

    #         # Try direct initialization first
    #         try:
    #             voikko = Voikko("fi")
    #             if voikko.analyze("testi"):
    #                 logger.info(
    #                     "Successfully initialized Voikko using system libraries"
    #                 )
    #                 return voikko
    #         except Exception as e:
    #             logger.debug(f"System library initialization failed: {e}")

    #         # Try with explicit paths
    #         for path in search_paths:
    #             if not os.path.exists(path):
    #                 continue

    #             try:
    #                 # Windows-specific handling
    #                 if platform == "win32":
    #                     self._add_dll_directory(path)

    #                 voikko = Voikko("fi", str(path))
    #                 if voikko.analyze("testi"):
    #                     logger.info(
    #                         f"Successfully initialized Voikko with path: {path}"
    #                     )
    #                     return voikko
    #             except Exception as e:
    #                 logger.debug(f"Failed initialization with {path}: {e}")

    #         logger.warning(
    #             "Could not initialize Voikko. Using fallback methods."
    #         )
    #         return None

    #     except Exception as e:
    #         logger.error(f"Voikko initialization failed: {str(e)}")
    #         return None

    # def _add_dll_directory(self, path: str) -> None:
    #     """Add directory to DLL search path on Windows."""
    #     if sys.platform == "win32":
    #         try:
    #             if hasattr(os, "add_dll_directory"):
    #                 os.add_dll_directory(path)
    #             else:
    #                 if path not in os.environ["PATH"]:
    #                     os.environ["PATH"] = (
    #                         path + os.pathsep + os.environ["PATH"]
    #                     )
    #             logger.info(f"Added {path} to DLL search path")
    #         except Exception as e:
    #             logger.error(f"Error adding DLL directory {path}: {str(e)}")

    def is_compound_word(self, word: str) -> bool:
        """Enhanced compound word detection for Finnish."""
        if not word:
            return False

        try:
            word_lower = word.lower()

            # First try Voikko analysis
            if self.voikko:
                analyses = self.voikko.analyze(word)
                if analyses:
                    # Check WORDBASES for compound structure
                    wordbases = analyses[0].get("WORDBASES", "")
                    if (
                        "+" in wordbases[1:]
                    ):  # Skip first + which marks word start
                        return True

            # Handle hyphenated words
            if "-" in word:
                return True

            # Check against core terms that are known compounds
            if word_lower in self.CORE_TERMS:
                for prefix in self.COMPOUND_PREFIXES:
                    if word_lower.startswith(prefix):
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking compound word {word}: {e}")
            return False

    def get_compound_parts(self, word: str) -> Optional[List[str]]:
        """Get compound parts with Voikko analysis."""
        if not word:
            return None

        try:
            # First try Voikko
            if self.voikko:
                analyses = self.voikko.analyze(word)
                if analyses and "+" in analyses[0].get("WORDBASES", ""):
                    parts = []
                    for part in analyses[0]["WORDBASES"].split("+"):
                        if "(" in part:
                            base = part.split("(")[1].split(")")[0]
                            if len(base) > 2:  # Skip short connectors
                                parts.append(base)
                    if len(parts) > 1:
                        return parts

            # Handle hyphenated words directly
            if "-" in word:
                return [
                    p.strip() for p in word.split("-") if len(p.strip()) > 2
                ]

            return None

        except Exception as e:
            logger.error(f"Error getting compound parts for {word}: {e}")
            return None

    def debug_voikko_analysis(self, word: str) -> None:
        """Debug helper to print detailed Voikko analysis."""
        try:
            analyses = self.voikko.analyze(word)
            logger.debug(f"\nDetailed Voikko analysis for word: '{word}'")
            logger.debug("-" * 50)

            if not analyses:
                logger.debug("No Voikko analysis available")
                return

            for i, analysis in enumerate(analyses):
                logger.debug(f"\nAnalysis #{i+1}:")
                for key, value in analysis.items():
                    logger.debug(f"  {key}: {value}")

        except Exception as e:
            logger.error(f"Error in debug analysis for {word}: {e}")

    def get_base_form(self, word: str) -> str:
        """Get base form with error handling."""
        if not word:
            return ""

        analysis = self._analyze_word_safe(word)
        if analysis and "BASEFORM" in analysis:
            base = analysis["BASEFORM"]
            logger.debug(f"Base form for {word}: {base}")
            return base.lower()
        return word.lower()

    def _analyze_word_safe(self, word: str) -> Optional[Dict[str, Any]]:
        """Thread-safe word analysis with failure tracking."""
        if not self.voikko or word in self._failed_words:
            return None

        try:
            if word in self._analysis_cache:
                return self._analysis_cache[word]

            with threading.Lock():  # Thread-safe Voikko access
                analysis = self.voikko.analyze(word)
                if not analysis:
                    self._failed_words.add(word)
                    return None

                result = analysis[0]
                self._analysis_cache[word] = result
                return result

        except Exception as e:
            logger.debug(f"Error analyzing word {word}: {str(e)}")
            self._failed_words.add(word)
            return None

    def _analyze_word(self, word: str) -> Optional[Dict[str, Any]]:
        """Cached word analysis."""
        if not self.voikko:
            return None

        try:
            # Check cache first
            if word in self._analysis_cache:
                return self._analysis_cache[word]

            analysis = self.voikko.analyze(word)
            if not analysis:
                return None

            # Cache and return first analysis
            result = analysis[0]
            self._analysis_cache[word] = result
            return result

        except Exception as e:
            logger.debug(f"Error analyzing word {word}: {str(e)}")
            return None

    def get_compound_parts(self, word: str) -> Optional[List[str]]:
        """Get compound parts safely."""
        analysis = self._analyze_word_safe(word)
        if not analysis:
            return None

        try:
            wordbases = analysis.get("WORDBASES", "")
            if not wordbases or "+" not in wordbases[1:]:
                return None

            parts = []
            for part in wordbases.split("+"):
                if "(" in part:
                    base = part.split("(")[1].split(")")[0]
                    if base and len(base) > 2:
                        parts.append(base)

            return parts if len(parts) > 1 else None

        except Exception as e:
            logger.debug(f"Error getting compound parts for {word}: {str(e)}")
            return None

    def should_keep_word(self, word: str) -> bool:
        """Word filtering with cached analysis."""
        try:
            if not super().should_keep_word(word):
                return False

            analysis = self._analyze_word(word)
            if not analysis:
                return False

            word_class = analysis.get("CLASS")
            if word_class == "teonsana":
                return False

            base_form = analysis.get("BASEFORM", "").lower()
            return not (
                base_form in self._stop_words
                or len(base_form) < self.config.get("min_word_length", 3)
            )
        except Exception as e:
            logger.debug(f"Error in should_keep_word for {word}: {str(e)}")
            return super().should_keep_word(word)

    def is_verb(self, word: str) -> bool:
        """Improved verb detection."""
        try:
            if not self.voikko:
                return False

            analyses = self.voikko.analyze(word)
            if not analyses:
                return False

            for analysis in analyses:
                # Check both CLASS and FSTOUTPUT
                if (
                    analysis.get("CLASS") == "teonsana"
                    or "[lt]" in analysis.get("FSTOUTPUT", "").lower()
                ):
                    return True

                # Check known verb forms
                base_form = analysis.get("BASEFORM", "").lower()
                known_verb_starts = {"para", "parane", "parantu", "kasva"}
                if any(
                    base_form.startswith(part) for part in known_verb_starts
                ):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking verb status for {word}: {e}")
            return False

    def get_word_info(self, word: str) -> Dict[str, Any]:
        """Get comprehensive word information from Voikko."""
        try:
            if not self.voikko:
                return {}

            analyses = self.voikko.analyze(word)
            if not analyses:
                return {}

            analysis = analyses[0]
            return {
                "base_form": analysis.get("BASEFORM", "").lower(),
                "class": analysis.get("CLASS"),
                "structure": analysis.get("STRUCTURE", ""),
                "word_bases": analysis.get("WORDBASES", ""),
                "is_compound": self.is_compound_word(word),
                "compound_parts": self.get_compound_parts(word),
            }

        except Exception as e:
            logger.error(f"Error getting word info for {word}: {e}")
            return {}

    def get_pos_tag(self, word: str) -> Optional[str]:
        """Get part-of-speech tag using Voikko.

        Maps Voikko classes to standard POS tags:
        - nimisana -> NN (nouns)
        - laatusana -> JJ (adjectives)
        - teonsana -> VB (verbs)
        - seikkasana -> RB (adverbs)
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
                "etunimi": "NNP",  # Proper noun (first name)
                "sukunimi": "NNP",  # Proper noun (last name)
                "paikannimi": "NNP",  # Proper noun (place name)
            }

            word_class = analyses[0].get("CLASS", "")
            return class_mapping.get(word_class)

        except Exception as e:
            logger.error(f"Error getting POS tag for '{word}': {e}")
            return None

    def _load_stop_words(self) -> Set[str]:
        """Load Finnish stopwords and add technical terms."""
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
                # Generic business/measurement terms
                "prosentti",
                "määrä",
                "osuus",
                "osa",
                "vaihe",
                "taso",
                "mittari",
                "yksikkö",
                "lukumäärä",
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

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Finnish text with proper morphological analysis."""
        if not text:
            return []

        try:
            # First preprocess the text
            text = self.preprocess_text(text)

            if self.voikko:
                tokens = self.voikko.tokens(text)
                # Filter for word tokens and process them
                words = []
                for token in tokens:
                    if (
                        hasattr(token, "tokenType") and token.tokenType == 1
                    ):  # Word tokens
                        word = token.tokenText.strip()
                        if word and self.should_keep_word(word):
                            words.append(word)
                return words
            else:
                # Fallback tokenization
                return [
                    word
                    for word in text.split()
                    if word.strip() and self.should_keep_word(word.strip())
                ]

        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return []

    def is_technical_term(self, word: str) -> bool:
        """Check if word is a recognized technical term."""
        # First check compound words for technical prefixes
        if self.is_compound_word(word):
            parts = self.get_compound_parts(word)
            if parts and parts[0] in self.COMPOUND_PREFIXES:
                return True

        # Check Voikko analysis for domain information
        try:
            if self.voikko:
                analyses = self.voikko.analyze(word)
                if analyses:
                    # Check if it's a recognized technical term
                    wordbases = analyses[0].get("WORDBASES", "")
                    return any(
                        prefix in wordbases.lower()
                        for prefix in [
                            "tekn",
                            "tiet",
                            "ohjelm",
                            "data",
                            "verkko",
                            "järj",
                        ]
                    )
        except Exception:
            pass

        return False

    # In FinnishTextProcessor class

    def process_compound_word(self, word: str) -> Optional[List[str]]:
        """Process compound word with Voikko analysis."""
        logger.debug(f"Processing compound: {word}")

        if not self.voikko:
            return None

        try:
            analyses = self.voikko.analyze(word)
            if not analyses:
                return None

            # Get both BASEFORM and WORDBASES
            analysis = analyses[0]
            base_form = analysis.get("BASEFORM", "")
            wordbases = analysis.get("WORDBASES", "")

            logger.debug(f"Base form: {base_form}")
            logger.debug(f"Word bases: {wordbases}")

            # Extract compound parts from WORDBASES
            if "+" in wordbases:
                parts = []
                for part in wordbases.split("+"):
                    if "(" in part:  # Format: "kone(kone)"
                        base = part.split("(")[1].rstrip(")")
                        if len(base) > 2:  # Skip short connecting parts
                            parts.append(base)
                            logger.debug(f"Found part: {base}")

                if len(parts) > 1:
                    logger.debug(f"Final compound parts: {parts}")
                    return parts

            return None

        except Exception as e:
            logger.error(f"Error processing compound: {e}")
            return None

    def preprocess_text(self, text: str) -> str:
        """Preprocess Finnish text with proper encoding."""
        if not isinstance(text, str):
            text = str(text)

        # Keep Finnish/Swedish letters and basic punctuation
        text = re.sub(r"[^a-zäöåA-ZÄÖÅ0-9\s\-.,]", " ", text, flags=re.UNICODE)

        # Normalize whitespace
        text = " ".join(text.split())

        return text.strip()


# src/core/language_processing/finnish.py


class VoikkoHandler:
    """Handler for Voikko initialization with proper cleanup."""

    def __init__(self):
        """Initialize handler."""
        self.voikko = None
        self.initialized = False  # Add explicit initialization

    def initialize(self, voikko_path: Optional[str] = None) -> Optional[Voikko]:
        """Initialize Voikko safely."""
        self._cleanup_existing()  # Clean up any existing instance

        try:
            # Try explicit paths first
            paths = self._get_search_paths(sys.platform, voikko_path)
            for path in paths:
                if not os.path.exists(path):
                    continue

                try:
                    if sys.platform == "win32":
                        self._add_dll_directory(path)

                    self.voikko = Voikko("fi", str(path))
                    self.voikko.analyze("testi")  # Test
                    self.initialized = True  # Mark as successfully initialized
                    logger.info(
                        f"Successfully initialized Voikko with path: {path}"
                    )
                    return self.voikko
                except Exception as e:
                    logger.debug(f"Failed to initialize with {path}: {e}")
                    self._cleanup_existing()
                    continue

            # Try system libraries as fallback
            self.voikko = Voikko("fi")
            self.voikko.analyze("testi")
            self.initialized = True  # Mark as successfully initialized
            logger.info(
                "Successfully initialized Voikko using system libraries"
            )
            return self.voikko

        except Exception as e:
            self._cleanup_existing()
            logger.debug(f"System library initialization failed: {e}")
            return None

    def _cleanup_existing(self):
        """Clean up existing Voikko instance."""
        if self.initialized and self.voikko:
            try:
                self.voikko.terminate()
            except Exception as e:
                logger.debug(f"Error during Voikko cleanup: {e}")
        self.voikko = None
        self.initialized = False

    def _get_search_paths(
        self, platform: str, custom_path: Optional[str] = None
    ) -> List[str]:
        """Get platform-specific Voikko search paths."""
        # Only return platform-specific paths
        if platform == "win32":
            base_paths = [
                r"C:\scripts\Voikko",
                r"C:\Program Files\Voikko",
                r"C:\Voikko",
                "~/Voikko",
            ]
        elif platform == "linux":
            base_paths = [
                "/usr/lib/voikko",
                "/usr/local/lib/voikko",
                "/usr/share/voikko",
                "~/voikko",
            ]
        elif platform == "darwin":
            base_paths = [
                "/usr/local/lib/voikko",
                "/opt/voikko",
                "~/voikko",
            ]
        else:
            base_paths = []  # Unsupported platform

        # Add custom path at the start if provided
        if custom_path:
            base_paths.insert(0, os.path.expanduser(custom_path))

        return [os.path.expanduser(p) for p in base_paths]

    def _add_dll_directory(self, path: str) -> None:
        """Add directory to DLL search path on Windows."""
        if sys.platform == "win32":
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(path)
                else:
                    if path not in os.environ["PATH"]:
                        os.environ["PATH"] = (
                            path + os.pathsep + os.environ["PATH"]
                        )
                logger.info(f"Added {path} to DLL search path")
            except Exception as e:
                logger.error(f"Error adding DLL directory {path}: {str(e)}")

    def __del__(self):
        """Clean up Voikko instance on deletion."""
        if hasattr(self, "initialized") and hasattr(self, "voikko"):
            if self.initialized and self.voikko:
                try:
                    self.voikko.terminate()
                except Exception as e:
                    logger.debug(f"Error during Voikko cleanup in __del__: {e}")
