# src/core/language_processing/finnish.py

# src/core/language_processing/finnish.py

import logging
import os
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
        "kone", "teko", "tieto", "ohjelmisto", "verkko", "järjestelmä",
        "käyttö", "kehitys", "palvelu", "asiakas", "laatu", "turva"
    }

    def __init__(
        self,
        language: str = "fi",
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        file_utils: Optional[FileUtils] = None
    ):
        """Initialize Finnish text processor.
        
        Args:
            language: Language code (should be 'fi')
            custom_stop_words: Optional set of additional stop words
            config: Configuration parameters including Voikko path
            file_utils: Optional FileUtils instance for file operations
        """
        super().__init__(language, custom_stop_words, config, file_utils)
        
        # Initialize Voikko
        voikko_path = config.get("voikko_path") if config else None
        self.voikko = self._initialize_voikko(voikko_path)
        
        if not self.voikko:
            logger.warning("Voikko initialization failed, some functionality will be limited")

    def _load_stop_words(self) -> Set[str]:
        """Load Finnish stopwords from file and add technical terms."""
        try:
            stop_words = set()
            
            # Load stopwords from file
            stopwords_path = self.get_data_path("configurations") / "stop_words" / "fi.txt"
            if stopwords_path.exists():
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    stop_words.update(
                        word.strip().lower()
                        for word in f
                        if word.strip()
                    )
                logger.info(f"Loaded {len(stop_words)} stopwords from {stopwords_path}")
            else:
                logger.warning(f"Stopwords file not found: {stopwords_path}")
            
            # Add common technical terms that should be excluded
            technical_stops = {
                # Common technical words that aren't meaningful alone
                "versio", "ohjelma", "järjestelmä", "sovellus", "tiedosto",
                "toiminto", "palvelu", "käyttö", "teksti", "tieto",
                
                # Common verbs in technical context
                "käyttää", "toimia", "suorittaa", "toteuttaa", "näyttää",
                "muokata", "lisätä", "poistaa", "hakea", "tallentaa",
                
                # Common adjectives in technical context
                "uusi", "vanha", "nykyinen", "aiempi", "seuraava",
                "erilainen", "samanlainen", "vastaava"
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

    def _initialize_voikko(self, voikko_path: Optional[str] = None) -> Optional[Voikko]:
        """Initialize Voikko with path handling."""
        try:
            # Get potential Voikko paths
            config_paths = []
            if voikko_path:
                config_paths.append(voikko_path)
                
            try:
                # Try to get paths from config
                config = self.load_config_file("language_processing.yaml")
                if paths := config.get("finnish", {}).get("voikko_paths", []):
                    config_paths.extend(paths)
            except Exception:
                pass
            
            # Default paths
            default_paths = [
                # Linux/WSL paths
                "/usr/lib/voikko",
                "/usr/local/lib/voikko",
                "/usr/share/voikko",
                # Home directory
                os.path.expanduser("~/voikko"),
                # Windows paths
                "C:/Program Files/Voikko",
                "C:/Voikko",
            ]
            
            # Combine all paths
            search_paths = config_paths + default_paths

            # Try direct initialization first (using system libraries)
            try:
                voikko = Voikko("fi")
                test_result = voikko.analyze("testi")
                if test_result:
                    logger.info("Successfully initialized Voikko using system libraries")
                    return voikko
            except Exception as e:
                logger.debug(f"Could not initialize Voikko using system libraries: {e}")

            # Try with explicit paths
            for path in search_paths:
                if not path:
                    continue
                
                try:
                    path = Path(path)
                    if not path.exists():
                        continue
                        
                    voikko = Voikko("fi", str(path))
                    # Test the initialization
                    test_result = voikko.analyze("testi")
                    if test_result:
                        logger.info(f"Successfully initialized Voikko with path: {path}")
                        return voikko
                        
                except Exception as e:
                    logger.debug(f"Failed to initialize Voikko with path {path}: {e}")
                    continue

            logger.error(
                "Could not initialize Voikko. Please ensure libvoikko and voikko-fi "
                "are installed. On Ubuntu/WSL: sudo apt-get install libvoikko-dev voikko-fi"
            )
            return None

        except Exception as e:
            logger.error(f"Failed to initialize Voikko: {str(e)}")
            return None

    # src/core/language_processing/finnish.py

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

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Finnish text."""
        if not isinstance(text, str):
            return []

        try:
            # First preprocess the text
            text = self.preprocess_text(text)
            
            if self.voikko:
                tokens = self.voikko.tokens(text)
                words = [
                    t.tokenText.strip()
                    for t in tokens
                    if hasattr(t, "tokenType") and t.tokenType == 1  # Word tokens
                ]
                return [w for w in words if w]
            else:
                # Fallback tokenization if Voikko is not available
                return super().tokenize(text)
            
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return super().tokenize(text)

    def is_compound_word(self, word: str) -> bool:
        """Check if word is a compound word."""
        if not self.voikko:
            return False

        try:
            analyses = self.voikko.analyze(word)
            if not analyses:
                return False

            analysis = analyses[0]
            return (
                "WORDBASES" in analysis
                and "+" in analysis["WORDBASES"]
            )

        except Exception as e:
            logger.error(f"Error checking compound word {word}: {e}")
            return False

    def get_compound_parts(self, word: str) -> List[str]:
        """Get parts of a compound word."""
        if not self.voikko:
            return [word]

        try:
            analyses = self.voikko.analyze(word)
            if not analyses:
                return [word]

            analysis = analyses[0]
            if "WORDBASES" not in analysis:
                return [word]

            word_bases = analysis["WORDBASES"]
            if "+" not in word_bases:
                return [word]

            # Extract the actual base forms from the analysis
            parts = []
            for part in word_bases.split("+"):
                if "(" in part:
                    base = part.split("(")[0]
                    parts.append(base)

            return parts if parts else [word]

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


# # src/core/language_processing/finnish.py

# import logging
# import os
# import re  # Add missing import
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Set

# from libvoikko import Voikko

# from .base import BaseTextProcessor

# logger = logging.getLogger(__name__)


# class FinnishTextProcessor(BaseTextProcessor):
#     """Finnish text processor using Voikko.

#     Handles Finnish-specific text processing including:
#     - Word base form extraction
#     - Compound word handling
#     - Finnish-specific token types
#     - Special case mappings
#     """

#     # Token type constants
#     WORD_TOKEN = 1
#     WHITESPACE_TOKEN = 3

#     # Word class mappings
#     WORD_CLASSES = {
#         "nimisana": "noun",
#         "teonsana": "verb",
#         "laatusana": "adjective",
#         "seikkasana": "adverb",
#     }

#     # Special case word mappings
#     WORD_MAPPINGS = {
#         "osata": "osaaminen",
#         "tilaama": "tilata",
#         "pakattu": "pakata",
#         "viivästynyt": "viivästyä",
#         # Add other special cases
#     }

#     # Common Finnish compound mappings
#     COMPOUND_MAPPINGS = {
#         "ohjelmistokehittäjä": ["ohjelmisto", "kehittäjä"],
#         "ohjelmistokehitys": ["ohjelmisto", "kehitys"],
#         "ohjelmistoprojekti": ["ohjelmisto", "projekti"],
#         "ohjelmisto": ["ohjelmisto"],  # Add base word itself
#         "verkkokauppa": ["verkko", "kauppa"],
#         "asiakasprojekti": ["asiakas", "projekti"],
#         "tietoturva": ["tieto", "turva"],
#         "käyttöliittymä": ["käyttö", "liittymä"],
#         "kehittäjä": ["kehittäjä"],  # Add base word itself
#     }

#     def __init__(
#         self,
#         language: str = "fi",
#         custom_stop_words: Optional[Set[str]] = None,
#         voikko_path: Optional[str] = None,
#         config: Optional[Dict[str, Any]] = None,
#     ):
#         """Initialize Finnish text processor.

#         Args:
#             # custom_stop_words: Additional stop words
#             # voikko_path: Path to Voikko installation
#             # config: Configuration parameters
#         """
#         super().__init__(language, custom_stop_words, config)
#         self.voikko = self._initialize_voikko(voikko_path)
        
#         if not self.voikko:
#             logger.warning("Voikko initialization failed, some functionality will be limited")

#         # Add any custom compound mappings from config
#         if config and "compound_mappings" in config:
#             self.COMPOUND_MAPPINGS.update(config["compound_mappings"])

#         # Common compound words to preserve
#         # list of Finnish compound words?
#         self.compounds = {
#             "ohjelmistokehittäjä": ["ohjelmisto", "kehittäjä"],
#             "asiakasprojekti": ["asiakas", "projekti"],
#             "verkkokauppa": ["verkko", "kauppa"],
#             "tietoturva": ["tieto", "turva"],
#         }

#     def get_base_form(self, word: str) -> str:
#         """Get base form of a Finnish word."""
#         try:
#             if not self.voikko:
#                 return word.lower()
                
#             analyses = self.voikko.analyze(word)
#             if analyses:
#                 return analyses[0].get("BASEFORM", word).lower()
                
#             return word.lower()
            
#         except Exception as e:
#             logger.error(f"Error getting base form for {word}: {e}")
#             return word.lower()

#     def _tokenize_text(self, text: str) -> List[str]:
#         """Finnish-specific tokenization using Voikko if available."""
#         if self.voikko:
#             try:
#                 # Use Voikko's tokenization
#                 tokens = self.voikko.tokens(text)
#                 return [t.tokenText for t in tokens if t.tokenType == Voikko.TOKEN_TYPE.WORD]
#             except Exception as e:
#                 self.logger.error(f"Voikko tokenization failed: {str(e)}")

#         # Fall back to base implementation if Voikko fails or is not available
#         return super()._tokenize_text(text)

#     def _initialize_voikko(self, voikko_path: Optional[str] = None) -> Optional[Voikko]:
#         """Initialize Voikko with WSL compatibility."""
#         try:
#             # Common paths in different environments
#             default_paths = [
#                 # WSL and Linux paths
#                 "/usr/lib/voikko",
#                 "/usr/local/lib/voikko",
#                 "/usr/share/voikko",
#                 # Home directory
#                 os.path.expanduser("~/voikko"),
#                 # Windows paths if running in Windows
#                 "C:/Program Files/Voikko",
#                 "C:/Voikko",
#             ]
            
#             # Add any custom path to the beginning of the list
#             search_paths = ([voikko_path] if voikko_path else []) + default_paths
            
#             # Try direct initialization first (using system libraries)
#             try:
#                 voikko = Voikko("fi")
#                 test_result = voikko.analyze("testi")
#                 if test_result:
#                     logger.info("Successfully initialized Voikko using system libraries")
#                     return voikko
#             except Exception as e:
#                 logger.debug(f"Could not initialize Voikko using system libraries: {e}")

#             # Try with explicit paths
#             for path in search_paths:
#                 if not path:
#                     continue
                
#                 try:
#                     path = Path(path)
#                     if not path.exists():
#                         continue
                        
#                     # Handle WSL path if needed
#                     if sys.platform == "win32":
#                         if hasattr(os, "add_dll_directory"):
#                             os.add_dll_directory(str(path))
                    
#                     voikko = Voikko("fi", str(path))
#                     # Test the initialization
#                     test_result = voikko.analyze("testi")
#                     if test_result:
#                         logger.info(f"Successfully initialized Voikko with path: {path}")
#                         return voikko
                        
#                 except Exception as e:
#                     logger.debug(f"Failed to initialize Voikko with path {path}: {e}")
#                     continue

#             logger.error(
#                 "Could not initialize Voikko. Please ensure libvoikko and voikko-fi "
#                 "are installed. On Ubuntu/WSL: sudo apt-get install libvoikko-dev voikko-fi"
#             )
#             return None

#         except Exception as e:
#             logger.error(f"Failed to initialize Voikko: {str(e)}")
#             return None

#     def _get_word_base_form(self, analyses: List[Dict[str, Any]]) -> str:
#         """Get the most appropriate base form from analyses."""
#         try:
#             best_analysis = self._select_best_analysis(analyses)
#             if not best_analysis:
#                 return analyses[0].get("BASEFORM", "").lower()

#             return best_analysis.get("BASEFORM", "").lower()

#         except Exception as e:
#             logger.error(f"Error getting word base form: {str(e)}")
#             return analyses[0].get("BASEFORM", "").lower() if analyses else ""

#     def _select_best_analysis(self, analyses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
#         """Select the best analysis based on word class and form."""
#         for analysis in analyses:
#             word_class = analysis.get("CLASS")
#             if not word_class:
#                 continue

#             # Prefer nouns
#             if word_class == "nimisana":
#                 return analysis

#             # Then verbs in basic form
#             if word_class == "teonsana" and not analysis.get("PARTICIPLE"):
#                 return analysis

#         return analyses[0] if analyses else None
    
#     def should_keep_word(self, word: str) -> bool:
#         """Determine if word should be kept."""
#         # First check base criteria from parent class
#         if not super().should_keep_word(word):
#             return False
            
#         # Get base form for stopword check
#         base_form = self.get_base_form(word)
#         if self.is_stop_word(base_form):
#             return False
            
#         return True

#     def is_compound_word(self, word: str) -> bool:
#         """Check if word is a compound word."""
#         if not self.voikko:
#             return False

#         analyses = self.voikko.analyze(word)
#         if not analyses:
#             return False

#         return "+" in analyses[0].get("WORDBASES", "")

#     def get_compound_parts(self, word: str) -> List[str]:
#         """Get parts of a compound word."""
#         if not self.voikko:
#             return [word]

#         analyses = self.voikko.analyze(word)
#         if not analyses:
#             return [word]

#         word_bases = analyses[0].get("WORDBASES", "")
#         if "+" not in word_bases:
#             return [word]

#         return [part.split("(")[0] for part in word_bases.split("+") if part and "(" in part]

#     def tokenize(self, text: str) -> List[str]:
#         """Tokenize Finnish text."""
#         if not isinstance(text, str):
#             return []

#         try:
#             if self.voikko:
#                 tokens = self.voikko.tokens(text)
#                 words = [
#                     t.tokenText.strip()
#                     for t in tokens
#                     if hasattr(t, "tokenType") and t.tokenType == 1
#                 ]
#                 return [w for w in words if w]
#             else:
#                 # Fallback tokenization if Voikko is not available
#                 return super().tokenize(text)
            
#         except Exception as e:
#             logger.error(f"Tokenization error: {e}")
#             return super().tokenize(text)

#     def process_text(self, text: str) -> str:
#         """Process Finnish text with compound word focus."""
#         if not text:
#             return ""

#         try:
#             logger.debug(f"Processing Finnish text: {text}")

#             # First look for compound words in original text
#             compounds = set()
#             text_lower = text.lower()
#             for compound, parts in self.COMPOUND_MAPPINGS.items():
#                 if compound in text_lower:
#                     logger.debug(f"Found compound: {compound}")
#                     compounds.update(parts)
#                     compounds.add(compound)  # Keep the original compound too

#             # Preprocess and tokenize
#             processed = self.preprocess_text(text)
#             tokens = self.tokenize(processed)
#             logger.debug(f"Tokens: {tokens}")

#             # Process individual tokens
#             processed_words = set()
#             for token in tokens:
#                 token_lower = token.lower()

#                 if self.is_stop_word(token_lower):
#                     continue

#                 # Check for compound words again at token level
#                 found_compound = False
#                 for compound, parts in self.COMPOUND_MAPPINGS.items():
#                     if compound in token_lower or token_lower in compound:
#                         processed_words.update(parts)
#                         processed_words.add(compound)
#                         found_compound = True
#                         logger.debug(f"Found compound in token: {token} -> {parts}")
#                         break

#                 if not found_compound:
#                     base = self.get_base_form(token)
#                     if base and len(base) >= self.config.get("min_word_length", 3):
#                         processed_words.add(base)

#             # Combine all found words
#             all_words = compounds.union(processed_words)
#             logger.debug(f"All processed words: {all_words}")

#             return " ".join(all_words)

#         except Exception as e:
#             logger.error(f"Error processing Finnish text: {e}")
#             return text