# src/core/language_processing/english.py

import logging
from typing import Any, Dict, List, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.core.language_processing.base import BaseTextProcessor
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class EnglishTextProcessor(BaseTextProcessor):
    """English text processor using NLTK."""

    # Update compound word parts dictionary
    COMPOUND_PARTS = {
        # Technical terms
        "pipeline": ["pipe", "line"],
        "devops": ["development", "operations"],
        "streamline": ["stream", "line"],
        "database": ["data", "base"],
        "middleware": ["middle", "ware"],
        "workflow": ["work", "flow"],
        # Business terms
        "stakeholder": ["stake", "holder"],
        "benchmark": ["bench", "mark"],
        "timeline": ["time", "line"],
        "framework": ["frame", "work"],
    }
    # Common business terms
    BUSINESS_TERMS = {
        # Financial terms
        "revenue",
        "profit",
        "margin",
        "cost",
        "growth",
        "sales",
        "market",
        "financial",
        "performance",
        "quarterly",
        "fiscal",
        "budget",
        "investment",
        # Business metrics
        "roi",
        "kpi",
        "metric",
        "benchmark",
        "target",
        # Customer related
        "customer",
        "client",
        "retention",
        "acquisition",
        "satisfaction",
        "engagement",
        "conversion",
        # Strategy terms
        "strategy",
        "initiative",
        "expansion",
        "optimization",
        "efficiency",
        "productivity",
        "implementation",
        # Market terms
        "market",
        "segment",
        "sector",
        "industry",
        "competitive",
        "opportunity",
        "penetration",
        "share",
    }

    def __init__(
        self,
        language: str = "en",
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        file_utils: Optional[FileUtils] = None,
    ):
        """Initialize English text processor.

        Args:
            language: Language code ('en')
            custom_stop_words: Optional set of additional stop words
            config: Configuration parameters
            file_utils: FileUtils instance for file operations
        """
        # Initialize contractions first
        self.contractions = {
            "i'm": "i",
            "you're": "you",
            "he's": "he",
            "she's": "she",
            "it's": "it",
            "we're": "we",
            "they're": "they",
            "i've": "i",
            "you've": "you",
            "we've": "we",
            "they've": "they",
            "won't": "will",
            "wouldn't": "would",
            "can't": "can",
            "cannot": "can",
            "couldn't": "could",
            "shouldn't": "should",
            "mustn't": "must"
        }

        # Initialize NLTK resources
        self._ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()

        # Call parent init after initializing our own attributes
        super().__init__(
            language=language,
            custom_stop_words=custom_stop_words,
            config=config,
            file_utils=file_utils
        )
        
        # Initialize stop words
        self.stop_words = self._load_stop_words()

    def _ensure_nltk_data(self):
        """Initialize NLTK with required data."""
        required_resources = {
            'punkt': 'tokenizers/punkt',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'wordnet': 'corpora/wordnet',
            'stopwords': 'corpora/stopwords'
        }
        
        try:
            # Verify required resources
            for resource, path in required_resources.items():
                try:
                    nltk.data.find(path)
                    logger.debug(f"Found NLTK resource: {resource}")
                except LookupError as e:
                    logger.error(f"Required NLTK resource '{resource}' not found. Please run setup_dev.sh to install all required resources.")
                    continue

            # Test core functionality
            try:
                # Test tokenizer
                test_text = "test"
                tokens = word_tokenize(test_text)
                if tokens:
                    logger.debug("Tokenizer working")

                # Test POS tagger
                try:
                    tags = nltk.pos_tag([test_text])
                    if tags:
                        logger.debug("POS tagger working")
                except LookupError:
                    logger.warning("POS tagger not available, falling back to basic noun assumption")

                # Test WordNet
                try:
                    from nltk.corpus import wordnet as wn
                    if wn.synsets(test_text):
                        logger.debug("WordNet working")
                except LookupError:
                    logger.warning("WordNet not available")

                # Test stopwords
                try:
                    from nltk.corpus import stopwords
                    if stopwords.words("english"):
                        logger.debug("Stopwords working")
                except LookupError:
                    logger.warning("Stopwords not available")

            except Exception as e:
                logger.error(f"Functionality test failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error verifying NLTK resources: {str(e)}")
            # Don't raise - allow fallback behavior for missing resources

    def _get_stop_words(self) -> Set[str]:
        """Get English stop words."""
        try:
            # Basic English stop words
            stop_words = {
                "a", "an", "and", "are", "as", "at", "be", "by", "for",
                "from", "has", "he", "in", "is", "it", "its", "of", "on",
                "that", "the", "to", "was", "were", "will", "with", "the",
                "i", "you", "he", "she", "it", "we", "they",
                "i'm", "you're"
            }
            
            # Add contractions and their base forms
            stop_words.update(self.contractions.keys())
            stop_words.update(self.contractions.values())
            
            return stop_words
            
        except Exception as e:
            logger.error(f"Error getting stop words: {str(e)}")
            return set()

    def _load_stop_words(self) -> Set[str]:
        """Load English stopwords from multiple sources."""
        try:
            stop_words = set()

            # 1. Load NLTK stopwords if available
            try:
                nltk.data.find('corpora/stopwords')
                nltk_stops = set(word.lower() for word in stopwords.words("english"))
                stop_words.update(nltk_stops)
            except LookupError:
                logger.debug("NLTK stopwords not available, using basic set")
                # Use our basic set if NLTK's not available
                stop_words.update(self._get_stop_words())

            # 2. Load additional stopwords from file if available
            if self.file_utils:
                try:
                    config_dir = self.file_utils.get_data_path("config")
                    stop_words_path = config_dir / "stop_words" / "en.txt"
                    if stop_words_path.exists():
                        with open(stop_words_path, "r", encoding="utf-8") as f:
                            file_stops = {line.strip().lower() for line in f if line.strip()}
                            stop_words.update(file_stops)
                except Exception as e:
                    logger.debug(f"Could not load additional stopwords from file: {e}")

            # 3. Add contractions and their base forms
            stop_words.update(self.contractions.keys())
            stop_words.update(self.contractions.values())

            # 4. Add common English words often not in NLTK's list
            additional_words = {
                # Modal verbs and auxiliaries
                "would", "could", "should", "must", "might", "may",
                # Common adverbs and conjunctions
                "really", "actually", "probably", "usually", "certainly",
                "however", "therefore", "moreover", "furthermore", "nevertheless",
                # Business/technical common words
                "etc", "eg", "ie", "example", "use", "using", "used",
                "new", "way", "ways", "like", "also", "though", "although",
                # Numbers and measurements
                "one", "two", "three", "many", "much", "several", "various",
            }
            stop_words.update(additional_words)

            return stop_words

        except Exception as e:
            logger.error(f"Error loading stopwords: {e}")
            # Fall back to basic stop words if loading fails
            return self._get_stop_words()

    def get_base_form(self, text: str) -> Optional[str]:
        """Get base form of a word or phrase."""
        try:
            # Tokenize the text
            tokens = word_tokenize(text.lower())
            
            # Get POS tags
            try:
                pos_tags = nltk.pos_tag(tokens)
            except LookupError:
                logger.warning("POS tagger not available, falling back to basic lemmatization")
                return ' '.join(self.lemmatizer.lemmatize(word) for word in tokens)
            
            # Lemmatize based on POS
            lemmatized = []
            for word, pos in pos_tags:
                # Convert Penn Treebank tag to WordNet tag
                tag = self._get_wordnet_pos(pos)
                if tag:
                    lemma = self.lemmatizer.lemmatize(word, tag)
                else:
                    lemma = self.lemmatizer.lemmatize(word)
                lemmatized.append(lemma)
            
            return ' '.join(lemmatized)
            
        except Exception as e:
            logger.error(f"Error getting base form for '{text}': {str(e)}")
            return text.strip()

    def handle_contractions(self, word: str) -> str:
        """Expand contractions to their base form."""
        return self.contractions.get(word.lower(), word)

    def _get_wordnet_pos(self, word: str) -> Optional[str]:
        """Get WordNet POS tag for a word."""
        try:
            if not word:
                return None

            pos = nltk.pos_tag([word])[0][1]
            tag_map = {
                "JJ": "a",  # Adjective
                "VB": "v",  # Verb
                "NN": "n",  # Noun
                "RB": "r",  # Adverb
            }
            for prefix, tag in tag_map.items():
                if pos.startswith(prefix):
                    return tag
            return None

        except Exception as e:
            logger.error(f"Error getting POS tag for '{word}': {str(e)}")
            return None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text using NLTK."""
        try:
            if not text:
                return []

            # First preprocess the text
            text = self.preprocess_text(text)

            # Use NLTK's tokenizer
            tokens = word_tokenize(text)

            # Process tokens
            processed_tokens = []
            for token in tokens:
                # Skip punctuation
                if self.is_punctuation(token):
                    continue

                # Handle contractions and check stopwords
                processed = self.handle_contractions(token)
                if processed and not self.is_stop_word(processed):
                    processed_tokens.append(processed)

            return processed_tokens

        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return []

    def should_keep_word(self, word: str) -> bool:
        """Determine if word should be kept based on English-specific rules."""
        if not word:
            return False

        word_lower = word.lower()

        # First check base criteria
        if not super().should_keep_word(word):
            return False

        # Always check lowercase version against stopwords
        if word_lower in self.stop_words:
            return False

        # Skip single letters except 'a' and 'i'
        if len(word_lower) == 1 and word_lower not in ["a", "i"]:
            return False

        return True

    # Add debug logging to EnglishTextProcessor methods:

    def is_compound_word(self, word: str) -> bool:
        """Enhanced compound word detection."""
        word_lower = word.lower()

        # Check predefined compounds
        if word_lower in self.COMPOUND_PARTS:
            return True

        # Check hyphenation
        if "-" in word:
            return True

        # Check camelCase and PascalCase
        if any(c.isupper() for c in word[1:]):
            return True

        # Check common compound patterns
        common_joining_parts = ["based", "driven", "oriented", "ready", "aware"]
        return any(part in word_lower for part in common_joining_parts)

    def get_compound_parts(self, word: str) -> List[str]:
        """Get parts of compound word with improved accuracy."""
        word_lower = word.lower()

        # Check predefined parts first
        if word_lower in self.COMPOUND_PARTS:
            return self.COMPOUND_PARTS[word_lower]

        # Handle hyphenated words
        if "-" in word:
            parts = [part.strip() for part in word.split("-")]
            return [part for part in parts if part]

        # Handle camelCase/PascalCase
        if any(c.isupper() for c in word[1:]):
            import re

            parts = re.findall("[A-Z][^A-Z]*", word)
            if parts:
                return [part.lower() for part in parts]

        # Default to original word if no compound structure found
        return [word]

    def _split_compound_word(self, word: str) -> List[str]:
        """Split compound word into parts."""
        # Base case
        if len(word) <= 3:
            return [word]

        # Try to find known word combinations
        for i in range(3, len(word) - 2):
            left = word[:i]
            right = word[i:]

            left_base = self.get_base_form(left)
            right_base = self.get_base_form(right)

            left_valid = (
                left_base in self._get_wordnet_words()
                and not self.is_stop_word(left_base)
            )
            right_valid = (
                right_base in self._get_wordnet_words()
                and not self.is_stop_word(right_base)
            )

            if left_valid and right_valid:
                return [left_base, right_base]

        return [word]

    def get_pos_tag(self, word: str) -> Optional[str]:
        """Get part-of-speech tag for a word."""
        try:
            if not word:
                return None

            # Clean the word - remove punctuation at the end
            word = word.strip().rstrip('.,!?:;')

            # Check technical terms first
            tech_pos_map = {
                "api": "NN",
                "cloud": "NN",
                "data": "NN",
                "code": "NN",
            }
            word_lower = word.lower()
            if word_lower in tech_pos_map:
                return tech_pos_map[word_lower]

            # Handle special cases first
            if word.isupper() or "-" in word:  # Acronym or technical term
                return "NN"

            # Get NLTK tag with error handling
            try:
                pos = nltk.pos_tag([word])[0][1]
                return pos
            except LookupError:
                logger.warning("POS tagger not available, falling back to basic noun assumption")
                return "NN"  # Default to noun
            except Exception as e:
                logger.error(f"Error in POS tagging: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error getting POS tag for '{word}': {str(e)}")
            return None

    def _evaluate_quality(self, word: str) -> float:
        """Enhanced quality evaluation with better business term handling."""
        if not self.language_processor:
            return 1.0

        word_lower = word.lower()
        quality_score = 1.0

        # Check if it's a business term
        if word_lower in self.BUSINESS_TERMS:
            return 1.0  # Always give full score to business terms
