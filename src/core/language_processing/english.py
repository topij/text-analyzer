# src/core/language_processing/english.py

import logging
from typing import Any, Dict, List, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.core.language_processing.base import BaseTextProcessor

# from FileUtils import FileUtils

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
    ):
        # Initialize NLTK first
        self._ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()

        # Call parent init
        super().__init__(language, custom_stop_words, config)

    def _ensure_nltk_data(self):
        """Initialize NLTK with required data."""
        try:
            for resource in [
                "punkt_tab",
                "averaged_perceptron_tagger",
                "wordnet",
            ]:
                try:
                    nltk.data.find(
                        f"corpora/{resource}"
                        if resource == "wordnet"
                        else f"tokenizers/{resource}"
                    )
                except LookupError:
                    nltk.download(resource, quiet=True)

            # Force WordNet to load
            from nltk.corpus import wordnet as wn

            wn.synsets("test")

        except Exception as e:
            logger.warning(f"NLTK initialization warning: {e}")

    def _load_stop_words(self) -> Set[str]:
        """Load English stopwords from multiple sources."""
        try:
            stop_words = set()

            # 1. Load NLTK stopwords
            nltk_stops = set(
                word.lower() for word in stopwords.words("english")
            )
            logger.debug(
                f"Loaded {len(nltk_stops)} NLTK stopwords"
            )  # Changed to DEBUG
            stop_words.update(nltk_stops)

            # 2. Load additional stopwords from file
            config_dir = self.get_data_path("configurations")
            stop_words_path = config_dir / "stop_words" / "en.txt"

            if stop_words_path.exists():
                with open(stop_words_path, "r", encoding="utf-8") as f:
                    file_stops = {
                        line.strip().lower() for line in f if line.strip()
                    }
                    logger.debug(
                        f"Loaded {len(file_stops)} additional stopwords from file"
                    )  # Changed to DEBUG
                    stop_words.update(file_stops)

            # 3. Add common contractions and special cases
            contractions = {
                "'s",
                "'t",
                "'re",
                "'ve",
                "'ll",
                "'d",
                "n't",
                "i'm",
                "you're",
                "he's",
                "she's",
                "it's",
                "we're",
                "they're",
                "i've",
                "you've",
                "we've",
                "they've",
                "won't",
                "wouldn't",
                "can't",
                "cannot",
                "couldn't",
                "mustn't",
                "shouldn't",
                "wouldn't",
            }
            stop_words.update(contractions)

            # Add common English words often not in NLTK's list
            additional_words = {
                # Modal verbs and auxiliaries
                "would",
                "could",
                "should",
                "must",
                "might",
                "may",
                # Common adverbs and conjunctions
                "really",
                "actually",
                "probably",
                "usually",
                "certainly",
                "however",
                "therefore",
                "moreover",
                "furthermore",
                "nevertheless",
                # Business/technical common words
                "etc",
                "eg",
                "ie",
                "example",
                "use",
                "using",
                "used",
                "new",
                "way",
                "ways",
                "like",
                "also",
                "though",
                "although",
                # Numbers and measurements
                "one",
                "two",
                "three",
                "many",
                "much",
                "several",
                "various",
                # Common adjectives
                "good",
                "better",
                "best",
                "bad",
                "worse",
                "worst",
                "big",
                "biggest",
                "small",
                "smaller",
                "smallest",
                "high",
                "higher",
                "highest",
                "low",
                "lower",
                "lowest",
            }
            stop_words.update(additional_words)

            logger.debug(
                f"Total English stopwords: {len(stop_words)}"
            )  # Changed to DEBUG
            return stop_words

        except Exception as e:
            logger.error(f"Error loading stopwords: {e}")
            return set()

    def get_base_form(self, word: str) -> str:
        """Get base form with proper initialization check."""
        if not word:
            return ""

        try:
            word = self.handle_contractions(word)

            # Get POS tag
            pos = self._get_wordnet_pos(word)
            if pos:
                return self.lemmatizer.lemmatize(word.lower(), pos=pos)

            # Try different POS tags
            lemmas = [
                self.lemmatizer.lemmatize(word.lower(), pos=p)
                for p in ["n", "v", "a", "r"]
            ]
            return min(lemmas, key=len)

        except Exception as e:
            logger.debug(f"Error getting base form for '{word}': {e}")
            return word.lower()

    def handle_contractions(self, word: str) -> str:
        """Handle contractions and possessives."""
        if not word:
            return ""

        word_lower = word.lower()

        # Split possessives from nouns
        if word_lower.endswith("'s"):
            return word[:-2]

        # Map common contractions
        contractions_map = {
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
            "mustn't": "must",
        }

        return contractions_map.get(word_lower, word)

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
        if word_lower in self._stop_words:
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
        """Get part-of-speech tag for a word.

        Args:
            word: Word to analyze

        Returns:
            Optional[str]: POS tag or None if not determinable
        """
        try:
            if not word:
                return None

            # Check technical terms first
            tech_pos_map = {
                "api": "NN",
                "cloud": "NN",
                "data": "NN",
                "code": "NN",
            }
            if word.lower() in tech_pos_map:
                return tech_pos_map[word.lower()]

            # Get NLTK tag
            pos = nltk.pos_tag([word])[0][1]

            # Handle special cases
            if word.isupper() or "-" in word:  # Acronym or technical term
                return "NN"

            return pos

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
