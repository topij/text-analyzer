# src/core/language_processing/english.py

import logging
from typing import Any, Dict, List, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.core.language_processing.base import BaseTextProcessor
# from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data."""
    required_packages = [
        "punkt_tab",
        "wordnet",
        "averaged_perceptron_tagger",
        "stopwords"
    ]
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK package {package}: {e}")

# Download resources when module is imported
download_nltk_data()

class EnglishTextProcessor(BaseTextProcessor):
    """English text processor using NLTK."""

    def __init__(
        self,
        language: str = "en",
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize English text processor."""
        # Initialize NLTK components first
        self.lemmatizer = WordNetLemmatizer()
        
        # Call parent init which will call _load_stop_words
        super().__init__(language, custom_stop_words, config)
        
        logger.info(f"Initialized English processor with {len(self._stop_words)} stopwords")

    def _load_stop_words(self) -> Set[str]:
        """Load English stopwords from multiple sources."""
        try:
            stop_words = set()
            
            # 1. Load NLTK stopwords
            nltk_stops = set(word.lower() for word in stopwords.words('english'))
            logger.debug(f"Loaded {len(nltk_stops)} NLTK stopwords")
            stop_words.update(nltk_stops)
            
            # 2. Load additional stopwords from file
            config_dir = self.get_data_path("configurations")
            stop_words_path = config_dir / "stop_words" / "en.txt"
            
            if stop_words_path.exists():
                with open(stop_words_path, 'r', encoding='utf-8') as f:
                    file_stops = {line.strip().lower() for line in f if line.strip()}
                    logger.debug(f"Loaded {len(file_stops)} additional stopwords from file")
                    stop_words.update(file_stops)
            
            # 3. Add common contractions and special cases
            contractions = {
                "'s", "'t", "'re", "'ve", "'ll", "'d", "n't",
                "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
                "i've", "you've", "we've", "they've",
                "won't", "wouldn't", "can't", "cannot", "couldn't",
                "mustn't", "shouldn't", "wouldn't"
            }
            stop_words.update(contractions)

            # Add common English words often not in NLTK's list
            additional_words = {
                # Modal verbs and auxiliaries
                "would", "could", "should", "must", "might", "may",
                
                # Common adverbs and conjunctions
                "really", "actually", "probably", "usually", "certainly",
                "however", "therefore", "moreover", "furthermore", "nevertheless",
                
                # Business/technical common words
                "etc", "eg", "ie", "example", "use", "using", "used", "new",
                "way", "ways", "like", "also", "though", "although",
                
                # Numbers and measurements
                "one", "two", "three", "many", "much", "several", "various",
                
                # Common adjectives
                "good", "better", "best", "bad", "worse", "worst",
                "big","biggest","small","smaller","smallest",
                "high", "higher", "highest", "low", "lower", "lowest"
            }
            stop_words.update(additional_words)
            
            logger.info(f"Initialized English processor with {len(stop_words)} stopwords")
            return stop_words
            
        except Exception as e:
            logger.error(f"Error loading stopwords: {e}")
            return set()

    def get_base_form(self, word: str) -> str:
        """Get base form (lemma) of an English word."""
        try:
            if not word:
                return ""
                
            word = self.handle_contractions(word)
            
            # Get POS tag
            pos_tag = self._get_wordnet_pos(word)

            # Lemmatize with POS tag if available
            if pos_tag:
                return self.lemmatizer.lemmatize(word.lower(), pos=pos_tag)

            # Try different POS tags to get shortest lemma
            lemmas = [
                self.lemmatizer.lemmatize(word.lower(), pos=pos)
                for pos in ['n', 'v', 'a', 'r']
            ]

            return min(lemmas, key=len)

        except Exception as e:
            logger.error(f"Error getting base form for '{word}': {str(e)}")
            return word.lower() if word else ""

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
            "i'm": "i", "you're": "you", "he's": "he", "she's": "she",
            "it's": "it", "we're": "we", "they're": "they",
            "i've": "i", "you've": "you", "we've": "we", "they've": "they",
            "won't": "will", "wouldn't": "would", "can't": "can",
            "cannot": "can", "couldn't": "could", "shouldn't": "should",
            "mustn't": "must"
        }
        
        return contractions_map.get(word_lower, word)

    def _get_wordnet_pos(self, word: str) -> Optional[str]:
        """Get WordNet POS tag for a word."""
        try:
            if not word:
                return None
                
            pos = nltk.pos_tag([word])[0][1]
            tag_map = {
                'JJ': 'a',  # Adjective
                'VB': 'v',  # Verb
                'NN': 'n',  # Noun
                'RB': 'r',  # Adverb
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
        if len(word_lower) == 1 and word_lower not in ['a', 'i']:
            return False

        return True