# src/core/language_processing/base.py

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Any
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseTextProcessor(ABC):
    """Abstract base class for language-specific text processors."""
    
    def __init__(
        self,
        language: str,
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize text processor.
        
        Args:
            language: Language code ('fi' or 'en')
            custom_stop_words: Optional set of additional stop words
            config: Configuration parameters
        """
        if not language:
            raise ValueError("Language code must be provided")
            
        self.language = language.lower()
        self.config = config or {}
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
            # First try to load from config directory
            config_dir = Path(self.config.get('config_dir', 'config'))
            stop_words_path = config_dir / "stop_words" / f"{self.language}.txt"
            
            if stop_words_path.exists():
                with open(stop_words_path, 'r', encoding='utf-8') as f:
                    stop_words = {line.strip().lower() for line in f if line.strip()}
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
        """Basic text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text with:
            - Converted to lowercase
            - Special characters removed (keeping language-specific letters)
            - Extra whitespace removed
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep language-specific letters
        if self.language == 'fi':
            # Keep Finnish/Swedish letters
            text = re.sub(r'[^a-zäöåA-ZÄÖÅ\s]', ' ', text)
        else:
            # Default English processing
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        logger.debug(f"Preprocessed text: '{text}'")
        return text
    
    def process_text(self, text: str) -> List[str]:
        """Process text to get base forms of non-stop words.
        
        Args:
            text: Input text
            
        Returns:
            List of base forms of non-stop words
        """
        logger.debug(f"Processing text: {text}")
        
        # Preprocess text
        text = self.preprocess_text(text)
        logger.debug(f"After preprocessing: {text}")
        
        # Tokenize
        tokens = self.tokenize(text)
        logger.debug(f"After tokenization: {len(tokens)} tokens")
        
        # Get base forms and filter stop words
        processed_tokens = []
        for token in tokens:
            if not self.is_stop_word(token):
                base_form = self.get_base_form(token)
                if base_form and len(base_form) >= self.config.get('min_word_length', 3):
                    processed_tokens.append(base_form)
        
        logger.debug(f"Final processed tokens: {len(processed_tokens)}")
        return processed_tokens
    
    def should_keep_word(self, word: str, base_form: str) -> bool:
        """Determine if word should be kept in results."""
        # Skip stop words
        if self.is_stop_word(word) or self.is_stop_word(base_form):
            return False
            
        # Check minimum length
        if len(base_form) < self.config.get('min_word_length', 3):
            return False
            
        # Check against excluded patterns
        excluded_patterns = self.config.get('excluded_patterns', [])
        if any(re.search(pattern, base_form) for pattern in excluded_patterns):
            return False
            
        return True