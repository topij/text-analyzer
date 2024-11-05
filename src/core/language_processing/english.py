# src/core/language_processing/english.py

from typing import List, Set, Optional, Dict, Any
import logging
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from .base import BaseTextProcessor

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK resources: {e}")

logger = logging.getLogger(__name__)

class EnglishTextProcessor(BaseTextProcessor):
    """English text processor using NLTK.
    
    Handles English-specific text processing including:
    - Word lemmatization
    - POS tagging
    - English-specific tokenization
    """
    
    def __init__(
        self,
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize English text processor.
        
        Args:
            custom_stop_words: Additional stop words
            config: Configuration parameters
        """
        super().__init__('en', custom_stop_words, config)
        self.lemmatizer = WordNetLemmatizer()
        
        # Default English stop words from NLTK if available
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self._stop_words.update(stopwords.words('english'))
            logger.debug("Added NLTK English stop words")
        except Exception as e:
            logger.warning(f"Could not load NLTK stop words: {e}")
    
    def get_base_form(self, word: str) -> str:
        """Get base form (lemma) of an English word."""
        try:
            # Get POS tag
            pos_tag = self._get_wordnet_pos(word)
            
            # Lemmatize with POS tag if available
            if pos_tag:
                return self.lemmatizer.lemmatize(word.lower(), pos=pos_tag)
            
            # Try different POS tags if no specific tag is found
            lemmas = [
                self.lemmatizer.lemmatize(word.lower(), pos=pos)
                for pos in ['n', 'v', 'a', 'r']
            ]
            
            # Return shortest lemma (usually most basic form)
            return min(lemmas, key=len)
            
        except Exception as e:
            logger.error(f"Error getting base form for '{word}': {str(e)}")
            return word.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text using NLTK."""
        try:
            return word_tokenize(text)
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return text.split()
    
    def _get_wordnet_pos(self, word: str) -> Optional[str]:
        """Get WordNet POS tag for a word."""
        try:
            # Get NLTK POS tag
            pos = nltk.pos_tag([word])[0][1]
            
            # Convert to WordNet POS tag
            tag_map = {
                'JJ': 'a',    # Adjective
                'VB': 'v',    # Verb
                'NN': 'n',    # Noun
                'RB': 'r',    # Adverb
            }
            
            # Get the first letter of the POS tag
            pos_prefix = pos[:2]
            return tag_map.get(pos_prefix)
            
        except Exception as e:
            logger.error(f"Error getting POS tag for '{word}': {str(e)}")
            return None
    
    def process_text(self, text: str) -> List[str]:
        """Process English text with POS tagging and lemmatization."""
        logger.debug(f"Processing text: {text}")
        
        try:
            # Preprocess
            text = self.preprocess_text(text)
            
            # Tokenize
            tokens = self.tokenize(text)
            
            # Get POS tags
            tagged_tokens = nltk.pos_tag(tokens)
            
            # Process tokens with POS information
            processed_tokens = []
            for word, pos in tagged_tokens:
                if not self.is_stop_word(word):
                    # Get WordNet POS tag
                    wordnet_pos = self._get_wordnet_pos(word)
                    
                    # Lemmatize with POS if available
                    if wordnet_pos:
                        base_form = self.lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)
                    else:
                        base_form = self.get_base_form(word)
                    
                    if self.should_keep_word(word, base_form):
                        processed_tokens.append(base_form)
            
            logger.debug(f"Processed {len(tokens)} tokens to {len(processed_tokens)} results")
            return processed_tokens
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return super().process_text(text)  # Fallback to basic processing
    
    def should_keep_word(self, word: str, base_form: str) -> bool:
        """Determine if word should be kept based on English-specific rules."""
        # First check base criteria
        if not super().should_keep_word(word, base_form):
            return False
        
        # Additional English-specific checks
        
        # Skip single letters except 'a' and 'i'
        if len(base_form) == 1 and base_form not in ['a', 'i']:
            return False
        
        # Skip common contractions
        contractions = {"'s", "'t", "'re", "'ve", "'ll", "'d"}
        if any(word.lower().endswith(c) for c in contractions):
            return False
        
        return True