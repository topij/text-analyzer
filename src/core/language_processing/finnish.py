# src/core/language_processing/finnish.py

from typing import List, Set, Optional, Dict, Any
import os
import logging
from pathlib import Path
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
    """
    
    # Token type constants
    WORD_TOKEN = 1
    WHITESPACE_TOKEN = 3
    
    # Word class mappings
    WORD_CLASSES = {
        'nimisana': 'noun',
        'teonsana': 'verb',
        'laatusana': 'adjective',
        'seikkasana': 'adverb'
    }
    
    # Special case word mappings
    WORD_MAPPINGS = {
        'osata': 'osaaminen',
        'tilaama': 'tilata',
        'pakattu': 'pakata',
        'viivästynyt': 'viivästyä',
        # Add other special cases
    }
    
    # Common compound words to preserve
    COMPOUNDS = {
        'asiakaspalvelu': 'asiakaspalvelu',
        'käyttäjätunnus': 'käyttäjätunnus',
        'ohjelmistokehittäjä': 'ohjelmistokehittäjä',
        # Add other compounds
    }
    
    def __init__(
        self,
        custom_stop_words: Optional[Set[str]] = None,
        voikko_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Finnish text processor.
        
        Args:
            custom_stop_words: Additional stop words
            voikko_path: Path to Voikko installation
            config: Configuration parameters
        """
        super().__init__('fi', custom_stop_words, config)
        
        # Initialize Voikko
        self.voikko = self._initialize_voikko(voikko_path)
        if not self.voikko:
            logger.warning("Using fallback text processing methods without Voikko")
    
    def get_base_form(self, word: str) -> str:
        """Get base form of a Finnish word."""
        try:
            if not self.voikko:
                return word.lower()
            
            # Get analysis
            analyses = self.voikko.analyze(word)
            if not analyses:
                return word.lower()
            
            # Get base form from best analysis
            base_form = self._get_word_base_form(analyses)
            
            # Check mappings and compounds
            if base_form in self.WORD_MAPPINGS:
                return self.WORD_MAPPINGS[base_form]
            if base_form in self.COMPOUNDS:
                return self.COMPOUNDS[base_form]
            
            return base_form
            
        except Exception as e:
            logger.error(f"Error getting base form for '{word}': {str(e)}")
            return word.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Finnish text."""
        if not text:
            return []
            
        try:
            if not self.voikko:
                return text.split()
            
            # Get tokens using Voikko
            tokens = self.voikko.tokens(text)
            
            # Filter for word tokens
            return [
                t.tokenText for t in tokens
                if hasattr(t, 'tokenType') and t.tokenType == self.WORD_TOKEN
            ]
            
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return text.split()
    
    def _initialize_voikko(self, voikko_path: Optional[str] = None) -> Optional[Voikko]:
        """Initialize Voikko with proper error handling."""
        try:
            if not voikko_path:
                voikko_path = os.environ.get('VOIKKO_PATH', '/usr/local/lib/voikko')
            
            # Add DLL directory on Windows
            if os.name == 'nt':
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(voikko_path)
                else:
                    os.environ['PATH'] = f"{voikko_path};{os.environ['PATH']}"
            
            # Initialize Voikko
            voikko = Voikko('fi', voikko_path)
            
            # Test initialization
            test_word = "testi"
            if voikko.analyze(test_word):
                logger.info("Voikko initialized successfully")
                return voikko
            else:
                logger.error("Voikko initialization test failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Voikko: {str(e)}")
            return None
    
    def _get_word_base_form(self, analyses: List[Dict[str, Any]]) -> str:
        """Get the most appropriate base form from analyses."""
        try:
            best_analysis = self._select_best_analysis(analyses)
            if not best_analysis:
                return analyses[0].get('BASEFORM', '').lower()
            
            return best_analysis.get('BASEFORM', '').lower()
            
        except Exception as e:
            logger.error(f"Error getting word base form: {str(e)}")
            return analyses[0].get('BASEFORM', '').lower() if analyses else ''
    
    def _select_best_analysis(self, analyses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best analysis based on word class and form."""
        for analysis in analyses:
            word_class = analysis.get('CLASS')
            if not word_class:
                continue
            
            # Prefer nouns
            if word_class == 'nimisana':
                return analysis
            
            # Then verbs in basic form
            if word_class == 'teonsana' and not analysis.get('PARTICIPLE'):
                return analysis
        
        return analyses[0] if analyses else None
    
    def is_compound_word(self, word: str) -> bool:
        """Check if word is a compound word."""
        if not self.voikko:
            return False
            
        analyses = self.voikko.analyze(word)
        if not analyses:
            return False
            
        return '+' in analyses[0].get('WORDBASES', '')
    
    def get_compound_parts(self, word: str) -> List[str]:
        """Get parts of a compound word."""
        if not self.voikko:
            return [word]
            
        analyses = self.voikko.analyze(word)
        if not analyses:
            return [word]
            
        word_bases = analyses[0].get('WORDBASES', '')
        if '+' not in word_bases:
            return [word]
            
        return [
            part.split('(')[0]
            for part in word_bases.split('+')
            if part and '(' in part
        ]