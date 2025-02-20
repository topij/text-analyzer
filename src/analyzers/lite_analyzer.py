"""Lightweight semantic analyzer that combines all analyses into a single LLM call."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import math

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer

from src.schemas import (
    KeywordAnalysisResult,
    ThemeAnalysisResult,
    CategoryAnalysisResult,
    CompleteAnalysisResult,
    KeywordInfo,
    ThemeInfo,
    CategoryMatch,
    Evidence,
    ThemeContext,
)
from src.loaders.parameter_handler import ParameterHandler
from src.core.config import AnalyzerConfig
from src.core.language_processing import create_text_processor
from FileUtils import FileUtils

logger = logging.getLogger(__name__)

class KeywordOutput(BaseModel):
    """Keyword with metadata."""
    keyword: str = Field(description="The keyword or phrase")
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    is_compound: bool = Field(description="Whether this is a compound word/phrase")
    frequency: Optional[int] = Field(default=0, description="Number of occurrences in text")

class CategoryOutput(BaseModel):
    """Category with metadata."""
    name: str = Field(description="Category name")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class LiteAnalysisOutput(BaseModel):
    """Combined output schema for lite semantic analysis."""
    
    keywords: List[KeywordOutput] = Field(
        default_factory=list,
        description="List of extracted keywords with scores and metadata"
    )
    compound_words: List[KeywordOutput] = Field(
        default_factory=list,
        description="List of extracted compound words/phrases with scores"
    )
    themes: List[str] = Field(
        default_factory=list,
        description="List of identified themes"
    )
    theme_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Hierarchical organization of themes"
    )
    categories: List[CategoryOutput] = Field(
        default_factory=list,
        description="List of matched categories with confidence scores"
    )

class LiteSemanticAnalyzer:
    """Lightweight semantic analyzer that performs all analyses in a single LLM call."""

    # Common compound word patterns
    COMPOUND_PATTERNS = {
        "en": {
            "suffixes": ["based", "driven", "oriented", "ready", "aware", "enabled", "centric", "focused"],
            "prefixes": ["auto", "cyber", "meta", "multi", "pre", "post", "re", "self", "semi"],
            "joiners": ["-", "/", "_"],
        },
        "fi": {
            "suffixes": ["pohjainen", "keskeinen", "lähtöinen", "painotteinen", "vetoinen"],
            "prefixes": ["esi", "jälki", "lähi", "etä", "itse", "yhteis"],
            "joiners": ["-", "/", "_"],
        }
    }

    # Domain-specific terms with boost factors
    DOMAIN_TERMS = {
        "technical": {
            "terms": {
                "cloud", "infrastructure", "platform", "kubernetes", "deployment",
                "pipeline", "integration", "monitoring", "microservices", "api",
                "devops", "architecture", "latency", "throughput", "availability",
                "reliability", "algorithm", "automation", "database", "network",
                "security", "software", "system", "technology", "artificial intelligence",
                "machine learning", "data science", "analytics", "cybersecurity",
                "blockchain", "computing", "digital", "encryption", "framework",
                "interface", "protocol", "quantum", "robotics", "virtualization"
            },
            "boost": 1.2
        },
        "business": {
            "terms": {
                "revenue", "cost", "profit", "margin", "growth", "efficiency",
                "performance", "optimization", "strategy", "operations",
                "analytics", "metrics", "investment", "market", "innovation",
                "productivity", "scalability", "sustainability", "transformation",
                "value", "agile", "competitive", "enterprise", "leadership",
                "management", "partnership", "quality", "solution", "stakeholder",
                "strategic"
            },
            "boost": 1.15
        }
    }

    # Common words to filter out
    FILTER_WORDS = {
        "en": {
            "verbs": {"use", "make", "do", "get", "take", "give", "have", "be", "provide", "enable", "drive"},
            "adjectives": {"good", "new", "big", "small", "high", "low", "many", "much", "some", "other", "different", "important", "significant"},
            "nouns": {"thing", "way", "time", "case", "example", "part", "kind", "type", "lot", "need", "fact", "point", "place"},
            "adverbs": {"very", "really", "quite", "rather", "too", "also", "just", "now", "then", "here", "there"}
        },
        "fi": {
            "verbs": {"käyttää", "tehdä", "ottaa", "antaa", "olla", "tarjota", "mahdollistaa", "edistää"},
            "adjectives": {"hyvä", "uusi", "iso", "pieni", "korkea", "matala", "moni", "eri", "tärkeä", "merkittävä", "positiivinen"},
            "nouns": {"asia", "tapa", "aika", "tapaus", "esimerkki", "osa", "tyyppi", "tarve", "paikka"},
            "adverbs": {"hyvin", "todella", "melko", "liian", "myös", "juuri", "nyt", "sitten", "täällä", "siellä"}
        }
    }

    # Cache for TF-IDF results
    _tfidf_cache = {}
    _cache_size = 1000  # Maximum number of cached results

    def __init__(
        self,
        llm: BaseChatModel,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        available_categories: Optional[Set[str]] = None,
        language: str = "en",
        config: Optional[Dict] = None,
        tfidf_weight: float = 0.5,
        custom_stop_words: Optional[Set[str]] = None,
        cache_size: int = 1000,
    ):
        """Initialize the analyzer.
        
        Args:
            llm: Language model to use
            parameter_file: Path to parameter file
            file_utils: FileUtils instance for file operations
            available_categories: Set of valid categories to choose from
            language: Language of the text to analyze ('en' or 'fi')
            config: Optional configuration dictionary
            tfidf_weight: Weight given to TF-IDF results (0.0-1.0)
            custom_stop_words: Optional set of additional stop words
            cache_size: Maximum number of cached TF-IDF results
        """
        self.llm = llm.with_structured_output(LiteAnalysisOutput)  # Use structured output directly with the LLM
        self.file_utils = file_utils
        self.available_categories = available_categories or set()
        self.language = language.lower()
        if self.language not in ["en", "fi"]:
            raise ValueError("Language must be 'en' or 'fi'")
            
        self.output_parser = PydanticOutputParser(pydantic_object=LiteAnalysisOutput)
        
        # Initialize TF-IDF vectorizer with language-specific settings
        self.tfidf_weight = tfidf_weight
        self.custom_stop_words = custom_stop_words or set()
        self._cache_size = cache_size
        
        # Language-specific TF-IDF configuration
        if self.language == "en":
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words="english",
                token_pattern=r"(?u)\b\w[\w-]*\w\b",
                min_df=2,  # Require at least 2 occurrences
                max_df=0.95,  # Ignore terms that appear in >95% of docs
                max_features=1000,  # Limit vocabulary size
                sublinear_tf=True  # Apply sublinear scaling to term frequencies
            )
        else:  # Finnish
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 4),
                token_pattern=r"(?u)\b\w[\w-]*\w\b",
                min_df=2,
                max_df=0.95,
                max_features=1000,
                sublinear_tf=True
            )
            
        if self.custom_stop_words:
            if not hasattr(self.tfidf_vectorizer, 'stop_words_'):
                self.tfidf_vectorizer.stop_words_ = set()
            self.tfidf_vectorizer.stop_words_.update(self.custom_stop_words)

        # Initialize parameters
        if parameter_file and file_utils:
            # Convert parameter file to Path if it's a string
            param_path = Path(parameter_file) if isinstance(parameter_file, str) else parameter_file
            
            # Use FileUtils to get the full path if needed
            if not param_path.is_absolute() and file_utils:
                param_path = file_utils.get_data_path("parameters") / param_path.name
                
            self.parameter_handler = ParameterHandler(param_path)
            self.parameters = self.parameter_handler.get_parameters()
            
            # Update language from parameters if not explicitly set
            if not language and self.parameters:
                self.language = self.parameters.general.language
        else:
            self.parameters = None
            
        # Initialize language processor
        self.language_processor = create_text_processor(
            language=self.language,
            config=config,
            file_utils=file_utils
        )
            
        # Initialize config
        self.config = config or {}
        if not self.config:
            if hasattr(self, 'parameters') and self.parameters:
                self.config = {
                    "language": self.language,
                    "min_keyword_length": self.parameters.general.min_keyword_length,
                    "min_confidence": self.parameters.general.min_confidence,
                    "focus_on": self.parameters.general.focus_on,
                    "include_compounds": self.parameters.general.include_compounds,
                }
            else:
                # Default configuration when no parameters are provided
                self.config = {
                    "language": self.language,
                    "min_keyword_length": 3,
                    "min_confidence": 0.6,
                    "focus_on": "general",
                    "include_compounds": True,
                }

        # Load additional configuration from parameters
        self.excluded_keywords = set()
        self.predefined_keywords = set()
        self.domain_context = None
        
        if hasattr(self, 'parameters') and self.parameters:
            # Load excluded keywords
            if hasattr(self.parameters, 'excluded_keywords'):
                self.excluded_keywords.update(
                    kw.lower() for kw in self.parameters.excluded_keywords
                )
            
            # Load predefined keywords with their domains
            if hasattr(self.parameters, 'predefined_keywords'):
                self.predefined_keywords = {
                    kw.lower() for kw in self.parameters.predefined_keywords 
                    if isinstance(kw, str)
                }
            
            # Load domain context
            if hasattr(self.parameters, 'domain_context'):
                self.domain_context = self.parameters.domain_context

    def calculate_dynamic_keyword_limit(
        self, text: str, base_limit: int = 8, max_limit: int = 15
    ) -> int:
        """Calculate dynamic keyword limit based on text length.
        
        Args:
            text: Input text
            base_limit: Minimum number of keywords
            max_limit: Maximum number of keywords
            
        Returns:
            Calculated keyword limit
        """
        word_count = len(text.split())
        return min(max(base_limit, word_count // 50), max_limit)

    def _is_compound_word(self, word: str) -> bool:
        """Check if a word is a compound word based on language-specific patterns.
        
        Args:
            word: Word to check
            
        Returns:
            True if the word is a compound word
        """
        if not word:
            return False
            
        patterns = self.COMPOUND_PATTERNS[self.language]
        word_lower = word.lower()
        
        # Check for joining characters
        if any(joiner in word for joiner in patterns["joiners"]):
            return True
            
        # Check for known suffixes and prefixes
        if any(word_lower.endswith(suffix) for suffix in patterns["suffixes"]):
            return True
            
        if any(word_lower.startswith(prefix) for prefix in patterns["prefixes"]):
            return True
            
        # Check for camelCase or PascalCase
        if any(c.isupper() for c in word[1:]):
            return True
            
        # Language-specific checks
        if self.language == "fi":
            # Finnish compound word patterns
            if len(word) > 12:  # Long words are likely compounds in Finnish
                return True
            # Check for typical compound word parts
            compound_parts = ["järjestelmä", "palvelu", "toiminta", "kehitys", "hallinta"]
            if any(part in word_lower for part in compound_parts):
                return True
        else:
            # English compound word patterns
            if "_" in word or any(c.isupper() for c in word[1:]):
                return True
                
        return False

    def _split_compound_word(self, word: str) -> List[str]:
        """Split a compound word into its components.
        
        Args:
            word: Compound word to split
            
        Returns:
            List of component words
        """
        if not word:
            return []
            
        # Handle hyphenated words
        if "-" in word:
            parts = [part.strip() for part in word.split("-") if part.strip()]
            return parts if parts else [word]
            
        # Handle camelCase and PascalCase
        if any(c.isupper() for c in word[1:]):
            import re
            parts = re.findall('[A-Z][^A-Z]*', word)
            if parts:
                return [part.lower() for part in parts]
                
        # Language-specific splitting
        if self.language == "fi":
            # Use language processor for Finnish compound splitting
            if self.language_processor:
                parts = self.language_processor.get_compound_parts(word)
                if parts:  # Check if parts is not None and not empty
                    return parts
        else:
            # English compound splitting
            patterns = self.COMPOUND_PATTERNS["en"]
            word_lower = word.lower()
            
            # Try to split by known patterns
            for suffix in patterns["suffixes"]:
                if word_lower.endswith(suffix):
                    base = word_lower[:-len(suffix)]
                    if base:  # Only return if base is not empty
                        return [base, suffix]
                    
            for prefix in patterns["prefixes"]:
                if word_lower.startswith(prefix):
                    rest = word_lower[len(prefix):]
                    if rest:  # Only return if rest is not empty
                        return [prefix, rest]
                    
        # If no splitting possible or splitting failed, return original word
        return [word]

    def _is_valid_keyword(self, word: str) -> bool:
        """Check if word is a valid keyword candidate."""
        if not word or len(word) < self.config.get("min_keyword_length", 3):
            return False

        word_lower = word.lower()
        
        # Check against excluded keywords
        if word_lower in self.excluded_keywords:
            return False
        
        # Check against filter words
        filter_words = self.FILTER_WORDS.get(self.language, {})
        for word_type, words in filter_words.items():
            if word_lower in words:
                return False

        if self.language_processor:
            # Skip stop words
            if self.language_processor.is_stop_word(word_lower):
                return False

            # Skip if base form is in filter words or excluded keywords
            base_form = self.language_processor.get_base_form(word_lower)
            if base_form:
                if base_form.lower() in self.excluded_keywords:
                    return False
                for word_type, words in filter_words.items():
                    if base_form in words:
                        return False

        return True

    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if a multi-word phrase is valid and meaningful."""
        try:
            # Split into words
            words = phrase.split()
            if len(words) < 2:
                return True  # Single words are valid
                
            # Skip phrases with stopwords in the middle
            if any(w.lower() in self.custom_stop_words for w in words[1:-1]):
                return False
                
            # Known high-value phrases are always valid
            known_phrases = {
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "cloud computing",
                "data analytics",
                "business intelligence",
                "neural network",
                "natural language",
                "computer vision",
                "internet of things",
                "blockchain technology",
                "quantum computing",
                "edge computing",
                "digital transformation",
                "cyber security",
                "data science",
                "big data",
                "real time",
                "open source",
                "user experience",
                "business model",
                "market analysis",
                "risk management",
                "strategic planning",
                "customer experience",
                "value proposition",
                "supply chain",
                "decision making"
            }
            if phrase.lower() in known_phrases:
                return True
                
            # Get POS tags
            pos_tags = nltk.pos_tag(words)
            
            # Valid phrase patterns with examples
            valid_patterns = [
                # Technical patterns
                (['JJ', 'NN'], ['artificial', 'intelligence']),  # Adjective + Noun
                (['NN', 'NN'], ['machine', 'learning']),  # Noun + Noun
                (['JJ', 'JJ', 'NN'], ['deep', 'neural', 'network']),  # Adj + Adj + Noun
                (['JJ', 'NN', 'NN'], ['digital', 'data', 'analysis']),  # Adj + Noun + Noun
                (['NN', 'NN', 'NN'], ['machine', 'learning', 'model']),  # Noun + Noun + Noun
                (['VBG', 'NN'], ['computing', 'system']),  # Gerund + Noun
                # Business patterns
                (['NN', 'NN', 'NN'], ['business', 'process', 'automation']),
                (['JJ', 'NN', 'NN'], ['strategic', 'business', 'planning']),
                (['NN', 'CC', 'NN'], ['costs', 'and', 'benefits']),
                # Special technical compounds
                (['NN', 'TO', 'NN'], ['business', 'to', 'business']),  # B2B style
                (['NN', 'IN', 'NN'], ['software', 'as', 'service']),  # SaaS style
                # Additional valid patterns
                (['NN', 'IN', 'NN'], ['internet', 'of', 'things']),  # IoT style
                (['NN', 'NN', 'NN'], ['supply', 'chain', 'management']),
                (['JJ', 'NN', 'NN'], ['artificial', 'neural', 'network'])
            ]
            
            # Get pattern of current phrase
            pattern = [tag for word, tag in pos_tags]
            
            # Check if pattern matches any valid pattern
            for valid_pattern, example in valid_patterns:
                if len(pattern) == len(valid_pattern):
                    # Exact match
                    if pattern == valid_pattern:
                        return True
                    # Allow some flexibility in noun phrases
                    if all(p.startswith('NN') for p in pattern):
                        return True
                    # Allow adjective + noun combinations
                    if len(pattern) == 2 and pattern[0] in ['JJ', 'VBG'] and pattern[1] == 'NN':
                        return True
                    
            # Special cases
            # Check for technical terms
            if self._is_technical_term(phrase) or any(self._is_technical_term(w) for w in words):
                return True
                
            # Check for hyphenated terms
            if '-' in phrase:
                return True
                
            # Check if all parts are in domain terms
            if all(w.lower() in self.DOMAIN_TERMS["technical"]["terms"] or 
                   w.lower() in self.DOMAIN_TERMS["business"]["terms"] 
                   for w in words):
                return True
                
            return False
            
        except Exception as e:
            logger.debug(f"Error checking phrase validity: {e}")
            return False  # Be conservative if check fails

    def _calculate_keyword_score(self, word: str, base_score: float) -> float:
        """Calculate final keyword score with various adjustments."""
        word_lower = word.lower()
        
        # Start with base score
        score = base_score
        
        # Predefined keywords get a significant boost
        if word_lower in self.predefined_keywords:
            score *= 1.3
            return min(score, 0.98)  # Cap at 0.98 for predefined keywords
        
        # Domain context boost
        if self.domain_context and self._matches_domain_context(word):
            score *= 1.2
        
        # Multi-word phrase scoring
        words = word.split()
        if len(words) > 1:
            if self._is_valid_phrase(word):
                score *= 1.15  # More moderate boost for valid phrases
            else:
                score *= 0.8  # Less aggressive penalty
        
        # Domain-specific boosts with more moderate adjustments
        for domain, info in self.DOMAIN_TERMS.items():
            if word_lower in info["terms"]:
                score *= 1.15  # Uniform boost for domain terms
                break
        
        # Technical term handling
        if self._is_technical_term(word):
            score *= 1.15  # More moderate boost
        
        # Proper noun handling
        if word[0].isupper() and not word.isupper():
            score *= 1.1  # More moderate boost
        
        # Length-based adjustments
        length = len(word)
        if length < 4:
            score *= 0.8
        elif length > 20:
            score *= 0.9
        
        # Frequency bonus with moderate scaling
        if hasattr(word, 'frequency') and word.frequency:
            freq_bonus = min(0.1, 0.03 * math.log(1 + word.frequency))
            score += freq_bonus
        
        return min(score, 0.95)

    def _matches_domain_context(self, word: str) -> bool:
        """Check if word matches the domain context."""
        if not self.domain_context or not word:
            return False
            
        word_lower = word.lower()
        
        # Get domain keywords from context
        domain_keywords = []
        if isinstance(self.domain_context, dict):
            # Try to get keywords from various possible dictionary keys
            if 'keywords' in self.domain_context:
                keywords = self.domain_context['keywords']
                if isinstance(keywords, list):
                    domain_keywords.extend(keywords)
                elif isinstance(keywords, str):
                    domain_keywords.append(keywords)
            
            # Fallback to collecting all string/list values as keywords
            for value in self.domain_context.values():
                if isinstance(value, str):
                    domain_keywords.append(value)
                elif isinstance(value, list):
                    domain_keywords.extend(str(item) for item in value if isinstance(item, (str, int, float)))
        elif isinstance(self.domain_context, str):
            domain_keywords = [self.domain_context]
        elif isinstance(self.domain_context, list):
            domain_keywords = [str(item) for item in self.domain_context if isinstance(item, (str, int, float))]
        
        # Ensure all keywords are strings and non-empty
        domain_keywords = [str(k).lower() for k in domain_keywords if k]
        
        if not domain_keywords:
            return False
        
        # Check direct match
        if word_lower in domain_keywords:
            return True
            
        # Check word parts against domain context
        words = word_lower.split()
        return any(
            any(part.lower() in domain_keywords for part in words)
            or any(k in word_lower for k in domain_keywords)
        )

    def extract_keywords_tfidf(self, text: str, max_keywords: int) -> List[KeywordOutput]:
        """Extract keywords using TF-IDF with caching."""
        try:
            # Check cache first
            cache_key = f"{text[:100]}_{max_keywords}"
            if cache_key in self._tfidf_cache:
                return self._tfidf_cache[cache_key]
            
            # Create a small corpus with the text and its sentences
            import nltk
            corpus = [text]
            try:
                sentences = nltk.sent_tokenize(text)
                corpus.extend(sentences)
            except Exception as e:
                logger.debug(f"Could not split text into sentences: {e}")
            
            # Configure TF-IDF parameters based on text length
            word_count = len(text.split())
            min_df = 1 if word_count < 50 else 2
            
            # Update vectorizer settings
            self.tfidf_vectorizer.min_df = min_df
            self.tfidf_vectorizer.max_df = 0.85
            
            # Add predefined keywords to vocabulary
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                vocab = self.tfidf_vectorizer.vocabulary_
                for keyword in self.predefined_keywords:
                    if isinstance(keyword, str) and keyword not in vocab:
                        vocab[keyword] = len(vocab)
            
            # Create a list of predefined keywords for the vocabulary
            predefined_vocab = [kw for kw in self.predefined_keywords if isinstance(kw, str)]
            
            # Update vectorizer settings with predefined vocabulary
            if predefined_vocab:
                self.tfidf_vectorizer.vocabulary = {word: idx for idx, word in enumerate(predefined_vocab)}
            
            # Fit and transform the text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]  # Use scores from the full text
            
            # Get keyword-score pairs with improved scoring
            keyword_scores = []
            max_score = max(tfidf_scores) if any(tfidf_scores) else 1.0
            
            # Track seen base forms to avoid duplicates
            seen_base_forms = set()
            
            # First process known phrases
            for word, score in zip(feature_names, tfidf_scores):
                if score > 0 and word.lower() in self.predefined_keywords:
                    normalized_score = 0.6 + (0.35 * score / max_score)  # Scale to 0.6-0.95 range
                    final_score = self._calculate_keyword_score(word, normalized_score)
                    
                    freq = sum(1 for s in corpus if word.lower() in s.lower())
                    
                    keyword_output = KeywordOutput(
                        keyword=word,
                        score=round(final_score, 2),
                        is_compound=True,
                        frequency=freq
                    )
                    keyword_scores.append((keyword_output, final_score))
                    seen_base_forms.add(word.lower())
            
            # Then process other terms
            for word, score in zip(feature_names, tfidf_scores):
                if score > 0 and word.lower() not in seen_base_forms:
                    # Skip invalid keywords
                    if not self._is_valid_keyword(word):
                        continue
                    
                    # Get base form if available
                    base_form = word
                    if self.language_processor:
                        base = self.language_processor.get_base_form(word)
                        if base:
                            base_form = base
                    
                    # Skip if we've seen this base form
                    if base_form.lower() in seen_base_forms:
                        continue
                    
                    # Calculate normalized score with various adjustments
                    normalized_score = 0.4 + (0.4 * score / max_score)  # Scale to 0.4-0.8 range
                    final_score = self._calculate_keyword_score(word, normalized_score)
                    
                    # Skip if score too low
                    if final_score < 0.4:  # Increased minimum threshold
                        continue
                    
                    # Calculate frequency
                    freq = sum(1 for s in corpus if word.lower() in s.lower())
                    
                    # Create KeywordOutput object
                    keyword_output = KeywordOutput(
                        keyword=base_form,
                        score=round(final_score, 2),
                        is_compound=len(word.split()) > 1 or self._is_compound_word(word),
                        frequency=freq
                    )
                    keyword_scores.append((keyword_output, final_score))
                    seen_base_forms.add(base_form.lower())
            
            # Sort by score and get top keywords
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Remove duplicates and overlapping terms
            result = []
            used_terms = set()
            
            for kw, score in keyword_scores:
                # Check if this keyword is contained in or contains any existing keyword
                overlaps = False
                kw_lower = kw.keyword.lower()
                
                for used in used_terms:
                    if kw_lower in used or used in kw_lower:
                        # Allow known phrases to override their parts
                        if kw_lower in self.predefined_keywords and kw_lower not in used:
                            overlaps = False
                            break
                        overlaps = True
                        break
                
                if not overlaps:
                    result.append(kw)
                    used_terms.add(kw_lower)
                    
                if len(result) >= max_keywords:
                    break
            
            # Update cache
            if len(self._tfidf_cache) >= self._cache_size:
                self._tfidf_cache.pop(next(iter(self._tfidf_cache)))
            self._tfidf_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {str(e)}")
            return []

    def _create_analysis_prompt(self, text: str, analysis_types: Optional[List[str]] = None) -> str:
        """Create the combined analysis prompt."""
        # Calculate dynamic keyword limit
        dynamic_limit = self.calculate_dynamic_keyword_limit(text)
        
        # Get TF-IDF keywords if needed
        tfidf_keywords = []
        if "keywords" in (analysis_types or ["keywords"]):
            tfidf_keywords = self.extract_keywords_tfidf(text, max_keywords=dynamic_limit)
            tfidf_info = [f"{kw.keyword} ({kw.score:.2f})" for kw in tfidf_keywords]
        
        # Base prompt with configuration context
        focus = self.config.get('focus_on', 'general')
        min_confidence = self.config.get('min_confidence', 0.6)
        min_length = self.config.get('min_keyword_length', 3)
        
        # Create base prompt
        prompt = f"""Analyze the following text using a theme-first approach. First identify the main themes and their relationships, then use these themes to guide the extraction of keywords and categories.

Configuration parameters:
- Language: {self.language}
- Minimum keyword length: {min_length}
- Minimum confidence threshold: {min_confidence}
- Analysis focus: {focus}
- Include compound words: {self.config.get('include_compounds', True)}
- Maximum keywords: {dynamic_limit}
- TF-IDF weight: {self.tfidf_weight}

Additional configuration:
- Predefined keywords: {list(self.predefined_keywords) if self.predefined_keywords else 'None'}
- Excluded keywords: {list(self.excluded_keywords) if self.excluded_keywords else 'None'}
- Available categories: {list(self.available_categories) if self.available_categories else 'None'}
- Domain context: {self.domain_context if self.domain_context else 'None'}

Text to analyze:
{text}

Follow these guidelines:
1. Theme Analysis:
   - Identify main themes and sub-themes
   - Create a hierarchical theme structure
   - Consider theme relationships and context
   - Aim for specific, descriptive themes

2. Theme-Enhanced Keyword Analysis:
   - Use identified themes to guide keyword extraction
   - Consider TF-IDF suggestions: {', '.join(tfidf_info) if tfidf_keywords else 'None'}
   - Include compound words and technical terms
   - Score based on theme relevance
   - Maximum keywords: {dynamic_limit}
   - Prioritize predefined keywords if relevant
   - Exclude any keywords from the excluded list
   - Consider domain context in scoring

3. Theme-Guided Category Analysis:
   - Match only with available categories if specified
   - Score based on thematic alignment
   - Consider theme relationships
   - Provide evidence for each category match
   - Focus on categories relevant to the domain context

Output requirements:
- Keywords must have scores between 0.0 and 1.0
- Include compound word detection
- Track keyword frequencies
- Ensure theme hierarchy is complete
- Categories must have confidence scores"""

        return prompt

    def _process_keywords(self, keywords: List[KeywordInfo], compound_words: List[str]) -> List[KeywordInfo]:
        """Process and clean keywords with improved scoring."""
        processed_keywords = []
        processed_terms = set()
        
        # Process multi-word expressions and compounds first
        for term in [*compound_words, *[k for k in keywords if len(k.split()) > 1]]:
            if not term:
                continue
                
            base_form = self.language_processor.get_base_form(term) if self.language_processor else term
            if not base_form or base_form.lower() in processed_terms:
                continue
                
            # Calculate confidence for multi-word terms
            word_count = len(term.split())
            base_score = 0.75  # Higher base score for multi-word terms
            
            # Additional scoring factors
            length_bonus = min(0.05 * word_count, 0.15)
            technical_bonus = 0.1 if self._is_technical_term(term) else 0
            proper_noun_bonus = 0.1 if any(w[0].isupper() for w in term.split()) else 0
            phrase_bonus = 0.1 if self._is_valid_phrase(term) else 0
            
            confidence = min(base_score + length_bonus + technical_bonus + proper_noun_bonus + phrase_bonus, 1.0)
            
            # Calculate frequency safely
            freq = sum(1 for k in keywords if k.keyword.lower() == term.lower())
            freq += sum(1 for c in compound_words if c.lower() == term.lower())
            
            processed_keywords.append(KeywordInfo(
                keyword=base_form,
                score=confidence,
                frequency=freq,
                is_compound=True,
                compound_parts=self._split_compound_word(term),
                metadata={
                    "word_count": word_count,
                    "is_technical": self._is_technical_term(term),
                    "is_valid_phrase": self._is_valid_phrase(term)
                }
            ))
            processed_terms.add(base_form.lower())
        
        # Process single keywords
        for keyword in [k for k in keywords if len(k.split()) == 1]:
            if not keyword:
                continue
                
            base_form = self.language_processor.get_base_form(keyword) if self.language_processor else keyword
            if not base_form or base_form.lower() in processed_terms or len(base_form) < 3:
                continue
            
            # Enhanced confidence calculation for single words
            base_score = 0.7  # Base score for single words
            length_bonus = min(0.05 * len(base_form) / 4, 0.1)
            proper_noun_bonus = 0.15 if base_form[0].isupper() else 0
            technical_bonus = 0.1 if self._is_technical_term(base_form) else 0
            
            # Calculate frequency safely
            freq = sum(1 for k in keywords if k.keyword.lower() == keyword.lower())
            frequency_bonus = min(0.05 * freq, 0.1)
            
            confidence = min(
                base_score + length_bonus + proper_noun_bonus + technical_bonus + frequency_bonus,
                1.0
            )
            
            processed_keywords.append(KeywordInfo(
                keyword=base_form,
                score=confidence,
                frequency=freq,
                is_compound=False,
                metadata={
                    "is_technical": self._is_technical_term(base_form),
                    "is_proper_noun": base_form[0].isupper()
                }
            ))
            processed_terms.add(base_form.lower())
        
        return sorted(processed_keywords, key=lambda x: (x.score, x.frequency), reverse=True)

    def _is_technical_term(self, word: str) -> bool:
        """Check if a word is likely a technical term."""
        technical_patterns = {
            "en": [
                r"^[A-Z]{2,}$",  # Acronyms
                r"\d+",  # Numbers
                r"^(api|sdk|ui|ux|ai|ml|nlp|http|sql|nosql)",  # Common tech abbreviations
                r"(format|protocol|framework|platform|service|system|engine|api|data|cloud)$"
            ],
            "fi": [
                r"^[A-Z]{2,}$",  # Acronyms
                r"\d+",  # Numbers
                r"(järjestelmä|palvelu|alusta|rajapinta|protokolla|moottori)$"
            ]
        }
        
        import re
        patterns = technical_patterns[self.language]
        return any(re.search(pattern, word.lower()) for pattern in patterns)

    def _get_base_form(self, word: str) -> str:
        """Get base form of a word with special handling for hyphenated compounds."""
        if not word:
            return word
            
        # Special handling for hyphenated words
        if '-' in word:
            parts = word.split('-')
            # Get base form for each part if language processor is available
            if self.language_processor:
                base_parts = [
                    self.language_processor.get_base_form(part) or part
                    for part in parts
                ]
                return '-'.join(base_parts)
            return word
            
        # Regular base form processing
        if self.language_processor:
            return self.language_processor.get_base_form(word) or word
        return word

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
    ) -> CompleteAnalysisResult:
        """Perform semantic analysis on the text."""
        start_time = datetime.now()
        
        try:
            # Detect language if not specified
            detected_language = detect(text)
            if detected_language != self.language:
                logger.warning(
                    f"Text language ({detected_language}) differs from analyzer language ({self.language})"
                )
            
            # Create and run the combined prompt
            prompt = self._create_analysis_prompt(text, analysis_types)
            parsed_output = await self.llm.ainvoke(prompt)
            
            # First process themes to use as context for other analyses
            themes = []
            for theme_name in parsed_output.themes:
                # Calculate theme confidence based on hierarchy position and specificity
                is_main_theme = any(theme_name == main for main in parsed_output.theme_hierarchy.keys())
                base_confidence = 0.8 if is_main_theme else 0.7
                specificity_bonus = 0.2 if len(theme_name.split()) > 1 else 0.1
                confidence = min(base_confidence + specificity_bonus, 1.0)
                
                themes.append(ThemeInfo(
                    name=theme_name,
                    description=theme_name,  # Using theme text as description
                    confidence=confidence,
                    score=confidence,
                ))
            
            theme_result = ThemeAnalysisResult(
                themes=themes,
                theme_hierarchy=parsed_output.theme_hierarchy,
                language=self.language,
                success=True
            )
            
            # Create theme context for enhancing other analyses
            theme_context = ThemeContext(
                main_themes=[theme.name for theme in themes if not any(
                    theme.name in children 
                    for children in parsed_output.theme_hierarchy.values()
                )],
                theme_hierarchy=parsed_output.theme_hierarchy,
                theme_descriptions={theme.name: theme.description for theme in themes},
                theme_confidence={theme.name: theme.confidence for theme in themes},
                theme_keywords={}  # Will be populated from keyword analysis
            )
            
            # Process keywords with theme context
            all_keywords = []
            keyword_counts = {}
            
            # First pass: collect frequencies and validate keywords
            for kw in parsed_output.keywords + parsed_output.compound_words:
                if not kw.keyword or not self._is_valid_keyword(kw.keyword):
                    continue
                keyword_counts[kw.keyword.lower()] = keyword_counts.get(kw.keyword.lower(), 0) + (kw.frequency or 1)
            
            # Second pass: create KeywordInfo objects with theme-enhanced scoring
            processed_keywords = set()  # Track processed keywords to avoid duplicates
            
            # Process regular keywords
            for kw in parsed_output.keywords:
                if not kw.keyword or kw.keyword.lower() in processed_keywords:
                    continue
                    
                # Get base form with special handling for compounds
                processed = self._get_base_form(kw.keyword)
                if not processed:
                    continue
                    
                # Calculate theme relevance
                theme_relevance = self._calculate_theme_keyword_relevance(processed, theme_context)
                
                # Base score adjustments
                base_score = kw.score
                if processed.lower() in self.predefined_keywords:
                    base_score = min(base_score * 1.3, 0.98)  # Boost predefined keywords
                
                # Theme-based score adjustment
                adjusted_score = base_score * (1.0 + theme_relevance * 0.3)  # Up to 30% boost
                
                # Create keyword info
                keyword_info = KeywordInfo(
                    keyword=processed,
                    score=min(adjusted_score, 1.0),
                    frequency=keyword_counts.get(kw.keyword.lower(), 1),
                    is_compound=kw.is_compound or len(processed.split()) > 1 or '-' in processed,
                    compound_parts=self._split_compound_word(processed) if kw.is_compound or '-' in processed else None,
                    metadata={
                        "is_technical": self._is_technical_term(processed),
                        "is_proper_noun": processed[0].isupper(),
                        "is_valid_phrase": len(processed.split()) > 1 and self._is_valid_phrase(processed),
                        "theme_relevance": theme_relevance,
                        "predefined": processed.lower() in self.predefined_keywords,
                        "domain_match": self._matches_domain_context(processed)
                    }
                )
                all_keywords.append(keyword_info)
                processed_keywords.add(processed.lower())
            
            # Process compound words
            for cw in parsed_output.compound_words:
                if not cw.keyword or cw.keyword.lower() in processed_keywords:
                    continue
                    
                # Get base form with special handling for compounds
                processed = self._get_base_form(cw.keyword)
                if not processed:
                    continue
                
                # Calculate theme relevance
                theme_relevance = self._calculate_theme_keyword_relevance(processed, theme_context)
                
                # Base score adjustments
                base_score = cw.score
                if processed.lower() in self.predefined_keywords:
                    base_score = min(base_score * 1.3, 0.98)  # Boost predefined keywords
                
                # Theme-based score adjustment
                adjusted_score = base_score * (1.0 + theme_relevance * 0.3)  # Up to 30% boost
                
                # Additional compound word bonus
                if self._is_compound_word(processed) or '-' in processed:
                    adjusted_score = min(adjusted_score * 1.1, 1.0)  # 10% boost for verified compounds
                
                # Create keyword info
                keyword_info = KeywordInfo(
                    keyword=processed,
                    score=min(adjusted_score, 1.0),
                    frequency=keyword_counts.get(cw.keyword.lower(), 1),
                    is_compound=True,
                    compound_parts=self._split_compound_word(processed),
                    metadata={
                        "is_technical": self._is_technical_term(processed),
                        "is_valid_phrase": self._is_valid_phrase(processed),
                        "theme_relevance": theme_relevance,
                        "predefined": processed.lower() in self.predefined_keywords,
                        "domain_match": self._matches_domain_context(processed)
                    }
                )
                all_keywords.append(keyword_info)
                processed_keywords.add(processed.lower())
            
            # Update theme_keywords in theme context
            for theme in themes:
                theme_keywords = [
                    kw.keyword for kw in all_keywords
                    if isinstance(kw.metadata, dict) and
                    kw.metadata.get("theme_relevance", 0) > 0.5 and
                    (theme.name.lower() in kw.keyword.lower() or
                     kw.keyword.lower() in theme.name.lower())
                ]
                if theme_keywords:
                    theme_context.theme_keywords[theme.name] = theme_keywords
            
            # Sort by score and remove duplicates while keeping highest scoring version
            seen_keywords = {}
            for kw in sorted(all_keywords, key=lambda x: (x.score, x.frequency), reverse=True):
                key = kw.keyword.lower()
                if key not in seen_keywords:
                    seen_keywords[key] = kw
            
            all_keywords = list(seen_keywords.values())
            
            keyword_result = KeywordAnalysisResult(
                keywords=all_keywords,
                compound_words=[kw.keyword for kw in all_keywords if kw.is_compound or '-' in kw.keyword],
                domain_keywords={},  # Not used in lite version
                language=self.language,
                success=True
            )
            
            # Process categories with theme-enhanced scoring
            category_matches = []
            for cat in parsed_output.categories:
                # Calculate semantic similarity with themes
                theme_similarities = [
                    self._calculate_theme_category_similarity(
                        cat.name.lower(),
                        theme.name.lower(),
                        theme.description.lower()
                    )
                    for theme in themes
                ]
                max_theme_similarity = max(theme_similarities) if theme_similarities else 0
                
                # Calculate evidence-based confidence
                evidence_list = []
                keyword_relevance = 0
                for kw in all_keywords:
                    if cat.name.lower() in kw.keyword.lower() or kw.keyword.lower() in cat.name.lower():
                        relevance = kw.score * (1 + kw.metadata.get("theme_relevance", 0) * 0.25)
                        evidence_list.append(Evidence(
                            text=kw.keyword,
                            relevance=min(relevance, 1.0)
                        ))
                        keyword_relevance = max(keyword_relevance, relevance)
                
                # Combined scoring
                base_confidence = 0.7
                theme_bonus = max_theme_similarity * 0.25  # Up to 25% boost from themes
                keyword_bonus = keyword_relevance * 0.1  # Up to 10% boost from keywords
                confidence = min(base_confidence + theme_bonus + keyword_bonus, 1.0)
                
                # Get related themes
                related_themes = [
                    theme.name for theme in themes
                    if self._calculate_theme_category_similarity(
                        cat.name.lower(),
                        theme.name.lower(),
                        theme.description.lower()
                    ) > 0.5
                ]
                
                category_matches.append(CategoryMatch(
                    name=cat.name,
                    confidence=confidence,
                    description=f"Category match: {cat.name}",
                    evidence=evidence_list,
                    themes=related_themes
                ))
            
            category_result = CategoryAnalysisResult(
                matches=category_matches,
                language=self.language,
                success=True
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Combine into complete result
            result = CompleteAnalysisResult(
                keywords=keyword_result,
                themes=theme_result,
                categories=category_result,
                language=self.language,
                success=True,
                processing_time=processing_time,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "detected_language": detected_language,
                    "analysis_type": "lite",
                    "language": self.language,
                    "config": self.config,
                    "tfidf_weight": self.tfidf_weight,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CompleteAnalysisResult(
                keywords=KeywordAnalysisResult(
                    language=self.language,
                    keywords=[],
                    compound_words=[],
                    domain_keywords={},
                    success=False,
                    error=str(e)
                ),
                themes=ThemeAnalysisResult(
                    language=self.language,
                    themes=[],
                    theme_hierarchy={},
                    success=False,
                    error=str(e)
                ),
                categories=CategoryAnalysisResult(
                    language=self.language,
                    matches=[],
                    success=False,
                    error=str(e)
                ),
                language=self.language,
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            )

    def _calculate_theme_keyword_relevance(self, keyword: str, theme_context: ThemeContext) -> float:
        """Calculate how relevant a keyword is to the identified themes."""
        if not theme_context or not theme_context.main_themes:
            return 0.0
            
        keyword_lower = keyword.lower()
        max_relevance = 0.0
        
        # Check direct matches in theme names and descriptions
        for theme in theme_context.main_themes:
            theme_lower = theme.lower()
            desc_lower = theme_context.theme_descriptions.get(theme, "").lower()
            
            # Direct match in theme name
            if keyword_lower in theme_lower or theme_lower in keyword_lower:
                confidence = theme_context.theme_confidence.get(theme, 0.8)
                max_relevance = max(max_relevance, confidence)
                continue
            
            # Match in theme description
            if keyword_lower in desc_lower:
                confidence = theme_context.theme_confidence.get(theme, 0.8) * 0.8
                max_relevance = max(max_relevance, confidence)
                continue
            
            # Check word-level similarity
            keyword_words = set(keyword_lower.split())
            theme_words = set(theme_lower.split())
            desc_words = set(desc_lower.split())
            
            word_overlap = len(keyword_words & (theme_words | desc_words))
            if word_overlap:
                overlap_score = word_overlap / len(keyword_words)
                confidence = theme_context.theme_confidence.get(theme, 0.8) * overlap_score * 0.7
                max_relevance = max(max_relevance, confidence)
        
        return max_relevance

    def _calculate_theme_category_similarity(
        self,
        category: str,
        theme: str,
        theme_desc: str
    ) -> float:
        """Calculate semantic similarity between category and theme."""
        # Direct match in theme name
        if category in theme or theme in category:
            return 0.8
        
        # Match in theme description
        if category in theme_desc:
            return 0.6
        
        # Word-level similarity
        category_words = set(category.split())
        theme_words = set(theme.split())
        desc_words = set(theme_desc.split())
        
        # Calculate overlap ratios
        theme_overlap = len(category_words & theme_words) / len(category_words)
        desc_overlap = len(category_words & desc_words) / len(category_words)
        
        # Combine overlap scores with weights
        similarity = theme_overlap * 0.7 + desc_overlap * 0.3
        
        return min(similarity, 0.7)  # Cap at 0.7 for partial matches 