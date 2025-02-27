# src/analyzers/keyword_analyzer.py

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import nltk

from src.config.manager import ConfigManager
from src.core.llm.factory import create_llm
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from src.core.language_processing.base import BaseTextProcessor
from src.core.config import AnalyzerConfig

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from pydantic import Field

from src.analyzers.base import AnalyzerOutput, TextAnalyzer, TextSection
from src.schemas import KeywordAnalysisResult, KeywordInfo, ThemeContext

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"


@dataclass
class DomainTerms:
    terms: Set[str]
    boost_factor: float = 1.0


class KeywordOutput(AnalyzerOutput):
    """Output model for keyword analysis."""

    keywords: List[KeywordInfo] = Field(default_factory=list)
    compound_words: List[str] = Field(default_factory=list)
    domain_keywords: Dict[str, List[str]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility


class KeywordAnalyzer(TextAnalyzer):
    """Analyzes text to extract keywords with position-aware weighting."""

    # Base domain terms (English)
    DOMAIN_TERMS = {
        DomainType.TECHNICAL: DomainTerms(
            {
                "cloud",
                "infrastructure",
                "platform",
                "kubernetes",
                "deployment",
                "pipeline",
                "integration",
                "monitoring",
                "microservices",
                "api",
                "devops",
                "architecture",
                "latency",
                "throughput",
                "availability",
                "reliability",
            },
            boost_factor=1.2,
        ),
        DomainType.BUSINESS: DomainTerms(
            {
                "revenue",
                "cost",
                "profit",
                "margin",
                "growth",
                "efficiency",
                "performance",
                "optimization",
                "strategy",
                "operations",
                "analytics",
                "metrics",
            },
            boost_factor=1.15,
        ),
    }

    FINNISH_DOMAIN_TERMS = {
        DomainType.TECHNICAL: DomainTerms(
            {
                # Cloud/Infrastructure
                "pilvipalvelu",
                "pilvipohjainen",
                "mikropalvelu",
                "infrastruktuuri",
                "konttiteknologia",
                "skaalautuvuus",
                # Integration/APIs
                "rajapinta",
                "integraatio",
                "käyttöönotto",
                # Performance/Ops
                "monitorointi",
                "vikasietoisuus",
                "kuormantasaus",
                "suorituskyky",
                "automaatio",
                "palvelutaso",
            },
            boost_factor=1.2,
        ),
        DomainType.BUSINESS: DomainTerms(
            {
                # Financial
                "liikevaihto",
                "tuotto",
                "kustannus",
                "markkinaosuus",
                "toistuvaislaskutus",
                "vuosineljännes",
                "investointi",
                # Customer
                "asiakaspysyvyys",
                "asiakashankinta",
                "asiakaskokemus",
                "asiakassegmentti",
                "asiakaskanta",
                # Market
                "markkinaosuus",
                "kilpailuasema",
                "markkina-asema",
                "markkinasegmentti",
                "markkinajohtaja",
            },
            boost_factor=1.15,
        ),
    }

    GENERIC_TERMS = {
        "fi": {
            "prosentti",
            "määrä",
            "osuus",
            "kasvaa",
            "vahvistua",
            "tehostua",
            "parantua",
            "merkittävä",
            "huomattava",
            "tärkeä",
            "kasvu",
            "vahva",
            "heikko",
            "suuri",
            "pieni",
        },
        "en": {
            "percent",
            "amount",
            "increase",
            "improve",
            "significant",
            "important",
            "growth",
            "strong",
            "weak",
            "large",
            "small",
        },
    }

    DEFAULT_WEIGHTS = {
        "statistical": 0.4,
        "llm": 0.6,
        "compound_bonus": 0.2,
        "domain_bonus": 0.15,
        "length_factor": 0.1,
        "generic_penalty": 0.3,
        "position_boost": 0.1,
        "frequency_factor": 0.15,
        "cluster_boost": 0.2,
    }

    # Technical term variations for better matching
    TERM_VARIATIONS = {
        "api": ["apis", "rest-api", "api-driven"],
        "cloud": ["cloud-based", "cloud-native", "multi-cloud"],
        "devops": ["dev-ops", "devsecops"],
        "data": ["dataset", "database", "datastore"],
        "ai": ["artificial-intelligence", "machine-learning", "ml"],
    }

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None,
    ):
        """Initialize analyzer with structured output support."""
        # Initialize analyzer config if not provided
        if llm is None:
            analyzer_config = AnalyzerConfig()
            llm = create_llm(config=analyzer_config)

            # Merge analyzer config with provided config
            if config is None:
                config = {}
            config = {**analyzer_config.config.get("analysis", {}), **config}

        super().__init__(llm, config)
        self.language_processor = language_processor

        # Initialize components with config
        self.weights = self._initialize_weights(config)
        self.min_confidence = config.get("min_confidence", 0.3)
        self.max_keywords = config.get("max_keywords", 10)
        self.language = self._get_language()

        # Create chain with structured output
        self.chain = self._create_chain()
        logger.debug(
            "KeywordAnalyzer initialized with new configuration system"
        )

    def _initialize_weights(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Initialize and validate weights."""
        weights = {
            "statistical": 0.4,
            "llm": 0.6,
            "compound_bonus": 0.2,
            "domain_bonus": 0.15,
        }
        if "weights" in config:
            weights.update(config["weights"])

        # Validate source weights sum to 1.0
        source_sum = weights["statistical"] + weights["llm"]
        if abs(source_sum - 1.0) > 0.001:
            logger.warning(
                f"Source weights sum to {source_sum}, normalizing..."
            )
            weights["statistical"] /= source_sum
            weights["llm"] /= source_sum

        return weights

    def _initialize_clustering_config(self, config: Optional[Dict]) -> Dict:
        """Initialize clustering configuration."""
        if not config:  # Handle None or empty config
            return {
                "similarity_threshold": 0.85,
                "max_cluster_size": 3,
                "boost_factor": 1.2,
                "domain_bonus": 0.1,
                "min_cluster_size": 2,
                "max_relation_distance": 2,
            }
        return config.get(
            "clustering",
            {
                "similarity_threshold": 0.85,
                "max_cluster_size": 3,
                "boost_factor": 1.2,
                "domain_bonus": 0.1,
                "min_cluster_size": 2,
                "max_relation_distance": 2,
            },
        )

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain with theme context support."""
        # Configure LLM for structured output first
        if hasattr(self.llm, "with_structured_output"):
            self.llm = self.llm.with_structured_output(KeywordOutput)

        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a keyword analysis expert. For the given text:
1. Identify specific and meaningful keywords
2. Consider compound words and technical terms
3. Calculate relevance scores
4. Group keywords by domain relevance

For Finnish text:
- Return keywords in their base dictionary form (nominative singular for nouns, infinitive for verbs)
- Preserve meaningful compound words as single keywords (e.g., "yrityskoulutuspaketti", "Python-ohjelmointi")
- Only split compound words if each part carries significant independent meaning
- For number expressions (e.g., "10 hengen"), convert the word part to base form ("10 henki")
- For hyphenated technical terms (e.g., "Python-ohjelmointi"), keep them as single keywords

{theme_context_prompt}

Focus on keywords that are clearly supported by the text content.
Do not generate generic keywords that could apply to any text."""
            ),
            (
                "human",
                """Analyze this text to identify keywords:
{text}

Language: {language}
{theme_context_str}

Required: 
1. Extract keywords with scores
2. Preserve meaningful compound words as single keywords
3. Consider domain relevance
4. For Finnish text, ensure all keywords are in their base dictionary form while keeping compound words intact when meaningful

Base all keywords on actual content and evidence from the text."""
            ),
        ])

        # Create chain with structured output
        chain = (
            {
                "text": RunnablePassthrough(),
                "language": lambda _: self._get_language(),
                "theme_context_str": lambda x: self._format_theme_context(x.get("theme_context") if isinstance(x, dict) else None),
                "theme_context_prompt": lambda x: self._get_theme_context_prompt(x.get("theme_context") if isinstance(x, dict) else None),
            }
            | template
            | self.llm
        )

        return chain

    def _format_theme_context(self, theme_context: Optional[ThemeContext]) -> str:
        """Format theme context for prompt if available."""
        if not theme_context or not theme_context.main_themes:
            return ""

        context_parts = ["Theme Context:"]
        
        # Add main themes with descriptions
        context_parts.append("\nMain Themes:")
        for theme in theme_context.main_themes:
            desc = theme_context.theme_descriptions.get(theme, "")
            conf = theme_context.theme_confidence.get(theme, 0.0)
            context_parts.append(f"- {theme} (confidence: {conf:.2f}): {desc}")
        
        # Add theme hierarchy if available
        if theme_context.theme_hierarchy:
            context_parts.append("\nTheme Relationships:")
            for parent, children in theme_context.theme_hierarchy.items():
                context_parts.append(f"- {parent} -> {', '.join(children)}")
        
        # Add theme keywords if available
        if theme_context.theme_keywords:
            context_parts.append("\nTheme Keywords:")
            for theme, keywords in theme_context.theme_keywords.items():
                if keywords:
                    context_parts.append(f"- {theme}: {', '.join(keywords)}")
        
        return "\n".join(context_parts)

    def _adjust_score_with_themes(
        self,
        keyword: str,
        base_score: float,
        theme_context: Optional[ThemeContext]
    ) -> float:
        """Adjust keyword score based on theme relevance."""
        if not theme_context or not theme_context.main_themes:
            return base_score

        max_theme_relevance = 0.0
        keyword_lower = keyword.lower()

        # Check direct keyword matches in theme keywords
        for theme, keywords in theme_context.theme_keywords.items():
            if any(kw.lower() in keyword_lower or keyword_lower in kw.lower() for kw in keywords):
                theme_conf = theme_context.theme_confidence.get(theme, 0.0)
                max_theme_relevance = max(max_theme_relevance, theme_conf)

        # Check relevance to theme descriptions
        for theme, desc in theme_context.theme_descriptions.items():
            if keyword_lower in desc.lower():
                theme_conf = theme_context.theme_confidence.get(theme, 0.0)
                max_theme_relevance = max(max_theme_relevance, theme_conf * 0.8)  # Slightly lower weight for description matches

        # Apply theme relevance boost (up to 30% boost for highly relevant keywords)
        theme_boost = 1.0 + (max_theme_relevance * 0.3)
        return min(base_score * theme_boost, 1.0)

    def _get_theme_context_prompt(self, theme_context: Optional[ThemeContext]) -> str:
        """Get theme-specific prompt section based on context availability."""
        if theme_context and theme_context.main_themes:
            return """If theme context is provided:
1. Use themes to guide keyword identification
2. Ensure keywords align with and support the identified themes
3. Consider theme hierarchy in scoring
4. Look for keywords that bridge multiple themes"""
        return ""  # Return empty string if no theme context

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

    def _calculate_score(
        self,
        word: str,
        base_score: float,
        domain: Optional[str] = None,
        compound_parts: Optional[List[str]] = None,
        theme_context: Optional[ThemeContext] = None,
        frequency: Optional[int] = None
    ) -> float:
        """Calculate final score with enhanced scoring factors."""
        # Start with base score
        score = base_score
        word_lower = word.lower()
        
        # Predefined keywords get a significant boost
        if word_lower in self.config.get("predefined_keywords", set()):
            score *= 1.3
            return min(score, 0.98)  # Cap at 0.98 for predefined keywords
        
        # Domain-specific boost
        if domain:
            score *= 1.2  # 20% boost for domain-relevant terms
        
        # Multi-word phrase scoring
        words = word.split()
        if len(words) > 1:
            if self._is_valid_phrase(word):
                score *= 1.15  # More moderate boost for valid phrases
            else:
                score *= 0.8  # Less aggressive penalty
        
        # Compound word handling
        if compound_parts:
            score *= 1.1  # 10% boost for verified compounds
        
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
        if frequency:
            freq_bonus = min(0.1, 0.03 * math.log(1 + frequency))
            score += freq_bonus
        
        # Theme-based scoring
        if theme_context and theme_context.main_themes:
            theme_relevance = self._calculate_theme_keyword_relevance(word, theme_context)
            score *= (1.0 + theme_relevance * 0.3)  # Up to 30% boost
        
        return min(score, 0.95)

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
        patterns = technical_patterns.get(self._get_language(), technical_patterns["en"])
        return any(re.search(pattern, word.lower()) for pattern in patterns)

    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if a multi-word phrase is valid and meaningful."""
        try:
            # Split into words
            words = phrase.split()
            if len(words) < 2:
                return True  # Single words are valid
                
            # Skip phrases with stopwords in the middle
            if self.language_processor and any(
                self.language_processor.is_stop_word(w.lower()) for w in words[1:-1]
            ):
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
                
            # Get POS tags if available
            if nltk:
                pos_tags = nltk.pos_tag(words)
                
                # Valid phrase patterns
                valid_patterns = [
                    ['JJ', 'NN'],  # Adjective + Noun
                    ['NN', 'NN'],  # Noun + Noun
                    ['JJ', 'JJ', 'NN'],  # Adj + Adj + Noun
                    ['JJ', 'NN', 'NN'],  # Adj + Noun + Noun
                    ['NN', 'NN', 'NN'],  # Noun + Noun + Noun
                    ['VBG', 'NN'],  # Gerund + Noun
                    ['NN', 'CC', 'NN'],  # Noun + Conjunction + Noun
                    ['NN', 'IN', 'NN'],  # Noun + Preposition + Noun
                ]
                
                pattern = [tag for word, tag in pos_tags]
                
                # Check pattern validity
                for valid_pattern in valid_patterns:
                    if len(pattern) == len(valid_pattern):
                        if pattern == valid_pattern:
                            return True
                        if all(p.startswith('NN') for p in pattern):
                            return True
                        if len(pattern) == 2 and pattern[0] in ['JJ', 'VBG'] and pattern[1] == 'NN':
                            return True
            
            # Special cases
            if self._is_technical_term(phrase) or any(self._is_technical_term(w) for w in words):
                return True
                
            if '-' in phrase:
                return True
                
            # Check if all parts are in domain terms
            if all(w.lower() in self.DOMAIN_TERMS[DomainType.TECHNICAL].terms or 
                   w.lower() in self.DOMAIN_TERMS[DomainType.BUSINESS].terms 
                   for w in words):
                return True
                
            return False
            
        except Exception as e:
            logger.debug(f"Error checking phrase validity: {e}")
            return False  # Be conservative if check fails

    def _create_error_result(self, error: str = "Analysis failed") -> KeywordAnalysisResult:
        """Create error result."""
        return KeywordAnalysisResult(
            keywords=[],
            compound_words=[],
            domain_keywords={},
            language=self._get_language(),
            success=False,
            error=error,
        )

    def _handle_error(self, error: str) -> KeywordOutput:
        """Create error output that matches model requirements."""
        return KeywordOutput(
            keywords=[],
            domain_keywords={},
            error=str(error),
            success=False,
            language=self._get_language(),
        )

    def _post_process_llm_output(self, response: Any) -> Dict[str, Any]:
        """Process LLM response and apply user-defined parameters as guidance."""
        try:
            # Parse base response
            parsed = self._parse_response(response)
            
            # Process keywords with user preferences as guidance
            processed_keywords = []
            for kw in parsed.get("keywords", []):
                keyword = kw.get("keyword", "").lower()
                score = kw.get("score", 0.0)
                
                # Skip only exact matches with excluded keywords
                if any(keyword == excluded.lower() for excluded in self.config.excluded_keywords):
                    continue
                    
                # Boost score for predefined keywords to prioritize them
                if any(predefined.lower() in keyword for predefined in self.config.predefined_keywords):
                    score = min(score * 1.2, 1.0)  # 20% boost for predefined keywords
                
                # Include keyword if it's either predefined or has sufficient score
                if score >= self.min_confidence:
                    processed_keywords.append({
                        "keyword": kw.get("keyword"),
                        "score": score,
                        "domain": kw.get("domain"),
                        "compound_parts": kw.get("compound_parts", []),
                        "is_predefined": any(predefined.lower() in keyword for predefined in self.config.predefined_keywords)
                    })
            
            # Sort keywords by score, but ensure predefined keywords get priority
            processed_keywords.sort(key=lambda x: (not x["is_predefined"], -x["score"]))
            
            # Remove the temporary is_predefined field before returning
            for kw in processed_keywords:
                kw.pop("is_predefined", None)
            
            return {
                "keywords": processed_keywords[:self.max_keywords],
                "domain_keywords": parsed.get("domain_keywords", {}),
                "language": parsed.get("language", self._get_language()),
                "success": parsed.get("success", True)
            }
            
        except Exception as e:
            logger.error(f"Error post-processing response: {e}")
            return self._handle_error(str(e))

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract content from responses
            if hasattr(response, "content"):
                data = json.loads(response.content)
            elif isinstance(response, str):
                data = json.loads(response)
            elif isinstance(response, dict):
                data = response
            else:
                return {
                    "keywords": [],
                    "compound_words": [],
                    "domain_keywords": {},
                    "success": True,
                    "language": self._get_language(),
                }

            # Transform keyword format to match Pydantic model
            transformed_keywords = []
            for kw in data.get("keywords", []):
                # Handle both string and dict formats for keywords
                if isinstance(kw, str):
                    # Get base form for the entire keyword phrase
                    keyword = kw.strip()
                    if self.language_processor:
                        keyword = self.language_processor.get_base_form(keyword)
                    transformed = {
                        "keyword": keyword,
                        "score": 0.5,
                        "domain": None,
                        "compound_parts": [],
                    }
                else:
                    # Get base form for the keyword and its parts
                    keyword = kw.get("keyword", kw.get("text", "")).strip()
                    if self.language_processor:
                        keyword = self.language_processor.get_base_form(keyword)
                    
                    # Process compound parts if available
                    compound_parts = kw.get("compound_parts", [])
                    if self.language_processor:
                        compound_parts = [
                            self.language_processor.get_base_form(part.strip())
                            for part in compound_parts
                        ]
                    
                    transformed = {
                        "keyword": keyword,
                        "score": kw.get("score", kw.get("confidence", 0.5)),
                        "domain": kw.get("domain"),
                        "compound_parts": compound_parts,
                    }
                transformed_keywords.append(transformed)

            # Convert compound words to base form
            compound_words = []
            for word in data.get("compound_words", []):
                if self.language_processor:
                    compound_words.append(self.language_processor.get_base_form(word.strip()))
                else:
                    compound_words.append(word.strip())

            # Convert domain keywords to base form
            domain_keywords = {}
            for domain, keywords in data.get("domain_keywords", {}).items():
                if self.language_processor:
                    domain_keywords[domain] = [
                        self.language_processor.get_base_form(kw.strip())
                        for kw in keywords
                    ]
                else:
                    domain_keywords[domain] = [kw.strip() for kw in keywords]

            return {
                "keywords": transformed_keywords,
                "compound_words": compound_words,
                "domain_keywords": domain_keywords,
                "language": data.get("language", self._get_language()),
                "success": data.get("success", True)
            }

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "keywords": [],
                "compound_words": [],
                "domain_keywords": {},
                "success": False,
                "language": self._get_language(),
                "error": str(e)
            }

    def _get_base_form(self, word: str) -> str:
        """Get base form of a word with special handling for compound words."""
        if not word:
            return word
            
        # Get language for specific handling
        language = self._get_language()
        
        # Special handling for Finnish compound words
        if language == "fi":
            # Handle number expressions (e.g., "10 hengen" -> "10 henki")
            if any(c.isdigit() for c in word):
                parts = word.split()
                if len(parts) > 1:
                    number_part = next((p for p in parts if any(c.isdigit() for c in p)), None)
                    word_parts = [p for p in parts if not any(c.isdigit() for c in p)]
                    if number_part and word_parts and self.language_processor:
                        base_words = [self.language_processor.get_base_form(p) or p for p in word_parts]
                        return f"{number_part} {' '.join(base_words)}"
            
            # Handle hyphenated words
            if '-' in word:
                parts = word.split('-')
                if self.language_processor:
                    base_parts = [
                        self.language_processor.get_base_form(part) or part
                        for part in parts
                    ]
                    return '-'.join(base_parts)
                return word
            
            # Use language processor for base form
            if self.language_processor:
                return self.language_processor.get_base_form(word) or word
            return word
        
        # Special handling for hyphenated words in other languages
        if '-' in word:
            parts = word.split('-')
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

    async def analyze(self, text: str) -> KeywordAnalysisResult:
        """Analyze text for keyword extraction."""
        try:
            if text is None:
                raise ValueError("Input text cannot be None")
            if not text.strip():
                return KeywordAnalysisResult(
                    keywords=[],
                    compound_words=[],
                    domain_keywords={},
                    language=self.language,
                    success=False,
                    error="Empty input text"
                )

            # Get LLM response
            llm_result = await self._get_llm_response(text)

            # Handle direct KeywordOutput
            if isinstance(llm_result, KeywordOutput):
                return KeywordAnalysisResult(
                    keywords=llm_result.keywords,
                    compound_words=llm_result.compound_words,
                    domain_keywords=llm_result.domain_keywords,
                    language=llm_result.language,
                    success=llm_result.success,
                    error=llm_result.error
                )

            # Handle string or dict response
            if isinstance(llm_result, str):
                try:
                    llm_result = json.loads(llm_result)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    return KeywordAnalysisResult(
                        keywords=[],
                        compound_words=[],
                        domain_keywords={},
                        language=self.language,
                        success=False,
                        error=f"Invalid LLM response format: {e}"
                    )

            if isinstance(llm_result, dict):
                try:
                    keyword_output = KeywordOutput(**llm_result)
                    return KeywordAnalysisResult(
                        keywords=keyword_output.keywords,
                        compound_words=keyword_output.compound_words,
                        domain_keywords=keyword_output.domain_keywords,
                        language=keyword_output.language,
                        success=keyword_output.success,
                        error=keyword_output.error
                    )
                except ValidationError as e:
                    logger.error(f"Failed to validate LLM response: {e}")
                    return KeywordAnalysisResult(
                        keywords=[],
                        compound_words=[],
                        domain_keywords={},
                        language=self.language,
                        success=False,
                        error=f"Invalid LLM response structure: {e}"
                    )

            # If we get here, the response format is invalid
            logger.error(f"Invalid LLM response type: {type(llm_result)}")
            return KeywordAnalysisResult(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language=self.language,
                success=False,
                error=f"Invalid LLM response type: {type(llm_result)}"
            )

        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            return KeywordAnalysisResult(
                keywords=[],
                compound_words=[],
                domain_keywords={},
                language=self.language,
                success=False,
                error=str(e)
            )

    def _group_by_domain(self, keywords: List[KeywordInfo]) -> Dict[str, List[str]]:
        """Group keywords by their domain."""
        domain_groups: Dict[str, List[str]] = defaultdict(list)
        
        for kw in keywords:
            if kw.domain:
                domain_groups[kw.domain].append(kw.keyword)
        
        return dict(domain_groups)

    async def _get_llm_response(self, text: str) -> Any:
        """Get LLM response with proper error handling."""
        try:
            if not text or not text.strip():
                raise ValueError("Empty input text")

            # Get response from chain
            result = await self.chain.ainvoke(text)
            logger.debug(f"LLM response: {result}")

            # Handle structured output
            if isinstance(result, KeywordOutput):
                return result

            # Handle AIMessage with structured output
            if isinstance(result, BaseMessage):
                if hasattr(result, "additional_kwargs") and "structured_output" in result.additional_kwargs:
                    return result.additional_kwargs["structured_output"]
                return result.content

            # Handle dict response
            if isinstance(result, dict):
                return result

            # Handle string response
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return result

            raise ValueError(f"Unexpected response type: {type(result)}")

        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return self._handle_error(str(e))