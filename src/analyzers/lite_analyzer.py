"""Lightweight semantic analyzer that combines all analyses into a single LLM call."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Union
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langdetect import detect

from src.schemas import (
    KeywordAnalysisResult,
    ThemeAnalysisResult,
    CategoryAnalysisResult,
    CompleteAnalysisResult,
    KeywordInfo,
    ThemeInfo,
    CategoryMatch,
)
from src.loaders.parameter_handler import ParameterHandler
from src.core.config import AnalyzerConfig
from src.core.language_processing import create_text_processor
from FileUtils import FileUtils

logger = logging.getLogger(__name__)

class LiteAnalysisOutput(BaseModel):
    """Combined output schema for lite semantic analysis."""
    
    keywords: List[str] = Field(
        default_factory=list,
        description="List of extracted keywords from the text"
    )
    compound_words: List[str] = Field(
        default_factory=list,
        description="List of extracted compound words or phrases"
    )
    themes: List[str] = Field(
        default_factory=list,
        description="List of identified themes"
    )
    theme_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Hierarchical organization of themes with parent themes as keys and related sub-themes as values"
    )
    categories: List[str] = Field(
        default_factory=list,
        description="List of identified categories that match the text"
    )

class LiteSemanticAnalyzer:
    """Lightweight semantic analyzer that performs all analyses in a single LLM call."""

    def __init__(
        self,
        llm: BaseChatModel,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        available_categories: Optional[Set[str]] = None,
        language: str = "en",
        config: Optional[Dict] = None,
    ):
        """Initialize the analyzer.
        
        Args:
            llm: Language model to use
            parameter_file: Path to parameter file
            file_utils: FileUtils instance for file operations
            available_categories: Set of valid categories to choose from
            language: Language of the text to analyze ('en' or 'fi')
            config: Optional configuration dictionary
        """
        self.llm = llm
        self.file_utils = file_utils
        self.available_categories = available_categories or set()
        self.language = language.lower()
        if self.language not in ["en", "fi"]:
            raise ValueError("Language must be 'en' or 'fi'")
            
        self.output_parser = PydanticOutputParser(pydantic_object=LiteAnalysisOutput)
        
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

    def _create_analysis_prompt(self, text: str, analysis_types: Optional[List[str]] = None) -> str:
        """Create the combined analysis prompt."""
        # Base prompt with configuration context
        focus = self.config.get('focus_on', 'general')
        min_confidence = self.config.get('min_confidence', 0.6)
        
        # Create base prompt
        prompt = f"""Analyze the following text and extract the requested information.

Configuration parameters:
- Minimum keyword length: {self.config.get('min_keyword_length', 3)}
- Minimum confidence threshold: {min_confidence}
- Analysis focus: {focus}
- Include compound words: {self.config.get('include_compounds', True)}
- Output language: {self.language} (provide all output in this language)

Text: {text}

Please provide a structured analysis with the following components:"""

        # Add analysis type specific instructions
        all_types = {"keywords", "themes", "categories"}
        types_to_run = set(analysis_types) if analysis_types else all_types

        if "keywords" in types_to_run:
            prompt += "\n1. Keywords:"
            for instr in [
                "Extract only the most significant and domain-specific keywords and phrases",
                f"Focus on {focus}-related terms and concepts",
                "Include both single words and multi-word expressions",
                "Provide confidence scores between 0.0 and 1.0 for each keyword",
                "Higher scores (>0.8) for exact matches and highly relevant terms",
                "Lower scores (0.6-0.8) for related or contextual terms",
                f"Only include keywords with confidence above {min_confidence}",
                f"Minimum length for single words: {self.config.get('min_keyword_length', 3)}",
                "Avoid generic verbs, common adjectives, and non-specific terms",
                "Return keywords in their singular form unless plurality is significant",
                "If a compound word is included, do not include its individual parts",
                f"Ensure all keywords are in the {self.language} language"
            ]:
                prompt += f"\n   - {instr}"

        if "themes" in types_to_run:
            prompt += "\n2. Themes:"
            for instr in [
                f"Identify the main themes present in the text, focusing on {focus} aspects",
                "Organize themes hierarchically, showing relationships between main themes and sub-themes",
                "Provide confidence scores between 0.0 and 1.0 for each theme",
                f"Only include themes with confidence above {min_confidence}",
                f"Ensure all themes are in the {self.language} language"
            ]:
                prompt += f"\n   - {instr}"

        if "categories" in types_to_run and self.available_categories:
            prompt += "\n3. Categories:"
            for instr in [
                "From the following categories, select those that best match the text",
                "Provide confidence scores between 0.0 and 1.0 for each category",
                f"Only include categories with confidence above {min_confidence}",
                "Base confidence on how well the text matches each category's scope"
            ]:
                prompt += f"\n   - {instr}"
            prompt += f"\n   - {', '.join(self.available_categories)}"

        # Add format instructions
        prompt += f"\n\n{self.output_parser.get_format_instructions()}"
        
        return prompt

    def _process_keywords(self, keywords: List[str], compound_words: List[str]) -> List[KeywordInfo]:
        """Process and clean keywords."""
        processed_keywords = []
        
        # Process compound words first
        processed_compounds = set()
        
        for compound in compound_words:
            # Handle compound words based on language
            if self.language == "fi":
                # Finnish processor returns a list of base forms
                base_form = self.language_processor.get_base_form(compound)
                if not base_form:
                    continue
                processed_compound = base_form
            else:
                # English processor just needs basic cleaning
                processed_compound = self.language_processor.get_base_form(compound)
                
            # Calculate confidence based on word count and specificity
            word_count = len(processed_compound.split())
            base_score = 0.7
            specificity_bonus = min(0.1 * word_count, 0.3)
            confidence = min(base_score + specificity_bonus, 1.0)
            
            processed_keywords.append(KeywordInfo(
                keyword=processed_compound,
                score=confidence,
                frequency=1
            ))
            # Add parts to processed set
            processed_compounds.update(processed_compound.lower().split())
            
        # Process individual keywords
        for keyword in keywords:
            # Skip if word is part of a compound
            if keyword.lower() in processed_compounds:
                continue
                
            # Use language processor for keyword processing
            processed_keyword = self.language_processor.get_base_form(keyword)
            if not processed_keyword:
                continue
            
            # Calculate confidence
            base_score = 0.6
            length_bonus = min(0.1 * (len(processed_keyword) / 5), 0.2)
            proper_noun_bonus = 0.2 if processed_keyword[0].isupper() else 0
            confidence = min(base_score + length_bonus + proper_noun_bonus, 1.0)
            
            processed_keywords.append(KeywordInfo(
                keyword=processed_keyword,
                score=confidence,
                frequency=1
            ))
            
        return processed_keywords

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
    ) -> CompleteAnalysisResult:
        """Perform semantic analysis on the text.
        
        Args:
            text: Text to analyze
            analysis_types: List of analysis types to perform. If None, performs all analyses.
            
        Returns:
            CompleteAnalysisResult containing all analysis results
        """
        start_time = datetime.now()
        
        try:
            # Create and run the combined prompt
            prompt = self._create_analysis_prompt(text, analysis_types)
            response = await self.llm.ainvoke(prompt)
            
            # Extract content from the message response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured output
            parsed_output = self.output_parser.parse(response_text)
            
            # Process keywords with intelligent scoring and lemmatization
            all_keywords = self._process_keywords(
                keywords=parsed_output.keywords,
                compound_words=parsed_output.compound_words
            )
            
            keyword_result = KeywordAnalysisResult(
                keywords=all_keywords,
                compound_words=[],  # No longer needed as we combine them
                domain_keywords={},  # Not used in lite version
                language=self.language,
                success=True
            )
            
            # Convert themes to ThemeInfo objects with varying confidence
            themes = []
            for theme in parsed_output.themes:
                # Calculate theme confidence based on hierarchy position and specificity
                is_main_theme = any(theme == main for main in parsed_output.theme_hierarchy.keys())
                base_confidence = 0.8 if is_main_theme else 0.7
                specificity_bonus = 0.2 if len(theme.split()) > 1 else 0.1
                confidence = min(base_confidence + specificity_bonus, 1.0)
                
                themes.append(ThemeInfo(
                    name=theme,
                    description=theme,  # Using theme text as description
                    confidence=confidence,
                    score=confidence,
                ))
            
            theme_result = ThemeAnalysisResult(
                themes=themes,
                theme_hierarchy=parsed_output.theme_hierarchy,
                language=self.language,
                success=True
            )
            
            # Process categories with confidence scoring
            category_matches = []
            for cat in parsed_output.categories:
                # Calculate category confidence based on theme overlap and keyword presence
                theme_overlap = sum(1 for theme in themes if cat.lower() in theme.name.lower())
                keyword_overlap = sum(1 for kw in all_keywords if cat.lower() in kw.keyword.lower())
                
                base_confidence = 0.7
                theme_bonus = min(0.1 * theme_overlap, 0.2)
                keyword_bonus = min(0.05 * keyword_overlap, 0.1)
                confidence = min(base_confidence + theme_bonus + keyword_bonus, 1.0)
                
                category_matches.append(CategoryMatch(
                    name=cat,
                    confidence=confidence,
                    description=f"Category match: {cat}",
                    evidence=[kw.keyword for kw in all_keywords if cat.lower() in kw.keyword.lower()],
                    themes=[theme.name for theme in themes if cat.lower() in theme.name.lower()]
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
                    "analysis_type": "lite",
                    "language": self.language,
                    "config": self.config,
                }
            )
            
            # Print results
            print(f"\nAnalysis Results for text in {self.language}:")
            if result.keywords and result.keywords.success:
                print("\nKeywords:")
                print(', '.join(kw.keyword for kw in result.keywords.keywords))
            
            if result.themes and result.themes.success:
                print("\nThemes:")
                print(f"- Main themes: {', '.join(theme.name for theme in result.themes.themes)}")
                print("\nTheme Hierarchy:")
                for main_theme, sub_themes in result.themes.theme_hierarchy.items():
                    print(f"- {main_theme}: {', '.join(sub_themes)}")
            
            if result.categories and result.categories.success:
                print("\nCategories:")
                print(f"- Matches: {', '.join(cat.name for cat in result.categories.matches)}")

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
                processing_time=processing_time
            ) 