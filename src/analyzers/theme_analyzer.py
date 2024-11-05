# src/analyzers/theme_analyzer.py

from typing import List, Dict, Any, Optional
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

from .base import TextAnalyzer, AnalyzerOutput

class ThemeOutput(AnalyzerOutput):
    """Output model for theme analysis."""
    themes: List[str] = Field(default_factory=list)
    theme_descriptions: Dict[str, str] = Field(default_factory=dict)
    theme_confidence: Dict[str, float] = Field(default_factory=dict)
    related_keywords: Dict[str, List[str]] = Field(default_factory=dict)

class ThemeAnalyzer(TextAnalyzer):
    """Analyzes text to identify main themes and topics."""
    
    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain for theme extraction."""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying main themes and topics in text.
            Focus on broad concepts and underlying messages rather than specific keywords.
            Identify themes that capture the key narratives and subject areas."""),
            ("human", """Analyze this text and identify main themes.
            
            Text: {text}
            Maximum Themes: {max_themes}
            Focus Areas: {focus_areas}
            
            Return themes in this JSON format:
            {
                "themes": ["theme1", "theme2", ...],
                "descriptions": {
                    "theme1": "brief description",
                    "theme2": "brief description",
                    ...
                },
                "confidence": {
                    "theme1": 0.95,
                    "theme2": 0.85,
                    ...
                },
                "related_keywords": {
                    "theme1": ["keyword1", "keyword2"],
                    "theme2": ["keyword3", "keyword4"],
                    ...
                }
            }""")
        ])
        
        chain = (
            {
                "text": RunnablePassthrough(),
                "max_themes": lambda _: self.config.get("max_themes", 3),
                "focus_areas": lambda _: self.config.get("focus_areas", "general topics")
            }
            | template 
            | self.llm
            | self._post_process_llm_output
        )
        
        return chain
    
    def _post_process_llm_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM output to standardized format."""
        try:
            if isinstance(output, str):
                import json
                output = json.loads(output)
            
            return {
                "themes": output.get("themes", []),
                "descriptions": output.get("descriptions", {}),
                "confidence": output.get("confidence", {}),
                "related_keywords": output.get("related_keywords", {})
            }
        except Exception as e:
            self.logger.error(f"Error processing LLM output: {e}")
            return {
                "themes": [],
                "descriptions": {},
                "confidence": {},
                "related_keywords": {}
            }
    
    def _validate_themes(
        self,
        themes: List[str],
        descriptions: Dict[str, str],
        confidence: Dict[str, float],
        related_keywords: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Validate and clean theme data."""
        valid_themes = []
        valid_descriptions = {}
        valid_confidence = {}
        valid_keywords = {}
        
        for theme in themes:
            if not theme or not isinstance(theme, str):
                continue
                
            theme = theme.strip()
            if not theme:
                continue
                
            valid_themes.append(theme)
            valid_descriptions[theme] = descriptions.get(theme, "").strip()
            valid_confidence[theme] = min(max(confidence.get(theme, 0.5), 0.0), 1.0)
            valid_keywords[theme] = [
                k.strip() for k in related_keywords.get(theme, [])
                if k and isinstance(k, str) and k.strip()
            ]
        
        return {
            "themes": valid_themes,
            "descriptions": valid_descriptions,
            "confidence": valid_confidence,
            "related_keywords": valid_keywords
        }
    
    async def analyze(
        self,
        text: str,
        max_themes: Optional[int] = None,
        focus_areas: Optional[str] = None,
        **kwargs
    ) -> ThemeOutput:
        """Analyze text to identify themes.
        
        Args:
            text: Input text to analyze
            max_themes: Maximum number of themes to identify
            focus_areas: Specific areas to focus on
            **kwargs: Additional parameters
            
        Returns:
            ThemeOutput: Analysis results
        """
        # Validate input
        if error := self._validate_input(text):
            return self._handle_error(error)
        
        try:
            # Update config with parameters
            if max_themes is not None:
                self.config["max_themes"] = max_themes
            if focus_areas is not None:
                self.config["focus_areas"] = focus_areas
            
            # Get themes from LLM
            result = await self.chain.ainvoke(text)
            
            # Validate and clean results
            validated = self._validate_themes(
                result.get("themes", []),
                result.get("descriptions", {}),
                result.get("confidence", {}),
                result.get("related_keywords", {})
            )
            
            return ThemeOutput(
                themes=validated["themes"],
                theme_descriptions=validated["descriptions"],
                theme_confidence=validated["confidence"],
                related_keywords=validated["related_keywords"],
                language=self._detect_language(text)
            )
            
        except Exception as e:
            self.logger.error(f"Error in theme analysis: {e}")
            return self._handle_error(str(e))