# src/analyzers/theme_analyzer.py

from typing import Any, Dict, List, Optional, Union
import logging
import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src.analyzers.base import TextAnalyzer, AnalyzerOutput

logger = logging.getLogger(__name__)

class ThemeInfo(BaseModel):
    """Information about an identified theme."""
    name: str
    description: str = ""
    confidence: float = 0.0
    keywords: List[str] = Field(default_factory=list)

class ThemeOutput(AnalyzerOutput):  # Inherit from AnalyzerOutput
    """Output model for theme analysis results."""
    themes: List[ThemeInfo] = Field(default_factory=list)
    theme_descriptions: Dict[str, str] = Field(default_factory=dict)
    theme_confidence: Dict[str, float] = Field(default_factory=dict)
    related_keywords: Dict[str, List[str]] = Field(default_factory=dict)

class ThemeAnalyzer(TextAnalyzer):
    """Analyzes text to identify main themes and topics."""
    
    def __init__(
        self,
        llm=None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(llm, config)
        self.chain = self._create_chain()

    def _create_chain(self) -> RunnableSequence:
        """Create LangChain processing chain for theme extraction."""
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a text analysis expert focusing on theme identification.
                Your task is to identify main themes in the given text and output them in a structured JSON format.
                Focus on clear, concrete themes rather than abstract concepts."""
            ),
            (
                "human",
                """Analyze this text and identify key themes.
                
                Text: {text}
                Maximum Themes: {max_themes}
                Focus Areas: {focus_areas}
                
                Output the results in this exact JSON format:
                {{
                    "themes": ["Theme 1", "Theme 2"],
                    "descriptions": {{
                        "Theme 1": "Description of theme 1",
                        "Theme 2": "Description of theme 2"
                    }},
                    "confidence": {{
                        "Theme 1": 0.85,
                        "Theme 2": 0.75
                    }},
                    "related_keywords": {{
                        "Theme 1": ["keyword1", "keyword2"],
                        "Theme 2": ["keyword3", "keyword4"]
                    }}
                }}"""
            )
        ])

        chain = (
            {
                "text": RunnablePassthrough(),
                "max_themes": lambda _: self.config.get("max_themes", 3),
                "focus_areas": lambda _: self.config.get("focus_areas", "business,finance,technical")
            }
            | template
            | self.llm
            | self._post_process_llm_output
        )

        return chain

    def _process_initial_content(self, content: Any) -> Dict[str, Any]:
        """Process the initial LLM output content."""
        try:
            # Parse JSON if needed
            if isinstance(content, str):
                parsed = json.loads(content)
            else:
                parsed = content
            
            # Create theme objects
            theme_objects = []
            descriptions = {}
            confidence_scores = {}
            keywords_dict = {}
            
            # Get theme data
            theme_names = parsed.get("themes", [])
            desc_dict = parsed.get("descriptions", {})
            conf_dict = parsed.get("confidence", {})
            kw_dict = parsed.get("related_keywords", {})
            
            # Process each theme
            for theme_name in theme_names:
                theme_info = ThemeInfo(
                    name=theme_name,
                    description=desc_dict.get(theme_name, ""),
                    confidence=float(conf_dict.get(theme_name, 0.0)),
                    keywords=kw_dict.get(theme_name, [])
                )
                theme_objects.append(theme_info)
                
                # Store metadata using string keys
                descriptions[theme_name] = theme_info.description
                confidence_scores[theme_name] = theme_info.confidence
                keywords_dict[theme_name] = theme_info.keywords
            
            return {
                "themes": theme_objects,
                "theme_descriptions": descriptions,
                "theme_confidence": confidence_scores,
                "related_keywords": keywords_dict
            }
            
        except Exception as e:
            logger.error(f"Error processing initial content: {str(e)}", exc_info=True)
            return None

    def _post_process_llm_output(self, output: Any) -> Dict[str, Any]:
        """Process LLM output to standardized format."""
        try:
            # Get raw content
            if hasattr(output, "content"):
                content = output.content
                logger.debug(f"Raw LLM output content: {content}")
                processed = self._process_initial_content(content)
            else:
                # Direct content
                processed = output
                
            if processed is None:
                return self._create_empty_output()
                
            logger.debug(f"Processed {len(processed['themes'])} themes successfully")
            return processed

        except Exception as e:
            logger.error(f"Error in post processing: {str(e)}", exc_info=True)
            return self._create_empty_output()

    def _create_empty_output(self) -> Dict[str, Any]:
        """Create empty output structure."""
        return {
            "themes": [],
            "theme_descriptions": {},
            "theme_confidence": {},
            "related_keywords": {}
        }

    async def analyze(self, text: str, **kwargs) -> ThemeOutput:
        """Analyze text to identify themes."""
        if not text:
            return ThemeOutput(error="Empty text", success=False)

        try:
            # Get LLM analysis
            logger.debug("Getting LLM analysis...")
            raw_result = await self.chain.ainvoke(text)
            
            # Process results
            logger.debug("Processing LLM results...")
            processed = self._post_process_llm_output(raw_result)
            
            # Check for themes
            if not processed["themes"]:
                return ThemeOutput(
                    error="No themes identified",
                    success=False,
                    language=self._detect_language(text)
                )

            # Create successful output
            return ThemeOutput(
                themes=processed["themes"],
                theme_descriptions=processed["theme_descriptions"],
                theme_confidence=processed["theme_confidence"],
                related_keywords=processed["related_keywords"],
                language=self._detect_language(text),
                success=True
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return ThemeOutput(
                error=str(e),
                success=False,
                language=self._detect_language(text)
            )

    def display_theme_results(self, results: Union[Dict[str, Any], ThemeOutput]) -> None:
        """Display theme analysis results."""
        print("\nTheme Analysis Results")
        print("=" * 50)
        
        try:
            # Convert to dict if needed
            if hasattr(results, "model_dump"):
                results = results.model_dump()
            elif hasattr(results, "dict"):
                results = results.dict()

            # Handle error case
            if results.get("error"):
                print(f"\nError: {results['error']}")
                return

            # Display themes
            themes = results.get("themes", [])
            if not themes:
                print("\nNo themes identified.")
                return

            print(f"\nIdentified {len(themes)} themes:\n")
            
            for i, theme in enumerate(themes, 1):
                # Extract theme data based on type
                if isinstance(theme, dict):
                    name = theme.get('name', 'Unnamed')
                    desc = theme.get('description', 'No description')
                    conf = theme.get('confidence', 0.0)
                    kw = theme.get('keywords', [])
                else:
                    name = theme.name
                    desc = theme.description
                    conf = theme.confidence
                    kw = theme.keywords

                # Print formatted theme info
                print(f"\nTheme {i}: {name}")
                print("-" * (len(f"Theme {i}: {name}") + 5))
                print(f"Description: {desc}")
                print(f"Confidence: {conf:.2f}")
                if kw:
                    print(f"Keywords: {', '.join(kw)}")

        except Exception as e:
            logger.error(f"Display error: {str(e)}", exc_info=True)
            print("Error displaying theme results.")