# API Reference

Complete API documentation for the Semantic Text Analyzer.

## Core Components

### SemanticAnalyzer

Main interface for text analysis.

```python
class SemanticAnalyzer:
    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs
    ) -> None
```

**Parameters:**
- `parameter_file`: Path to Excel parameter file
- `file_utils`: Optional FileUtils instance
- `llm`: Optional language model instance
- `**kwargs`: Additional configuration options

**Methods:**

```python
async def analyze(
    self,
    text: str,
    analysis_types: Optional[List[str]] = None,
    timeout: float = 60.0,
    **kwargs
) -> CompleteAnalysisResult
```
Performs complete text analysis.

**Parameters:**
- `text`: Input text to analyze
- `analysis_types`: List of analysis types (["keywords", "themes", "categories"])
- `timeout`: Maximum execution time in seconds
- `**kwargs`: Additional analysis parameters

**Returns:** `CompleteAnalysisResult`

```python
async def analyze_batch(
    self,
    texts: List[str],
    batch_size: int = 3,
    timeout: float = 30.0,
    **kwargs
) -> List[CompleteAnalysisResult]
```
Process multiple texts with controlled concurrency.

### KeywordAnalyzer

```python
class KeywordAnalyzer:
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict] = None,
        language_processor: Optional[BaseTextProcessor] = None
    )
```

**Methods:**

```python
async def analyze(self, text: str) -> KeywordOutput
```
Extracts keywords from text.

### ThemeAnalyzer

```python
class ThemeAnalyzer:
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict] = None,
        language_processor: Optional[BaseTextProcessor] = None
    )
```

**Methods:**

```python
async def analyze(self, text: str) -> ThemeOutput
```
Identifies themes in text.

### CategoryAnalyzer

```python
class CategoryAnalyzer:
    def __init__(
        self,
        categories: Dict[str, CategoryConfig],
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict] = None,
        language_processor: Optional[BaseTextProcessor] = None
    )
```

**Methods:**

```python
async def analyze(self, text: str) -> CategoryOutput
```
Classifies text into categories.

## Data Models

### CompleteAnalysisResult

```python
class CompleteAnalysisResult(BaseModel):
    keywords: KeywordAnalysisResult
    themes: ThemeAnalysisResult
    categories: CategoryAnalysisResult
    language: str
    success: bool
    error: Optional[str]
    processing_time: float
```

### KeywordInfo

```python
class KeywordInfo(BaseModel):
    keyword: str
    score: float
    domain: Optional[str]
    compound_parts: Optional[List[str]]
```

### ThemeInfo

```python
class ThemeInfo(BaseModel):
    name: str
    description: str
    confidence: float
    keywords: List[str]
    parent_theme: Optional[str]
```

### CategoryMatch

```python
class CategoryMatch(BaseModel):
    name: str
    confidence: float
    description: str
    evidence: List[Evidence]
    themes: List[str]
```

## Language Processing

### BaseTextProcessor

```python
class BaseTextProcessor(ABC):
    def __init__(
        self,
        language: str,
        custom_stop_words: Optional[Set[str]] = None,
        config: Optional[Dict[str, Any]] = None
    )
```

**Abstract Methods:**
```python
@abstractmethod
def get_base_form(self, word: str) -> str:
    """Get base form of a word."""
    pass

@abstractmethod
def tokenize(self, text: str) -> List[str]:
    """Tokenize text into words."""
    pass

@abstractmethod
def is_compound_word(self, word: str) -> bool:
    """Check if word is a compound word."""
    pass

@abstractmethod
def get_compound_parts(self, word: str) -> Optional[List[str]]:
    """Get parts of compound word."""
    pass
```

### EnglishTextProcessor

Implements `BaseTextProcessor` for English language.

### FinnishTextProcessor

Implements `BaseTextProcessor` for Finnish language with Voikko support.

## Configuration

### AnalyzerConfig

```python
class AnalyzerConfig:
    def __init__(
        self,
        file_utils: Optional[FileUtils] = None
    )
```

**Methods:**

```python
def get_provider_config(
    self,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]
```

## File Operations

### FileUtils

```python
class FileUtils:
    @classmethod
    def create_azure_utils(
        cls,
        connection_string: str,
        **kwargs
    ) -> 'FileUtils'
```

**Methods:**

```python
def save_data_to_storage(
    self,
    data: Dict[str, Any],
    output_filetype: OutputFileType,
    file_name: str,
    output_type: str = "processed",
    include_timestamp: bool = True,
    **kwargs
) -> Tuple[Dict[str, Path], Optional[str]]
```

## Utility Functions

```python
def create_text_processor(
    language: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    file_utils: Optional[FileUtils] = None
) -> BaseTextProcessor
```

```python
def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseChatModel
```

## Exceptions

```python
class ValidationError(Exception):
    def __init__(self, message: str, details: Dict[str, Any]):
        self.message = message
        self.details = details
        super().__init__(message)
```

## Constants

```python
VALID_ANALYSIS_TYPES = {"keywords", "themes", "categories"}

DEFAULT_CONFIG = {
    "default_language": "en",
    "content_column": "content",
    "models": {
        "default_provider": "azure",
        "default_model": "gpt-4o-mini"
    }
}
```

## Type Hints

```python
from typing import (
    Dict, List, Optional, Set, Union, Any, Tuple,
    TypeVar, Generic, Callable, Awaitable
)

TextProcessor = TypeVar('TextProcessor', bound=BaseTextProcessor)
LLMType = TypeVar('LLMType', bound=BaseChatModel)
```

For more examples and usage patterns, see [EXAMPLES.md](EXAMPLES.md).