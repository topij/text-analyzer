# API Reference

## Environment Management

### EnvironmentManager

The `EnvironmentManager` class provides centralized environment setup and management.

```python
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Initialize
config = EnvironmentConfig(
    log_level="INFO",
    config_dir="config",
    project_root=None,  # Auto-detected
    custom_directory_structure=None  # Optional
)
env_manager = EnvironmentManager(config)

# Get initialized components
components = env_manager.get_components()
```

## Core Classes

### SemanticAnalyzer

Main interface for text analysis.

```python
class SemanticAnalyzer:
    def __init__(
        self,
        parameter_file: Optional[Union[str, Path]] = None,
        file_utils: Optional[FileUtils] = None,
        config_manager: Optional[ConfigManager] = None,
        **kwargs
    ) -> None:
        """Initialize analyzer with optional parameter file and components."""

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        **kwargs
    ) -> CompleteAnalysisResult:
        """Analyze text with specified analysis types."""

    async def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 3,
        analysis_types: Optional[List[str]] = None,
        **kwargs
    ) -> List[CompleteAnalysisResult]:
        """Process multiple texts with controlled concurrency."""

    def save_results(
        self,
        results: Union[CompleteAnalysisResult, List[CompleteAnalysisResult]],
        output_file: str
    ) -> None:
        """Save analysis results to file."""
```

### KeywordAnalyzer

```python
class KeywordAnalyzer:
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None
    ):
        """Initialize keyword analyzer."""

    async def analyze(
        self,
        text: str,
        **kwargs
    ) -> KeywordAnalysisResult:
        """Extract keywords from text."""
```

### ThemeAnalyzer

```python
class ThemeAnalyzer:
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None
    ):
        """Initialize theme analyzer."""

    async def analyze(
        self,
        text: str,
        **kwargs
    ) -> ThemeAnalysisResult:
        """Identify themes in text."""
```

### CategoryAnalyzer

```python
class CategoryAnalyzer:
    def __init__(
        self,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None
    ):
        """Initialize category analyzer."""

    async def analyze(
        self,
        text: str,
        **kwargs
    ) -> CategoryAnalysisResult:
        """Classify text into categories."""
```

### ConfigManager

Configuration management with FileUtils integration.

```python
class ConfigManager:
    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_dir: str = "config",
        project_root: Optional[Path] = None,
        custom_directory_structure: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration manager."""

    def load_configurations(self) -> None:
        """Load configurations from all sources."""
```

## Data Models

### CompleteAnalysisResult

```python
class CompleteAnalysisResult(BaseModel):
    keywords: Optional[KeywordAnalysisResult]
    themes: Optional[ThemeAnalysisResult]
    categories: Optional[CategoryAnalysisResult]
    language: str
    success: bool
    error: Optional[str]
    processing_time: float
```

### KeywordAnalysisResult

```python
class KeywordAnalysisResult(BaseModel):
    keywords: List[KeywordInfo]
    language: str
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

### ThemeAnalysisResult

```python
class ThemeAnalysisResult(BaseModel):
    themes: List[ThemeInfo]
    language: str
    processing_time: float
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

### CategoryAnalysisResult

```python
class CategoryAnalysisResult(BaseModel):
    categories: List[CategoryMatch]
    language: str
    processing_time: float
```

### CategoryMatch

```python
class CategoryMatch(BaseModel):
    name: str
    confidence: float
    description: Optional[str]
    evidence: List[str]
    themes: Optional[List[str]]
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
    ):
        """Initialize text processor."""

    @abstractmethod
    def get_base_form(self, word: str) -> str:
        """Get base form of a word."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""

    @abstractmethod
    def is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word."""
```

## File Operations

All file operations should use the FileUtils class:

```python
from FileUtils import FileUtils

file_utils = FileUtils()

# Load file
data = file_utils.load_single_file(file_path)

# Save file
saved_files, _ = file_utils.save_data_to_storage(
    data=data,
    output_filetype="xlsx",
    output_type="processed",
    file_name="output"
)
```

For complete examples and usage patterns, see [EXAMPLES.md](EXAMPLES.md).