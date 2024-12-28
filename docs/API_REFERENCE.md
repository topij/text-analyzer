# API Reference

## Environment Management

### EnvironmentManager

The `EnvironmentManager` class provides centralized environment setup and management.

```python
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

# Initialize
config = EnvironmentConfig(log_level="INFO")
env_manager = EnvironmentManager(config)

# Methods
env_manager.verify_environment()  # Check environment setup
env_manager.display_configuration()  # Show current config
env_manager.get_llm_info(analyzer)  # Get LLM details
```

### LoggingManager

The `LoggingManager` class handles logging configuration and management.

```python
from src.nb_helpers.logging_manager import LoggingManager

# Initialize
logging_manager = LoggingManager()

# Methods
logging_manager.configure_logging(level="INFO")
logging_manager.setup_debug_logging(module_name)
logging_manager.verify_logging_setup()
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
        llm: Optional[BaseChatModel] = None,
        categories: Optional[Dict[str, CategoryConfig]] = None,
        **kwargs
    ) -> None

    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[str]] = None,
        timeout: float = 60.0,
        **kwargs
    ) -> CompleteAnalysisResult:
        """Analyze text with specified analysis types."""

    async def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 3,
        timeout: float = 30.0,
        **kwargs
    ) -> List[CompleteAnalysisResult]:
        """Process multiple texts with controlled concurrency."""

    def save_results(
        self,
        results: CompleteAnalysisResult,
        output_file: str,
        output_type: str = "processed"
    ) -> Path:
        """Save analysis results to file."""

    @classmethod
    def from_excel(
        cls,
        content_file: Union[str, Path],
        parameter_file: Union[str, Path],
        **kwargs
    ) -> 'ExcelSemanticAnalyzer':
        """Create Excel-aware analyzer instance."""
```

### KeywordAnalyzer

```python
class KeywordAnalyzer:
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None
    )

    async def analyze(self, text: str) -> KeywordOutput:
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
    )

    async def analyze(self, text: str) -> ThemeOutput:
        """Identify themes in text."""
```

### CategoryAnalyzer

```python
class CategoryAnalyzer:
    def __init__(
        self,
        categories: Dict[str, CategoryConfig],
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
        language_processor: Optional[BaseTextProcessor] = None
    )

    async def analyze(self, text: str) -> CategoryOutput:
        """Classify text into categories."""
```

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

    @abstractmethod
    def get_base_form(self, word: str) -> str:
        """Get base form of a word."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""

    @abstractmethod
    def is_compound_word(self, word: str) -> bool:
        """Check if word is a compound word."""
```

## Configuration

### ConfigManager

```python
class ConfigManager:
    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_dir: str = "config",
        project_root: Optional[Path] = None,
        custom_directory_structure: Optional[Dict[str, Any]] = None
    )

    def get_config(self) -> GlobalConfig:
        """Get complete configuration."""

    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration."""

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
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