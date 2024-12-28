# Troubleshooting Guide

## Installation Issues

### Conda Environment Creation

**Issue**: Error creating conda environment
```
ResolvePackageNotFound: Package not found
```

**Solution**:
```bash
# Update conda
conda update -n base -c defaults conda

# Create environment without optional deps
conda create -n semantic-analyzer python=3.9
conda activate semantic-analyzer
pip install -e .
```

### Voikko Installation

#### Windows
**Issue**: DLL load failed
```
Error: DLL load failed while importing libvoikko
```

**Solution**:
1. Verify Visual C++ Redistributable installation
2. Check PATH includes Voikko directory
3. Set environment variable:
```powershell
setx VOIKKO_PATH "C:\scripts\Voikko"
```

#### Linux
**Issue**: Library not found
```
ImportError: libvoikko.so.1: cannot open shared object file
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libvoikko-dev

# Fedora
sudo dnf install libvoikko-devel
```

## Runtime Issues

### LLM Connection

**Issue**: Failed to connect to LLM service

**Solution**:
```python
# Verify API keys
import os
print("OpenAI API Key set:", bool(os.getenv("OPENAI_API_KEY")))
print("Azure OpenAI Key set:", bool(os.getenv("AZURE_OPENAI_API_KEY")))

# Test connection
from src.core.llm.factory import create_llm
llm = create_llm()
response = await llm.apredict("Test message")
```

### Memory Issues

**Issue**: Out of memory during batch processing

**Solution**:
1. Reduce batch size:
```python
results = await analyzer.analyze_batch(
    texts=texts,
    batch_size=2  # Reduce from default
)
```

2. Enable garbage collection:
```python
import gc
gc.collect()
```

### Performance Issues

**Issue**: Slow analysis speed

**Solution**:
1. Enable caching:
```python
analyzer = SemanticAnalyzer(
    config={"features": {"use_caching": True}}
)
```

2. Optimize batch processing:
```python
config = {
    "analysis": {
        "batch_size": 5,
        "max_workers": 2,
        "timeout": 30.0
    }
}
```

## Language Processing Issues

### Finnish Text Analysis

**Issue**: Poor compound word detection

**Solution**:
1. Verify Voikko:
```python
from libvoikko import Voikko
v = Voikko("fi")
analysis = v.analyze("koneoppimismalli")
print(analysis)
```

2. Check compound settings:
```python
config = {
    "languages": {
        "fi": {
            "preserve_compounds": True,
            "min_compound_length": 3
        }
    }
}
```

### File Operations Issues

**Issue**: FileUtils errors

**Solution**:
```python
from FileUtils import FileUtils
fu = FileUtils()

# Check paths
print("Raw path:", fu.get_data_path("raw"))
print("Processed path:", fu.get_data_path("processed"))

# Verify permissions
import os
path = fu.get_data_path("raw")
print("Exists:", os.path.exists(path))
print("Writable:", os.access(path, os.W_OK))
```

## Environment Setup Issues

### Environment Verification Failed

Use the environment manager to diagnose the issue:

```python
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

# Initialize with debug logging
config = EnvironmentConfig(log_level="DEBUG")
env_manager = EnvironmentManager(config)

# Check environment status
status = env_manager.verify_environment()
print("Environment Status:", status)

# Display current configuration
env_manager.display_configuration()
```

Common causes:
1. Missing or invalid `.env` file
2. Incorrect project root detection
3. Missing required directories
4. Invalid configuration values

### Logging Issues

Use the logging manager to verify setup:

```python
from src.nb_helpers.logging_manager import LoggingManager

# Initialize logging manager
logging_manager = LoggingManager()

# Verify logging setup
logging_manager.verify_logging_setup(show_hierarchy=True)

# Enable debug logging for troubleshooting
logging_manager.setup_debug_logging("src.analyzers")
```

Common issues:
1. Incorrect log level configuration
2. Missing log handlers
3. Permission issues with log files
4. Conflicting logger configurations

## Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Configure specific loggers
logging.getLogger('semantic_analyzer').setLevel(logging.DEBUG)
logging.getLogger('FileUtils').setLevel(logging.DEBUG)
```

## Common Error Messages

### "No module named 'semantic_analyzer'"
```bash
# Verify installation
pip list | grep semantic-analyzer

# Reinstall package
pip install -e .
```

### "Language 'fi' not supported"
```bash
# Verify Voikko
python -c "from libvoikko import Voikko; v=Voikko('fi')"

# Check path
echo $VOIKKO_PATH  # Linux/macOS
echo %VOIKKO_PATH% # Windows
```

## Getting Help

1. Generate environment info:
```python
from semantic_analyzer import verify_environment
print(verify_environment())
```

2. Check logs:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('semantic_analyzer')
```

For configuration issues, see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).