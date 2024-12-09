# Troubleshooting Guide

Common issues and solutions when working with the Semantic Text Analyzer.

## Installation Issues

### Conda Environment Creation Failed

**Issue**: Error creating conda environment from environment.yaml

**Solution**:
1. Verify conda is up to date:
```bash
conda update -n base -c defaults conda
```

2. Try creating environment without optional dependencies:
```bash
conda create -n semantic-analyzer python=3.9
conda activate semantic-analyzer
pip install -e .
```

3. Install dependencies manually:
```bash
conda install -c conda-forge pandas numpy pyyaml
```

### Voikko Installation Problems

**Windows**:
```
Error: DLL load failed while importing libvoikko
```

**Solution**:
1. Verify Visual C++ Redistributable is installed
2. Check PATH environment variable includes Voikko directory
3. Try alternative installation path:
```powershell
setx VOIKKO_PATH "C:\Program Files\Voikko"
```

**Linux**:
```
ImportError: libvoikko.so.1: cannot open shared object file
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libvoikko-dev

# Fedora
sudo dnf install libvoikko-devel

# Check library path
sudo ldconfig
```

## Runtime Issues

### LLM Connection Problems

**Issue**: Failed to connect to OpenAI/Azure

**Solution**:
1. Verify API keys:
```python
import os
print("OpenAI API Key set:", bool(os.getenv("OPENAI_API_KEY")))
print("Azure OpenAI Key set:", bool(os.getenv("AZURE_OPENAI_API_KEY")))
```

2. Check endpoints:
```python
from semantic_analyzer import verify_environment
verify_environment()
```

3. Test connection directly:
```python
from src.core.llm.factory import create_llm
llm = create_llm()
response = await llm.apredict("Test message")
```

### Memory Issues

**Issue**: Out of memory with large batch processing

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

3. Monitor memory usage:
```python
from memory_profiler import profile

@profile
def process_texts(texts):
    # Your processing code
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
    "processing": {
        "batch_size": 5,
        "max_workers": 2,
        "timeout": 30.0
    }
}
```

3. Profile code:
```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    results = await analyzer.analyze(text)
    
stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats()
```

## Language Processing Issues

### Finnish Text Analysis Problems

**Issue**: Poor Finnish compound word detection

**Solution**:
1. Verify Voikko installation (linux):
```python
from libvoikko import Voikko
v = Voikko("fi")
analysis = v.analyze("koneoppimismalli")
print(analysis)
```
TODO: installation verification for Windows DLL and dictionaries 

2. Check compound word settings:
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

### English Text Analysis Issues

**Issue**: Incorrect keyword extraction

**Solution**:
1. Adjust confidence thresholds:
```python
config = {
    "analysis": {
        "keywords": {
            "min_confidence": 0.3,
            "weights": {
                "statistical": 0.4,
                "llm": 0.6
            }
        }
    }
}
```

2. Enable POS tagging:
```python
config = {
    "features": {
        "enable_pos_tagging": True
    }
}
```

## FileUtils Issues

### Storage Access Problems

**Issue**: Cannot save/load files

**Solution**:
1. Verify paths:
```python
from FileUtils import FileUtils
fu = FileUtils()
print(fu.get_data_path("raw"))
print(fu.get_data_path("processed"))
```

2. Check permissions:
```python
import os
path = fu.get_data_path("raw")
print("Exists:", os.path.exists(path))
print("Writable:", os.access(path, os.W_OK))
```

### Azure Storage Issues

**Issue**: Cannot access Azure storage

**Solution**:
1. Verify connection string:
```python
from FileUtils import FileUtils
fu = FileUtils.create_azure_utils(
    connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
```

2. Check container access:
```python
containers = fu.list_containers()
print("Available containers:", containers)
```

## Debug Mode

Enable debug mode for more detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Configure specific loggers
logging.getLogger('semantic_analyzer').setLevel(logging.DEBUG)
logging.getLogger('FileUtils').setLevel(logging.DEBUG)
```

## Common Error Messages

### "No module named 'semantic_analyzer'"

**Solution**:
```bash
# Verify installation
pip list | grep semantic-analyzer

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip install -e .
```

### "Language 'fi' not supported"

**Solution**:
```bash
# Verify Voikko installation
python -c "from libvoikko import Voikko; v=Voikko('fi')"

# Check Voikko path
echo $VOIKKO_PATH  # Linux/macOS
echo %VOIKKO_PATH% # Windows
```

## Getting Help

1. Check logs:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('semantic_analyzer')
```

2. Generate environment info:
```python
from semantic_analyzer import get_environment_info
print(get_environment_info())
```

3. Create minimal example:
```python
from semantic_analyzer import SemanticAnalyzer

async def minimal_test():
    analyzer = SemanticAnalyzer()
    return await analyzer.analyze("Test text")
```

<!-- 4. File an issue on GitHub with:
- Environment information
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces -->