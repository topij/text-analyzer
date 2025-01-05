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
1. Download Voikko installer from official site
2. Install to a known location (e.g., C:\Program Files\Voikko)
3. Set environment variable:
```powershell
setx VOIKKO_PATH "C:\Program Files\Voikko"
```

#### Linux
**Issue**: Library not found
```
ImportError: libvoikko.so.1: cannot open shared object file
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libvoikko-dev voikko-fi

# Fedora
sudo dnf install libvoikko-devel voikko-fi
```

## Runtime Issues

### LLM Connection

**Issue**: Failed to connect to LLM service

**Solution**:
1. Check environment variables:
```python
import os
print("OpenAI API Key set:", bool(os.getenv("OPENAI_API_KEY")))
print("Azure OpenAI Key set:", bool(os.getenv("AZURE_OPENAI_API_KEY")))
print("Azure OpenAI Endpoint:", bool(os.getenv("AZURE_OPENAI_ENDPOINT")))
```

2. Test LLM connection:
```python
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Initialize environment
env_manager = EnvironmentManager(EnvironmentConfig())
components = env_manager.get_components()

# Initialize analyzer
analyzer = SemanticAnalyzer(
    file_utils=components["file_utils"],
    config_manager=components["config_manager"]
)

# Test analysis
result = await analyzer.analyze("Test message")
print("Analysis successful:", result.success)
```

### Memory Issues

**Issue**: Out of memory during batch processing

**Solution**:
1. Reduce batch size:
```python
results = await analyzer.analyze_batch(
    texts=texts,
    batch_size=2  # Default is 3
)
```

2. Process in smaller chunks:
```python
from itertools import islice

def chunk_texts(texts, size=100):
    it = iter(texts)
    return iter(lambda: list(islice(it, size)), [])

for chunk in chunk_texts(all_texts, size=100):
    results = await analyzer.analyze_batch(
        texts=chunk,
        batch_size=2
    )
    # Process results here
```

### Performance Issues

**Issue**: Slow analysis speed

**Solution**:
1. Optimize batch size:
```python
# Balance between memory usage and speed
results = await analyzer.analyze_batch(
    texts=texts,
    batch_size=3,  # Adjust based on your system
    analysis_types=["keywords"]  # Only request needed analysis types
)
```

2. Use focused analysis:
```python
# Only analyze what you need
result = await analyzer.analyze(
    text="Your text",
    analysis_types=["keywords"]  # Skip themes and categories if not needed
)
```

## Language Processing Issues

### Finnish Text Analysis

**Issue**: Poor compound word detection

**Solution**:
1. Verify Voikko installation:
```python
from libvoikko import Voikko
try:
    v = Voikko("fi")
    print("Voikko initialized successfully")
    
    # Test analysis
    word = "koneoppimismalli"
    analysis = v.analyze(word)
    print(f"Analysis for {word}:", analysis)
except Exception as e:
    print(f"Voikko error: {str(e)}")
```

2. Check parameter file settings:
```yaml
# In parameters_fi.xlsx
parameter            | value
--------------------|-------
language           | fi
include_compounds   | true
min_keyword_length  | 3
```

### File Operations Issues

**Issue**: File access errors

**Solution**:
1. Check environment setup:
```python
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Initialize with explicit paths
config = EnvironmentConfig(
    config_dir="config",
    project_root="/path/to/project"
)
env_manager = EnvironmentManager(config)
components = env_manager.get_components()

# Check file utils
fu = components["file_utils"]
print("Config directory:", fu.get_config_dir())
print("Project root:", fu.get_project_root())
```

2. Verify file permissions:
```python
import os
from pathlib import Path

def check_path(path: Path):
    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")
    print(f"Is file: {path.is_file()}")
    print(f"Is dir: {path.is_dir()}")
    print(f"Readable: {os.access(path, os.R_OK)}")
    print(f"Writable: {os.access(path, os.W_OK)}")

# Check parameter file
check_path(Path("parameters.xlsx"))

# Check output directory
check_path(Path("output"))
```

## Common Issues

### Missing Components

**Issue**: Component not initialized error

**Solution**:
```python
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

# Initialize environment
env_manager = EnvironmentManager(EnvironmentConfig())
components = env_manager.get_components()

# Check required components
required = ["file_utils", "config_manager"]
for component in required:
    if component not in components:
        print(f"Missing component: {component}")
    else:
        print(f"Found component: {component}")
```

### Parameter File Issues

**Issue**: Invalid parameter file format

**Solution**:
1. Check sheet names:
```python
import pandas as pd

def verify_parameter_file(file_path: str):
    try:
        # Read Excel file
        xl = pd.ExcelFile(file_path)
        
        # Check sheets
        required = "General Parameters"
        if required not in xl.sheet_names:
            print(f"Missing required sheet: {required}")
        
        # Check required parameters
        params = pd.read_excel(file_path, sheet_name=required)
        required_params = [
            "max_keywords",
            "language",
            "focus_on"
        ]
        
        for param in required_params:
            if param not in params["parameter"].values:
                print(f"Missing required parameter: {param}")
                
    except Exception as e:
        print(f"Error reading parameter file: {str(e)}")

# Verify parameter file
verify_parameter_file("parameters.xlsx")
```

For more detailed configuration information, see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).