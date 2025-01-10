# Voikko Installation Guide

## Overview
Voikko is a linguistic tool used for Finnish language processing in the semantic text analyzer. This guide covers installation, configuration, and troubleshooting across different operating systems.

## Installation

### Automated Setup (Recommended)

The easiest way to install Voikko is using the provided setup script:

```bash
# Run the setup script
scripts/setup_dev.sh
```

The script will automatically:
- Install Voikko and its dependencies
- Configure the environment
- Set up required paths

### Manual Installation

#### macOS
1. **Using Homebrew**
   ```bash
   # Install Voikko
   brew install libvoikko
   ```

2. **Verify Installation**
   ```bash
   # Check library location
   ls -l /opt/homebrew/lib/libvoikko*
   ```

3. **Set Environment Variables**
   ```bash
   # Add to your .env file
   VOIKKO_LIBRARY_PATH='/opt/homebrew/lib/libvoikko.dylib'
   VOIKKO_DICT_PATH='/opt/homebrew/lib/voikko'
   ```

### Windows

1. **Download and Install**
   - Download the latest Voikko installer from [Voikko Releases](https://www.puimula.org/voikko/win/)
   - Run the installer and follow the installation wizard
   - Default installation path: `C:\Program Files\Voikko`

2. **Set Environment Variable**
   ```powershell
   # Set VOIKKO_PATH environment variable
   setx VOIKKO_PATH "C:\Program Files\Voikko"
   ```

3. **Verify Installation**
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

### Linux

1. **Using Package Manager**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libvoikko-dev voikko-fi
   
   # Fedora
   sudo dnf install libvoikko-devel voikko-fi
   ```

2. **Verify Installation**
   ```bash
   # Check library
   find /usr -name "libvoikko.so*"
   
   # Check dictionary
   find /usr -name "voikko"
   ```

3. **Test Installation**
   ```python
   from libvoikko import Voikko
   
   try:
       v = Voikko("fi")
       print("Voikko initialized successfully")
   except Exception as e:
       print(f"Voikko error: {str(e)}")
   ```

## Configuration

### Environment Variables

1. **Windows**
   - `VOIKKO_PATH`: Points to Voikko installation directory
   ```powershell
   setx VOIKKO_PATH "C:\Program Files\Voikko"
   ```

2. **Linux**
   - Usually not required as libraries are installed in standard locations
   - If needed:
   ```bash
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   ```

### Parameter File Configuration

Configure Finnish language settings in your parameter file:

```yaml
# parameters_fi.xlsx
parameter            | value
--------------------|-------
language            | fi
include_compounds   | true
min_keyword_length  | 3
```

## Troubleshooting

### Common Issues

1. **Library Not Found**
   ```
   Error: DLL load failed / libvoikko.so.1: cannot open shared object file
   ```
   **Solution:**
   - macOS: Check Homebrew installation and paths
   ```bash
   # Verify installation
   brew list libvoikko
   # Check library location
   ls -l /opt/homebrew/lib/libvoikko*
   ```
   - Windows: Verify `VOIKKO_PATH` environment variable
   - Linux: Check library installation with package manager
   ```bash
   # Ubuntu/Debian
   sudo apt-get install --reinstall libvoikko-dev voikko-fi
   
   # Fedora
   sudo dnf reinstall libvoikko-devel voikko-fi
   ```

2. **Dictionary Not Found**
   ```
   Warning: Could not find Voikko dictionaries
   ```
   **Solution:**
   - macOS: Check dictionary path
   ```bash
   # Verify dictionary location
   ls -l /opt/homebrew/lib/voikko
   # Reinstall if needed
   brew reinstall libvoikko
   ```
   - Windows: Check installation directory
   - Linux: Verify dictionary package installation
   ```bash
   # Ubuntu/Debian
   sudo apt-get install --reinstall voikko-fi
   ```

3. **Initialization Failed**
   ```python
   # Verify installation
   from libvoikko import Voikko
   
   def verify_voikko():
       try:
           v = Voikko("fi")
           print("Initialization: Success")
           
           # Test basic analysis
           word = "koneoppimismalli"
           analysis = v.analyze(word)
           print(f"Analysis: Success")
           print(f"Results for '{word}':", analysis)
           
           return True
       except Exception as e:
           print(f"Error: {str(e)}")
           return False
   
   # Run verification
   verify_voikko()
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Set debug level for Voikko-related components
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("semantic_analyzer.analyzers.finnish").setLevel(logging.DEBUG)
```

## Usage Examples

### Basic Analysis

```python
from libvoikko import Voikko

def analyze_word(word: str):
    v = Voikko("fi")
    analysis = v.analyze(word)
    
    print(f"Analysis for '{word}':")
    for result in analysis:
        print("\nProperties:")
        for key, value in result.items():
            print(f"  {key}: {value}")

# Test with compound word
analyze_word("koneoppimismalli")
```

### Compound Word Analysis

```python
from src.semantic_analyzer import SemanticAnalyzer
from src.core.managers.environment_manager import EnvironmentManager
from src.core.config import EnvironmentConfig

async def analyze_finnish_text():
    # Set up environment
    env_manager = EnvironmentManager(EnvironmentConfig())
    components = env_manager.get_components()
    
    # Initialize analyzer with Finnish parameters
    analyzer = SemanticAnalyzer(
        parameter_file="parameters_fi.xlsx",
        file_utils=components["file_utils"]
    )
    
    # Analyze text with compounds
    text = """Koneoppimismalleja koulutetaan suurilla datajoukolla 
              tunnistamaan kaavoja. Neuroverkon arkkitehtuuri 
              sisältää useita kerroksia."""
    
    result = await analyzer.analyze(
        text=text,
        analysis_types=["keywords", "themes"]
    )
    
    if result.success and result.keywords:
        print("\nCompound Words:")
        for kw in result.keywords.keywords:
            if kw.compound_parts:
                print(f"• {kw.keyword}")
                print(f"  Parts: {' + '.join(kw.compound_parts)}")
                print(f"  Score: {kw.score:.2f}")
```

## Additional Resources

- [Voikko Official Documentation](https://voikko.puimula.org/)