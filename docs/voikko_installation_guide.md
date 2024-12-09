# Voikko Installation and Configuration Guide

## NOTE: guide in progress

## Overview
Voikko is a linguistic tool used for Finnish language processing in the semantic text analyzer. This guide covers installation, configuration, and troubleshooting across different environments.

## Installation

### Windows

1. **Manual Installation**
   ```powershell
   # Create installation directory
   mkdir C:\scripts\Voikko

   # Download Voikko files
   # Place the following files in C:\scripts\Voikko:
   # - libvoikko-1.dll
   # - Dictionary files in '5' subdirectory
   ```

2. **Configure Environment**
   ```yaml
   # In config.yaml
   languages:
     fi:
       voikko:
         paths:
           win32:
             lib_path: "C:/scripts/Voikko/libvoikko-1.dll"
             dict_path: "C:/scripts/Voikko"
   ```

### Linux/Azure

1. **Using Package Manager**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libvoikko1 voikko-fi

   # Verify installation
   find /usr -name "libvoikko.so.1"
   find /usr -name "voikko"
   ```

2. **Configure Environment**
   ```yaml
   # In config.yaml
   languages:
     fi:
       voikko:
         paths:
           linux:
             lib_path: "/usr/lib/x86_64-linux-gnu/libvoikko.so.1"
             dict_path: "/usr/lib/voikko"
   ```

### Azure ML Environment

1. **Installation**
   ```bash
   pip install libvoikko
   ```

2. **Environment Setup**
   ```bash
   # Set required environment variables
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   export VOIKKO_DICTIONARY_PATH=/usr/share/voikko
   ```

## Configuration

### Default Search Paths

Voikko searches for dictionaries in the following order:
1. Path given in configuration
2. Path specified by `VOIKKO_DICTIONARY_PATH` environment variable
3. Platform-specific default locations:
   - Linux: `/usr/lib/voikko`, `/usr/share/voikko`
   - Windows: `C:\scripts\Voikko`, `C:\Program Files\Voikko`

### Configuration Options

```yaml
# config.yaml
languages:
  fi:
    voikko:
      paths:
        win32:
          lib_path: "C:/scripts/Voikko/libvoikko-1.dll"
          dict_path: "C:/scripts/Voikko"
        linux:
          lib_path: "/usr/lib/x86_64-linux-gnu/libvoikko.so.1"
          dict_path: "/usr/lib/voikko"
```

## Troubleshooting

### Common Issues

1. **Library Not Found**
   ```
   Error: libvoikko.so.1: cannot open shared object file: No such file or directory
   ```
   **Solution:**
   - Check library path in configuration
   - Verify library installation
   - Set `LD_LIBRARY_PATH` on Linux

2. **Dictionary Not Found**
   ```
   Warning: Could not find Voikko dictionaries
   ```
   **Solution:**
   - Verify dictionary installation path
   - Check directory structure (should contain version subdirectory, for example, '5')
   - Set `VOIKKO_DICTIONARY_PATH` environment variable

3. **Initialization Failure**
   ```
   Error: Voikko initialization failed
   ```
   **Solution:**
   - Check both library and dictionary paths
   - Verify file permissions
   - Enable debug logging for more details

### Debugging

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.getLogger("VoikkoHandler").setLevel(logging.DEBUG)
```

### Environment Verification

Use these commands to verify your installation:

```python
from libvoikko import Voikko

## TODO

```

## Best Practices

1. **Configuration Management**
   - Keep paths in configuration files, not hardcoded
   - Use environment-specific configurations
   - Include fallback paths

2. **Error Handling**
   - Implement graceful fallback for missing functionality
   - Log initialization failures with detailed information
   - Check Voikko availability before usage

3. **Testing**
   - Test initialization in all target environments
   - Verify dictionary loading
   - Include fallback mode tests

## Additional Resources

- [Voikko Documentation](https://voikko.puimula.org/)
- [Libvoikko Python API](https://github.com/voikko/python-libvoikko)
- Project Issue Tracker: Report installation issues here

## Support

For additional help:
1. Check the debug logs with increased verbosity
2. Verify environment variables and paths
3. Contact project maintainers with:
   - Environment details
   - Configuration used
   - Error messages and logs