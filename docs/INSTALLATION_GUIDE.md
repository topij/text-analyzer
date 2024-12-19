# Installation Guide

Complete installation instructions for the Semantic Text Analyzer.

## Prerequisites

- Python 3.9+
- Conda (recommended) or virtualenv
- Git

## Basic Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/semantic-text-analyzer.git
cd semantic-text-analyzer
```

2. Create conda environment:
```bash
conda env create -f environment.yaml
conda activate semantic-analyzer
```

3. Install in development mode:
```bash
pip install -e .
```

## Language Support Setup

### Finnish Language Support (Voikko)

#### Linux
```bash
# Ubuntu/Debian (if needed)
sudo apt-get update
sudo apt-get install libvoikko-dev voikko-fi

```

#### Windows
1. Download Voikko installer from [Official Site](https://voikko.puimula.org/windows.html)
2. Install, for example, to `C:\Program Files\Voikko` or `C:\scripts\Voikko`
3. Add installation directory to PATH
4. Set environment variable:
```powershell
setx VOIKKO_PATH "C:\scripts\Voikko"
```

#### macOS
```bash
brew install libvoikko
brew install voikko-fi
```

## Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Required for OpenAI
OPENAI_API_KEY=your-key-here

# Required for Azure OpenAI
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Required for Anthropic
ANTHROPIC_API_KEY=your-key-here
```

## Verify Installation

Run verification script:
```python
from semantic_analyzer import verify_environment
verify_environment()
```

## Package Dependencies

Key dependencies from environment.yaml:
```yaml
dependencies:
  - python=3.9
  - pandas>=2.0.0
  - numpy>=1.24.0
  - pyyaml>=6.0
  - jsonschema>=4.17.0
  - python-dotenv>=1.0.0
  - openpyxl>=3.1.0
  - scikit-learn>=1.2.0
  - nltk>=3.8.0
  - langdetect>=1.0.9
  - langchain
  - langchain-openai
  - langchain-anthropic
  - aiohttp>=3.9.0
  - libvoikko
```

The environment.yaml includes FileUtils and its dependencies:
```yaml
dependencies:
  - python=3.9
  - pip
  - pip:
    - "FileUtils[all] @ git+https://github.com/topij/FileUtils.git"
```

## Azure Integration Setup

1. Install Azure dependencies:
```bash
pip install -e ".[azure]"
```

2. Configure Azure credentials:
```bash
az login
az account set --subscription <subscription-id>
```

### Storage Configuration

#### Local Storage (Default)
No additional configuration needed. FileUtils will automatically:
- Create necessary project directories
- Handle file paths consistently across platforms
- Manage data organization

#### Azure Storage Setup
1. Set environment variables:
```bash
# Azure Storage
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
```

2. Or update config.yaml:
```yaml
storage:
  default_type: "azure"
  azure:
    enabled: true
    container_mapping:
      raw: "raw-data"
      processed: "processed-data"
      parameters: "parameters"
```

### Verify FileUtils Installation

Run in Python:
```python
from FileUtils import FileUtils
from semantic_analyzer import verify_environment

# Check FileUtils version
print(FileUtils.__version__)
```

## Environment Variables

Create a `.env` file:
```env
# OpenAI API
OPENAI_API_KEY=your-key-here

# Azure OpenAI (if using)
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Anthropic (if using)
ANTHROPIC_API_KEY=your-key-here

# Finnish support
VOIKKO_PATH=/path/to/voikko  # Linux/macOS
VOIKKO_PATH=C:\scripts\Voikko  # Windows
```

## Troubleshooting

### Common Installation Issues

1. Voikko Installation
   - Linux: Check `libvoikko-dev` installation
   - Verify VOIKKO_PATH environment variable

2. Package Conflicts
   - Use `conda clean --all` before installation
   - Create fresh environment if conflicts persist
   - Check for conflicting packages with `pip check`

3. Azure Integration
   - Verify Azure CLI installation
   - Check credential configuration
   - Confirm subscription access

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Next Steps

- Review [Getting Started](GETTING_STARTED.md) for initial usage
- Configure analyzer using [Configuration Guide](CONFIGURATION_GUIDE.md)
- Check [Examples](EXAMPLES.md) for usage patterns