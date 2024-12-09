# Installation Guide

This guide covers installing the Semantic Text Analyzer from GitHub and setting up all required dependencies.

## Prerequisites

- Python 3.9 or higher
- Conda (recommended) or virtualenv
- Git

## Basic Installation

1. Clone the repository:
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

## FileUtils Setup

### Automatic Installation

FileUtils is automatically installed as part of the environment setup:
```bash
# Create conda environment
conda env create -f environment.yaml
```

The environment.yaml includes FileUtils and its dependencies:
```yaml
dependencies:
  - python=3.9
  - pip
  - pip:
    - "FileUtils[all] @ git+https://github.com/topij/FileUtils.git"
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

# Verify analyzer environment
result = verify_environment()
print("FileUtils status:", "OK" if result else "Failed")
```

## Language Support Setup

### Finnish Language Support (Voikko)

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libvoikko-dev voikko-fi

# Fedora
sudo dnf install libvoikko-devel malaga-suomi-voikko
```

#### Windows
1. Download Voikko installer from [Official Site](https://voikko.puimula.org/windows.html)
2. Install to `C:\Program Files\Voikko` or `C:\scripts\Voikko`
3. Add installation directory to PATH
4. Set environment variables:
```powershell
setx VOIKKO_PATH "C:\scripts\Voikko"
```

#### macOS
```bash
brew install libvoikko
brew install voikko-fi
```

## Development Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

## Azure Integration Setup

For Azure support:

1. Install Azure dependencies:
```bash
pip install -e ".[azure]"
```

2. Configure Azure credentials:
```bash
az login
az account set --subscription <subscription-id>
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

## Verify Installation

Run verification script:
```bash
python -m semantic_analyzer.verify_installation
```

Or run in Python:
```python
from semantic_analyzer import verify_environment

result = verify_environment()
print("Installation status:", "OK" if result else "Failed")
```

## Docker Installation (not tested)

```dockerfile
FROM continuumio/miniconda3

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libvoikko-dev voikko-fi && \
    rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/yourusername/semantic-text-analyzer.git .

# Create conda environment
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "semantic-analyzer", "/bin/bash", "-c"]

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["conda", "run", "-n", "semantic-analyzer", "python"]
```

## Troubleshooting

### Common Issues

1. Voikko Installation Issues
   - Windows: Ensure Visual C++ build tools are installed
   - Linux: Check `libvoikko-dev` installation
   - Path issues: Verify VOIKKO_PATH environment variable

2. Package Dependency Conflicts
   - Use `conda create --name semantic-analyzer-clean --file requirements.txt`
   - Check for conflicting packages with `pip check`

3. Azure Integration Issues
   - Verify Azure CLI installation
   - Check credential configuration
   - Confirm subscription access

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Next Steps

- Review [GETTING_STARTED.md](GETTING_STARTED.md) for usage
- Set up [Azure integration](AZURE_GUIDE.md)
- Configure the analyzer using [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)