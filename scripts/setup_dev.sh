#!/bin/bash

# Setup script for semantic text analyzer development environment

# Function to log messages
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check environment variables
check_environment() {
    log_msg "Checking environment configuration..."
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_msg "Error: .env file not found"
        log_msg "Please create a .env file with required API keys. Example:"
        echo "
# Required: Choose one of these options
# Option 1: OpenAI
OPENAI_API_KEY='your-key-here'

# Option 2: Azure OpenAI
AZURE_OPENAI_API_KEY='your-key-here'
AZURE_OPENAI_ENDPOINT='your-endpoint'

# Optional: For Finnish language support
VOIKKO_LIBRARY_PATH='/opt/homebrew/lib/libvoikko.dylib'  # macOS default path
VOIKKO_DICT_PATH='/opt/homebrew/lib/voikko'  # macOS dictionary path"
        return 1
    fi
    
    # Source the .env file
    set -a
    source .env
    set +a
    
    # Check for required API keys
    if [ -z "$OPENAI_API_KEY" ] && ( [ -z "$AZURE_OPENAI_API_KEY" ] || [ -z "$AZURE_OPENAI_ENDPOINT" ] ); then
        log_msg "Error: Missing required API configuration"
        log_msg "Please ensure either OPENAI_API_KEY or both AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set in .env"
        return 1
    fi
    
    # Validate OpenAI API key format if provided
    if [ -n "$OPENAI_API_KEY" ]; then
        # Remove any quotes from the key for validation
        CLEAN_KEY=$(echo "$OPENAI_API_KEY" | tr -d "'\"")
        
        # Debug: Print key format without revealing the full key
        PREFIX=${CLEAN_KEY:0:8}  # Get "sk-proj-" part
        KEY_LENGTH=${#CLEAN_KEY}
        #log_msg "Debug: Key starts with: $PREFIX, total length: $KEY_LENGTH chars"
        
        # Check if key starts with sk-proj-
        if [[ "$CLEAN_KEY" =~ ^sk-proj- ]]; then
            # Key format is correct, check length
            if [ ${#CLEAN_KEY} -lt 40 ]; then
                log_msg "Warning: OPENAI_API_KEY seems too short (should be at least 40 characters in total)"
                read -p "Continue anyway? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    return 1
                fi
            fi
        else
            log_msg "Warning: OPENAI_API_KEY format looks incorrect (should start with 'sk-proj-' for project-specific keys)"
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                return 1
            fi
        fi
    fi
    
    # Validate Azure OpenAI endpoint format if provided
    if [ -n "$AZURE_OPENAI_ENDPOINT" ] && [[ ! "$AZURE_OPENAI_ENDPOINT" =~ ^https:// ]]; then
        log_msg "Warning: AZURE_OPENAI_ENDPOINT should start with 'https://'"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    log_msg "Environment configuration looks good"
    return 0
}

# Function to setup Voikko for Mac
setup_voikko_mac() {
    log_msg "Setting up Voikko for Mac..."
    
    # Install libvoikko using Homebrew if not already installed
    if ! command_exists voikkospell; then
        if command_exists brew; then
            brew install libvoikko
        else
            log_msg "Error: Homebrew is required for Mac installation. Please install it first."
            return 1
        fi
    fi
    
    # Create Spelling directory if it doesn't exist
    mkdir -p ~/Library/Spelling/voikko
    
    # Download and install Finnish dictionary
    DICT_URL="https://www.puimula.org/htp/testing/voikko-snapshot/dict-morpho.zip"
    TEMP_DIR=$(mktemp -d)
    
    log_msg "Downloading Voikko dictionary..."
    if curl -L -o "$TEMP_DIR/dict-morpho.zip" "$DICT_URL"; then
        cd "$TEMP_DIR" || exit
        unzip dict-morpho.zip
        cp -r 2/mor-morpho/* ~/Library/Spelling/voikko/
        cd - || exit
        rm -rf "$TEMP_DIR"
        log_msg "Voikko dictionary installed successfully"
    else
        log_msg "Error: Failed to download Voikko dictionary"
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    return 0
}

# Function to setup Voikko for Linux
setup_voikko_linux() {
    log_msg "Setting up Voikko for Linux..."
    
    # Install libvoikko and Finnish dictionary
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y libvoikko-dev voikko-fi python3-libvoikko
    elif command_exists dnf; then
        sudo dnf install -y libvoikko-devel voikko-fi python3-libvoikko
    else
        log_msg "Error: Unsupported Linux distribution"
        return 1
    fi
    
    return 0
}

# Function to setup Voikko for Windows
setup_voikko_windows() {
    log_msg "Setting up Voikko for Windows..."
    
    # Download and install Voikko
    VOIKKO_WIN_URL="https://www.puimula.org/voikko/win/libvoikko-4.3.2.msi"
    TEMP_DIR=$(mktemp -d)
    
    log_msg "Downloading Voikko installer..."
    if curl -L -o "$TEMP_DIR/libvoikko.msi" "$VOIKKO_WIN_URL"; then
        msiexec /i "$TEMP_DIR/libvoikko.msi" /quiet
        rm -rf "$TEMP_DIR"
        log_msg "Voikko installed successfully"
    else
        log_msg "Error: Failed to download Voikko installer"
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    return 0
}

# Function to setup NLTK data
setup_nltk() {
    log_msg "Setting up NLTK data..."
    
    # Install NLTK package first
    log_msg "Installing NLTK package..."
    if ! pip install -U nltk; then
        log_msg "Error: Failed to install NLTK package"
        return 1
    fi
    
    # Create a Python script for NLTK setup
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << 'EOF'
import nltk
import ssl
import os
import zipfile
from pathlib import Path

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path to conda environment
nltk_data_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path
nltk.data.path.insert(0, nltk_data_path)

# Download essential NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)

# Get NLTK data path
nltk_data_path = nltk.data.path[0]

# Ensure all downloaded zip files are extracted
corpora_path = Path(nltk_data_path) / 'corpora'
if corpora_path.exists():
    for zip_file in corpora_path.glob('*.zip'):
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(corpora_path)
            print(f"Extracted {zip_file.name}")
        except Exception as e:
            print(f"Error extracting {zip_file.name}: {e}")

# Verify wordnet installation
try:
    from nltk.corpus import wordnet
    synsets = wordnet.synsets('test')
    print("Wordnet verification successful")
except Exception as e:
    print(f"Error verifying wordnet: {e}")
    exit(1)
EOF
    
    # Get conda environment Python path
    CONDA_PYTHON="$CONDA_PREFIX/bin/python"
    
    # Run the NLTK setup script with conda environment's Python
    if [ -x "$CONDA_PYTHON" ]; then
        "$CONDA_PYTHON" "$TEMP_SCRIPT"
        local result=$?
        rm -f "$TEMP_SCRIPT"
        
        if [ $result -eq 0 ]; then
            log_msg "NLTK setup completed successfully"
            return 0
        else
            log_msg "Error: NLTK setup failed"
            return 1
        fi
    else
        log_msg "Error: Could not find Python in conda environment"
        rm -f "$TEMP_SCRIPT"
        return 1
    fi
}

# Main setup function
main() {
    log_msg "Starting setup..."
    
    # Check environment configuration first
    if ! check_environment; then
        log_msg "Environment check failed. Please fix the issues and try again."
        exit 1
    fi
    
    # Check Python version
    if ! command_exists python3; then
        log_msg "Error: Python 3 is required"
        exit 1
    fi
    
    # Check conda
    if ! command_exists conda; then
        log_msg "Error: conda is required"
        exit 1
    fi
    
    # Create conda environment
    log_msg "Creating conda environment..."
    if ! (conda env create -f environment.yaml || conda env update -f environment.yaml); then
        log_msg "Error: Failed to create/update conda environment"
        exit 1
    fi
    
    # Activate conda environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda activate semantic-analyzer; then
        log_msg "Error: Failed to activate conda environment"
        exit 1
    fi
    
    # Verify conda environment is active
    if [[ "$CONDA_DEFAULT_ENV" != "semantic-analyzer" ]]; then
        log_msg "Error: Failed to activate semantic-analyzer environment"
        exit 1
    fi
    
    # Setup Voikko based on platform
    local setup_result=0
    case "$(uname -s)" in
        Darwin*)
            setup_voikko_mac || setup_result=$?
            ;;
        Linux*)
            setup_voikko_linux || setup_result=$?
            ;;
        MINGW*|MSYS*|CYGWIN*)
            setup_voikko_windows || setup_result=$?
            ;;
        *)
            log_msg "Error: Unsupported operating system"
            exit 1
            ;;
    esac
    
    if [ $setup_result -ne 0 ]; then
        log_msg "Error: Voikko setup failed"
        exit 1
    fi
    
    # Setup NLTK data
    if ! setup_nltk; then
        log_msg "Error: NLTK setup failed"
        exit 1
    fi
    
    # Initialize project structure and generate test data
    log_msg "Initializing project structure..."
    if ! "$CONDA_PREFIX/bin/python" "scripts/setup_project.py"; then
        log_msg "Error: Project initialization failed"
        exit 1
    fi
    
    # Initialize UI components
    log_msg "Setting up UI components..."
    if ! "$CONDA_PREFIX/bin/python" "ui/setup_ui.py"; then
        log_msg "Error: UI setup failed"
        exit 1
    fi
    
    log_msg "Setup completed successfully!"
    echo
    echo "╭──────────────────────────────────────────────────╮"
    echo "│ Remember to activate the environment before use: │"
    echo "│ $ conda activate semantic-analyzer               │"
    echo "│                                                  │"
    echo "│ Check README.md for getting started guide!       │"
    echo "╰──────────────────────────────────────────────────╯"
    echo
    return 0
}

# Run main function
main