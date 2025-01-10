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
    
    # Create a Python script for NLTK setup
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << 'EOF'
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download essential NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
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
    
    log_msg "Setup completed successfully!"
    return 0
}

# Run main function
main