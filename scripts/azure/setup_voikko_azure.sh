#!/bin/bash
# scripts/azure/setup_voikko.sh

# Azure ML setup script for libvoikko
echo "Setting up libvoikko in Azure ML environment..."

# Function to log messages with timestamps
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Try to install without root (Azure ML environments often don't have sudo)
install_voikko() {
    local install_dir="$HOME/.local"
    mkdir -p "$install_dir"

    # Check conda environment
    if [ -z "$CONDA_PREFIX" ]; then
        log_msg "Error: No conda environment detected"
        return 1
    fi

    # Install libvoikko using conda
    log_msg "Attempting to install libvoikko via conda..."
    conda install -y -c conda-forge libvoikko voikko-fi

    # If conda install fails, try pip
    if [ $? -ne 0 ]; then
        log_msg "Conda install failed, trying pip..."
        pip install --user libvoikko
    fi

    # Set up environment variables if not already set
    grep -q "VOIKKO_DICTIONARY_PATH" ~/.bashrc || echo "export VOIKKO_DICTIONARY_PATH=$CONDA_PREFIX/share/voikko" >> ~/.bashrc
    grep -q "LD_LIBRARY_PATH.*$CONDA_PREFIX/lib" ~/.bashrc || echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

    # Create symbolic links if needed
    if [ -d "$CONDA_PREFIX/lib" ] && [ ! -f "$CONDA_PREFIX/lib/libvoikko.so.1" ]; then
        if [ -f "$CONDA_PREFIX/lib/libvoikko.so" ]; then
            ln -s "$CONDA_PREFIX/lib/libvoikko.so" "$CONDA_PREFIX/lib/libvoikko.so.1"
            log_msg "Created symbolic link for libvoikko.so.1"
        fi
    fi
}

# Main installation
log_msg "Starting libvoikko installation..."

install_voikko
INSTALL_STATUS=$?

# Verify installation
if [ $INSTALL_STATUS -eq 0 ] && { [ -f "$CONDA_PREFIX/lib/libvoikko.so" ] || [ -f "$CONDA_PREFIX/lib/libvoikko.so.1" ]; }; then
    log_msg "Installation successful!"
    
    # Source updated environment
    source ~/.bashrc
    
    # Verify dictionary installation
    if [ -d "$CONDA_PREFIX/share/voikko" ]; then
        log_msg "Voikko dictionaries found at $CONDA_PREFIX/share/voikko"
    else
        log_msg "Warning: Voikko dictionaries not found"
    fi
else
    log_msg "Warning: Installation may not have completed successfully"
fi

log_msg "Setup complete!"