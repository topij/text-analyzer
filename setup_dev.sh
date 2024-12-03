#!/bin/bash

# Initialize conda for bash shell if needed
if ! command -v conda &> /dev/null; then
    echo "Initializing conda for bash..."
    eval "$(conda shell.bash hook)"
fi

# Remove existing environment if it exists
conda deactivate 2>/dev/null
conda env remove -n semantic-analyzer 2>/dev/null

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yaml

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate semantic-analyzer

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.template .env
    echo "Please edit .env file with your API keys"
fi

# Create test .env if it doesn't exist
if [ ! -f .env.test ]; then
    echo "Creating .env.test file..."
    cp .env.template .env.test
    echo "Please edit .env.test file with your test API keys"
fi

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
mkdir -p .git/hooks

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# Run tests
echo "Running tests..."
python -m pytest tests/ -v || exit 1

# Run formatters
echo "Running formatters..."
python -m black . || exit 1
python -m isort . || exit 1

# Run linters
echo "Running linters..."
python -m flake8 . || exit 1
python -m mypy . || exit 1
EOF

chmod +x .git/hooks/pre-commit

# Verify installation
echo "Verifying installation..."
pip list

echo "Development environment setup complete!"
echo "Please edit .env and .env.test files with your API keys"
echo ""
echo "To activate the environment, run:"
echo "conda activate semantic-analyzer"