@echo off
setlocal

:: Create conda environment
echo Creating conda environment...
call conda env create -f environment.yaml

:: Activate environment
echo Activating environment...
call conda activate semantic-analyzer

:: Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

:: Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    copy .env.template .env
    echo Please edit .env file with your API keys
)

:: Create test .env if it doesn't exist
if not exist .env.test (
    echo Creating .env.test file...
    copy .env.template .env.test
    echo Please edit .env.test file with your test API keys
)

:: Set up pre-commit hooks
echo Setting up pre-commit hooks...
if not exist .git\hooks mkdir .git\hooks

(
echo @echo off
echo :: Run tests
echo pytest tests/ -v || exit /b 1
echo.
echo :: Run formatters
echo black . || exit /b 1
echo isort . || exit /b 1
echo.
echo :: Run linters
echo flake8 . || exit /b 1
echo mypy . || exit /b 1
) > .git\hooks\pre-commit.bat

echo Development environment setup complete!
echo Please edit .env and .env.test files with your API keys

endlocal