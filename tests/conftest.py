# tests/conftest.py
# tests/conftest.py

import os
from pathlib import Path

import nltk
import pytest
from dotenv import load_dotenv

from src.core.config import AnalyzerConfig
from src.utils.FileUtils.file_utils import FileUtils

from .config import get_test_config
from .mocks.mock_llm import MockLLM


def setup_nltk():
    """Download required NLTK data."""
    try:
        # Use FileUtils to get data directory for NLTK
        utils = FileUtils()
        nltk_data_dir = utils.get_data_path("configurations") / "nltk_data"
        nltk_data_dir.mkdir(exist_ok=True)

        # Set NLTK data path
        nltk.data.path.append(str(nltk_data_dir))

        # Required NLTK resources
        resources = [
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/wordnet", "wordnet"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/stopwords", "stopwords"),
        ]

        # Download resources if not already present
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
                print(f"Found NLTK resource: {resource_name}")
            except LookupError:
                print(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, download_dir=str(nltk_data_dir), quiet=True)

    except Exception as e:
        print(f"Warning: Error setting up NLTK: {e}")
        print("Some text processing features might be limited.")


def setup_test_files(file_utils: FileUtils):
    """Set up test files."""
    # Create config directories
    config_dir = file_utils.get_data_path("configurations")
    stop_words_dir = config_dir / "stop_words"
    stop_words_dir.mkdir(exist_ok=True)

    # Create stop words files
    stop_words = {
        "en": """
a
an
and
are
as
at
be
by
for
from
has
he
in
is
it
its
of
on
that
the
to
was
were
will
with""",
        "fi": """
ja
on
ei
se
että
olla
joka
sen
hän
ne
sitä
tämä
kun
oli
myös
jos
sekä
niin
vain
mutta""",
    }

    for lang, content in stop_words.items():
        stop_words_file = stop_words_dir / f"{lang}.txt"
        stop_words_file.write_text(content.strip(), encoding="utf-8")

    # Create semantic analyzer config
    config = get_test_config()
    file_utils.config.update(config)

    # Save configuration
    config_file = config_dir / "config.yaml"
    file_utils.save_yaml(file_utils.config, config_file, include_timestamp=False)


def pytest_configure(config):
    """Configure test environment."""
    # Load environment variables
    for env_file in [".env.test", ".env"]:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"Loaded environment from {env_file}")

    # Set test environment
    os.environ["ENVIRONMENT"] = "test"

    # Setup NLTK
    setup_nltk()

    # Setup test files
    file_utils = FileUtils()
    setup_test_files(file_utils)


@pytest.fixture(scope="session")
def nltk_data():
    """Fixture to ensure NLTK data is available."""
    setup_nltk()


@pytest.fixture(scope="session")
def file_utils():
    """Fixture for FileUtils instance."""
    utils = FileUtils()
    setup_test_files(utils)  # Ensure files are set up
    return utils


@pytest.fixture(scope="session")
def analyzer_config(file_utils, nltk_data):  # Add nltk_data dependency
    """Fixture for AnalyzerConfig instance."""
    return AnalyzerConfig(file_utils=file_utils)


@pytest.fixture(scope="session")
def mock_llm():
    """Fixture for mock LLM."""
    return MockLLM()


@pytest.fixture(scope="session")
def test_categories():
    """Fixture for test categories."""
    return {
        "technical": "Technical content about programming and technology",
        "business": "Business and financial content",
        "general": "General or other content",
    }


@pytest.fixture(scope="session")
def voikko_path():
    """Fixture for Voikko path."""
    return os.getenv("VOIKKO_PATH")


@pytest.fixture(scope="function")
def test_data_dir(file_utils):
    """Fixture for test data directory."""
    data_dir = file_utils.get_data_path("raw") / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="function")
def cleanup_test_files(test_data_dir):
    """Fixture to clean up test files after tests."""
    yield
    # Clean up test files
    for file in test_data_dir.glob("*"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Error cleaning up {file}: {e}")


@pytest.fixture(scope="session")
def sample_texts():
    """Fixture for test texts."""
    return {
        "technical": """
            Python is a high-level programming language known for its simplicity and readability.
            It supports multiple programming paradigms including procedural, object-oriented, and
            functional programming.
        """,
        "business": """
            The company's Q3 results exceeded expectations with revenue growth of 15%.
            Customer acquisition costs decreased while retention rates improved significantly.
        """,
        "finnish": """
            Ohjelmistokehittäjä työskentelee asiakasprojektissa kehittäen uusia ominaisuuksia.
            Tekninen toteutus vaatii erityistä huomiota tietoturvan osalta.
        """,
    }


@pytest.fixture(scope="session")
def mock_categories():
    """Fixture for test categories."""
    return {
        "technical": "Content about technology, programming, and software development",
        "business": "Content about business operations, finance, and management",
        "general": "General content that doesn't fit other categories",
    }


@pytest.fixture(scope="session")
def expected_keywords():
    """Fixture for expected keywords."""
    return {
        "technical": ["python", "programming", "language"],
        "business": ["revenue", "growth", "acquisition"],
        "finnish": ["ohjelmistokehittäjä", "tekninen", "tietoturva"],
    }


def setup_test_environment(file_utils):
    """Set up test environment."""
    # Create stop words files
    config_dir = file_utils.get_data_path("configurations")
    stop_words_dir = config_dir / "stop_words"
    stop_words_dir.mkdir(exist_ok=True)

    for lang, words in {
        "en": ["a", "an", "and", "the"],
        "fi": ["ja", "on", "ei"],
    }.items():
        stop_words_file = stop_words_dir / f"{lang}.txt"
        stop_words_file.write_text("\n".join(words), encoding="utf-8")
