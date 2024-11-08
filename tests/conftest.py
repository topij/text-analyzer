# tests/conftest.py

import asyncio
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import Dict  # ,Generator #Any

import pandas as pd
import pytest
from dotenv import load_dotenv

from src.core.config import AnalyzerConfig
from src.models.parameter_models import ParameterSheets
from src.utils.FileUtils.file_utils import FileUtils
from tests.mocks.mock_llm import MockLLM
from tests.test_parameter_loading import create_test_excel


def pytest_configure(config):
    """Configure test environment."""
    # Load environment variables
    for env_file in [".env.test", ".env"]:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"Loaded environment from {env_file}")

    # Set test environment
    os.environ["ENVIRONMENT"] = "test"


@pytest.fixture(scope="session")
def file_utils():
    """Fixture for FileUtils instance."""
    return FileUtils()


@pytest.fixture(scope="session")
def analyzer_config(file_utils):
    """Fixture for AnalyzerConfig instance."""
    return AnalyzerConfig(file_utils=file_utils)


@pytest.fixture(scope="session")
def mock_llm():
    """Fixture for mock LLM."""
    return MockLLM()


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create test data directory."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def parameter_files(test_data_dir) -> Dict[str, Path]:
    """Create parameter test files."""
    files = {}

    # English parameters
    en_file = test_data_dir / "parameters_en.xlsx"
    with pd.ExcelWriter(en_file) as writer:
        # General Parameters
        pd.DataFrame(
            {
                "parameter": [
                    "max_kws",
                    "max_themes",
                    "focus_on",
                    "language",
                    "additional_context",
                    "column_name_to_analyze",
                    "min_keyword_length",
                    "include_compounds",
                ],
                "value": [
                    8,
                    3,
                    "education and career-related topics",
                    "en",
                    "Customer service conversations",
                    "text",
                    3,
                    True,
                ],
            }
        ).to_excel(writer, sheet_name=ParameterSheets.GENERAL.value, index=False)

        # Categories
        pd.DataFrame(
            {
                "category": ["education_type", "career_planning", "course_content"],
                "description": [
                    "Type of education (online, in-person)",
                    "Career and employment topics",
                    "Course content and requirements",
                ],
                "keywords": ["online,remote,classroom", "career,job,employment", "curriculum,materials,prerequisites"],
                "threshold": [0.5, 0.6, 0.4],
            }
        ).to_excel(writer, sheet_name=ParameterSheets.CATEGORIES.value, index=False)

        # Predefined Keywords
        pd.DataFrame(
            {
                "keyword": ["programming", "marketing", "career change"],
                "importance": [1.0, 0.8, 0.9],
                "domain": ["tech", "business", "career"],
                "notes": ["Technical skill", "Business skill", "Career development"],
            }
        ).to_excel(writer, sheet_name=ParameterSheets.KEYWORDS.value, index=False)

        # Excluded Keywords
        pd.DataFrame({"keyword": ["education", "study"], "reason": ["too general", "too vague"]}).to_excel(
            writer, sheet_name=ParameterSheets.EXCLUDED.value, index=False
        )

    files["en"] = en_file

    # Finnish parameters
    fi_file = test_data_dir / "parameters_fi.xlsx"
    with pd.ExcelWriter(fi_file) as writer:
        # Yleiset säännöt (General Parameters)
        pd.DataFrame(
            {
                "parametri": [
                    "max_kws",
                    "max_themes",
                    "focus_on",
                    "language",
                    "additional_context",
                    "column_name_to_analyze",
                    "min_keyword_length",
                    "include_compounds",
                ],
                "arvo": [
                    8,
                    3,
                    "koulutukseen ja työelämään liittyvät aiheet",
                    "fi",
                    "Koulutuspalveluiden asiakaspalvelukeskustelut",
                    "keskustelu",
                    3,
                    True,
                ],
            }
        ).to_excel(writer, sheet_name="yleiset säännöt", index=False)

        # Kategoriat (Categories)
        pd.DataFrame(
            {
                "kategoria": ["koulutusmuoto", "urasuunnittelu", "koulutuksen_sisalto"],
                "kuvaus": [
                    "Koulutuksen toteutustapa (verkko, lähi)",
                    "Uraan ja työllistymiseen liittyvät aiheet",
                    "Koulutuksen sisältöön liittyvät asiat",
                ],
                "avainsanat": ["verkko,lähi,etä", "ura,työ,työllisyys", "sisältö,materiaali,vaatimukset"],
                "kynnysarvo": [0.5, 0.6, 0.4],
            }
        ).to_excel(writer, sheet_name="kategoriat", index=False)

        # Haettavat avainsanat (Predefined Keywords)
        pd.DataFrame(
            {
                "avainsana": ["ohjelmointi", "markkinointi", "alanvaihto"],
                "tärkeys": [1.0, 0.8, 0.9],
                "aihepiiri": ["tekniikka", "liiketoiminta", "ura"],
                "muistiinpanot": ["Tekninen taito", "Liiketoimintataito", "Urakehitys"],
            }
        ).to_excel(writer, sheet_name="haettavat avainsanat", index=False)

        # Älä käytä (Excluded Keywords)
        pd.DataFrame({"avainsana": ["koulutus", "opiskelu"], "syy": ["liian yleinen", "liian epämääräinen"]}).to_excel(
            writer, sheet_name="älä käytä", index=False
        )

    files["fi"] = fi_file

    return files


@pytest.fixture(scope="session")
def test_texts() -> Dict[str, str]:
    """Sample texts for testing."""
    return {
        "en": {
            "technical": """
                Python is a high-level programming language known for its simplicity.
                It supports multiple programming paradigms including procedural and
                object-oriented programming.
            """,
            "business": """
                Looking for online programming courses with flexible scheduling.
                I have experience in marketing but want to transition to IT.
                What would you recommend for career changers?
            """,
            "empty": "",
            "invalid": None,
        },
        "fi": {
            "technical": """
                Python on korkean tason ohjelmointikieli, joka tunnetaan selkeydestään.
                Se tukee useita ohjelmointiparadigmoja kuten proseduraalista ja
                olio-ohjelmointia.
            """,
            "business": """
                Etsin joustavan aikataulun verkko-ohjelmointikursseja.
                Minulla on kokemusta markkinoinnista, mutta haluan siirtyä IT-alalle.
                Mitä suosittelisitte alanvaihtajalle?
            """,
            "empty": "",
            "invalid": None,
        },
    }


@pytest.fixture(scope="session")
def test_categories() -> Dict[str, str]:
    """Fixture for test categories."""
    return {
        "technical": "Technical content about programming and technology",
        "business": "Business and financial content",
        "general": "General or other content",
    }


@pytest.fixture
def cleanup_test_files(test_data_dir):
    """Fixture to clean up test files after tests."""
    yield
    # Clean up test files
    for file in test_data_dir.glob("*"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Error cleaning up {file}: {e}")


def setup_test_environment(file_utils):
    """Set up test environment."""
    # Create stop words files
    config_dir = file_utils.get_data_path("configurations")
    stop_words_dir = config_dir / "stop_words"
    stop_words_dir.mkdir(exist_ok=True)

    # Add stop words for both languages
    stop_words = {"en": ["a", "an", "and", "the"], "fi": ["ja", "on", "ei"]}

    for lang, words in stop_words.items():
        stop_words_file = stop_words_dir / f"{lang}.txt"
        stop_words_file.write_text("\n".join(words), encoding="utf-8")


# Set up test environment at start
@pytest.fixture(scope="session", autouse=True)
def setup_environment(file_utils):
    """Set up test environment automatically."""
    setup_test_environment(file_utils)


@pytest.fixture(scope="session", autouse=True)
def setup_nltk():
    """Set up NLTK data."""
    import nltk

    nltk_resources = [
        "punkt_tab",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",  # Added missing tagger
        "stopwords",
    ]
    for resource in nltk_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            pytest.skip(f"Failed to download NLTK resource {resource}: {e}")


@pytest.fixture(scope="function")
def sample_excel_path(tmp_path) -> Path:
    """Create sample English parameter file."""
    test_file = tmp_path / "parameters.xlsx"
    create_test_excel(test_file, "en")
    return test_file


@pytest.fixture(scope="function")
def sample_finnish_excel_path(tmp_path) -> Path:
    """Create sample Finnish parameter file."""
    test_file = tmp_path / "parametrit.xlsx"
    create_test_excel(test_file, "fi")
    return test_file


# Windows-specific event loop setup
@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure event loop policy for the test session."""
    if platform.system() == "Windows":
        # Use WindowsSelectorEventLoopPolicy on Windows
        policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)


@pytest.fixture
def event_loop(request):
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop

    # Clean up
    if loop.is_running():
        loop.call_soon(loop.stop)
        loop.run_forever()

    # Close all transports
    for task in asyncio.all_tasks(loop):
        if not task.done() and not task.cancelled():
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, Exception):
                pass

    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


# Filter warnings
# Filter all asyncio-related warnings
warnings.filterwarnings("ignore", module="asyncio.*")

# Filter pytest-asyncio deprecation warning
warnings.filterwarnings(
    "ignore", message=".*event_loop fixture provided by pytest-asyncio has been redefined.*", module="pytest_asyncio.*"
)
