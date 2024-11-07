# tests/test_semantic_analyzer.py

import pytest
from typing import Dict #, Any
from pathlib import Path

import asyncio
from contextlib import asynccontextmanager

import asyncio
# import concurrent.futures
# from functools import partial

from src.semantic_analyzer.analyzer import SemanticAnalyzer
from src.loaders.models import CategoryConfig
from tests.test_parameter_loading import create_test_excel

@pytest.fixture
def test_categories() -> Dict[str, CategoryConfig]:
    """Fixture for test categories."""
    return {
        "technical": CategoryConfig(
            description="Technical content",
            keywords=["programming", "software", "technology"],
            threshold=0.7
        ),
        "business": CategoryConfig(
            description="Business content",
            keywords=["finance", "marketing", "strategy"],
            threshold=0.6
        )
    }

@pytest.fixture
def test_texts():
    """Sample texts for testing."""
    return {
        "en": "Looking for online programming courses. Any recommendations?",
        "fi": "Etsin verkko-ohjelmointikursseja. Suosituksia?"
    }

@pytest.fixture
def parameter_files(tmp_path):
    """Create parameter files for testing."""
    # Create English parameters
    en_params = create_test_parameters("en", tmp_path)
    
    # Create Finnish parameters
    fi_params = create_test_parameters("fi", tmp_path)
    
    return {"en": en_params, "fi": fi_params}

@pytest.mark.asyncio(scope="function")
async def test_analysis_with_parameters(test_texts, parameter_files):
    """Test complete analysis with parameters."""
    analyzer_en = SemanticAnalyzer(parameter_file=parameter_files["en"])
    try:
        results_en = await asyncio.wait_for(
            analyzer_en.analyze(test_texts["en"]),
            timeout=30
        )

        # Verify structure first
        assert "keywords" in results_en
        assert isinstance(results_en["keywords"], dict)
        assert "keywords" in results_en["keywords"]
        assert isinstance(results_en["keywords"]["keywords"], list)

        # Then check content
        keywords = results_en["keywords"]["keywords"]
        assert "programming" in keywords
        assert any(kw.lower() == "online" for kw in keywords)
        assert "education" not in keywords

    except asyncio.TimeoutError:
        pytest.fail("Analysis timed out")
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
    finally:
        await cleanup_pending_tasks()

@asynccontextmanager
async def cleanup_tasks():
    """Context manager to ensure proper task cleanup."""
    try:
        yield
    finally:
        # Get all tasks tied to the current loop
        pending = asyncio.all_tasks()
        for task in pending:
            task.cancel()
        
        # Wait for cancellation
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

@pytest.mark.asyncio
async def test_analysis_with_parameters(test_texts, parameter_files):
    """Test complete analysis with parameters."""
    analyzer_en = SemanticAnalyzer(parameter_file=parameter_files["en"])
    try:
        results_en = await asyncio.wait_for(
            analyzer_en.analyze(test_texts["en"]),
            timeout=30
        )

        # Verify structure first
        assert "keywords" in results_en
        assert isinstance(results_en["keywords"], dict)
        assert "keywords" in results_en["keywords"]
        assert isinstance(results_en["keywords"]["keywords"], list)

        # Then check content
        keywords = results_en["keywords"]["keywords"]
        assert "programming" in keywords
        assert any(kw.lower() == "online" for kw in keywords)
        assert "education" not in keywords

    except asyncio.TimeoutError:
        pytest.fail("Analysis timed out")
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
    finally:
        await cleanup_pending_tasks()


@pytest.mark.asyncio(scope="function")
async def test_parameter_override(test_texts, parameter_files):
    """Test overriding parameters at runtime."""
    analyzer = SemanticAnalyzer(parameter_file=parameter_files["en"])
    try:
        results = await asyncio.wait_for(
            analyzer.analyze(
                test_texts["en"],
                keyword_params={"max_keywords": 3}
            ),
            timeout=10.0
        )

        # Check results
        assert "keywords" in results
        assert "keywords" in results["keywords"]
        assert len(results["keywords"]["keywords"]) <= 3

    except asyncio.TimeoutError:
        pytest.fail("Analysis timed out")
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
    finally:
        await cleanup_pending_tasks()

@pytest.mark.asyncio(scope="function")
async def test_error_handling_with_parameters(parameter_files, tmp_path):
    """Test error handling with parameters."""
    test_file = tmp_path / "parameters.xlsx"
    create_test_excel(test_file, "en")
    
    analyzer = SemanticAnalyzer(parameter_file=test_file)

    # Test with empty text
    results = await analyzer.analyze("")
    assert "error" in results["keywords"]

    # Test with invalid parameters
    with pytest.raises(ValueError):
        invalid_params = {"max_keywords": -1}  # Invalid value
        analyzer.keyword_analyzer.validate_parameters(invalid_params)  # Add validation method

def create_test_parameters(language: str, output_dir: Path) -> Path:
    """Create test parameter files."""
    # Implementation of test file creation
    # (Similar to the parameter file generator we created earlier)
    pass

async def cleanup_pending_tasks():
    """Clean up any pending tasks."""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

@pytest.mark.asyncio(scope="function")
async def test_batch_analysis_with_parameters(parameter_files):
    """Test batch analysis with parameters."""
    texts = [
        "Looking for programming courses",
        "Interested in marketing training",
        "Need business development advice"
    ]

    analyzer = SemanticAnalyzer(parameter_file=parameter_files["en"])
    try:
        tasks = [analyzer.analyze(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, dict)
            assert "keywords" in result
            assert "keywords" in result["keywords"]
            assert isinstance(result["keywords"]["keywords"], list)
            assert "education" not in result["keywords"]["keywords"]

    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
    finally:
        await cleanup_pending_tasks()