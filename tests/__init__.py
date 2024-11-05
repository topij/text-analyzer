# tests/__init__.py

"""Test suite for semantic text analyzer."""

import os
from pathlib import Path

# Set up test constants
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
CONFIG_DIR = TEST_DIR / "config"

# Ensure test directories exist
FIXTURES_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Sample texts for testing
SAMPLE_TEXTS = {
    "technical": """
        Python is a high-level programming language known for its simplicity and readability.
        It supports multiple programming paradigms including procedural, object-oriented, and
        functional programming. The language features a dynamic type system and automatic
        memory management.
    """,
    
    "business": """
        The company's Q3 results exceeded expectations with revenue growth of 15%.
        Customer acquisition costs decreased while retention rates improved significantly.
        The board has approved a new strategic initiative focusing on market expansion.
    """,
    
    "finnish": """
        Ohjelmistokehittäjä työskentelee asiakasprojektissa kehittäen uusia ominaisuuksia
        verkkokauppajärjestelmään. Tekninen toteutus vaatii erityistä huomiota
        tietoturvan ja käyttäjäystävällisyyden osalta.
    """
}