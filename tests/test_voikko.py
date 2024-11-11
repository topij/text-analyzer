# test_voikko.py

import logging
import sys
from pathlib import Path

project_root = str(Path().resolve())
print(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.language_processing import create_text_processor

logging.basicConfig(level=logging.INFO)

def test_voikko_installation():
    """Test Voikko initialization and basic functionality."""
    print("Testing Voikko Installation")
    print("=" * 50)
    
    processor = create_text_processor(language="fi")
    
    # Test text
    test_text = "Koneoppiminen on tekoälyn osa-alue."
    
    print("\nTest Capabilities:")
    
    # Test tokenization
    print("\n1. Tokenization:")
    tokens = processor.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    # Test base form
    print("\n2. Base form extraction:")
    for word in tokens:
        base = processor.get_base_form(word)
        print(f"{word:15} -> {base}")
    
    # Test stopword filtering
    print("\n3. Stopword detection:")
    test_words = ["on", "ja", "koneoppiminen", "tekoäly"]
    for word in test_words:
        print(f"{word:15} -> {'stopword' if processor.is_stop_word(word) else 'keep'}")

if __name__ == "__main__":
    test_voikko_installation()