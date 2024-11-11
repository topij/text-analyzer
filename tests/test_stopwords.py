# test_stopwords.py

import logging
import sys
from pathlib import Path

project_root = str(Path().resolve())
print(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.language_processing import create_text_processor

logging.basicConfig(level=logging.INFO)

def test_stopwords():
    """Test NLTK stopwords functionality."""
    print("Testing English Stopwords")
    print("=" * 50)
    
    processor = create_text_processor(language="en")
    
    # Test text with various cases
    test_text = """
    The company's Q3 results exceeded expectations with 15% revenue growth.
    They've been using machine learning and AI for data analysis.
    However, it's important to note that these are preliminary results.
    """
    
    print("\nOriginal text:")
    print(test_text)
    
    # Show tokenization
    tokens = processor.tokenize(test_text)
    print("\nTokens:")
    print(tokens)
    
    # Show which words are stopped
    print("\nWord classification:")
    for token in tokens:
        status = "stopword" if processor.is_stop_word(token) else "keep"
        base = processor.get_base_form(token)
        print(f"{token:15} -> {status:10} (base form: {base})")
    
    # Test common cases
    print("\nTesting common cases:")
    test_words = [
        "the", "and", "is", "that", "with", "using",
        "however", "company's", "they've", "important",
        "analysis", "results", "machine", "learning"
    ]
    
    for word in test_words:
        status = "stopword" if processor.is_stop_word(word) else "keep"
        base = processor.get_base_form(word)
        print(f"{word:15} -> {status:10} (base form: {base})")

if __name__ == "__main__":
    test_stopwords()