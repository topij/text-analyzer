# debug_imports.py
import sys
from pathlib import Path
import langchain_core

print(f"langchain_core version: {langchain_core.__version__}")
print(f"langchain_core path: {langchain_core.__file__}")

# List contents of language_models directory
models_dir = Path(langchain_core.__file__).parent / "language_models"
print("\nContents of language_models directory:")
for file in models_dir.glob("*.py"):
    print(f"- {file.name}")

# Try different import paths
try:
    from langchain_core.language_models import FakeChatModel

    print("\nSuccessfully imported FakeChatModel")
except ImportError as e:
    print(f"\nFailed to import FakeChatModel: {e}")

try:
    from langchain_core.language_models.fake_chat_models import FakeChatModel

    print("\nSuccessfully imported FakeChatModel from fake_chat_models")
except ImportError as e:
    print(f"\nFailed to import from fake_chat_models: {e}")
