# test_simple.py
import sys
import os

print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying imports...")

try:
    from FileUtils import FileUtils

    print("FileUtils import: OK")

    # Get module location
    import FileUtils as fu_module

    print(f"FileUtils module location: {os.path.dirname(fu_module.__file__)}")
except ImportError as e:
    print(f"FileUtils import failed: {e}")

# Test basic functionality
try:
    utils = FileUtils()
    print("\nFileUtils initialization: OK")
    print(f"Project root: {utils.project_root}")
except Exception as e:
    print(f"\nFileUtils initialization failed: {e}")
