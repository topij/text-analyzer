# check_install.py
import site
import sys

print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

print("\nSite packages:")
for p in site.getsitepackages():
    print(f"  {p}")

# Try to find FileUtils
try:
    import FileUtils

    print(f"\nFileUtils location: {FileUtils.__file__}")
except ImportError as e:
    print(f"\nFileUtils import error: {e}")
