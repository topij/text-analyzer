import sys

print("\nPython path:")
for p in sys.path:
    print(f"  {p}")


def test_imports():
    try:
        # Try installed package first
        from FileUtils import FileUtils as PackageFileUtils

        print("Package import working!")
    except ImportError as e:
        print(f"Package import failed: {e}")

    try:
        # Try local import
        from src.utils.FileUtils import FileUtils as LocalFileUtils

        print("Local import working!")
    except ImportError as e:
        print(f"Local import failed: {e}")


if __name__ == "__main__":
    test_imports()
