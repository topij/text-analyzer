# Adding FileUtils as a Submodule

## 1. Initial Setup

First, initialize the submodule in your project:

```bash
# Navigate to your project root
cd semantic-text-analyzer

# Add the submodule
git submodule add https://github.com/topij/FileUtils.git src/utils/FileUtils

# Initialize and fetch the submodule content
git submodule init
git submodule update
```

## 2. Update .gitmodules

A `.gitmodules` file will be automatically created in your project root. It should look like this:

```ini
[submodule "src/utils/FileUtils"]
    path = src/utils/FileUtils
    url = https://github.com/topij/FileUtils.git
```

## 3. Create __init__.py

Create an __init__.py file in the utils directory to make it a package:

```bash
# Create the utils directory if it doesn't exist
mkdir -p src/utils

# Create the __init__.py file
touch src/utils/__init__.py
```

Add the following content to `src/utils/__init__.py`:

```python
from .FileUtils.file_utils import FileUtils

__all__ = ["FileUtils"]
```

## 4. Update environment.yaml

Add any FileUtils-specific dependencies to your environment.yaml:

```yaml
dependencies:
  # ... existing dependencies ...
  - pyyaml>=6.0
  - pathlib>=1.0  # If not already included
```

## 5. Using FileUtils in Your Project

You can now import FileUtils in your code:

```python
from FileUtils import FileUtils

# Initialize FileUtils
file_utils = FileUtils()

# Use FileUtils methods
file_utils.save_yaml(...)
file_utils.load_yaml(...)
```

## 6. Updating the Submodule

To update the FileUtils submodule to the latest version:

```bash
# Navigate to the submodule directory
cd src/utils/FileUtils

# Fetch the latest changes
git fetch

# Merge the latest changes
git merge origin/main

# Return to project root
cd ../../..

# Commit the submodule update
git add src/utils/FileUtils
git commit -m "Update FileUtils submodule"
```

## 7. For Other Developers

When cloning your project, other developers should use:

```bash
# Clone the main project with submodules
git clone --recurse-submodules https://github.com/your-username/semantic-analyzer.git

# Or if already cloned, initialize submodules
git submodule init
git submodule update
```

## Common Issues and Solutions

1. If the submodule appears empty:
```bash
git submodule update --init --recursive
```

2. If you need to change the submodule URL:
```bash
git submodule set-url src/utils/FileUtils https://new-url.git
```

3. If you need to remove the submodule:
```bash
# Remove the submodule entry from .git/config
git submodule deinit -f src/utils/FileUtils

# Remove the submodule from .git/modules
rm -rf .git/modules/src/utils/FileUtils

# Remove the submodule directory
git rm -f src/utils/FileUtils
```