# scripts/migrate_config.py

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import shutil
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigMigrationTool:
    """Tool for migrating old configuration patterns to new ones."""

    # Skip directories and files
    SKIP_PATTERNS = {
        "_archive",
        "old_",
        "config_old",
        "config_no_azure",
    }

    # More precise patterns with context
    PATTERNS = {
        # Import patterns
        r"^from src\.core\.config import AnalyzerConfig(?!\s*as\s*ConfigManager)": "from src.core.config_management import ConfigManager",
        # Instance creation with variable capture
        r"(\w+)\s*=\s*AnalyzerConfig\((.*?)\)": r"\1 = ConfigManager(\2)",
        # Config access patterns with variable capture
        r'(\w+)\.config\["models"\](?!\s*=)': r"\1.get_model_config()",
        r'(\w+)\.config\["analysis"\](?!\s*=)': r"\1.get_analyzer_config()",
        r'(\w+)\.config\.get\("models",\s*({[^}]*}|\[.*?\]|[^,)]+)\)': r"\1.get_model_config().get(\2)",
        # Logging configurations
        r"logging\.basicConfig\(\s*level\s*=([^,)]+)(?:,\s*format\s*=([^,)]+))?\)": "# Logging is now handled by ConfigManager\n# Old code: logging.basicConfig(level=\1\2)",
    }

    def __init__(self, project_root: Path, dry_run: bool = True):
        """Initialize the migration tool.

        Args:
            project_root: Root directory of the project
            dry_run: If True, only show what would be changed
        """
        self.project_root = project_root
        self.dry_run = dry_run
        self.modified_files: Set[Path] = set()
        self.backup_dir = (
            project_root
            / "backups"
            / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        # Skip the migration script itself
        if file_path.name == "migrate_config.py":
            return False

        # Skip backup directory
        if str(file_path).startswith(str(self.backup_dir)):
            return False

        # Skip archive and old files
        path_str = str(file_path)
        if any(pattern in path_str for pattern in self.SKIP_PATTERNS):
            return False

        # Skip common directories to ignore
        ignore_dirs = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            "backups",
        }
        if any(part in ignore_dirs for part in file_path.parts):
            return False

        # Only process Python files
        return file_path.suffix == ".py"

    def create_backup(self, file_path: Path) -> None:
        """Create backup of file before modification."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)

        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    def find_python_files(self) -> List[Path]:
        """Find all Python files in project."""
        return [
            path
            for path in self.project_root.rglob("*.py")
            if self.should_process_file(path)
        ]

    def process_file(self, file_path: Path) -> Tuple[str, bool]:
        """Process a single file with improved pattern matching."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        modified = False

        # Only process files that actually use AnalyzerConfig
        if (
            "AnalyzerConfig" not in content
            and "logging.basicConfig" not in content
        ):
            return content, False

        # Apply patterns with context awareness
        for pattern, replacement in self.PATTERNS.items():
            if re.search(pattern, content, re.MULTILINE):
                new_content = re.sub(
                    pattern, replacement, content, flags=re.MULTILINE
                )
                if new_content != content:
                    content = new_content
                    modified = True

        # Add imports only if we made changes and they're needed
        if modified and "ConfigManager" not in content:
            # Find import section
            import_section_end = 0
            for match in re.finditer(
                r"^(?:import|from)\s+.*$", content, re.MULTILINE
            ):
                import_section_end = max(import_section_end, match.end())

            if import_section_end:
                # Add after last import
                content = (
                    content[:import_section_end]
                    + "\nfrom src.core.config_management import ConfigManager"
                    + content[import_section_end:]
                )
            else:
                # Add at beginning
                content = (
                    "from src.core.config_management import ConfigManager\n\n"
                    + content
                )

        # Additional safety check - don't modify if changes look too aggressive
        if modified:
            changes = sum(
                1
                for a, b in zip(
                    original_content.splitlines(), content.splitlines()
                )
                if a != b
            )
            if changes > 10:  # Threshold for suspicious number of changes
                logger.warning(
                    f"Too many changes ({changes}) in {file_path}, skipping"
                )
                return original_content, False

        return content, modified

    def migrate(self) -> None:
        """Perform the migration."""
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to process")

        for file_path in python_files:
            try:
                new_content, modified = self.process_file(file_path)

                if modified:
                    self.modified_files.add(file_path)
                    if not self.dry_run:
                        # Create backup
                        self.create_backup(file_path)
                        # Write changes
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        logger.info(f"Updated {file_path}")
                    else:
                        logger.info(f"Would update {file_path} (dry run)")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Summary
        logger.info("\nMigration Summary:")
        logger.info(
            f"{'Would modify' if self.dry_run else 'Modified'} {len(self.modified_files)} files:"
        )
        for file_path in sorted(self.modified_files):
            logger.info(f"  - {file_path.relative_to(self.project_root)}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate configuration patterns to new format"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    args = parser.parse_args()

    migrator = ConfigMigrationTool(args.project_root, args.dry_run)
    migrator.migrate()


if __name__ == "__main__":
    main()
