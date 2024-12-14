# scripts/setup_project.py

import shutil
from pathlib import Path
import logging
from typing import Optional
from src.core.config_management import ConfigManager


def setup_project(project_root: Optional[Path] = None):
    """Set up initial project structure with configurations."""
    project_root = Path(project_root) if project_root else Path().resolve()

    # Create initial directories needed for setup
    (project_root / "data" / "config").mkdir(parents=True, exist_ok=True)

    # Copy project-specific FileUtils config
    config_source = project_root / "src" / "config" / "fileutils_config.yaml"
    config_dest = project_root / "data" / "config" / "fileutils_config.yaml"

    if config_source.exists():
        shutil.copy2(config_source, config_dest)
        print(f"Copied FileUtils configuration to {config_dest}")
    else:
        raise FileNotFoundError(
            f"FileUtils config template not found at {config_source}"
        )

    # Initialize ConfigManager with project-specific config
    config_manager = ConfigManager(project_root=project_root)

    # Create directory structure
    for parent_dir, subdirs in config_manager.get_directory_structure().items():
        parent_path = project_root / parent_dir
        parent_path.mkdir(exist_ok=True)

        if isinstance(subdirs, (list, dict)):
            for subdir in subdirs:
                (parent_path / subdir).mkdir(exist_ok=True)

    print("Project setup complete!")


if __name__ == "__main__":
    setup_project()
