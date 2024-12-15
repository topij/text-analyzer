# src/nb_helpers/environment.py
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.nb_helpers.logging import configure_logging
from FileUtils import FileUtils


class EnvironmentSetup:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        log_level: Optional[str] = None,
    ):
        self.project_root = project_root or Path().resolve().parent
        # Initialize FileUtils with optional log level
        self.file_utils = FileUtils(log_level=log_level)
        self.required_env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

    def verify(self) -> bool:
        # Store current logging level
        root_level = logging.getLogger().level
        handler_level = (
            logging.getLogger().handlers[0].level
            if logging.getLogger().handlers
            else None
        )

        env_loaded = load_dotenv(self.project_root / ".env")
        checks = self._run_checks(env_loaded)
        self._display_results(checks)

        # Restore logging levels
        if root_level:
            logging.getLogger().setLevel(root_level)
        if handler_level and logging.getLogger().handlers:
            logging.getLogger().handlers[0].setLevel(handler_level)

        return all(result for _, result in checks.items())

    def _run_checks(self, env_loaded: bool) -> Dict[str, bool]:
        return {
            **self._check_basic_setup(env_loaded),
            **self._check_env_vars(),
            **self._check_paths(),
        }

    def _check_basic_setup(self, env_loaded: bool) -> Dict[str, bool]:
        return {
            "Project root in path": str(self.project_root) in sys.path,
            "Can import src": "src" in sys.modules,
            "FileUtils initialized": hasattr(self.file_utils, "project_root"),
            ".env file loaded": env_loaded,
        }

    def _check_env_vars(self) -> Dict[str, bool]:
        return {
            f"{var} set": os.getenv(var) is not None
            for var in self.required_env_vars
        }

    def _check_paths(self) -> Dict[str, bool]:
        paths = {
            "Raw data": self.file_utils.get_data_path("raw"),
            "Processed data": self.file_utils.get_data_path("processed"),
            "Configuration": self.file_utils.get_data_path("config"),
            "Main config.yaml": self.project_root / "config.yaml",
        }
        return {f"{name} exists": path.exists() for name, path in paths.items()}

    def _display_results(self, checks: Dict[str, bool]) -> None:
        sections = {
            "Basic Setup": {
                k: v
                for k, v in checks.items()
                if "in path" in k or "initialized" in k or "loaded" in k
            },
            "Environment Variables": {
                k: v for k, v in checks.items() if "set" in k
            },
            "Project Structure": {
                k: v for k, v in checks.items() if "exists" in k
            },
        }

        print("Environment Check Results:")
        print("=" * 50)

        for section, section_checks in sections.items():
            print(f"\n{section}:")
            print("-" * len(section))
            for check, result in section_checks.items():
                status = "✓" if result else "✗"
                print(f"{status} {check}")

        print("\n" + "=" * 50)
        all_passed = all(checks.values())
        print(
            "Environment Status:", "Ready ✓" if all_passed else "Setup needed ✗"
        )


def verify_environment(log_level: Optional[str] = None) -> bool:
    """
    Verify environment setup with optional logging level control.

    Args:
        log_level: Optional logging level to use (e.g., "DEBUG", "INFO")
                  If None, uses FileUtils default
    """
    setup = EnvironmentSetup(log_level=log_level)
    return setup.verify()


def setup_notebook_env(log_level: Optional[str] = None) -> None:
    """Setup notebook environment with optional logging level control."""
    logger = logging.getLogger(__name__)

    setup = EnvironmentSetup(log_level=log_level)
    if not str(setup.project_root) in sys.path:
        sys.path.append(str(setup.project_root))

    # Log current levels before any changes
    logger.debug(
        "Before environment setup - Root logger level: %s",
        logging.getLevelName(logging.getLogger().level),
    )

    # We're not calling configure_logging() anymore
    # Remove or comment out: configure_logging()

    # Log levels after setup
    logger.debug(
        "After environment setup - Root logger level: %s",
        logging.getLevelName(logging.getLogger().level),
    )
