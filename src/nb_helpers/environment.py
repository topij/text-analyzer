# src/nb_helpers/environment.py
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from src.utils.FileUtils.file_utils import FileUtils
from src.nb_helpers.logging import configure_logging
import os
import sys

class EnvironmentSetup:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path().resolve().parent
        self.file_utils = FileUtils()
        self.required_env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        
    def verify(self) -> bool:
        env_loaded = load_dotenv(self.project_root / ".env")
        checks = self._run_checks(env_loaded)
        self._display_results(checks)
        return all(result for _, result in checks.items())

    def _run_checks(self, env_loaded: bool) -> Dict[str, bool]:
        return {
            **self._check_basic_setup(env_loaded),
            **self._check_env_vars(),
            **self._check_paths()
        }
    
    def _check_basic_setup(self, env_loaded: bool) -> Dict[str, bool]:
        return {
            "Project root in path": str(self.project_root) in sys.path,
            "Can import src": "src" in sys.modules,
            "FileUtils initialized": hasattr(self.file_utils, "project_root"),
            ".env file loaded": env_loaded
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
            "Configuration": self.file_utils.get_data_path("configurations"),
            "Main config.yaml": self.project_root / "config.yaml"
        }
        return {f"{name} exists": path.exists() for name, path in paths.items()}
    
    def _display_results(self, checks: Dict[str, bool]) -> None:
        sections = {
            "Basic Setup": {k: v for k, v in checks.items() if "in path" in k or "initialized" in k or "loaded" in k},
            "Environment Variables": {k: v for k, v in checks.items() if "set" in k},
            "Project Structure": {k: v for k, v in checks.items() if "exists" in k}
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
        print("Environment Status:", "Ready ✓" if all_passed else "Setup needed ✗")

def verify_environment() -> bool:
    return EnvironmentSetup().verify()

def setup_notebook_env() -> None:
    setup = EnvironmentSetup()
    if not str(setup.project_root) in sys.path:
        sys.path.append(str(setup.project_root))
    # configure_logging()  # from logging module