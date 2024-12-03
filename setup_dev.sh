#!/usr/bin/env python3
"""
Setup script for semantic-text-analyzer.
Handles both local (Windows/Linux) and Azure ML environments.
"""

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SetupManager:
    """Manages setup process for semantic-text-analyzer."""
    
    def __init__(self, env_type: str = "local"):
        """Initialize setup manager.
        
        Args:
            env_type: Type of environment ("local" or "azure")
        """
        self.env_type = env_type
        self.project_root = Path(__file__).parent.resolve()
        self.is_windows = platform.system().lower() == "windows"

    def run_setup(self, force: bool = False) -> bool:
        """Run complete setup process.
        
        Args:
            force: Force recreation of environment
            
        Returns:
            bool: Setup success status
        """
        try:
            logger.info(f"Starting setup in {self.env_type} environment")
            
            # Verify system requirements
            self._verify_requirements()
            
            # Set up environment
            if self.env_type == "local":
                success = self._setup_local_environment(force)
            else:
                success = self._setup_azure_environment()
            
            if success:
                self._setup_git_hooks()
                self._create_env_files()
                logger.info("Setup completed successfully")
            
            return success

        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False

    def _verify_requirements(self) -> None:
        """Verify system requirements are met."""
        # Check Python version
        py_version = sys.version_info
        if py_version.major != 3 or py_version.minor < 9:
            raise RuntimeError("Python 3.9 or higher required")
            
        # Check conda installation
        if not self._check_command("conda"):
            raise RuntimeError("Conda not found")
            
        # Check git installation
        if not self._check_command("git"):
            raise RuntimeError("Git not found")

    def _setup_local_environment(self, force: bool) -> bool:
        """Set up local development environment."""
        env_name = "semantic-analyzer"
        
        # Remove existing environment if forced
        if force:
            self._run_command(["conda", "env", "remove", "-n", env_name])
        
        # Create conda environment
        logger.info("Creating conda environment...")
        result = self._run_command([
            "conda", "env", "create",
            "-f", "environment.yaml",
            "-n", env_name
        ])
        
        if not result:
            return False
        
        # Download NLTK data
        logger.info("Downloading NLTK data...")
        self._run_python_code(
            "import nltk; "
            "nltk.download('punkt'); "
            "nltk.download('averaged_perceptron_tagger'); "
            "nltk.download('wordnet')"
        )
        
        # Handle Voikko setup for Windows
        if self.is_windows:
            self._setup_voikko()
        
        return True

    def _setup_azure_environment(self) -> bool:
        """Set up Azure ML environment."""
        try:
            # Import Azure-specific modules
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            # Initialize Azure ML workspace
            credential = DefaultAzureCredential()
            ml_client = MLClient.from_config(credential=credential)
            
            # Create compute cluster if needed
            self._ensure_compute_cluster(ml_client)
            
            return True
            
        except Exception as e:
            logger.error(f"Azure setup failed: {str(e)}")
            return False

    def _setup_voikko(self) -> None:
        """Set up Voikko for Windows environment."""
        voikko_path = Path("C:/scripts/Voikko")
        if not voikko_path.exists():
            logger.info("Setting up Voikko...")
            voikko_path.parent.mkdir(parents=True, exist_ok=True)
            # TODO: Add Voikko download and installation
            # This would require implementing proper Voikko binary management

    def _setup_git_hooks(self) -> None:
        """Set up Git hooks for development."""
        hooks_dir = self.project_root / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pre-commit hook
        hook_path = hooks_dir / "pre-commit"
        with open(hook_path, "w") as f:
            f.write("""#!/bin/bash
echo "Running pre-commit checks..."

# Run tests
python -m pytest tests/ -v || exit 1

# Run formatters
python -m black . || exit 1
python -m isort . || exit 1

# Run linters
python -m flake8 . || exit 1
python -m mypy . || exit 1
""")
        
        # Make hook executable
        hook_path.chmod(0o755)

    def _create_env_files(self) -> None:
        """Create environment files if they don't exist."""
        for env_file in [".env", ".env.test"]:
            env_path = self.project_root / env_file
            if not env_path.exists():
                template_path = self.project_root / ".env.template"
                if template_path.exists():
                    template_path.copy(env_path)
                    logger.info(f"Created {env_file} from template")

    @staticmethod
    def _check_command(cmd: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([cmd, "--version"], capture_output=True)
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _run_command(cmd: List[str]) -> bool:
        """Run a shell command."""
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return False

    @staticmethod
    def _run_python_code(code: str) -> bool:
        """Run Python code in current environment."""
        try:
            subprocess.run([sys.executable, "-c", code], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Python code execution failed: {e}")
            return False

def main():
    """Main setup entry point."""
    parser = argparse.ArgumentParser(
        description="Setup semantic-text-analyzer environment"
    )
    parser.add_argument(
        "--env",
        choices=["local", "azure"],
        default="local",
        help="Environment type"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force environment recreation"
    )
    
    args = parser.parse_args()
    
    setup_manager = SetupManager(args.env)
    success = setup_manager.run_setup(args.force)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()