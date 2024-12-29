"""Base configuration functionality."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from FileUtils import FileUtils
from src.config.models import GlobalConfig

logger = logging.getLogger(__name__)


class BaseConfigManager:
    """Base configuration manager with FileUtils integration."""

    DEFAULT_DIRECTORY_STRUCTURE = {
        "data": ["raw", "interim", "processed", "config", "parameters", "logs"],
        "notebooks": [],
        "docs": [],
        "scripts": [],
        "src": [],
        "reports": [],
        "models": [],
    }

    def __init__(
        self,
        file_utils: Optional[FileUtils] = None,
        config_dir: str = "config",
        project_root: Optional[Path] = None,
    ):
        """Initialize base configuration manager."""
        # FileUtils must be provided - no fallback to creating new instance
        if file_utils is None:
            raise ValueError(
                "FileUtils instance must be provided to BaseConfigManager. "
                "Use EnvironmentManager to get a shared FileUtils instance."
            )
        
        self.file_utils = file_utils
        self.project_root = (
            Path(project_root) if project_root else self.file_utils.project_root
        )
        self.config_dir = config_dir
        self._config = None

    def init_paths(self) -> None:
        """Initialize configuration paths."""
        try:
            # Set up config directory
            self.config_path = self.project_root / self.config_dir
            if not self.config_path.exists():
                self.config_path.mkdir(parents=True)
                logger.info(f"Created config directory: {self.config_path}")

            # Set up data paths
            self.data_path = self.project_root / "data"
            if not self.data_path.exists():
                self.data_path.mkdir(parents=True)
                logger.info(f"Created data directory: {self.data_path}")

        except Exception as e:
            logger.error(f"Failed to initialize paths: {e}")
            raise

    def init_environment(self) -> None:
        """Initialize environment variables."""
        try:
            # Check for .env files
            env_paths = [
                self.project_root / ".env",
                self.project_root / ".env.local",
                Path.home() / ".env",
            ]

            for env_path in env_paths:
                if env_path.exists():
                    logger.info(f"Found .env file: {env_path}")
                    break
            else:
                logger.warning("No .env file found")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise

    def verify_paths(self) -> Dict[str, bool]:
        """Verify all required paths exist."""
        return {
            "project_root": self.project_root.exists(),
            "config_dir": self.config_path.exists(),
            "data_dir": self.data_path.exists(),
        }

    def get_config(self) -> GlobalConfig:
        """Get the current configuration.
        
        Returns:
            GlobalConfig: The current configuration
            
        Raises:
            ValueError: If configuration is not loaded
        """
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_configurations() first.")
        logger.debug(f"Returning config: {self._config}")
        return self._config
