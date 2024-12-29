"""Core environment management functionality."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from dotenv import load_dotenv

from FileUtils import FileUtils
from src.config.manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for environment setup."""
    env_type: Optional[str] = None
    project_root: Optional[Path] = None
    log_level: str = "INFO"
    config_dir: str = "config"
    custom_directory_structure: Optional[Dict[str, Any]] = None


class EnvironmentManager:
    """Centralized manager for environment setup and configuration."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Optional[EnvironmentConfig] = None):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize environment manager if not already initialized."""
        # Only initialize once
        if EnvironmentManager._initialized:
            return
            
        try:
            self.config = config or EnvironmentConfig()
            
            # Set up logging first
            log_level = os.getenv("APP_LOGGING_LEVEL", self.config.log_level)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.debug(f"Logging initialized at level {log_level}")
            
            # Initialize core utilities first
            self._init_file_utils()
            
            # Initialize other components
            self._init_config_manager()
            self._components = {
                "file_utils": self.file_utils,
                "config_manager": self.config_manager,
                "project_root": self.project_root,
            }
            
            EnvironmentManager._initialized = True
            logger.info("Environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize environment: {e}") from e

    def _init_file_utils(self) -> None:
        """Initialize FileUtils with proper configuration."""
        try:
            # Set up project root
            if self.config.project_root:
                self.project_root = Path(self.config.project_root)
            else:
                # Try to find project root by climbing up
                current = Path().resolve()
                root = None
                while current.parent != current:
                    if (current / "data").exists():
                        root = current
                        break
                    current = current.parent
                
                if not root:
                    raise ValueError("Could not determine project root")
                self.project_root = root

            if not self.project_root.exists():
                raise ValueError(f"Project root does not exist: {self.project_root}")

            # Initialize FileUtils with project root
            self.file_utils = FileUtils(project_root=self.project_root)
            logger.debug(f"FileUtils initialized with project root: {self.project_root}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FileUtils: {e}")
            raise

    def _init_config_manager(self) -> None:
        """Initialize ConfigManager with shared FileUtils instance."""
        try:
            self.config_manager = ConfigManager(
                file_utils=self.file_utils,
                config_dir=self.config.config_dir,
                project_root=self.project_root,
                custom_directory_structure=self.config.custom_directory_structure,
            )
            # Load configurations
            self.config_manager.load_configurations()
            logger.debug("ConfigManager initialized and configurations loaded")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConfigManager: {e}")
            raise

    def get_components(self) -> Dict[str, Any]:
        """Get all registered components."""
        if not hasattr(self, "_components"):
            self._components = {}
        return self._components

    def verify_environment(self) -> Dict[str, bool]:
        """Verify environment setup and return status."""
        status = {
            "project_root_exists": self.project_root.exists(),
            "config_dir_exists": (self.project_root / self.config.config_dir).exists(),
            "data_dir_exists": (self.project_root / "data").exists(),
            "env_file_exists": any(p.exists() for p in [
                self.project_root / ".env",
                self.project_root / ".env.local",
                Path.home() / ".env"
            ]),
            "components_initialized": self._initialized,
        }
        return status

    @classmethod
    def get_llm_info(cls, analyzer: 'SemanticAnalyzer', detailed: bool = False) -> str:
        """Get information about the active LLM configuration.
        
        Args:
            analyzer: Semantic analyzer instance
            detailed: Whether to include detailed configuration
            
        Returns:
            String containing LLM information
        """
        try:
            # Get current config and LLM info from analyzer's config
            config = analyzer.analyzer_config.config.get("models", {})
            if not config:
                return "No LLM configuration found"

            provider = config.get("default_provider")
            model = config.get("default_model")
            if not provider or not model:
                return "Incomplete LLM configuration"

            info = f"Active LLM: {provider} ({model})"

            if detailed:
                # Get global parameters
                params = config.get("parameters", {})
                if params:
                    info += "\nGlobal Parameters:"
                    for key, value in params.items():
                        display_key = key.replace("_", " ").title()
                        info += f"\n- {display_key}: {value}"

                # Get provider-specific parameters
                provider_config = config.get("providers", {}).get(provider, {})
                if provider_config:
                    # Get model-specific parameters
                    model_config = provider_config.get("available_models", {}).get(model, {})
                    if model_config:
                        info += f"\n\nModel Configuration:"
                        for key, value in model_config.items():
                            display_key = key.replace("_", " ").title()
                            info += f"\n- {display_key}: {value}"

                    # Add provider-specific info
                    if "api_type" in provider_config:
                        info += f"\nAPI Type: {provider_config['api_type']}"
                    if "api_version" in provider_config:
                        info += f"\nAPI Version: {provider_config['api_version']}"

            return info

        except Exception as e:
            logger.error(f"Error getting LLM info: {e}")
            return "Unable to get LLM information"

    @classmethod
    def get_instance(cls) -> 'EnvironmentManager':
        """Get the singleton instance.
        
        Returns:
            EnvironmentManager instance
            
        Raises:
            RuntimeError: If EnvironmentManager not initialized
        """
        if cls._instance is None:
            raise RuntimeError(
                "EnvironmentManager not initialized. "
                "Create an instance with config first."
            )
        return cls._instance