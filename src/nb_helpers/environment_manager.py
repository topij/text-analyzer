import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from dotenv import load_dotenv

from FileUtils import FileUtils
from src.config.manager import ConfigManager
from src.config.models import (
    GlobalConfig,
    LoggingConfig,
    ModelConfig,
    LanguageConfig,
)
from src.core.config import AnalyzerConfig
from src.semantic_analyzer.analyzer import SemanticAnalyzer
from src.nb_helpers.logging_manager import LoggingManager

logger = logging.getLogger(__name__)


class EnvironmentType(str, Enum):
    """Environment types for analysis."""
    LOCAL = "local"
    AZURE = "azure"


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
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize environment manager."""
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self.config = config or EnvironmentConfig()
        self._logging_manager = LoggingManager()
        self._initialized_components: Dict[str, Any] = {}
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up the environment with all necessary components."""
        try:
            # Configure logging first
            self._logging_manager.configure_logging(self.config.log_level)
            
            # Set up project root
            self.project_root = self._setup_project_root()
            logger.info(f"Project root: {self.project_root}")
            
            # Load environment variables
            self._load_environment()
            
            # Initialize core components
            self._init_components()
            
            self._initialized = True
            logger.info("Environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize environment: {e}") from e

    def _setup_project_root(self) -> Path:
        """Set up and validate project root."""
        if self.config.project_root:
            root = Path(self.config.project_root)
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
        
        if not root.exists():
            raise ValueError(f"Project root does not exist: {root}")
        
        return root

    def _load_environment(self) -> None:
        """Load environment variables."""
        env_file = self.project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.debug("Loaded environment variables from .env file")

    def _init_components(self) -> None:
        """Initialize core components."""
        try:
            # Initialize FileUtils
            file_utils = FileUtils(
                project_root=self.project_root,
                log_level=self.config.log_level,
                directory_structure=self.config.custom_directory_structure
                or ConfigManager.DEFAULT_DIRECTORY_STRUCTURE,
            )
            
            # Initialize ConfigManager
            config_manager = ConfigManager(
                file_utils=file_utils,
                config_dir=self.config.config_dir,
                project_root=self.project_root,
                custom_directory_structure=self.config.custom_directory_structure,
            )
            
            # Store initialized components
            self._initialized_components.update({
                "file_utils": file_utils,
                "config_manager": config_manager,
                "project_root": self.project_root,
                "data_dir": self.project_root / "data",
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def verify_environment(self) -> Dict[str, bool]:
        """Verify environment setup and return status."""
        status = {
            "project_structure": False,
            "configuration": False,
            "environment_vars": False,
            "directory_structure": False,
        }
        
        try:
            config_manager = self._initialized_components.get("config_manager")
            file_utils = self._initialized_components.get("file_utils")
            
            # Check project structure
            if self.project_root and self.project_root.exists():
                status["project_structure"] = True
            
            # Check configuration
            if config_manager and config_manager.config:
                status["configuration"] = True
            
            # Check environment variables
            required_vars = ["OPENAI_API_KEY"]  # Add more as needed
            status["environment_vars"] = all(
                os.getenv(var) for var in required_vars
            )
            
            # Check directory structure
            if file_utils and file_utils.verify_directory_structure():
                status["directory_structure"] = True
                
        except Exception as e:
            logger.error(f"Environment verification failed: {e}")
            
        return status

    def display_configuration(self) -> None:
        """Display current configuration settings."""
        config_manager = self._initialized_components.get("config_manager")
        if not config_manager:
            logger.warning("No ConfigManager available")
            return

        print("\nCurrent Configuration:")
        print("-" * 50)
        
        # Display global config
        if global_config := config_manager.config.get("global"):
            print("\nGlobal Configuration:")
            for key, value in global_config.items():
                print(f"  {key}: {value}")
        
        # Display model config
        if model_config := config_manager.config.get("model"):
            print("\nModel Configuration:")
            for key, value in model_config.items():
                print(f"  {key}: {value}")
        
        # Display directory structure
        print("\nDirectory Structure:")
        self._display_directory_structure(
            config_manager.directory_structure
        )

    def _display_directory_structure(
        self, structure: Dict[str, Any], indent: int = 2
    ) -> None:
        """Display directory structure with proper indentation."""
        for key, value in structure.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                self._display_directory_structure(value, indent + 2)
            else:
                print(f"{prefix}{key}")

    @property
    def components(self) -> Dict[str, Any]:
        """Get initialized components."""
        return self._initialized_components.copy()

    def get_llm_info(
        self, analyzer: SemanticAnalyzer, detailed: bool = False
    ) -> Dict[str, Any]:
        """Get information about the active LLM configuration."""
        info = {
            "provider": analyzer.llm.__class__.__name__,
            "model": getattr(analyzer.llm, "model_name", "unknown"),
        }
        
        if detailed:
            info.update({
                "temperature": getattr(analyzer.llm, "temperature", None),
                "max_tokens": getattr(analyzer.llm, "max_tokens", None),
                "streaming": getattr(analyzer.llm, "streaming", None),
            })
        
        return info

    def get_available_providers(
        self, analyzer: SemanticAnalyzer
    ) -> Dict[str, List[str]]:
        """Get available LLM providers and their configurations."""
        return analyzer.get_available_providers()

    def change_llm_provider(
        self,
        analyzer: SemanticAnalyzer,
        provider: str,
        model: Optional[str] = None
    ) -> None:
        """Change LLM provider and optionally model."""
        analyzer.change_llm_provider(provider, model)
