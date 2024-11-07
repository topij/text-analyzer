# src/core/config.py

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging
from src.utils.FileUtils.file_utils import FileUtils

logger = logging.getLogger(__name__)

class AnalyzerConfig:
    """Configuration handler that uses FileUtils and environment variables."""
    
    DEFAULT_CONFIG = {
        "default_language": "en",
        "content_column": "content",
        "analysis": {
            "keywords": {
                "max_keywords": 5,
                "min_keyword_length": 3,
                "include_compounds": True
            },
            "themes": {
                "max_themes": 3,
                "min_confidence": 0.5,
                "include_hierarchy": True
            },
            "categories": {
                "max_categories": 3,
                "min_confidence": 0.3,
                "require_evidence": True
            }
        },
        "models": {
            "default_provider": "openai",
            "default_model": "gpt-4o-mini",  # Updated default model
            "parameters": {
                "temperature": 0.0,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        },
        "features": {
            "use_caching": True,
            "use_async": True,
            "use_batching": True,
            "enable_finnish_support": True
        }
    }
    
    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize configuration handler.
        
        Args:
            file_utils: Optional FileUtils instance
        """
        # Load environment variables
        self._load_env_vars()
        
        # Initialize FileUtils
        self.file_utils = file_utils or FileUtils()
        
        # Load and merge configurations
        self.config = self._load_config()
        
        # Validate required variables
        self._validate_required_vars()
        
        # Initialize logging using FileUtils configuration
        self._setup_logging()
    
    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""
        for env_file in ['.env', '.env.local']:
            if Path(env_file).exists():
                load_dotenv(env_file)
                logger.debug(f"Loaded environment from {env_file}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configurations from different sources."""
        config = self.DEFAULT_CONFIG.copy()
        
        try:
            # Load from FileUtils
            file_config = self.file_utils.config.get('semantic_analyzer', {})
            config = self._deep_merge(config, file_config)
            
            # Override with environment variables if present
            env_config = self._load_env_config()
            if env_config:
                config = self._deep_merge(config, env_config)
                
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}. Using defaults.")
            
        return config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _load_env_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from environment variables."""
        config = {}
        
        # Model configuration from environment
        if model := os.getenv('SEMANTIC_ANALYZER_MODEL'):
            config['models'] = {
                'default_model': model
            }
        
        # Feature flags from environment
        for feature in ['use_caching', 'use_async', 'use_batching']:
            if value := os.getenv(f'SEMANTIC_ANALYZER_{feature.upper()}'):
                if 'features' not in config:
                    config['features'] = {}
                config['features'][feature] = value.lower() == 'true'
        
        return config if config else None
    
    def _validate_required_vars(self) -> None:
        """Validate required environment variables."""
        required_vars = ['OPENAI_API_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
    
    def _setup_logging(self) -> None:
        """Set up logging using FileUtils configuration."""
        # FileUtils already handles logging setup
        pass

    # Public interface methods remain the same
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('models', self.DEFAULT_CONFIG['models'])
    
    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type."""
        return self.config.get('analysis', {}).get(
            analyzer_type,
            self.DEFAULT_CONFIG['analysis'].get(analyzer_type, {})
        )
    
    def get_features(self) -> Dict[str, bool]:
        """Get feature flags."""
        return self.config.get('features', self.DEFAULT_CONFIG['features'])
    
    @property
    def default_language(self) -> str:
        """Get default language."""
        return self.config.get('default_language', self.DEFAULT_CONFIG['default_language'])
    
    @property
    def content_column(self) -> str:
        """Get content column name."""
        return self.config.get('content_column', self.DEFAULT_CONFIG['content_column'])
    
    def save_results(
        self,
        data: Dict[str, Any],
        filename: str,
        output_type: str = "processed"
    ) -> Path:
        """Save results using FileUtils."""
        return self.file_utils.save_yaml(
            data=data,
            file_path=filename,
            output_type=output_type,
            include_timestamp=self.file_utils.config.get('include_timestamp', True)
        )