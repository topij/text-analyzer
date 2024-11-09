# src/semantic_analyzer/parameters.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from src.schemas import AnalysisParameters

logger = logging.getLogger(__name__)

class ParameterManager:
    """Manages analysis parameters with validation and defaults."""
    
    DEFAULT_CONFIG = {
        "language": "en",
        "max_keywords": 8,
        "min_confidence": 0.3,
        "categories": {
            "general": {
                "description": "General content category",
                "keywords": ["general", "content", "information"],
                "threshold": 0.3
            }
        },
        "analysis": {
            "keywords": {
                "min_length": 3,
                "include_compounds": True,
                "weights": {
                    "statistical": 0.4,
                    "llm": 0.6
                }
            },
            "themes": {
                "max_themes": 3,
                "min_confidence": 0.5,
                "include_hierarchy": True
            },
            "categories": {
                "min_confidence": 0.3,
                "require_evidence": True,
                "max_categories": 3
            }
        }
    }

    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        **override_params
    ):
        """Initialize parameter manager.
        
        Args:
            config_path: Optional path to YAML configuration file
            **override_params: Parameters to override defaults or file config
        """
        self.config_path = Path(config_path) if config_path else None
        self.parameters = self._load_parameters(override_params)
        
    def _load_parameters(self, override_params: Dict[str, Any]) -> AnalysisParameters:
        """Load and validate parameters from all sources.
        
        Args:
            override_params: Parameters to override defaults/file config
            
        Returns:
            AnalysisParameters: Validated parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Start with defaults
        config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                config.update(file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}. Using defaults.")
        
        # Apply overrides
        if override_params:
            config.update(override_params)
        
        try:
            # Validate using Pydantic model
            parameters = AnalysisParameters(**config)
            self._validate_categories(parameters)
            return parameters
            
        except ValidationError as e:
            logger.error(f"Parameter validation failed: {e}")
            raise
            
    def _validate_categories(self, parameters: AnalysisParameters) -> None:
        """Additional validation for category configurations.
        
        Args:
            parameters: Parameters to validate
            
        Raises:
            ValueError: If category configuration is invalid
        """
        for category, config in parameters.categories.items():
            if not config.get("description"):
                raise ValueError(f"Category {category} missing description")
            if not config.get("keywords"):
                raise ValueError(f"Category {category} missing keywords")
                
    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type.
        
        Args:
            analyzer_type: Type of analyzer ("keywords", "themes", "categories")
            
        Returns:
            Dict[str, Any]: Configuration for the specified analyzer
        """
        return self.parameters.dict().get("analysis", {}).get(
            analyzer_type, 
            self.DEFAULT_CONFIG["analysis"][analyzer_type]
        )
        
    def update_parameters(self, **kwargs) -> AnalysisParameters:
        """Update parameters with new values.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            AnalysisParameters: Updated parameters
            
        Raises:
            ValidationError: If new parameters are invalid
        """
        config = self.parameters.dict()
        config.update(kwargs)
        self.parameters = AnalysisParameters(**config)
        return self.parameters
        
    @property
    def default_language(self) -> str:
        """Get default language."""
        return self.parameters.language
        
    @property
    def categories(self) -> Dict[str, Dict[str, Any]]:
        """Get category configurations."""
        return self.parameters.categories
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return self.parameters.dict()