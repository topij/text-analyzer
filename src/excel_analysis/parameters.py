# src/excel_analysis/parameters.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.loaders.models import CategoryConfig, ParameterSet
from src.loaders.parameter_handler import ParameterHandler
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class AnalysisParameters:
    """Container for analysis-specific parameters with validation."""

    # Valid analysis types and their required parameters
    REQUIRED_PARAMETERS = {
        "keywords": {"max_keywords", "min_keyword_length", "include_compounds"},
        "themes": {"max_themes", "min_confidence"},
        "categories": {"categories", "min_confidence"},
    }

    def __init__(
        self,
        parameter_file: Union[str, Path],
        file_utils: Optional[FileUtils] = None,
    ):
        """Initialize parameters from Excel file.

        Args:
            parameter_file: Path to Excel parameter file
            file_utils: Optional FileUtils instance
        """
        self.file_utils = file_utils or FileUtils()
        self.parameter_handler = ParameterHandler(parameter_file)
        self.parameters = self.parameter_handler.get_parameters()

        # Cache analyzer configs
        self._analyzer_configs = {}

    def get_analyzer_config(self, analyzer_type: str) -> Dict[str, Any]:
        """Get configuration for specific analyzer type.

        Args:
            analyzer_type: Type of analyzer ("keywords", "themes", "categories")

        Returns:
            Dict containing analyzer configuration

        Raises:
            ValueError: If analyzer type is invalid
        """
        if analyzer_type not in self.REQUIRED_PARAMETERS:
            raise ValueError(f"Invalid analyzer type: {analyzer_type}")

        # Return cached config if available
        if analyzer_type in self._analyzer_configs:
            return self._analyzer_configs[analyzer_type]

        # Build analyzer-specific config
        config = {
            "language": self.parameters.general.language,
            "min_confidence": self.parameters.general.min_confidence,
            "focus_on": self.parameters.general.focus_on,
        }

        # Add analyzer-specific parameters
        if analyzer_type == "keywords":
            config.update(
                {
                    "max_keywords": self.parameters.general.max_keywords,
                    "min_keyword_length": self.parameters.general.min_keyword_length,
                    "include_compounds": self.parameters.general.include_compounds,
                    "weights": self.parameters.analysis_settings.weights.model_dump(),
                }
            )
        elif analyzer_type == "themes":
            config.update(
                {
                    "max_themes": self.parameters.general.max_themes,
                    "theme_analysis": self.parameters.analysis_settings.theme_analysis.model_dump(),
                }
            )
        elif analyzer_type == "categories":
            config.update({"categories": self.parameters.categories})

        # Cache and return
        self._analyzer_configs[analyzer_type] = config
        return config

    def validate_for_analysis(
        self, analysis_types: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate parameters for requested analysis types.

        Args:
            analysis_types: List of analysis types to validate for

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate analysis types
        invalid_types = set(analysis_types) - set(
            self.REQUIRED_PARAMETERS.keys()
        )
        if invalid_types:
            errors.append(f"Invalid analysis types: {invalid_types}")

        # Check required parameters for each type
        for analysis_type in analysis_types:
            if analysis_type not in self.REQUIRED_PARAMETERS:
                continue

            config = self.get_analyzer_config(analysis_type)
            missing_params = self.REQUIRED_PARAMETERS[analysis_type] - set(
                config.keys()
            )

            if missing_params:
                errors.append(
                    f"Missing required parameters for {analysis_type}: {missing_params}"
                )

            # Validate specific parameter values
            try:
                self._validate_parameter_values(analysis_type, config)
            except ValidationError as e:
                errors.extend(str(err) for err in e.errors())

        return len(errors) == 0, errors

    def _validate_parameter_values(
        self, analyzer_type: str, config: Dict[str, Any]
    ) -> None:
        """Validate specific parameter values for an analyzer type."""
        if analyzer_type == "keywords":
            if config["max_keywords"] < 1 or config["max_keywords"] > 20:
                raise ValidationError("max_keywords must be between 1 and 20")

            if config["min_keyword_length"] < 2:
                raise ValidationError("min_keyword_length must be at least 2")

        elif analyzer_type == "themes":
            if config["max_themes"] < 1 or config["max_themes"] > 10:
                raise ValidationError("max_themes must be between 1 and 10")

        # Common validations
        if "min_confidence" in config:
            if config["min_confidence"] < 0.0 or config["min_confidence"] > 1.0:
                raise ValidationError(
                    "min_confidence must be between 0.0 and 1.0"
                )

    def update_parameters(self, **kwargs) -> None:
        """Update parameters with new values.

        Args:
            **kwargs: Parameter updates by section
        """
        self.parameters = self.parameter_handler.update_parameters(**kwargs)
        # Clear config cache
        self._analyzer_configs.clear()
