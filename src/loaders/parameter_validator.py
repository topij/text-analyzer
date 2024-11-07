# src/loaders/parameter_validator.py

from typing import Dict, Any, List, Tuple

from .models import (
    GeneralParameters,
    ParameterSet,
    PredefinedKeyword,
    CategoryConfig
)

class ParameterValidator:
    """Validator for analysis parameters."""
    
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate parameters and return validation results."""
        warnings = []
        errors = []
        
        try:
            # Convert to ParameterSet
            if not isinstance(params, ParameterSet):
                param_set = ParameterSet(**params)
            else:
                param_set = params
                
            # Check general parameters
            general = param_set.general
            if general.max_keywords > 15:
                errors.append("max_keywords cannot exceed 15")
                
            if general.min_keyword_length < 2:
                errors.append("min_keyword_length must be at least 2")
                
            # Check confidence thresholds
            if general.min_confidence < 0.1:
                warnings.append("Very low minimum confidence threshold")
            elif general.min_confidence > 0.9:
                warnings.append("Very high minimum confidence threshold")
            
            # Check categories
            if hasattr(param_set, 'categories'):
                for cat_name, cat in param_set.categories.items():
                    if cat.threshold < 0.1:
                        warnings.append(f"Very low threshold for category {cat_name}")
            
            # Check for keyword conflicts
            if param_set.predefined_keywords and param_set.excluded_keywords:
                conflicts = set(param_set.predefined_keywords) & param_set.excluded_keywords
                if conflicts:
                    warnings.append(f"Keywords appear in both predefined and excluded: {conflicts}")
            
            return len(errors) == 0, warnings, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, warnings, errors
    
    def validate_parameters(self, params: Dict[str, Any]) -> GeneralParameters:
        """Validate and convert general parameters."""
        if isinstance(params, dict) and 'general' in params:
            return GeneralParameters(**params['general'])
        return GeneralParameters(**params)