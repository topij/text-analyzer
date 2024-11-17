# src/loaders/parameter_validation.py

from typing import Any, Dict, List, Set, Tuple, Optional
from pydantic import BaseModel, Field, field_validator
import re
import logging

logger = logging.getLogger(__name__)


class ValidationRule(BaseModel):
    """Single validation rule."""

    rule_type: str
    field_path: str
    condition: Any
    error_level: str = "error"  # "error" or "warning"
    message: str

    @field_validator("error_level")
    def validate_level(cls, v):
        if v not in ["error", "warning"]:
            raise ValueError("error_level must be 'error' or 'warning'")
        return v


# src/loaders/parameter_validation.py

from typing import Any, Dict, List, Set, Tuple, Optional
from pydantic import BaseModel, Field, field_validator
import re
import logging

logger = logging.getLogger(__name__)


class ValidationRule(BaseModel):
    """Single validation rule."""

    rule_type: str
    field_path: str
    condition: Any
    error_level: str = "error"  # "error" or "warning"
    message: str

    @field_validator("error_level")
    def validate_level(cls, v):
        if v not in ["error", "warning"]:
            raise ValueError("error_level must be 'error' or 'warning'")
        return v


class ParameterValidation:
    """Enhanced parameter validation rules."""

    # Value range rules
    VALUE_RANGES = {
        "general.max_keywords": (1, 20),
        "general.min_keyword_length": (2, 10),
        "general.max_themes": (1, 10),
        "general.min_confidence": (0.0, 1.0),
        "analysis_settings.theme_analysis.min_confidence": (0.0, 1.0),
        "categories.*.threshold": (0.0, 1.0),
    }

    # Required fields with types
    REQUIRED_FIELDS = {"general.max_keywords": int, "general.focus_on": str, "general.column_name_to_analyze": str}

    # Pattern validations - Fixed regex patterns
    PATTERNS = {
        "general.language": r"^(en|fi)$",
        # Allow commas, spaces, dashes, underscores, and alphanumeric characters
        "categories.*.keywords": r"^[a-zA-Z0-9,\s\-_]+$",
    }

    # Interdependent field validations
    FIELD_DEPENDENCIES = [
        {
            "fields": ["analysis_settings.weights.statistical", "analysis_settings.weights.llm"],
            "condition": lambda x: abs(sum(x.values()) - 1.0) < 0.001,
            "message": "Weight values must sum to 1.0",
        }
    ]

    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate parameters with comprehensive rules."""
        warnings = []
        errors = []

        try:
            # Check required fields
            self._validate_required_fields(params, errors)

            # Check value ranges
            self._validate_ranges(params, errors, warnings)

            # Check patterns
            self._validate_patterns(params, errors)

            # Check field dependencies
            try:
                self._validate_dependencies(params, errors)
            except Exception as e:
                logger.error(f"Error in dependency validation: {str(e)}")
                errors.append(f"Dependency validation error: {str(e)}")

            # Check keyword uniqueness and conflicts
            self._validate_keywords(params, warnings)

            # Check category hierarchy
            self._validate_category_hierarchy(params, warnings)

            return len(errors) == 0, warnings, errors

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, warnings, errors

    def _validate_required_fields(self, params: Dict[str, Any], errors: List[str]) -> None:
        """Validate required fields existence and types."""
        for field_path, expected_type in self.REQUIRED_FIELDS.items():
            try:
                value = self._get_nested_value(params, field_path)
                if value is None:
                    errors.append(f"Missing required field: {field_path}")
                elif not isinstance(value, expected_type):
                    errors.append(f"Invalid type for {field_path}: expected {expected_type.__name__}")
            except Exception as e:
                logger.error(f"Error validating field {field_path}: {str(e)}")
                errors.append(f"Error validating {field_path}: {str(e)}")

    def _validate_patterns(self, params: Dict[str, Any], errors: List[str]) -> None:
        """Validate pattern matching."""
        for field_path, pattern in self.PATTERNS.items():
            try:
                if "*" in field_path:
                    self._validate_wildcard_pattern(field_path, pattern, params, errors)
                else:
                    value = self._get_nested_value(params, field_path)
                    if value is not None:
                        self._check_pattern_match(value, pattern, field_path, errors)
            except Exception as e:
                logger.error(f"Pattern validation error for {field_path}: {str(e)}")
                errors.append(f"Pattern validation error: {str(e)}")

    def _validate_wildcard_pattern(
        self, field_path: str, pattern: str, params: Dict[str, Any], errors: List[str]
    ) -> None:
        """Handle wildcard pattern validation."""
        parent_path = field_path.split(".*")[0]
        parent = self._get_nested_value(params, parent_path)

        if isinstance(parent, dict):
            for key, obj in parent.items():
                try:
                    child_path = field_path.replace("*", key)
                    # For categories.*.keywords, we need to check the 'keywords' field in the category object
                    if field_path == "categories.*.keywords":
                        value = obj.get("keywords", None)
                        if isinstance(value, list):
                            # Join list elements with commas for validation
                            value = ",".join(str(v) for v in value)
                    else:
                        value = obj

                    if value is not None:
                        self._check_pattern_match(value, pattern, child_path, errors)
                except Exception as e:
                    logger.error(f"Error in wildcard validation for {child_path}: {str(e)}")

    def _validate_dependencies(self, params: Dict[str, Any], errors: List[str]) -> None:
        """Validate interdependent fields."""
        for dep in self.FIELD_DEPENDENCIES:
            try:
                field_values = {}
                for field in dep["fields"]:
                    value = self._get_nested_value(params, field)
                    if value is not None:
                        field_values[field] = value

                if len(field_values) == len(dep["fields"]):
                    if not dep["condition"](field_values):
                        errors.append(dep["message"])
            except Exception as e:
                logger.error(f"Error validating dependencies: {str(e)}")
                errors.append(f"Dependency validation error: {str(e)}")

    def _validate_keywords(self, params: Dict[str, Any], warnings: List[str]) -> None:
        """Validate keyword uniqueness and conflicts."""
        try:
            # Get all keywords from different sources
            predefined = set(params.get("predefined_keywords", {}).keys())
            excluded = params.get("excluded_keywords", set())

            # Collect category keywords
            category_keywords = set()
            categories = params.get("categories", {})
            for category in categories.values():
                if isinstance(category, dict) and "keywords" in category:
                    if isinstance(category["keywords"], list):
                        category_keywords.update(category["keywords"])
                    elif isinstance(category["keywords"], str):
                        # Handle comma-separated string case
                        keywords = [k.strip() for k in category["keywords"].split(",")]
                        category_keywords.update(keywords)

            # Check for conflicts
            conflicts = predefined & excluded
            if conflicts:
                warnings.append(f"Keywords appear in both predefined and excluded lists: {conflicts}")

            # Check for duplicates across categories
            for cat1 in categories:
                for cat2 in categories:
                    if cat1 < cat2:  # Compare each pair once
                        cat1_keywords = set(categories[cat1].get("keywords", []))
                        cat2_keywords = set(categories[cat2].get("keywords", []))
                        common = cat1_keywords & cat2_keywords
                        if common:
                            warnings.append(f"Common keywords found in categories {cat1} and {cat2}: {common}")

        except Exception as e:
            logger.error(f"Error validating keywords: {str(e)}")
            warnings.append(f"Keyword validation warning: {str(e)}")

    def _validate_category_hierarchy(self, params: Dict[str, Any], warnings: List[str]) -> None:
        """Validate category hierarchy consistency."""
        try:
            categories = params.get("categories", {})
            visited = set()

            def check_cycle(category: str, path: Set[str]) -> bool:
                if category in path:
                    return True
                if category in visited:
                    return False

                visited.add(category)
                path.add(category)

                parent = categories.get(category, {}).get("parent")
                if parent:
                    if parent not in categories:
                        warnings.append(f"Category {category} references non-existent parent {parent}")
                        return False
                    return check_cycle(parent, path)

                path.remove(category)
                return False

            # Check each category
            for category in categories:
                if check_cycle(category, set()):
                    warnings.append(f"Circular dependency detected in category hierarchy involving {category}")

        except Exception as e:
            logger.error(f"Error validating category hierarchy: {str(e)}")
            warnings.append(f"Category hierarchy validation warning: {str(e)}")

    @staticmethod
    def _get_nested_value(params: Dict[str, Any], path: str) -> Any:
        """Get nested dictionary value using dot notation path."""
        current = params
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            if part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _check_pattern_match(value: Any, pattern: str, field_path: str, errors: List[str]) -> None:
        """Check if value matches pattern."""
        try:
            value_str = str(value)
            if not re.match(pattern, value_str):
                logger.debug(f"Pattern match failed for {field_path}: value='{value_str}', pattern='{pattern}'")
                errors.append(f"Invalid format for {field_path}: must match pattern {pattern}")
        except Exception as e:
            logger.error(f"Pattern matching error for {field_path}: {str(e)}")
            errors.append(f"Pattern matching error for {field_path}: {str(e)}")

    def _validate_ranges(self, params: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validate numeric values are within specified ranges."""
        try:
            for field_path, (min_val, max_val) in self.VALUE_RANGES.items():
                if "*" in field_path:
                    self._validate_wildcard_range(field_path, min_val, max_val, params, errors, warnings)
                else:
                    value = self._get_nested_value(params, field_path)
                    if value is not None:
                        self._check_range(field_path, value, min_val, max_val, errors, warnings)
        except Exception as e:
            logger.error(f"Error in range validation: {str(e)}")
            errors.append(f"Range validation error: {str(e)}")

    def _validate_wildcard_range(
        self,
        field_path: str,
        min_val: float,
        max_val: float,
        params: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Handle wildcard range validation for collection fields."""
        try:
            parent_path = field_path.split(".*")[0]
            parent = self._get_nested_value(params, parent_path)

            if isinstance(parent, dict):
                for key, obj in parent.items():
                    child_path = field_path.replace("*", key)
                    # For categories.*.threshold, we need to check the threshold value in the category object
                    if field_path == "categories.*.threshold":
                        value = obj.get("threshold", None)
                    else:
                        value = self._get_nested_value(obj, field_path.split(".")[-1])

                    if value is not None:
                        self._check_range(child_path, value, min_val, max_val, errors, warnings)
        except Exception as e:
            logger.error(f"Error in wildcard range validation: {str(e)}")
            errors.append(f"Wildcard range validation error: {str(e)}")

    def _check_range(
        self, field_path: str, value: Any, min_val: float, max_val: float, errors: List[str], warnings: List[str]
    ) -> None:
        """Check if a numeric value is within the specified range."""
        try:
            # Convert value to float for comparison
            num_value = float(value)

            # Check for out of range values
            if num_value < min_val:
                errors.append(f"Value for {field_path} ({num_value}) is below minimum {min_val}")
            elif num_value > max_val:
                errors.append(f"Value for {field_path} ({num_value}) is above maximum {max_val}")

            # Add warnings for values close to limits
            elif num_value < min_val * 1.1:  # Within 10% of minimum
                warnings.append(f"Value for {field_path} ({num_value}) is close to minimum {min_val}")
            elif num_value > max_val * 0.9:  # Within 10% of maximum
                warnings.append(f"Value for {field_path} ({num_value}) is close to maximum {max_val}")

        except (TypeError, ValueError) as e:
            logger.error(f"Invalid numeric value for {field_path}: {str(e)}")
            errors.append(f"Invalid numeric value for {field_path}: must be a number between {min_val} and {max_val}")
