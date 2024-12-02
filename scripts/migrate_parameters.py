# scripts/migrate_parameters.py

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Add project root to Python path
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.loaders.models import (
    AnalysisSettings,
    CategoryConfig,
    GeneralParameters,
    ParameterSet,
    PredefinedKeyword,
)
from src.loaders.parameter_config import ParameterSheets
from src.loaders.parameter_handler import ParameterHandler
from FileUtils import FileUtils

logger = logging.getLogger(__name__)


class ParameterMigration:
    """Handles migration of parameter files to new format."""

    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize migration handler."""
        # Get current logging level
        root_logger = logging.getLogger()
        current_level = logging.getLevelName(root_logger.getEffectiveLevel())

        self.file_utils = file_utils or FileUtils(log_level=current_level)
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def migrate_file(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        backup: bool = True,
    ) -> Tuple[bool, str]:
        """Migrate single parameter file to new format.

        Args:
            input_file: Path to old parameter file
            output_file: Optional path for new file
            backup: Whether to create backup

        Returns:
            Tuple[bool, str]: Success status and output path
        """
        try:
            logger.info(f"Starting migration of {input_file}")

            # Create backup if requested
            if backup:
                backup_path = self._create_backup(input_file)
                logger.info(f"Created backup at {backup_path}")

            # Load old parameter file
            old_params = self._load_old_parameters(input_file)

            # Convert to new format
            new_params = self._convert_parameters(old_params)

            # Validate new parameters
            handler = ParameterHandler()
            is_valid, warnings, errors = handler.validate()

            self.warnings.extend(warnings)
            if errors:
                self.errors.extend(errors)
                raise ValueError(
                    f"Validation errors in converted parameters: {errors}"
                )

            # Save new parameters
            output_path = output_file or input_file.with_suffix(".new.xlsx")
            self._save_new_parameters(new_params, output_path)

            logger.info(f"Successfully migrated parameters to {output_path}")
            return True, str(output_path)

        except Exception as e:
            error_msg = f"Error migrating {input_file}: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False, error_msg

    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of original file."""
        backup_path = file_path.with_suffix(f".bak{file_path.suffix}")
        self.file_utils.save_data_to_disk(
            data=self.file_utils.load_excel_sheets(file_path),
            output_filetype="xlsx",
            file_name=backup_path.stem,
        )
        return backup_path

    def _create_example_parameters(
        self,
        output_file: Union[str, Path],
        output_type: str = "parameters",
        language: str = "en",
    ) -> Path:
        """Create example parameter file with all required fields.

        This is an internal method used by the public convenience function.

        Args:
            output_file: Path to save the example file
            language: Language code (en/fi)

        Returns:
            Path: Path to created file
        """

        sheets = {
            ParameterSheets.get_sheet_name("GENERAL", language): pd.DataFrame(
                [
                    {
                        "parameter": "max_keywords",
                        "value": 10,
                        "description": "Maximum keywords to extract (1-20)",
                    },
                    {
                        "parameter": "focus_on",
                        "value": "technical and business content",
                        "description": "Analysis focus area",
                    },
                    {
                        "parameter": "column_name_to_analyze",
                        "value": "content",
                        "description": "Name of the content column",
                    },
                    {
                        "parameter": "min_keyword_length",
                        "value": 3,
                        "description": "Minimum keyword length",
                    },
                    {
                        "parameter": "include_compounds",
                        "value": True,
                        "description": "Include compound words",
                    },
                    {
                        "parameter": "language",
                        "value": language,
                        "description": "Content language",
                    },
                ]
            ),
            ParameterSheets.get_sheet_name(
                "CATEGORIES", language
            ): pd.DataFrame(
                [
                    {
                        "category": "technical_content",
                        "description": "Technical and software development content",
                        "keywords": "software,development,api,programming,technical,code",
                        "threshold": 0.6,
                        "parent": None,
                    },
                    {
                        "category": "business_content",
                        "description": "Business and financial content",
                        "keywords": "revenue,sales,market,growth,financial,business",
                        "threshold": 0.6,
                        "parent": None,
                    },
                    {
                        "category": "educational_content",
                        "description": "Educational and training content",
                        "keywords": "learning,education,training,teaching,skills",
                        "threshold": 0.5,
                        "parent": None,
                    },
                ]
            ),
            ParameterSheets.get_sheet_name("KEYWORDS", language): pd.DataFrame(
                [
                    {
                        "keyword": "machine learning",
                        "importance": 1.0,
                        "domain": "technical",
                    },
                    {
                        "keyword": "cloud computing",
                        "importance": 0.9,
                        "domain": "technical",
                    },
                    {
                        "keyword": "revenue growth",
                        "importance": 0.9,
                        "domain": "business",
                    },
                ]
            ),
            ParameterSheets.get_sheet_name("EXCLUDED", language): pd.DataFrame(
                [
                    {"keyword": "the", "reason": "Common word"},
                    {"keyword": "and", "reason": "Common word"},
                    {"keyword": "for", "reason": "Common word"},
                ]
            ),
            ParameterSheets.get_sheet_name("SETTINGS", language): pd.DataFrame(
                [
                    {
                        "setting": "theme_analysis.min_confidence",
                        "value": 0.5,
                        "description": "Minimum confidence for theme detection",
                    },
                    {
                        "setting": "weights.statistical",
                        "value": 0.4,
                        "description": "Weight for statistical analysis",
                    },
                    {
                        "setting": "weights.llm",
                        "value": 0.6,
                        "description": "Weight for LLM analysis",
                    },
                ]
            ),
        }

        output_path = Path(output_file)

        # Save using FileUtils and handle return value correctly
        result, metadata = self.file_utils.save_data_to_disk(
            data=sheets,
            output_filetype="xlsx",
            file_name=output_path.stem,
            output_type=output_type,
            include_timestamp=False,
        )

        # Result is a dictionary with filenames as keys
        if isinstance(result, dict):
            saved_path = Path(next(iter(result.values())))
        else:
            saved_path = Path(result)

        logger.info(f"Created example parameter file: {saved_path}")
        return saved_path

        # # Save using FileUtils
        # output_path = Path(output_file)
        # self.file_utils.save_data_to_disk(
        #     data=sheets, output_filetype="xlsx", file_name=output_path.stem, include_timestamp=False
        # )

        # logger.info(f"Created example parameter file: {output_path}")
        # return output_path

    def _load_old_parameters(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Load parameters from old format."""
        return self.file_utils.load_excel_sheets(file_path)

    def _convert_parameters(
        self, old_params: Dict[str, pd.DataFrame]
    ) -> ParameterSet:
        """Convert old parameters to new format."""
        # Extract language from file content or name
        language = self._detect_language(old_params)

        # Convert general parameters
        general_params = self._convert_general_parameters(
            old_params.get(ParameterSheets.get_sheet_name("GENERAL", language))
        )

        # Convert categories
        categories = self._convert_categories(
            old_params.get(
                ParameterSheets.get_sheet_name("CATEGORIES", language)
            )
        )

        # Convert keywords
        keywords = self._convert_keywords(
            old_params.get(ParameterSheets.get_sheet_name("KEYWORDS", language))
        )

        # Convert excluded keywords
        excluded = self._convert_excluded_keywords(
            old_params.get(ParameterSheets.get_sheet_name("EXCLUDED", language))
        )

        # Convert analysis settings
        settings = self._convert_settings(
            old_params.get(ParameterSheets.get_sheet_name("SETTINGS", language))
        )

        # Create new parameter set
        return ParameterSet(
            general=general_params,
            categories=categories,
            predefined_keywords=keywords,
            excluded_keywords=excluded,
            analysis_settings=settings,
        )

    def _detect_language(self, params: Dict[str, pd.DataFrame]) -> str:
        """Detect language from parameter sheets."""
        # Try to detect from sheet names
        if any(
            "suomi" in name.lower() or "finnish" in name.lower()
            for name in params.keys()
        ):
            return "fi"
        return "en"

    def _convert_general_parameters(
        self, df: Optional[pd.DataFrame]
    ) -> GeneralParameters:
        """Convert general parameters."""
        if df is None or df.empty:
            return GeneralParameters()

        params = {}
        param_col = "parameter"
        value_col = "value"

        if param_col in df.columns and value_col in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row[param_col]) and pd.notna(row[value_col]):
                    params[str(row[param_col]).strip()] = row[value_col]

        return GeneralParameters(**params)

    def _convert_categories(
        self, df: Optional[pd.DataFrame]
    ) -> Dict[str, CategoryConfig]:
        """Convert category configurations."""
        categories = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if "category" in df.columns and pd.notna(row["category"]):
                    cat_name = str(row["category"]).strip()
                    categories[cat_name] = CategoryConfig(
                        description=row.get("description", ""),
                        keywords=self._split_list(row.get("keywords", "")),
                        threshold=float(row.get("threshold", 0.5)),
                        parent=(
                            row.get("parent")
                            if pd.notna(row.get("parent"))
                            else None
                        ),
                    )
        return categories

    def _split_list(self, value: Any) -> List[str]:
        """Split comma-separated string into list."""
        if pd.isna(value):
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def _save_new_parameters(
        self, params: ParameterSet, output_path: Path
    ) -> None:
        """Save parameters in new format."""
        sheets = {
            "General Parameters": self._create_general_sheet(params.general),
            "Categories": self._create_categories_sheet(params.categories),
            "Keywords": self._create_keywords_sheet(params.predefined_keywords),
            "Excluded Keywords": pd.DataFrame(
                {"keyword": list(params.excluded_keywords)}
            ),
            "Analysis Settings": self._create_settings_sheet(
                params.analysis_settings
            ),
        }

        self.file_utils.save_data_to_disk(
            data=sheets, output_filetype="xlsx", file_name=output_path.stem
        )

    def _create_general_sheet(self, params: GeneralParameters) -> pd.DataFrame:
        """Create general parameters sheet."""
        return pd.DataFrame(
            [
                {"parameter": k, "value": v}
                for k, v in params.model_dump().items()
            ]
        )

    def _create_categories_sheet(
        self, categories: Dict[str, CategoryConfig]
    ) -> pd.DataFrame:
        """Create categories sheet."""
        rows = []
        for name, cat in categories.items():
            rows.append(
                {
                    "category": name,
                    "description": cat.description,
                    "keywords": ",".join(cat.keywords),
                    "threshold": cat.threshold,
                    "parent": cat.parent or "",
                }
            )
        return pd.DataFrame(rows)

    def get_migration_status(self) -> Tuple[List[str], List[str]]:
        """Get warnings and errors from migration."""
        return self.warnings, self.errors


def main():
    """Run parameter migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate parameter files to new format"
    )
    parser.add_argument(
        "input_files", nargs="+", help="Input parameter files to migrate"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Don't create backups"
    )
    parser.add_argument(
        "-o", "--output-dir", help="Output directory for new files"
    )

    args = parser.parse_args()

    # Set up logging
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    migrator = ParameterMigration()
    results = []

    for input_file in args.input_files:
        input_path = Path(input_file)
        if args.output_dir:
            output_path = Path(args.output_dir) / input_path.name
        else:
            output_path = None

        success, output = migrator.migrate_file(
            input_path, output_path, not args.no_backup
        )
        results.append((input_file, success, output))

    # Print results
    print("\nMigration Results:")
    print("-" * 50)
    for input_file, success, output in results:
        status = "✓" if success else "✗"
        print(f"{status} {input_file} -> {output}")

    # Print warnings and errors
    warnings, errors = migrator.get_migration_status()
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")


def create_example_parameters(
    output_file: str,
    language: str = "en",
    output_type: str = "parameters",  # Add output_type parameter with default
) -> Path:
    """Create example parameter file.

    This is a convenience function that creates an example parameter file
    with predefined settings suitable for getting started with the analyzer.

    Args:
        output_file: Name or path for the output file
        language: Language code ("en" or "fi")
        output_type: Directory to save parameters (default: "parameters")

    Returns:
        Path: Path to created parameter file

    Example:
        >>> params_file = create_example_parameters("parameters_en.xlsx")
        >>> handler = ParameterHandler(params_file)
        >>> params = handler.get_parameters()
    """
    migrator = ParameterMigration()
    return migrator._create_example_parameters(
        output_file, output_type, language
    )


if __name__ == "__main__":
    main()
