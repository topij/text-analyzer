# scripts/test_data_generator.py

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")

import pandas as pd
from src.loaders.parameter_config import ParameterSheets
from src.loaders.parameter_handler import ParameterHandler
from FileUtils import FileUtils, OutputFileType

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generator for test data and parameter files."""

    def __init__(self, file_utils: Optional[FileUtils] = None):
        """Initialize generator with FileUtils."""
        self.file_utils = file_utils or FileUtils()

    def generate_all(self, force: bool = False) -> Dict[str, List[Path]]:
        """Generate all test files.

        Args:
            force: If True, overwrite existing files

        Returns:
            Dict containing paths to generated files by type
        """
        logger.info("Starting test data generation...")
        files = {"parameters": [], "content": []}

        try:
            # Generate parameter files
            logger.info("Generating parameter files...")
            param_files = self.generate_parameter_files(force=force)
            files["parameters"].extend(param_files.values())

            # Generate content files
            logger.info("Generating test content...")
            content_files = self.generate_test_content(force=force)
            files["content"].extend(content_files.values())

            logger.info("Test data generation completed successfully")
            return files

        except Exception as e:
            logger.error(f"Error generating test data: {e}")
            raise

    def generate_test_content(self, force: bool = False) -> Dict[str, Path]:
        """Generate test content in both languages."""
        output_dir = self.file_utils.get_data_path("raw")
        files = {}

        content = {
            "en": {
                "technical": [
                    "Machine learning models are trained using large datasets to recognize patterns. Neural networks enable complex pattern recognition.",
                    "Cloud computing services provide scalable infrastructure for deployments. APIs enable system integration.",
                ],
                "business": [
                    "Q3 financial results show 15% revenue growth. Market expansion strategy focuses on emerging sectors.",
                    "Strategic partnerships drive innovation. Customer satisfaction metrics show positive trends.",
                ],
            },
            "fi": {
                "technical": [
                    "Koneoppimismallit koulutetaan suurilla datajoukolla tunnistamaan kaavoja. Neuroverkot mahdollistavat monimutkaisen hahmontunnistuksen.",
                    "Pilvipalvelut tarjoavat skaalautuvan infrastruktuurin. Rajapinnat mahdollistavat järjestelmäintegraation.",
                ],
                "business": [
                    "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun. Markkinalaajennusstrategia keskittyy uusiin sektoreihin.",
                    "Strategiset kumppanuudet edistävät innovaatiota. Asiakastyytyväisyysmittarit osoittavat positiivista kehitystä.",
                ],
            },
        }

        for lang, texts in content.items():
            df = pd.DataFrame(
                [
                    {
                        "id": f"{content_type}_{idx+1}",
                        "type": content_type,
                        "language": lang,
                        "content": text,
                    }
                    for content_type, content_texts in texts.items()
                    for idx, text in enumerate(content_texts)
                ]
            )

            file_name = f"test_content_{lang}"
            if not force and (output_dir / f"{file_name}.xlsx").exists():
                logger.warning(
                    f"Content file {file_name}.xlsx already exists, skipping..."
                )
                files[lang] = output_dir / f"{file_name}.xlsx"
                continue

            result = self.file_utils.save_data_to_storage(
                data={file_name: df},
                output_type="raw",
                file_name=file_name,
                output_filetype=OutputFileType.XLSX,
                include_timestamp=False,
            )

            saved_path = Path(next(iter(result[0].values())))
            files[lang] = saved_path

        return files

    def generate_parameter_files(self, force: bool = False) -> Dict[str, Path]:
        """Generate parameter files for both languages."""
        output_dir = self.file_utils.get_data_path("parameters")
        files = {}

        # Create parameter sheets for both languages
        for language in ["en", "fi"]:
            sheets = {
                # Required general parameters sheet
                ParameterSheets.get_sheet_name(
                    "general", language
                ): self._create_general_params(language),
                # Optional sheets
                ParameterSheets.get_sheet_name(
                    "keywords", language
                ): self._create_keywords_params(language),
                ParameterSheets.get_sheet_name(
                    "excluded", language
                ): self._create_excluded_params(language),
                ParameterSheets.get_sheet_name(
                    "categories", language
                ): self._create_categories_params(language),
                ParameterSheets.get_sheet_name(
                    "domains", language
                ): self._create_domains_params(language),
                ParameterSheets.get_sheet_name(
                    "settings", language
                ): self._create_settings_params(language),
            }

            file_path = self._save_parameters(
                language, sheets, output_dir, force
            )
            files[language] = file_path

        return files

    def _create_general_params(self, language: str) -> pd.DataFrame:
        """Create general parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("general", language)
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ][language]

        internal_params = {
            # Required parameters
            "column_name_to_analyze": "content",
            "focus_on": "general content analysis",
            "max_keywords": 10,
            # Optional parameters
            "language": language,
            "min_keyword_length": 3,
            "include_compounds": True,
            "max_themes": 3,
            "min_confidence": 0.3,
        }

        display_params = {}
        for internal_name, value in internal_params.items():
            display_name = next(
                excel_name
                for excel_name, mapped_name in param_mappings.items()
                if mapped_name == internal_name
            )
            display_params[display_name] = value

        return pd.DataFrame(
            {
                column_names["parameter"]: list(display_params.keys()),
                column_names["value"]: list(display_params.values()),
                column_names["description"]: [
                    (
                        f"Description for {k}"
                        if language == "en"
                        else f"Kuvaus: {k}"
                    )
                    for k in display_params.keys()
                ],
            }
        )

    def _create_excluded_params(self, language: str) -> pd.DataFrame:
        """Create excluded keywords DataFrame."""
        column_names = ParameterSheets.get_column_names("excluded", language)

        excluded_data = {
            "en": [
                ("the", "Common word"),
                ("and", "Common word"),
                ("with", "Common word"),
                ("for", "Common word"),
                ("new", "Common word"),
            ],
            "fi": [
                ("ja", "Yleinen sana"),
                ("tai", "Yleinen sana"),
                ("sekä", "Yleinen sana"),
                ("kanssa", "Yleinen sana"),
                ("uusi", "Yleinen sana"),
            ],
        }

        data = excluded_data[language]
        return pd.DataFrame(
            {
                column_names["keyword"]: [item[0] for item in data],
                column_names["reason"]: [item[1] for item in data],
            }
        )

    def _create_domains_params(self, language: str) -> pd.DataFrame:
        """Create domains parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("domains", language)

        domains_data = {
            "en": [
                (
                    "technical",
                    "Technical content",
                    "software,development,algorithm,system",
                    "Focus on technical implementation details",
                    "using,implementation,basic,simple",
                ),
                (
                    "business",
                    "Business content",
                    "revenue,growth,market,financial",
                    "Focus on business performance metrics",
                    "new,current,various,general",
                ),
            ],
            "fi": [
                (
                    "tekninen",
                    "Tekninen sisältö",
                    "ohjelmisto,kehitys,algoritmi,järjestelmä",
                    "Keskittyy teknisiin toteutusyksityiskohtiin",
                    "käyttäen,toteutus,perus,yksinkertainen",
                ),
                (
                    "liiketoiminta",
                    "Liiketoimintasisältö",
                    "liikevaihto,kasvu,markkina,talous",
                    "Keskittyy liiketoiminnan suorituskykyyn",
                    "uusi,nykyinen,erilainen,yleinen",
                ),
            ],
        }

        data = domains_data[language]
        return pd.DataFrame(
            {
                column_names["name"]: [item[0] for item in data],
                column_names["description"]: [item[1] for item in data],
                column_names["key_terms"]: [item[2] for item in data],
                column_names["context"]: [item[3] for item in data],
                column_names["stopwords"]: [item[4] for item in data],
            }
        )

    def _create_keywords_params(self, language: str) -> pd.DataFrame:
        """Create keywords parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("keywords", language)

        keywords_data = {
            "en": [
                ("machine learning", 1.0, "technical"),
                ("data analysis", 0.9, "technical"),
                ("cloud computing", 0.8, "technical"),
                ("business intelligence", 0.9, "business"),
                ("market analysis", 0.8, "business"),
            ],
            "fi": [
                ("koneoppiminen", 1.0, "tekninen"),
                ("data-analyysi", 0.9, "tekninen"),
                ("pilvipalvelut", 0.8, "tekninen"),
                ("liiketoimintatiedon hallinta", 0.9, "liiketoiminta"),
                ("markkina-analyysi", 0.8, "liiketoiminta"),
            ],
        }

        data = keywords_data[language]
        return pd.DataFrame(
            {
                column_names["keyword"]: [item[0] for item in data],
                column_names["importance"]: [item[1] for item in data],
                column_names["domain"]: [item[2] for item in data],
            }
        )

    def _create_categories_params(self, language: str) -> pd.DataFrame:
        """Create categories parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("categories", language)

        categories_data = {
            "en": [
                (
                    "technical",
                    "Technical content",
                    "machine learning,api,data,system",
                    0.6,
                ),
                (
                    "business",
                    "Business content",
                    "revenue,growth,market,profit",
                    0.6,
                ),
            ],
            "fi": [
                (
                    "tekninen",
                    "Tekninen sisältö",
                    "koneoppiminen,järjestelmä,rajapinta,data",
                    0.6,
                ),
                (
                    "liiketoiminta",
                    "Liiketoimintasisältö",
                    "liikevaihto,kasvu,markkina,tuotto",
                    0.6,
                ),
            ],
        }

        data = categories_data[language]
        return pd.DataFrame(
            {
                column_names["category"]: [item[0] for item in data],
                column_names["description"]: [item[1] for item in data],
                column_names["keywords"]: [item[2] for item in data],
                column_names["threshold"]: [item[3] for item in data],
            }
        )

    def _create_settings_params(self, language: str) -> pd.DataFrame:
        """Create settings parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("settings", language)
        param_mappings = ParameterSheets.PARAMETER_MAPPING["settings"][
            "parameters"
        ][language]

        settings = {
            "theme_analysis.min_confidence": 0.5,
            "weights.statistical": 0.4,
            "weights.llm": 0.6,
        }

        display_settings = {}
        for internal_name, value in settings.items():
            display_name = next(
                excel_name
                for excel_name, mapped_name in param_mappings.items()
                if mapped_name == internal_name
            )
            display_settings[display_name] = value

        return pd.DataFrame(
            {
                column_names["setting"]: list(display_settings.keys()),
                column_names["value"]: list(display_settings.values()),
                column_names["description"]: [
                    (
                        f"Description for {k}"
                        if language == "en"
                        else f"Kuvaus: {k}"
                    )
                    for k in display_settings.keys()
                ],
            }
        )

    def generate_test_content(self, force: bool = False) -> Dict[str, Path]:
        """Generate test content in both languages."""
        output_dir = self.file_utils.get_data_path("raw")
        files = {}

        content = {
            "en": {
                "technical": [
                    "Machine learning models are trained using large datasets to recognize patterns. Neural networks enable complex pattern recognition.",
                    "Cloud computing services provide scalable infrastructure for deployments. APIs enable system integration.",
                ],
                "business": [
                    "Q3 financial results show 15% revenue growth. Market expansion strategy focuses on emerging sectors.",
                    "Strategic partnerships drive innovation. Customer satisfaction metrics show positive trends.",
                ],
            },
            "fi": {
                "technical": [
                    "Koneoppimismallit koulutetaan suurilla datajoukolla tunnistamaan kaavoja. Neuroverkot mahdollistavat monimutkaisen hahmontunnistuksen.",
                    "Pilvipalvelut tarjoavat skaalautuvan infrastruktuurin. Rajapinnat mahdollistavat järjestelmäintegraation.",
                ],
                "business": [
                    "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun. Markkinalaajennusstrategia keskittyy uusiin sektoreihin.",
                    "Strategiset kumppanuudet edistävät innovaatiota. Asiakastyytyväisyysmittarit osoittavat positiivista kehitystä.",
                ],
            },
        }

        for lang, texts in content.items():
            df = pd.DataFrame(
                [
                    {
                        "id": f"{content_type}_{idx+1}",
                        "type": content_type,
                        "language": lang,
                        "content": text,
                    }
                    for content_type, content_texts in texts.items()
                    for idx, text in enumerate(content_texts)
                ]
            )

            file_name = f"test_content_{lang}"
            if not force and (output_dir / f"{file_name}.xlsx").exists():
                logger.warning(
                    f"Content file {file_name}.xlsx already exists, skipping..."
                )
                files[lang] = output_dir / f"{file_name}.xlsx"
                continue

            result = self.file_utils.save_data_to_storage(
                data={file_name: df},
                output_type="raw",
                file_name=file_name,
                output_filetype=OutputFileType.XLSX,
                include_timestamp=False,
            )

            saved_path = Path(next(iter(result[0].values())))
            files[lang] = saved_path

        return files

    def _save_parameters(
        self,
        language: str,
        sheets: Dict[str, pd.DataFrame],
        output_dir: Path,
        force: bool,
    ) -> Path:
        """Save parameter sheets to Excel file."""
        file_name = f"parameters_{language}"

        if not force and (output_dir / f"{file_name}.xlsx").exists():
            logger.warning(
                f"Parameter file {file_name}.xlsx already exists, skipping..."
            )
            return output_dir / f"{file_name}.xlsx"

        result = self.file_utils.save_data_to_storage(
            data=sheets,
            output_type="parameters",
            file_name=file_name,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=False,
        )

        saved_path = Path(next(iter(result[0].values())))

        # Validate generated file
        handler = ParameterHandler(saved_path)
        is_valid, warnings, errors = handler.validate()
        if not is_valid:
            raise ValueError(
                f"Generated parameter file {saved_path} failed validation: {errors}"
            )

        return saved_path


def main():
    """Generate all test data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test data for semantic analyzer"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite existing files"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generator = TestDataGenerator()

    try:
        files = generator.generate_all(force=args.force)
        print("\nGenerated files:")
        for file_type, paths in files.items():
            print(f"\n{file_type.title()}:")
            for path in paths:
                print(f"  - {path}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
