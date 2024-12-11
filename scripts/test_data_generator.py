# scripts/test_data_generator.py

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to Python path
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")

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

        files = {"parameters": [], "content": [], "metadata": []}

        try:
            # Generate parameter files
            logger.info("Generating parameter files...")
            param_files = self.generate_parameter_files(force=force)
            files["parameters"].extend(param_files.values())

            # Generate content files
            logger.info("Generating test content...")
            content_files = self.generate_test_content(force=force)
            files["content"].extend(content_files.values())

            # Generate metadata
            logger.info("Generating metadata...")
            metadata_file = self._save_metadata(files)
            files["metadata"].append(metadata_file)

            logger.info("Test data generation completed successfully")
            return files

        except Exception as e:
            logger.error(f"Error generating test data: {e}")
            raise

    def generate_parameter_files(self, force: bool = False) -> Dict[str, Path]:
        """Generate parameter files for both languages."""
        output_dir = self.file_utils.get_data_path("parameters")
        files = {}

        # English parameters
        en_sheets = {
            ParameterSheets.get_sheet_name(
                "general", "en"
            ): self._create_general_params("en"),
            ParameterSheets.get_sheet_name(
                "keywords", "en"
            ): self._create_keywords_params("en"),
            ParameterSheets.get_sheet_name(
                "excluded", "en"
            ): self._create_excluded_params("en"),
            ParameterSheets.get_sheet_name(
                "categories", "en"
            ): self._create_categories_params("en"),
            ParameterSheets.get_sheet_name(
                "domains", "en"
            ): self._create_domains_params("en"),
            ParameterSheets.get_sheet_name(
                "settings", "en"
            ): self._create_settings_params("en"),
        }

        # Finnish parameters
        fi_sheets = {
            ParameterSheets.get_sheet_name(
                "general", "fi"
            ): self._create_general_params("fi"),
            ParameterSheets.get_sheet_name(
                "keywords", "fi"
            ): self._create_keywords_params("fi"),
            ParameterSheets.get_sheet_name(
                "excluded", "fi"
            ): self._create_excluded_params("fi"),
            ParameterSheets.get_sheet_name(
                "categories", "fi"
            ): self._create_categories_params("fi"),
            ParameterSheets.get_sheet_name(
                "domains", "fi"
            ): self._create_domains_params("fi"),
            ParameterSheets.get_sheet_name(
                "settings", "fi"
            ): self._create_settings_params("fi"),
        }

        # Save files using FileUtils
        for lang, sheets in [("en", en_sheets), ("fi", fi_sheets)]:
            file_path = self._save_parameters(lang, sheets, output_dir, force)
            files[lang] = file_path

        return files

    # def _create_general_params(self, language: str) -> pd.DataFrame:
    #     """Create general parameters DataFrame."""
    #     column_names = ParameterSheets.get_column_names("general", language)
    #     param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
    #         "parameters"
    #     ][language]

    #     # Map internal names to display names
    #     params = {
    #         param_mappings.get("max_keywords", "max_keywords"): 8,
    #         param_mappings.get("language", "language"): language,
    #         param_mappings.get(
    #             "focus_on", "focus_on"
    #         ): "business and technical content analysis",
    #         param_mappings.get("min_keyword_length", "min_keyword_length"): 3,
    #         param_mappings.get("include_compounds", "include_compounds"): True,
    #         param_mappings.get(
    #             "column_name_to_analyze", "column_name_to_analyze"
    #         ): "content",
    #         param_mappings.get("max_themes", "max_themes"): 3,
    #         param_mappings.get("min_confidence", "min_confidence"): 0.3,
    #     }

    #     return pd.DataFrame(
    #         {
    #             column_names["parameter"]: list(params.keys()),
    #             column_names["value"]: list(params.values()),
    #             column_names["description"]: [
    #                 f"Description for {k}" for k in params.keys()
    #             ],
    #         }
    #     )
    def _create_general_params(self, language: str) -> pd.DataFrame:
        """Create general parameters DataFrame using ParameterSheets mappings."""
        column_names = ParameterSheets.get_column_names("general", language)
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"][
            "parameters"
        ][language]

        # Start with internal parameter names and values
        internal_params = {
            "max_keywords": 8,
            "language": language,
            "focus_on": "business and technical content analysis",
            "min_keyword_length": 3,
            "include_compounds": True,
            "column_name_to_analyze": "content",
            "max_themes": 3,
            "min_confidence": 0.3,
        }

        # Map internal names to display names using PARAMETER_MAPPING
        display_params = {}
        for internal_name, value in internal_params.items():
            # Find the display name from mappings
            display_name = next(
                excel_name
                for excel_name, mapped_name in param_mappings.items()
                if mapped_name == internal_name
            )
            display_params[display_name] = value

        # Create DataFrame with mapped column names from ParameterSheets
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

    def _create_keywords_params(self, language: str) -> pd.DataFrame:
        """Create keywords parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("keywords", language)

        if language == "fi":
            data = {
                column_names["keyword"]: [
                    "koneoppiminen",
                    "data-analyysi",
                    "pilvipalvelut",
                    "liiketoimintatiedon hallinta",
                    "markkina-analyysi",
                ],
                column_names["importance"]: [1.0, 0.9, 0.8, 0.9, 0.8],
                column_names["domain"]: [
                    "tekninen",
                    "tekninen",
                    "tekninen",
                    "liiketoiminta",
                    "liiketoiminta",
                ],
            }
        else:
            data = {
                column_names["keyword"]: [
                    "machine learning",
                    "data analysis",
                    "cloud computing",
                    "business intelligence",
                    "market analysis",
                ],
                column_names["importance"]: [1.0, 0.9, 0.8, 0.9, 0.8],
                column_names["domain"]: [
                    "technical",
                    "technical",
                    "technical",
                    "business",
                    "business",
                ],
            }

        return pd.DataFrame(data)

    def _create_excluded_params(self, language: str) -> pd.DataFrame:
        """Create excluded keywords DataFrame."""
        column_names = ParameterSheets.get_column_names("excluded", language)

        if language == "fi":
            data = {
                column_names["keyword"]: [
                    "ja",
                    "tai",
                    "sekä",
                    "kanssa",
                    "uusi",
                ],
                column_names["reason"]: ["Yleinen sana"] * 5,
            }
        else:
            data = {
                column_names["keyword"]: ["the", "and", "with", "for", "new"],
                column_names["reason"]: ["Common word"] * 5,
            }

        return pd.DataFrame(data)

    def _create_categories_params(self, language: str) -> pd.DataFrame:
        """Create categories parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("categories", language)

        if language == "fi":
            data = {
                column_names["category"]: [
                    "tekninen_sisältö",
                    "liiketoiminta_sisältö",
                ],
                column_names["description"]: [
                    "Tekninen ja ohjelmistokehitys sisältö",
                    "Liiketoiminta ja talousanalyysi sisältö",
                ],
                column_names["keywords"]: [
                    "ohjelmisto,kehitys,rajapinta,ohjelmointi",
                    "liikevaihto,myynti,markkina,kasvu",
                ],
                column_names["threshold"]: [0.6, 0.6],
                column_names["parent"]: [None, None],
            }
        else:
            data = {
                column_names["category"]: [
                    "technical_content",
                    "business_content",
                ],
                column_names["description"]: [
                    "Technical and software development content",
                    "Business and financial analysis content",
                ],
                column_names["keywords"]: [
                    "software,development,api,programming",
                    "revenue,sales,market,growth",
                ],
                column_names["threshold"]: [0.6, 0.6],
                column_names["parent"]: [None, None],
            }

        return pd.DataFrame(data)

    def _create_domains_params(self, language: str) -> pd.DataFrame:
        """Create domains parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("domains", language)

        if language == "fi":
            data = {
                column_names["name"]: ["tekninen", "liiketoiminta"],
                column_names["description"]: [
                    "Tekninen ja ohjelmistokehitys sisältö",
                    "Liiketoiminta ja talousanalyysi sisältö",
                ],
                column_names["key_terms"]: [
                    "ohjelmisto,kehitys,algoritmi,järjestelmä",
                    "liikevaihto,kasvu,markkina,talous",
                ],
                column_names["context"]: [
                    "Keskittyy teknisiin toteutusyksityiskohtiin",
                    "Keskittyy liiketoiminnan suorituskykyyn",
                ],
                column_names["stopwords"]: [
                    "käyttäen,toteutus,perus,yksinkertainen",
                    "uusi,nykyinen,erilainen,yleinen",
                ],
            }
        else:
            data = {
                column_names["name"]: ["technical", "business"],
                column_names["description"]: [
                    "Technical and software development content",
                    "Business and financial content",
                ],
                column_names["key_terms"]: [
                    "software,development,algorithm,system",
                    "revenue,growth,market,financial",
                ],
                column_names["context"]: [
                    "Focus on technical implementation details",
                    "Focus on business performance metrics",
                ],
                column_names["stopwords"]: [
                    "using,implementation,basic,simple",
                    "new,current,various,general",
                ],
            }

        return pd.DataFrame(data)

    def _create_settings_params(self, language: str) -> pd.DataFrame:
        """Create settings parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("settings", language)
        param_mappings = ParameterSheets.PARAMETER_MAPPING["settings"][
            "parameters"
        ][language]

        settings = [
            ("theme_analysis.min_confidence", 0.5),
            ("weights.statistical", 0.4),
            ("weights.llm", 0.6),
        ]

        return pd.DataFrame(
            {
                column_names["setting"]: [
                    param_mappings.get(s[0], s[0]) for s in settings
                ],
                column_names["value"]: [s[1] for s in settings],
                column_names["description"]: [
                    f"Description for {s[0]}" for s in settings
                ],
            }
        )

    def _save_parameters(
        self,
        language: str,
        sheets: Dict[str, pd.DataFrame],
        output_dir: Path,
        force: bool,
    ) -> Path:
        """Save parameter sheets to Excel file."""
        file_name = f"parameters_{language}"
        file_path = output_dir / f"{file_name}.xlsx"

        if not force and file_path.exists():
            logger.warning(
                f"Parameter file {file_path} already exists, skipping..."
            )
            return file_path

        result = self.file_utils.save_data_to_storage(
            data=sheets,
            output_type="parameters",
            file_name=file_name,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=False,
        )

        # Get first file path from result
        saved_path = Path(next(iter(result[0].values())))

        # Validate generated file
        handler = ParameterHandler(saved_path)
        is_valid, warnings, errors = handler.validate()

        if not is_valid:
            raise ValueError(
                f"Generated parameter file {saved_path} failed validation: {errors}"
            )

        return saved_path

    def generate_test_content(self, force: bool = False) -> Dict[str, Path]:
        """Generate test content in both languages."""
        output_dir = self.file_utils.get_data_path("raw")
        files = {}

        # Define test content with category-focused examples
        content = {
            "en": {
                "technical": [
                    "Machine learning models are trained using large datasets to recognize patterns. The neural network architecture includes multiple layers for feature extraction. Data preprocessing and feature engineering are crucial steps in the pipeline.",
                    "Cloud computing services provide scalable infrastructure for deployments. Microservices architecture enables modular and maintainable system design.",
                    "Version control systems track changes in source code repositories. Continuous integration ensures code quality and automated testing.",
                ],
                "business": [
                    "Q3 financial results show 15% revenue growth and improved profit margins. Customer acquisition costs decreased while retention rates increased.",
                    "Strategic partnerships drive innovation and market penetration. Investment in R&D resulted in three new product launches.",
                    "Operational efficiency improved through process automation. Customer satisfaction metrics show positive trend year-over-year.",
                ],
            },
            "fi": {
                "technical": [
                    "Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja. Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen.",
                    "Pilvipalvelut tarjoavat skaalautuvan infrastruktuurin käyttöönottoon. Mikropalveluarkkitehtuuri mahdollistaa modulaarisen järjestelmäsuunnittelun.",
                    "Versionhallintajärjestelmät seuraavat lähdekoodin muutoksia. Jatkuva integraatio varmistaa koodin laadun ja automaattitestauksen.",
                ],
                "business": [
                    "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet. Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani.",
                    "Strategiset kumppanuudet edistävät innovaatiota ja markkinapenetraatiota. T&K-investoinnit johtivat kolmeen uuteen tuotelanseeraukseen.",
                    "Toiminnan tehokkuus parani prosessiautomaation avulla. Asiakastyytyväisyysmittarit osoittavat positiivista kehitystä.",
                ],
            },
        }

        # Save content files
        for lang, texts in content.items():
            # Create rows for DataFrame
            # Create rows for DataFrame
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

            # Get first file path from result
            saved_path = Path(next(iter(result[0].values())))
            files[lang] = saved_path

        return files

    def _save_metadata(self, files: Dict[str, List[Path]]) -> Path:
        """Save metadata about generated files."""
        metadata = {
            "files": {
                file_type: [str(p) for p in paths]
                for file_type, paths in files.items()
            },
            "generation_time": pd.Timestamp.now().isoformat(),
            "parameter_sheets": {
                name: ParameterSheets.get_sheet_name(name, "en")
                for name in [
                    "general",
                    "keywords",
                    "excluded",
                    "categories",
                    "domains",
                    "settings",
                ]
            },
        }

        # Convert metadata to DataFrame for storage
        metadata_df = pd.DataFrame([metadata])

        result, _ = self.file_utils.save_data_to_storage(
            data=metadata_df,
            output_filetype=OutputFileType.XLSX,  # or CSV if preferred
            output_type="parameters",
            file_name="test_data_metadata",
            include_timestamp=False,
        )

        # Return first saved path
        return Path(next(iter(result.values())))


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

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
