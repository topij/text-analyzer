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

        # Define topics and their content
        topics = {
            "business": {
                "en": [
                    "Q3 financial results show 15% revenue growth. Market expansion strategy focuses on emerging sectors.",
                    "Strategic partnerships drive innovation. Customer satisfaction metrics show positive trends.",
                    "New digital transformation initiative launched to improve operational efficiency.",
                    "Market analysis indicates strong growth potential in APAC region.",
                ],
                "fi": [
                    "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun. Markkinalaajennusstrategia keskittyy uusiin sektoreihin.",
                    "Strategiset kumppanuudet edistävät innovaatiota. Asiakastyytyväisyysmittarit osoittavat positiivista kehitystä.",
                    "Uusi digitaalinen transformaatiohanke käynnistetty toiminnan tehostamiseksi.",
                    "Markkina-analyysi osoittaa vahvaa kasvupotentiaalia APAC-alueella.",
                ]
            },
            "support": {
                "en": [
                    "I'm having trouble logging into the admin dashboard. The system keeps showing 'Invalid credentials' even though I'm sure the password is correct.",
                    "The export functionality is not working in the reporting module. When I click export to CSV, nothing happens.",
                    "Getting error code E1234 when trying to sync data between mobile and desktop apps. Need urgent assistance.",
                    "Can't access the API documentation. The link in the developer portal returns a 404 error.",
                ],
                "fi": [
                    "Minulla on ongelmia kirjautua hallintapaneeliin. Järjestelmä näyttää 'Virheelliset tunnukset' vaikka salasana on varmasti oikein.",
                    "Raporttien vientiominaisuus ei toimi. Kun klikkaan 'Vie CSV-muotoon', mitään ei tapahdu.",
                    "Saan virhekoodin E1234 yrittäessäni synkronoida tietoja mobiili- ja työpöytäsovellusten välillä. Tarvitsen kiireellistä apua.",
                    "En pääse käsiksi API-dokumentaatioon. Kehittäjäportaalin linkki palauttaa 404-virheen.",
                ]
            },
            "training": {
                "en": [
                    "Hi, I'm interested in the Advanced Data Science course. What are the prerequisites and when does the next cohort start?",
                    "Could you provide more information about the certification process for the Cloud Architecture program?",
                    "I need to reschedule my upcoming Python Programming workshop. What's the process for this?",
                    "Are there any group discounts available for corporate training packages? We have a team of 10 developers.",
                ],
                "fi": [
                    "Hei, olen kiinnostunut Edistynyt Data Science -kurssista. Mitkä ovat esitietovaatimukset ja milloin seuraava ryhmä alkaa?",
                    "Voisitteko kertoa lisätietoja Cloud Architecture -ohjelman sertifiointiprosessista?",
                    "Minun täytyy siirtää tuleva Python-ohjelmoinnin työpaja. Mikä on prosessi tähän?",
                    "Onko yrityskoulutuspaketteihin saatavilla ryhmäalennuksia? Meillä on 10 hengen kehittäjätiimi.",
                ]
            }
        }

        # Generate content files for each topic and language
        for topic, content in topics.items():
            for lang, texts in content.items():
                df = pd.DataFrame(
                    [
                        {
                            "id": f"{topic}_{idx+1}",
                            "type": topic,
                            "language": lang,
                            "content": text,
                        }
                        for idx, text in enumerate(texts)
                    ]
                )

                file_name = f"{topic}_test_content_{lang}"
                if not force and (output_dir / f"{file_name}.xlsx").exists():
                    logger.warning(
                        f"Content file {file_name}.xlsx already exists, skipping..."
                    )
                    files[f"{topic}_{lang}"] = output_dir / f"{file_name}.xlsx"
                    continue

                result = self.file_utils.save_data_to_storage(
                    data={file_name: df},
                    output_type="raw",
                    file_name=file_name,
                    output_filetype=OutputFileType.XLSX,
                    include_timestamp=False,
                )

                saved_path = Path(next(iter(result[0].values())))
                files[f"{topic}_{lang}"] = saved_path

        return files

    def generate_parameter_files(self, force: bool = False) -> Dict[str, Path]:
        """Generate parameter files for each topic and language."""
        output_dir = self.file_utils.get_data_path("parameters")
        files = {}

        # Define topics and their parameters
        topics = ["business", "support", "training"]
        
        for topic in topics:
            for language in ["en", "fi"]:
                sheets = {
                    # Required general parameters sheet
                    ParameterSheets.get_sheet_name(
                        "general", language
                    ): self._create_general_params(language, topic),
                    # Optional sheets
                    ParameterSheets.get_sheet_name(
                        "keywords", language
                    ): self._create_keywords_params(language, topic),
                    ParameterSheets.get_sheet_name(
                        "excluded", language
                    ): self._create_excluded_params(language, topic),
                    ParameterSheets.get_sheet_name(
                        "categories", language
                    ): self._create_categories_params(language, topic),
                    ParameterSheets.get_sheet_name(
                        "domains", language
                    ): self._create_domains_params(language, topic),
                    ParameterSheets.get_sheet_name(
                        "settings", language
                    ): self._create_settings_params(language),
                }

                file_name = f"{topic}_parameters_{language}"
                if not force and (output_dir / f"{file_name}.xlsx").exists():
                    logger.warning(
                        f"Parameter file {file_name}.xlsx already exists, skipping..."
                    )
                    files[f"{topic}_{language}"] = output_dir / f"{file_name}.xlsx"
                    continue

                result = self.file_utils.save_data_to_storage(
                    data=sheets,
                    output_type="parameters",
                    file_name=file_name,
                    output_filetype=OutputFileType.XLSX,
                    include_timestamp=False,
                )

                saved_path = Path(next(iter(result[0].values())))
                files[f"{topic}_{language}"] = saved_path

                # Validate generated file
                handler = ParameterHandler(saved_path)
                is_valid, warnings, errors = handler.validate()
                if not is_valid:
                    raise ValueError(
                        f"Generated parameter file {saved_path} failed validation: {errors}"
                    )

        return files

    def _create_general_params(self, language: str, topic: str) -> pd.DataFrame:
        """Create general parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("general", language)
        param_mappings = ParameterSheets.PARAMETER_MAPPING["general"]["parameters"][language]

        # Topic-specific focus areas
        focus_areas = {
            "business": "business content analysis",
            "support": "technical support analysis",
            "training": "training services analysis"
        }

        internal_params = {
            # Required parameters
            "column_name_to_analyze": "content",
            "focus_on": focus_areas.get(topic, "general content analysis"),
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
                    f"Parameter for {topic} analysis: {k}"
                    if language == "en"
                    else f"Parametri {topic}-analyysille: {k}"
                    for k in display_params.keys()
                ],
            }
        )

    def _create_excluded_params(self, language: str, topic: str) -> pd.DataFrame:
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

    def _create_domains_params(self, language: str, topic: str) -> pd.DataFrame:
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

    def _create_keywords_params(self, language: str, topic: str) -> pd.DataFrame:
        """Create keywords parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("keywords", language)

        # Topic-specific keywords
        topic_keywords = {
            "business": {
                "en": [
                    ("revenue", 0.9, "Financial"),
                    ("growth", 0.9, "Business"),
                    ("market", 0.8, "Business"),
                    ("strategy", 0.8, "Strategy"),
                    ("metrics", 0.7, "Analytics"),
                    ("performance", 0.8, "Business"),
                    ("innovation", 0.7, "Strategy"),
                    ("transformation", 0.8, "Strategy"),
                ],
                "fi": [
                    ("liikevaihto", 0.9, "Talous"),
                    ("kasvu", 0.9, "Liiketoiminta"),
                    ("markkina", 0.8, "Liiketoiminta"),
                    ("strategia", 0.8, "Strategia"),
                    ("mittarit", 0.7, "Analytiikka"),
                    ("suorituskyky", 0.8, "Liiketoiminta"),
                    ("innovaatio", 0.7, "Strategia"),
                    ("transformaatio", 0.8, "Strategia"),
                ]
            },
            "support": {
                "en": [
                    ("login", 0.9, "Authentication"),
                    ("error", 0.8, "System Issues"),
                    ("bug", 0.8, "System Issues"),
                    ("access", 0.8, "Authentication"),
                    ("credentials", 0.8, "Authentication"),
                    ("sync", 0.7, "System Features"),
                    ("documentation", 0.7, "Resources"),
                    ("API", 0.8, "Development"),
                ],
                "fi": [
                    ("kirjautuminen", 0.9, "Tunnistautuminen"),
                    ("virhe", 0.8, "Järjestelmäongelmat"),
                    ("vika", 0.8, "Järjestelmäongelmat"),
                    ("pääsy", 0.8, "Tunnistautuminen"),
                    ("tunnukset", 0.8, "Tunnistautuminen"),
                    ("synkronointi", 0.7, "Järjestelmäominaisuudet"),
                    ("dokumentaatio", 0.7, "Resurssit"),
                    ("rajapinta", 0.8, "Kehitys"),
                ]
            },
            "training": {
                "en": [
                    ("course", 0.9, "Education"),
                    ("workshop", 0.8, "Education"),
                    ("certification", 0.9, "Education"),
                    ("prerequisites", 0.7, "Education"),
                    ("training", 0.8, "Education"),
                    ("schedule", 0.7, "Planning"),
                    ("enrollment", 0.8, "Administration"),
                    ("discount", 0.7, "Pricing"),
                ],
                "fi": [
                    ("kurssi", 0.9, "Koulutus"),
                    ("työpaja", 0.8, "Koulutus"),
                    ("sertifiointi", 0.9, "Koulutus"),
                    ("esitietovaatimukset", 0.7, "Koulutus"),
                    ("koulutus", 0.8, "Koulutus"),
                    ("aikataulu", 0.7, "Suunnittelu"),
                    ("ilmoittautuminen", 0.8, "Hallinto"),
                    ("alennus", 0.7, "Hinnoittelu"),
                ]
            }
        }

        data = topic_keywords.get(topic, {}).get(language, [])
        return pd.DataFrame(
            {
                column_names["keyword"]: [item[0] for item in data],
                column_names["importance"]: [item[1] for item in data],
                column_names["domain"]: [item[2] for item in data],
            }
        )

    def _create_categories_params(self, language: str, topic: str) -> pd.DataFrame:
        """Create categories parameters DataFrame."""
        column_names = ParameterSheets.get_column_names("categories", language)

        # Topic-specific categories
        topic_categories = {
            "business": {
                "en": [
                    ("financial_performance", "Financial results and metrics", "revenue,growth,profit,margin,performance", 0.6),
                    ("market_strategy", "Market and strategy related", "market,strategy,expansion,innovation", 0.6),
                    ("operational_efficiency", "Operational improvements", "efficiency,optimization,transformation,process", 0.6),
                ],
                "fi": [
                    ("taloudellinen_tulos", "Taloudelliset tulokset ja mittarit", "liikevaihto,kasvu,tuotto,kannattavuus", 0.6),
                    ("markkinastrategia", "Markkinoihin ja strategiaan liittyvä", "markkina,strategia,laajennus,innovaatio", 0.6),
                    ("toiminnan_tehokkuus", "Toiminnan parannukset", "tehokkuus,optimointi,transformaatio,prosessi", 0.6),
                ]
            },
            "support": {
                "en": [
                    ("login_issues", "Authentication and access problems", "login,password,credentials,authentication,access", 0.6),
                    ("system_errors", "Technical errors and system failures", "error,bug,crash,failure,malfunction", 0.6),
                    ("documentation_issues", "Documentation and API problems", "documentation,api,guide,manual,reference", 0.6),
                ],
                "fi": [
                    ("kirjautumisongelmat", "Tunnistautumis- ja käyttöoikeusongelmat", "kirjautuminen,salasana,tunnukset,pääsy", 0.6),
                    ("järjestelmävirheet", "Tekniset virheet ja häiriöt", "virhe,vika,kaatuminen,häiriö", 0.6),
                    ("dokumentaatio-ongelmat", "Dokumentaatio- ja API-ongelmat", "dokumentaatio,api,ohje,manuaali", 0.6),
                ]
            },
            "training": {
                "en": [
                    ("course_inquiry", "Course information requests", "course,training,workshop,program,schedule", 0.6),
                    ("certification", "Certification related inquiries", "certification,qualification,exam,assessment", 0.6),
                    ("pricing_inquiry", "Pricing and payment questions", "price,cost,payment,discount,fee", 0.6),
                ],
                "fi": [
                    ("kurssitiedustelu", "Kurssitietojen kyselyt", "kurssi,koulutus,työpaja,ohjelma,aikataulu", 0.6),
                    ("sertifiointi", "Sertifiointiin liittyvät kyselyt", "sertifiointi,pätevyys,koe,arviointi", 0.6),
                    ("hintatiedustelu", "Hinnoittelu- ja maksukysymykset", "hinta,kustannus,maksu,alennus", 0.6),
                ]
            }
        }

        data = topic_categories.get(topic, {}).get(language, [])
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
        param_mappings = ParameterSheets.PARAMETER_MAPPING["settings"]["parameters"][language]

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
    """Run test data generation."""
    import argparse
    from src.config.manager import ConfigManager
    from src.core.managers.environment_manager import EnvironmentManager, EnvironmentConfig
    from FileUtils import FileUtils

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate test data for semantic analyzer")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    args = parser.parse_args()

    try:
        # Create FileUtils instance
        project_root = Path(__file__).resolve().parent.parent
        file_utils = FileUtils(project_root=project_root)

        # Initialize environment with ConfigManager
        config_manager = ConfigManager(file_utils=file_utils)
        config_manager.init_environment()
        config_manager.init_paths()
        config_manager.init_file_utils()
        config_manager.load_configurations()

        # Initialize environment manager with shared FileUtils
        env_config = EnvironmentConfig(
            env_type="development",
            project_root=project_root,
            log_level="INFO"
        )
        environment = EnvironmentManager(env_config)
        
        # Set up shared components
        environment.file_utils = file_utils
        environment.config_manager = config_manager
        
        # Make sure the environment is properly initialized
        EnvironmentManager._instance = environment

        # Generate test data using the shared environment
        logger.info("Initializing project structure...")
        generator = TestDataGenerator(file_utils=file_utils)
        files = generator.generate_all(force=args.force)
        logger.info(f"Generated files: {files}")
        return 0

    except Exception as e:
        logger.error(f"Error during project setup: {str(e)}")
        return 1


if __name__ == "__main__":
    main()
