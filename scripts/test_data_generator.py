# scripts/test_data_generator.py

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to Python path
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")


from src.loaders.models import CategoryConfig, GeneralParameters

# Update imports
from src.loaders.parameter_config import ParameterSheets
from src.loaders.parameter_handler import ParameterHandler
from FileUtils import FileUtils, OutputFileType

# from src.loaders.parameter_config import ParameterSheets, ParameterConfigurations

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
        # Update parameter file generation to match new structure
        # Rest of implementation...

        """Generate parameter files in both languages."""
        output_dir = self.file_utils.get_data_path("parameters")
        files = {}

        # English parameters
        en_sheets = {
            ParameterSheets.GENERAL_EN.value: pd.DataFrame(
                {
                    "parameter": [
                        "max_keywords",
                        "language",
                        "focus_on",
                        "min_keyword_length",
                        "include_compounds",
                        "column_name_to_analyze",
                        "max_themes",
                        "min_confidence",
                    ],
                    "value": [
                        8,
                        "en",
                        "business and technical content analysis",
                        3,
                        True,
                        "content",
                        3,
                        0.3,
                    ],
                    "description": [
                        "Maximum keywords to extract",
                        "Content language",
                        "Analysis focus area",
                        "Minimum keyword length",
                        "Handle compound words",
                        "Content column name",
                        "Maximum themes to identify",
                        "Minimum confidence threshold",
                    ],
                }
            ),
            ParameterSheets.KEYWORDS_EN.value: pd.DataFrame(
                {
                    "keyword": [
                        "machine learning",
                        "data analysis",
                        "cloud computing",
                        "business intelligence",
                        "market analysis",
                        "revenue growth",
                    ],
                    "importance": [1.0, 0.9, 0.8, 0.9, 0.8, 0.9],
                    "domain": [
                        "technical",
                        "technical",
                        "technical",
                        "business",
                        "business",
                        "business",
                    ],
                }
            ),
            ParameterSheets.EXCLUDED_EN.value: pd.DataFrame(
                {
                    "keyword": ["the", "and", "with", "for", "new", "using"],
                    "reason": ["Common word"] * 6,
                }
            ),
            # New: Categories sheet
            ParameterSheets.CATEGORIES_EN.value: pd.DataFrame(
                {
                    "category": [
                        "technical_content",
                        "business_content",
                        "educational_content",
                        "online_learning",
                        "classroom_learning",
                        "professional_development",
                    ],
                    "description": [
                        "Technical and software development content",
                        "Business and financial analysis content",
                        "General educational and learning content",
                        "Online and e-learning specific content",
                        "Traditional classroom learning content",
                        "Professional skills development content",
                    ],
                    "keywords": [
                        "software,development,api,programming,technical,code",
                        "revenue,sales,market,growth,business,financial",
                        "learning,education,training,teaching,curriculum",
                        "elearning,virtual,remote,online,digital",
                        "classroom,lecture,workshop,in-person,hands-on",
                        "skills,competence,career,professional,development",
                    ],
                    "threshold": [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                    "parent": [
                        None,
                        None,
                        None,
                        "educational_content",
                        "educational_content",
                        None,
                    ],
                }
            ),
            ParameterSheets.DOMAINS_EN.value: pd.DataFrame(
                {
                    "name": ["technical", "business", "educational"],
                    "description": [
                        "Technical and software development content",
                        "Business and financial content",
                        "Educational and training content",
                    ],
                    "key_terms": [
                        "software,development,algorithm,system",
                        "revenue,growth,market,financial",
                        "learning,training,education,skills",
                    ],
                    "context": [
                        "Focus on technical implementation details",
                        "Focus on business performance metrics",
                        "Focus on learning and development aspects",
                    ],
                    "stopwords": [
                        "using,implementation,basic,simple",
                        "new,current,various,general",
                        "basic,simple,just,like",
                    ],
                }
            ),
            ParameterSheets.SETTINGS_EN.value: pd.DataFrame(
                {
                    "setting": [
                        # Theme analysis settings
                        "theme_analysis.enabled",
                        "theme_analysis.min_confidence",
                        # Weights for different analysis components
                        "weights.statistical",
                        "weights.llm",
                        "weights.compound_bonus",
                        "weights.domain_bonus",
                        # Analysis thresholds
                        "thresholds.keyword_similarity",
                        "thresholds.theme_relevance",
                        "thresholds.category_match",
                    ],
                    "value": [
                        True,  # theme_analysis.enabled
                        0.5,  # theme_analysis.min_confidence
                        0.4,  # weights.statistical
                        0.6,  # weights.llm
                        0.2,  # weights.compound_bonus
                        0.15,  # weights.domain_bonus
                        0.85,  # thresholds.keyword_similarity
                        0.6,  # thresholds.theme_relevance
                        0.5,  # thresholds.category_match
                    ],
                    "description": [
                        "Enable theme analysis",
                        "Minimum confidence for theme detection",
                        "Weight for statistical analysis",
                        "Weight for LLM analysis",
                        "Bonus weight for compound words",
                        "Bonus weight for domain matches",
                        "Minimum similarity for keyword grouping",
                        "Minimum relevance for theme evidence",
                        "Minimum threshold for category matches",
                    ],
                }
            ),
        }

        # Finnish parameters
        fi_sheets = {
            ParameterSheets.GENERAL_FI.value: pd.DataFrame(
                {
                    "parametri": [
                        "maksimi_avainsanat",
                        "kieli",
                        "keskity_aihepiiriin",
                        "min_sanan_pituus",
                        "sisällytä_yhdyssanat",
                        "analysoitava_sarake",
                        "maksimi_teemat",
                        "min_luottamus",
                    ],
                    "arvo": [
                        8,
                        "fi",
                        "tekninen ja liiketoiminta-analyysi",
                        3,
                        True,
                        "sisältö",
                        3,
                        0.3,
                    ],
                    "kuvaus": [
                        "Poimittavien avainsanojen maksimimäärä",
                        "Sisällön kieli",
                        "Analyysin painopistealue",
                        "Avainsanan minimipituus",
                        "Käsittele yhdyssanat",
                        "Analysoitavan sisällön sarake",
                        "Maksimi teemojen määrä",
                        "Minimivarmuustaso",
                    ],
                }
            ),
            ParameterSheets.KEYWORDS_FI.value: pd.DataFrame(
                {
                    "avainsana": [
                        "koneoppiminen",
                        "data-analyysi",
                        "pilvipalvelut",
                        "liiketoimintatiedon hallinta",
                        "markkina-analyysi",
                        "liikevaihdon kasvu",
                    ],
                    "tärkeys": [1.0, 0.9, 0.8, 0.9, 0.8, 0.9],
                    "aihepiiri": [
                        "tekninen",
                        "tekninen",
                        "tekninen",
                        "liiketoiminta",
                        "liiketoiminta",
                        "liiketoiminta",
                    ],
                }
            ),
            ParameterSheets.EXCLUDED_FI.value: pd.DataFrame(
                {
                    "avainsana": [
                        "ja",
                        "tai",
                        "sekä",
                        "kanssa",
                        "uusi",
                        "käyttäen",
                    ],
                    "syy": ["Yleinen sana"] * 6,
                }
            ),
            # New: Finnish Categories sheet
            ParameterSheets.CATEGORIES_FI.value: pd.DataFrame(
                {
                    "kategoria": [
                        "tekninen_sisältö",
                        "liiketoiminta_sisältö",
                        "koulutussisältö",
                        "verkko_oppiminen",
                        "lähiopetus",
                        "ammatillinen_kehitys",
                    ],
                    "kuvaus": [
                        "Tekninen ja ohjelmistokehitys sisältö",
                        "Liiketoiminta ja talousanalyysi sisältö",
                        "Yleinen koulutus ja oppimissisältö",
                        "Verkko-oppimisen ja e-oppimisen sisältö",
                        "Perinteinen lähiopetussisältö",
                        "Ammatillisen osaamisen kehittämissisältö",
                    ],
                    "avainsanat": [
                        "ohjelmisto,kehitys,rajapinta,ohjelmointi,tekninen,koodi",
                        "liikevaihto,myynti,markkina,kasvu,liiketoiminta,talous",
                        "oppiminen,koulutus,opetus,opetussuunnitelma",
                        "verkko-oppiminen,virtuaalinen,etä,online,digitaalinen",
                        "lähiopetus,luento,työpaja,läsnäolo,käytäntö",
                        "osaaminen,pätevyys,ura,ammatillinen,kehitys",
                    ],
                    "kynnysarvo": [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                    "yläkategoria": [
                        None,
                        None,
                        None,
                        "koulutussisältö",
                        "koulutussisältö",
                        None,
                    ],
                }
            ),
            ParameterSheets.DOMAINS_FI.value: pd.DataFrame(
                {
                    "nimi": ["tekninen", "liiketoiminta", "koulutus"],
                    "kuvaus": [
                        "Tekninen ja ohjelmistokehityssisältö",
                        "Liiketoiminta ja taloussisältö",
                        "Koulutus ja oppimissisältö",
                    ],
                    "keskeiset_termit": [
                        "ohjelmisto,kehitys,algoritmi,järjestelmä",
                        "liikevaihto,kasvu,markkina,talous",
                        "oppiminen,koulutus,opetus,osaaminen",
                    ],
                    "konteksti": [
                        "Keskittyy teknisiin toteutusyksityiskohtiin",
                        "Keskittyy liiketoiminnan suorituskykyyn",
                        "Keskittyy oppimiseen ja kehitykseen",
                    ],
                    "ohitettavat_sanat": [
                        "käyttäen,toteutus,perus,yksinkertainen",
                        "uusi,nykyinen,erilainen,yleinen",
                        "perus,yksinkertainen,vain,kuten",
                    ],
                }
            ),
            ParameterSheets.SETTINGS_FI.value: pd.DataFrame(
                {
                    "asetus": [
                        # Theme analysis settings
                        "teema_analyysi.käytössä",
                        "teema_analyysi.min_luottamus",
                        # Weights for different components
                        "painot.tilastollinen",
                        "painot.llm",
                        "painot.yhdyssana_bonus",
                        "painot.aihepiiri_bonus",
                        # Analysis thresholds
                        "kynnysarvot.avainsana_samankaltaisuus",
                        "kynnysarvot.teema_relevanssi",
                        "kynnysarvot.kategoria_osuma",
                    ],
                    "arvo": [
                        True,  # teema_analyysi.käytössä
                        0.5,  # teema_analyysi.min_luottamus
                        0.4,  # painot.tilastollinen
                        0.6,  # painot.llm
                        0.2,  # painot.yhdyssana_bonus
                        0.15,  # painot.aihepiiri_bonus
                        0.85,  # kynnysarvot.avainsana_samankaltaisuus
                        0.6,  # kynnysarvot.teema_relevanssi
                        0.5,  # kynnysarvot.kategoria_osuma
                    ],
                    "kuvaus": [
                        "Teema-analyysin käyttö",
                        "Teeman tunnistuksen minimivarmuus",
                        "Tilastollisen analyysin paino",
                        "LLM-analyysin paino",
                        "Yhdyssanojen bonuspaino",
                        "Aihepiirivastaavuuden bonuspaino",
                        "Avainsanojen samankaltaisuusraja",
                        "Teemojen relevanssikynnys",
                        "Kategoriaosuman minimikynnys",
                    ],
                }
            ),
        }

        # Save files using FileUtils
        for lang, sheets in [("en", en_sheets), ("fi", fi_sheets)]:
            file_path = self._save_parameters(lang, sheets, output_dir, force)
            files[lang] = file_path

        return files

    def generate_test_content(self, force: bool = False) -> Dict[str, Path]:
        """Generate test content in both languages."""
        output_dir = self.file_utils.get_data_path("raw")
        files = {}

        # Define test content with category-focused examples
        content = {
            "en": {
                "technical": [
                    "Machine learning models are trained using large datasets to recognize patterns. The neural network architecture includes multiple layers for feature extraction. Data preprocessing and feature engineering are crucial steps in the pipeline.",
                    "Cloud computing services provide scalable infrastructure for deployments. Microservices architecture enables modular and maintainable system design. API endpoints handle authentication and data validation requirements.",
                    "Version control systems track changes in source code repositories. Continuous integration ensures code quality and automated testing. Documentation covers API usage and system architecture details.",
                ],
                "business": [
                    "Q3 financial results show 15% revenue growth and improved profit margins. Customer acquisition costs decreased while retention rates increased. Market expansion strategy focuses on emerging technology sectors.",
                    "Strategic partnerships drive innovation and market penetration. Investment in R&D resulted in three new product launches. Sales performance exceeded targets in key market segments.",
                    "Operational efficiency improved through process automation. Customer satisfaction metrics show positive trend year-over-year. Cost optimization initiatives delivered 12% savings in Q4.",
                ],
                "educational": [
                    "The online learning platform features interactive modules and self-paced progress tracking. Virtual classrooms enable real-time collaboration between students and instructors. Digital assessment tools provide immediate feedback on learning outcomes.",
                    "Classroom workshops combine theoretical concepts with hands-on exercises. Small group activities promote peer learning and knowledge sharing. Practice sessions reinforce key learning objectives.",
                    "Professional development programs focus on industry-relevant skills. Mentoring sessions provide guidance for career advancement. Competency assessments track progress towards learning goals.",
                ],
            },
            "fi": {
                "technical": [
                    "Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja. Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen. Datan esikäsittely ja piirteiden suunnittelu ovat keskeisiä vaiheita prosessissa.",
                    "Pilvipalvelut tarjoavat skaalautuvan infrastruktuurin käyttöönottoon. Mikropalveluarkkitehtuuri mahdollistaa modulaarisen järjestelmäsuunnittelun. Rajapintapalvelut hoitavat autentikoinnin ja datan validoinnin.",
                    "Versionhallintajärjestelmät seuraavat lähdekoodin muutoksia. Jatkuva integraatio varmistaa koodin laadun ja automaattitestauksen. Dokumentaatio kattaa rajapintojen käytön ja järjestelmäarkkitehtuurin.",
                ],
                "business": [
                    "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet. Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani. Markkinalaajennusstrategia keskittyy nouseviin teknologiasektoreihin.",
                    "Strategiset kumppanuudet edistävät innovaatiota ja markkinapenetraatiota. T&K-investoinnit johtivat kolmeen uuteen tuotelanseeraukseen. Myyntitulos ylitti tavoitteet keskeisissä markkinasegmenteissä.",
                    "Toiminnan tehokkuus parani prosessiautomaation avulla. Asiakastyytyväisyysmittarit osoittavat positiivista kehitystä. Kustannusoptimointi tuotti 12% säästöt Q4:llä.",
                ],
                "educational": [
                    "Verkko-oppimisalusta sisältää interaktiivisia moduuleja ja oman tahdin edistymisen seurannan. Virtuaaliluokat mahdollistavat reaaliaikaisen yhteistyön opiskelijoiden ja ohjaajien välillä. Digitaaliset arviointityökalut antavat välitöntä palautetta oppimistuloksista.",
                    "Lähiopetuksen työpajat yhdistävät teoreettiset käsitteet käytännön harjoituksiin. Pienryhmätoiminta edistää vertaisoppimista ja tiedon jakamista. Harjoitukset vahvistavat keskeisiä oppimistavoitteita.",
                    "Ammatillisen kehityksen ohjelmat keskittyvät toimialan kannalta olennaisiin taitoihin. Mentorointi tarjoaa ohjausta urakehitykseen. Osaamisarvioinnit seuraavat edistymistä kohti oppimistavoitteita.",
                ],
            },
        }

        # Save content files
        for lang, texts in content.items():
            file_path = self._save_content(lang, texts, output_dir, force)
            files[lang] = file_path

        return files

    def _validate_generated_parameters(self, file_path: Path) -> bool:
        """Validate generated parameter file."""
        try:
            handler = ParameterHandler(file_path)
            is_valid, warnings, errors = handler.validate()

            if warnings:
                logger.warning(f"Validation warnings for {file_path}:")
                for warning in warnings:
                    logger.warning(f"- {warning}")

            if not is_valid:
                logger.error(f"Validation failed for {file_path}:")
                for error in errors:
                    logger.error(f"- {error}")

            return is_valid

        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            return False

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

        result = self.file_utils.save_data_to_disk(
            data=sheets,
            output_type="parameters",
            file_name=file_name,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=False,
        )

        # Validate generated file
        if not self._validate_generated_parameters(file_path):
            raise ValueError(
                f"Generated parameter file {file_path} failed validation"
            )

        return file_path

    def _save_content(
        self,
        language: str,
        texts: Dict[str, List[str]],
        output_dir: Path,
        force: bool,
    ) -> Path:
        """Save test content to Excel file."""
        file_name = f"test_content_{language}"
        file_path = output_dir / f"{file_name}.xlsx"

        if not force and file_path.exists():
            logger.warning(
                f"Content file {file_path} already exists, skipping..."
            )
            return file_path

        # Create DataFrame with metadata
        rows = []
        for content_type, content_texts in texts.items():
            for idx, text in enumerate(content_texts, 1):
                rows.append(
                    {
                        "id": f"{content_type}_{idx}",
                        "type": content_type,
                        "language": language,
                        "content": text,
                    }
                )

        df = pd.DataFrame(rows)

        result = self.file_utils.save_data_to_disk(
            data={file_name: df},
            output_type="raw",
            file_name=file_name,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=False,
        )

        return file_path

    def _save_metadata(self, files: Dict[str, List[Path]]) -> Path:
        """Save metadata about generated files."""
        metadata = {
            "files": {
                file_type: [str(p) for p in paths]
                for file_type, paths in files.items()
            },
            "generation_time": pd.Timestamp.now().isoformat(),
        }

        result = self.file_utils.save_yaml(
            data=metadata,
            file_path="test_data_metadata",
            output_type="parameters",
            include_timestamp=False,
        )

        return result


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
