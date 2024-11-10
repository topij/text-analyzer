# scripts/test_data_generator.py

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import sys

# Add project root to Python path
project_root = str(Path().resolve())
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")

from src.utils.FileUtils.file_utils import FileUtils, OutputFileType
from src.loaders.parameter_config import ParameterSheets, ParameterConfigurations

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
        
        files = {
            "parameters": [],
            "content": [],
            "metadata": []
        }
        
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
        """Generate parameter files in both languages."""
        output_dir = self.file_utils.get_data_path("configurations")
        files = {}

        # English parameters
        en_sheets = {
            ParameterSheets.GENERAL_EN.value: pd.DataFrame({
                "parameter": [
                    "max_keywords", "language", "focus_on", "min_keyword_length",
                    "include_compounds", "column_name_to_analyze"
                ],
                "value": [
                    8, "en", "business and technical content analysis",
                    3, True, "content"
                ],
                "description": [
                    "Maximum keywords to extract",
                    "Content language",
                    "Analysis focus area",
                    "Minimum keyword length",
                    "Handle compound words",
                    "Content column name"
                ]
            }),
            ParameterSheets.KEYWORDS_EN.value: pd.DataFrame({
                "keyword": [
                    "machine learning", "data analysis", "cloud computing",
                    "business intelligence", "market analysis", "revenue growth"
                ],
                "importance": [1.0, 0.9, 0.8, 0.9, 0.8, 0.9],
                "domain": [
                    "technical", "technical", "technical",
                    "business", "business", "business"
                ]
            }),
            ParameterSheets.EXCLUDED_EN.value: pd.DataFrame({
            "keyword": ["the", "and", "with", "for", "new", "using"],
            "reason": ["Common word"] * 6
            }),
            ParameterSheets.DOMAINS_EN.value: pd.DataFrame({
                "name": ["technical", "business"],
                "description": [
                    "Technical and software development content",
                    "Business and financial content"
                ],
                "key_terms": [
                    "software,development,algorithm,system",
                    "revenue,growth,market,financial"
                ],
                "context": [
                    "Focus on technical implementation details",
                    "Focus on business performance metrics"
                ],
                "stopwords": [
                    "using,implementation,basic,simple",
                    "new,current,various,general"
                ]
            })
        }

        # Finnish parameters
        fi_sheets = {
            ParameterSheets.GENERAL_FI.value: pd.DataFrame({
                "parametri": [
                    "maksimi_avainsanat", "kieli", "keskity_aihepiiriin",
                    "min_sanan_pituus", "sisällytä_yhdyssanat", "analysoitava_sarake"
                ],
                "arvo": [
                    8, "fi", "tekninen ja liiketoiminta-analyysi",
                    3, True, "sisältö"
                ],
                "kuvaus": [
                    "Poimittavien avainsanojen maksimimäärä",
                    "Sisällön kieli",
                    "Analyysin painopistealue",
                    "Avainsanan minimipituus",
                    "Käsittele yhdyssanat",
                    "Analysoitavan sisällön sarake"
                ]
            }),
            ParameterSheets.EXCLUDED_FI.value: pd.DataFrame({
                "avainsana": ["ja", "tai", "sekä", "kanssa", "uusi", "käyttäen"],
                "syy": ["Yleinen sana"] * 6
            }),
            ParameterSheets.DOMAINS_FI.value: pd.DataFrame({
                "nimi": ["tekninen", "liiketoiminta"],
                "kuvaus": [
                    "Tekninen ja ohjelmistokehityssisältö",
                    "Liiketoiminta ja taloussisältö"
                ],
                "keskeiset_termit": [
                    "ohjelmisto,kehitys,algoritmi,järjestelmä",
                    "liikevaihto,kasvu,markkina,talous"
                ],
                "konteksti": [
                    "Keskittyy teknisiin toteutusyksityiskohtiin",
                    "Keskittyy liiketoiminnan suorituskykyyn"
                ],
                "ohitettavat_sanat": [
                    "käyttäen,toteutus,perus,yksinkertainen",
                    "uusi,nykyinen,erilainen,yleinen"
                ]
            })
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

        # Define test content
        content = {
            "en": {
                "technical": [
                    "Machine learning models are trained using large datasets to recognize patterns. The neural network architecture includes multiple layers for feature extraction. Data preprocessing and feature engineering are crucial steps in the pipeline. ",
                    "Cloud computing services provide scalable infrastructure for deployments. Microservices architecture enables modular and maintainable system design. API endpoints handle authentication and data validation requirements. ",
                    "Version control systems track changes in source code repositories. Continuous integration ensures code quality and automated testing. Documentation covers API usage and system architecture details. "],
                "business": [
                    "Q3 financial results show 15% revenue growth and improved profit margins. Customer acquisition costs decreased while retention rates increased. Market expansion strategy focuses on emerging technology sectors. ",
                    "Strategic partnerships drive innovation and market penetration. Investment in R&D resulted in three new product launches. Sales performance exceeded targets in key market segments. ",
                    "Operational efficiency improved through process automation. Customer satisfaction metrics show positive trend year-over-year. Cost optimization initiatives delivered 12% savings in Q4. "],
                "general": [
                    "Team collaboration improved with new communication tools. Project timeline adjustments accommodate additional requirements. Resource allocation ensures optimal task distribution. ",
                    "Knowledge sharing sessions enhance team capabilities. Regular updates maintain stakeholder engagement levels. Quality assurance practices meet industry standards. "]
            },
            "fi": {
                "technical": [
                    "Koneoppimismalleja koulutetaan suurilla datajoukolla tunnistamaan kaavoja. Neuroverkon arkkitehtuuri sisältää useita kerroksia piirteiden erottamiseen. Datan esikäsittely ja piirteiden suunnittelu ovat keskeisiä vaiheita prosessissa. ",
                    "Pilvipalvelut tarjoavat skaalautuvan infrastruktuurin käyttöönottoon. Mikropalveluarkkitehtuuri mahdollistaa modulaarisen järjestelmäsuunnittelun. Rajapintapalvelut hoitavat autentikoinnin ja datan validoinnin. ",
                    "Versionhallintajärjestelmät seuraavat lähdekoodin muutoksia. Jatkuva integraatio varmistaa koodin laadun ja automaattitestauksen. Dokumentaatio kattaa rajapintojen käytön ja järjestelmäarkkitehtuurin. "],
                "business": [
                    "Q3 taloudelliset tulokset osoittavat 15% liikevaihdon kasvun ja parantuneet katteet. Asiakashankinnan kustannukset laskivat ja asiakaspysyvyys parani. Markkinalaajennusstrategia keskittyy nouseviin teknologiasektoreihin. ",
                    "Strategiset kumppanuudet edistävät innovaatiota ja markkinapenetraatiota. T&K-investoinnit johtivat kolmeen uuteen tuotelanseeraukseen. Myyntitulos ylitti tavoitteet keskeisissä markkinasegmenteissä. ",
                    "Toiminnan tehokkuus parani prosessiautomaation avulla. Asiakastyytyväisyysmittarit osoittavat positiivista kehitystä. Kustannusoptimointi tuotti 12% säästöt Q4:llä. "],
                "general": [
                    "Tiimin yhteistyö parani uusien viestintätyökalujen myötä. Projektiaikataulua mukautettiin lisävaatimusten vuoksi. Resurssien kohdentaminen varmistaa optimaalisen tehtäväjaon. ",
                    "Tiedonjakotilaisuudet kehittävät tiimin osaamista. Säännölliset päivitykset ylläpitävät sidosryhmien sitoutumista. Laadunvarmistuskäytännöt täyttävät toimialan standardit. "]
            }
        }

        # Save content files
        for lang, texts in content.items():
            file_path = self._save_content(lang, texts, output_dir, force)
            files[lang] = file_path

        return files

    def _save_parameters(
        self, 
        language: str, 
        sheets: Dict[str, pd.DataFrame],
        output_dir: Path,
        force: bool
    ) -> Path:
        """Save parameter sheets to Excel file."""
        file_name = f"parameters_{language}"
        file_path = output_dir / f"{file_name}.xlsx"
        
        if not force and file_path.exists():
            logger.warning(f"Parameter file {file_path} already exists, skipping...")
            return file_path

        result = self.file_utils.save_data_to_disk(
            data=sheets,
            output_type="configurations",
            file_name=file_name,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=False
        )
        
        return file_path

    def _save_content(
        self,
        language: str,
        texts: Dict[str, List[str]],
        output_dir: Path,
        force: bool
    ) -> Path:
        """Save test content to Excel file."""
        file_name = f"test_content_{language}"
        file_path = output_dir / f"{file_name}.xlsx"
        
        if not force and file_path.exists():
            logger.warning(f"Content file {file_path} already exists, skipping...")
            return file_path

        # Create DataFrame with metadata
        rows = []
        for content_type, content_texts in texts.items():
            for idx, text in enumerate(content_texts, 1):
                rows.append({
                    "id": f"{content_type}_{idx}",
                    "type": content_type,
                    "language": language,
                    "content": text
                })
        
        df = pd.DataFrame(rows)
        
        result = self.file_utils.save_data_to_disk(
            data={file_name: df},
            output_type="raw",
            file_name=file_name,
            output_filetype=OutputFileType.XLSX,
            include_timestamp=False
        )
        
        return file_path

    def _save_metadata(self, files: Dict[str, List[Path]]) -> Path:
        """Save metadata about generated files."""
        metadata = {
            "files": {
                file_type: [str(p) for p in paths]
                for file_type, paths in files.items()
            },
            "generation_time": pd.Timestamp.now().isoformat()
        }
        
        result = self.file_utils.save_yaml(
            data=metadata,
            file_path="test_data_metadata",
            output_type="configurations",
            include_timestamp=False
        )
        
        return result

def main():
    """Generate all test data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test data for semantic analyzer")
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force overwrite existing files"
    )
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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