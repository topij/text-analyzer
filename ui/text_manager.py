import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from FileUtils import FileUtils, OutputFileType

class UITextManager:
    """Manages UI text content and translations"""
    
    def __init__(self, config_path: Optional[Path] = None, file_utils: Optional[FileUtils] = None):
        """Initialize with optional custom config path"""
        self.file_utils = file_utils or FileUtils()
        
        # Always use config directory from FileUtils
        if config_path is None:
            config_path = self.file_utils.get_data_path("config") / "ui_texts.yaml"
            
        if not config_path.exists():
            logger.warning(f"UI text configuration file not found at {config_path}. Creating default configuration.")
            self._create_default_config(config_path)
            
        try:
            # Load YAML configuration using FileUtils
            config_data = None
            if config_path.exists():
                config_data = self.file_utils.load_yaml(config_path)
            
            # If no valid config data, create default
            if not config_data:
                config_data = self._get_default_config()
                
            self.config = config_data
            self.default_language = self.config['default_language']
            logger.info(f"Loaded UI text configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading UI text configuration: {str(e)}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default UI text configuration"""
        return {
            "default_language": "en",
            "categories": {
                "titles": "Main UI titles and headers",
                "messages": "Status and information messages",
                "buttons": "Button labels",
                "help_texts": "Help and documentation texts",
                "labels": "Input field labels",
                "placeholders": "Input field placeholder texts",
                "tabs": "Tab labels and titles",
                "table_headers": "Table column headers",
                "result_labels": "Analysis result labels and descriptions"
            },
            "texts": {
                "titles": {
                    "main": {
                        "en": "Text Analyzer",
                        "fi": "Tekstianalysaattori"
                    },
                    "help": {
                        "en": "Help & Documentation",
                        "fi": "Ohje & Dokumentaatio"
                    },
                    "parameters": {
                        "en": "Parameter Settings",
                        "fi": "Parametrien asetukset"
                    },
                    "file_upload": {
                        "en": "File Upload",
                        "fi": "Tiedoston lataus"
                    },
                    "results": {
                        "en": "Analysis Results",
                        "fi": "Analyysin tulokset"
                    },
                    "language_settings": {
                        "en": "Language Settings",
                        "fi": "Kieliasetukset"
                    },
                    "sample_files": {
                        "en": "Sample Files",
                        "fi": "Esimerkkitiedostot"
                    },
                    "preview_data": {
                        "en": "Check uploaded data",
                        "fi": "Tarkista ladatut tiedot"
                    },
                    "content_info": {
                        "en": "Content Information",
                        "fi": "Sisällön tiedot"
                    },
                    "parameter_info": {
                        "en": "Parameter Information",
                        "fi": "Parametrien tiedot"
                    },
                    "basic_info": {
                        "en": "Basic Information",
                        "fi": "Perustiedot"
                    },
                    "column_details": {
                        "en": "Column Details",
                        "fi": "Sarakkeiden tiedot"
                    },
                    "current_settings": {
                        "en": "Current Settings",
                        "fi": "Nykyiset asetukset"
                    },
                    "data_preview": {
                        "en": "Data Preview",
                        "fi": "Datan esikatselu"
                    },
                    "content_details": {
                        "en": "Content Details",
                        "fi": "Sisällön tiedot"
                    },
                    "parameter_details": {
                        "en": "Parameter Details",
                        "fi": "Parametrien tiedot"
                    },
                    "check_uploaded_data": {
                        "en": "Check uploaded data",
                        "fi": "Tarkista ladatut tiedot"
                    },
                    "basic_information": {
                        "en": "Basic Information",
                        "fi": "Perustiedot"
                    },
                    "column_information": {
                        "en": "Column Details",
                        "fi": "Sarakkeiden tiedot"
                    },
                    "data_sample": {
                        "en": "Data Sample",
                        "fi": "Datan esimerkki"
                    }
                },
                "messages": {
                    "analyzing": {
                        "en": "Analyzing text...",
                        "fi": "Analysoidaan tekstiä..."
                    },
                    "processing_text": {
                        "en": "Processing text {current} of {total}...",
                        "fi": "Käsitellään tekstiä {current}/{total}..."
                    },
                    "exporting": {
                        "en": "Preparing export...",
                        "fi": "Valmistellaan vientiä..."
                    },
                    "export_complete": {
                        "en": "Export completed successfully",
                        "fi": "Vienti valmistui onnistuneesti"
                    },
                    "error": {
                        "en": "❌ Error: {error}",
                        "fi": "❌ Virhe: {error}"
                    },
                    "texts_loaded": {
                        "en": "✅ Texts loaded successfully",
                        "fi": "✅ Tekstit ladattu onnistuneesti"
                    },
                    "parameters_loaded": {
                        "en": "✅ Parameters loaded successfully",
                        "fi": "✅ Parametrit ladattu onnistuneesti"
                    },
                    "error_load": {
                        "en": "❌ Error loading file: {error}",
                        "fi": "❌ Virhe tiedoston latauksessa: {error}"
                    },
                    "analysis_complete": {
                        "en": "✅ Analysis completed successfully",
                        "fi": "✅ Analyysi valmis"
                    },
                    "analysis_partial": {
                        "en": "⚠️ Analysis completed with {success_count} successful out of {total_texts} texts",
                        "fi": "⚠️ Analyysi valmistui {success_count} onnistuneella {total_texts} tekstistä"
                    },
                    "analysis_failed_all": {
                        "en": "❌ Analysis failed for all {total_texts} texts",
                        "fi": "❌ Analyysi epäonnistui kaikille {total_texts} tekstille"
                    },
                    "analysis_failed_text": {
                        "en": "Analysis failed for text: {text}...",
                        "fi": "Analyysi epäonnistui tekstille: {text}..."
                    },
                    "export_success": {
                        "en": "Results exported successfully to {filename}",
                        "fi": "Tulokset viety onnistuneesti tiedostoon {filename}"
                    },
                    "export_error": {
                        "en": "Error exporting results: {error}",
                        "fi": "Virhe tulosten viennissä: {error}"
                    },
                    "no_content_file": {
                        "en": "No content file uploaded yet.",
                        "fi": "Sisältötiedostoa ei ole vielä ladattu."
                    },
                    "no_parameter_file": {
                        "en": "No parameter file uploaded yet.",
                        "fi": "Parametritiedostoa ei ole vielä ladattu."
                    },
                    "no_results": {
                        "en": "No analysis results available yet.",
                        "fi": "Analyysin tuloksia ei ole vielä saatavilla."
                    },
                    "export_in_progress": {
                        "en": "Exporting results...",
                        "fi": "Viedään tuloksia..."
                    },
                    "messages.export_complete]": {
                        "en": "Exporting ready",
                        "fi": "Tulokset tallennettu"
                    }
                },
                "tabs": {
                    "keywords": {
                        "en": "Keywords",
                        "fi": "Avainsanat"
                    },
                    "themes": {
                        "en": "Themes",
                        "fi": "Teemat"
                    },
                    "categories": {
                        "en": "Categories",
                        "fi": "Kategoriat"
                    },
                    "content_details": {
                        "en": "Content Details",
                        "fi": "Sisällön tiedot"
                    },
                    "parameter_details": {
                        "en": "Parameter Details",
                        "fi": "Parametrien tiedot"
                    }
                },
                "buttons": {
                    "analyze": {
                        "en": "🔍 Analyze",
                        "fi": "🔍 Analysoi"
                    },
                    "reset_parameters": {
                        "en": "↺ Reset Parameters",
                        "fi": "↺ Palauta oletusarvot"
                    },
                    "export_results": {
                        "en": "📥 Export Results",
                        "fi": "📥 Vie tulokset"
                    },
                    "download_sample_content": {
                        "en": "📥 Download Sample Content",
                        "fi": "📥 Lataa esimerkkisisältö"
                    },
                    "download_sample_parameters": {
                        "en": "📥 Download Sample Parameters",
                        "fi": "📥 Lataa esimerkkiparametrit"
                    }
                },
                "labels": {
                    "max_keywords": {
                        "en": "Max Keywords",
                        "fi": "Avainsanojen maksimimäärä"
                    },
                    "max_themes": {
                        "en": "Max Themes",
                        "fi": "Teemojen maksimimäärä"
                    },
                    "focus": {
                        "en": "Analysis Focus",
                        "fi": "Analyysin fokus"
                    },
                    "content_column": {
                        "en": "Content Column",
                        "fi": "Sisältösarake"
                    },
                    "interface_language": {
                        "en": "Interface Language",
                        "fi": "Käyttöliittymän kieli"
                    },
                    "upload_texts": {
                        "en": "Upload text file for analysis",
                        "fi": "Lataa tekstitiedosto analysoitavaksi"
                    },
                    "choose_texts": {
                        "en": "Choose text file",
                        "fi": "Valitse tekstitiedosto"
                    },
                    "upload_parameters": {
                        "en": "Upload parameter file (optional)",
                        "fi": "Lataa parametritiedosto (valinnainen)"
                    },
                    "choose_parameters": {
                        "en": "Choose parameter file",
                        "fi": "Valitse parametritiedosto"
                    },
                    "show_data_preview": {
                        "en": "Show data preview",
                        "fi": "Näytä datan esikatselu"
                    },
                    "show_all_parameters": {
                        "en": "Show all parameters",
                        "fi": "Näytä kaikki parametrit"
                    },
                    "number_of_rows": {
                        "en": "Number of Rows",
                        "fi": "Rivien määrä"
                    },
                    "number_of_columns": {
                        "en": "Number of Columns",
                        "fi": "Sarakkeiden määrä"
                    },
                    "selected_text_column": {
                        "en": "Selected Text Column",
                        "fi": "Valittu tekstisarake"
                    },
                    "download_results": {
                        "en": "📥 Download Results",
                        "fi": "📥 Lataa tulokset"
                    },
                    "data_preview": {
                        "en": "Data Preview",
                        "fi": "Datan esikatselu"
                    },
                    "basic_info": {
                        "en": "Basic Information",
                        "fi": "Perustiedot"
                    },
                    "column_details": {
                        "en": "Column Details",
                        "fi": "Sarakkeiden tiedot"
                    },
                    "parameter_info": {
                        "en": "Parameter Information",
                        "fi": "Parametrien tiedot"
                    },
                    "current_settings": {
                        "en": "Current Settings",
                        "fi": "Nykyiset asetukset"
                    },
                    "column_name": {
                        "en": "Column Name",
                        "fi": "Sarakkeen nimi"
                    },
                    "data_type": {
                        "en": "Data Type",
                        "fi": "Tietotyyppi"
                    },
                    "non_null_count": {
                        "en": "Non-Null Count",
                        "fi": "Ei-tyhjät arvot"
                    },
                    "null_count": {
                        "en": "Null Count",
                        "fi": "Tyhjät arvot"
                    },
                    "all_parameters": {
                        "en": "All Parameters",
                        "fi": "Kaikki parametrit"
                    },
                    "error_loading_parameters": {
                        "en": "Error loading full parameters: {error}",
                        "fi": "Virhe parametrien latauksessa: {error}"
                    }
                },
                "help_texts": {
                    "what_is_title": {
                        "en": "What is Text Analyzer?",
                        "fi": "Mikä on Tekstianalysaattori?"
                    },
                    "what_is": {
                        "en": "Text Analyzer is a powerful tool for semantic text analysis. It extracts keywords, identifies themes, and categorizes content using advanced natural language processing.",
                        "fi": "Tekstianalysaattori on tehokas työkalu semanttiseen tekstianalyysiin. Se poimii avainsanoja, tunnistaa teemoja ja luokittelee sisältöä käyttäen edistynyttä luonnollisen kielen käsittelyä."
                    },
                    "file_requirements_title": {
                        "en": "File Requirements",
                        "fi": "Tiedostovaatimukset"
                    },
                    "file_requirements": {
                        "en": "- Text files should be in XLSX format\n- Maximum file size is 20MB\n- Text should be in Finnish or English",
                        "fi": "- Tekstitiedostojen tulee olla XLSX-muodossa\n- Tiedoston maksimikoko on 20MB\n- Tekstin tulee olla suomeksi tai englanniksi"
                    },
                    "max_keywords_help": {
                        "en": "Maximum number of keywords to extract from each text",
                        "fi": "Maksimimäärä avainsanoja, jotka poimitaan kustakin tekstistä"
                    },
                    "max_themes_help": {
                        "en": "Maximum number of themes to identify from each text",
                        "fi": "Maksimimäärä teemoja, jotka tunnistetaan kustakin tekstistä"
                    },
                    "focus_on_help": {
                        "en": "Specify what aspects of the text to focus on during analysis",
                        "fi": "Määritä mihin tekstin osa-alueisiin analyysi keskittyy"
                    },
                    "content_column_help": {
                        "en": "Select the column containing the text content to analyze",
                        "fi": "Valitse sarake, joka sisältää analysoitavan tekstisisällön"
                    }
                },
                "table_headers": {
                    "text_content": {
                        "en": "Text Content",
                        "fi": "Tekstisisältö"
                    },
                    "keywords": {
                        "en": "Keywords",
                        "fi": "Avainsanat"
                    },
                    "themes": {
                        "en": "Themes",
                        "fi": "Teemat"
                    },
                    "categories": {
                        "en": "Categories",
                        "fi": "Kategoriat"
                    },
                    "column_name": {
                        "en": "Column Name",
                        "fi": "Sarakkeen nimi"
                    },
                    "data_type": {
                        "en": "Data Type",
                        "fi": "Tietotyyppi"
                    },
                    "non_null_count": {
                        "en": "Non-Null Count",
                        "fi": "Ei-tyhjät arvot"
                    },
                    "null_count": {
                        "en": "Null Count",
                        "fi": "Tyhjät arvot"
                    },
                    "text_preview": {
                        "en": "Text Preview",
                        "fi": "Tekstin esikatselu"
                    },
                    "parameter_name": {
                        "en": "Parameter Name",
                        "fi": "Parametrin nimi"
                    },
                    "value": {
                        "en": "Value",
                        "fi": "Arvo"
                    },
                    "evidence": {
                        "en": "Evidence",
                        "fi": "Näyttö"
                    },
                    "confidence": {
                        "en": "Confidence",
                        "fi": "Luottamus"
                    }
                },
                "result_labels": {
                    "confidence_score": {
                        "en": "Confidence Score",
                        "fi": "Luottamusarvo"
                    },
                    "related_keywords": {
                        "en": "Related Keywords",
                        "fi": "Liittyvät avainsanat"
                    },
                    "theme_hierarchy": {
                        "en": "Theme Hierarchy",
                        "fi": "Teemahierarkia"
                    },
                    "category_evidence": {
                        "en": "Category Evidence",
                        "fi": "Kategorian näyttö"
                    }
                }
            },
            "placeholders": {
                "drag_drop_text": {
                    "en": "Drag and drop file here",
                    "fi": "Raahaa ja pudota tiedosto tähän"
                },
                "file_limit_text": {
                    "en": "Limit 20MB per file • XLSX",
                    "fi": "Maksimikoko 20MB per tiedosto • XLSX"
                },
                "file_limit_text_params": {
                    "en": "Limit 20MB per file • XLSX",
                    "fi": "Maksimikoko 20MB per tiedosto • XLSX"
                },
                "not_set": {
                    "en": "Not set",
                    "fi": "Ei asetettu"
                },
                "no_evidence": {
                    "en": "No evidence available",
                    "fi": "Ei saatavilla olevaa näyttöä"
                },
                "no_results": {
                    "en": "No results available",
                    "fi": "Ei tuloksia saatavilla"
                },
                "no_themes": {
                    "en": "No themes identified",
                    "fi": "Ei tunnistettuja teemoja"
                },
                "no_categories": {
                    "en": "No categories matched",
                    "fi": "Ei sopivia kategorioita"
                },
                "no_keywords": {
                    "en": "No keywords extracted",
                    "fi": "Ei poimittuja avainsanoja"
                }
            }
        }
    
    def _create_default_config(self, config_path: Path) -> None:
        """Create default UI text configuration file"""
        try:
            # Get default configuration
            default_config = self._get_default_config()
            
            # Create config directory using FileUtils
            config_dir = self.file_utils.get_data_path("config")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure all required categories and texts exist
            for category in default_config['texts']:
                if category not in default_config['categories']:
                    default_config['categories'][category] = f"{category.replace('_', ' ').title()} texts"
            
            # Save configuration using FileUtils
            self.file_utils.save_yaml(config_path, default_config)
            
            # Verify the written file
            written_config = self.file_utils.load_yaml(config_path)
            
            # Verify all categories and texts
            for category, items in default_config['texts'].items():
                if category not in written_config.get('texts', {}):
                    raise ValueError(f"Missing category in written file: {category}")
                for key in items:
                    if key not in written_config['texts'][category]:
                        raise ValueError(f"Missing key in category {category}: {key}")
                    for lang in ['en', 'fi']:
                        if lang not in written_config['texts'][category][key]:
                            raise ValueError(f"Missing {lang} translation for {category}.{key}")
            
            logger.info(f"Created default UI text configuration at {config_path}")
            
        except Exception as e:
            logger.error(f"Error creating default UI text configuration: {str(e)}")
            raise
    
    def get_text(self, category: str, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get translated text for given category and key.
        
        Args:
            category: Text category (e.g., 'titles', 'messages')
            key: Specific text key within the category
            language: Target language code (defaults to default_language)
            **kwargs: Format parameters for text templates
            
        Returns:
            Translated text string
        """
        language = language or self.default_language
        try:
            text = self.config['texts'][category][key][language]
            return text.format(**kwargs) if kwargs else text
        except KeyError:
            logger.warning(f"Text not found for {category}.{key} in {language}")
            # Try fallback to default language
            try:
                text = self.config['texts'][category][key][self.default_language]
                return text.format(**kwargs) if kwargs else text
            except KeyError:
                return f"[{category}.{key}]"

    def export_to_excel(self, output_path: Optional[Path] = None) -> Path:
        """
        Export UI texts to Excel for easy editing.
        
        Returns:
            Path to the created Excel file
        """
        try:
            # Use config directory from FileUtils
            if output_path is None:
                output_path = self.file_utils.get_data_path("config") / "texts.xlsx"
            
            # Get default configuration to ensure all texts are included
            default_config = self._get_default_config()
            
            # Create a list to store all text entries
            rows = []
            
            # Process all categories from both default and current config
            all_categories = set(default_config['texts'].keys()) | set(self.config.get('texts', {}).keys())
            
            for category in sorted(all_categories):
                # Get all keys for this category
                default_items = default_config['texts'].get(category, {})
                current_items = self.config.get('texts', {}).get(category, {})
                all_keys = set(default_items.keys()) | set(current_items.keys())
                
                for key in sorted(all_keys):
                    # Get translations, preferring current over default
                    translations = default_items.get(key, {})
                    if key in current_items:
                        translations.update(current_items[key])
                    
                    row = {
                        'category': category,
                        'key': key,
                        'en': translations.get('en', ''),
                        'fi': translations.get('fi', ''),
                        'description': self.config.get('categories', {}).get(category, 
                                     default_config['categories'].get(category, ''))
                    }
                    rows.append(row)
            
            # Create DataFrame with all texts
            df = pd.DataFrame(rows)
            
            # Save using FileUtils
            saved_files, _ = self.file_utils.save_data_to_storage(
                data={"texts": df},
                file_name=output_path.stem,
                output_type="config",
                output_filetype=OutputFileType.XLSX,
                include_timestamp=False
            )
            
            result_path = Path(list(saved_files.values())[0])
            logger.info(f"Exported UI texts to {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error exporting UI texts to Excel: {str(e)}")
            raise

    def import_from_excel(self, file_path: Path) -> None:
        """
        Import UI texts from Excel file.
        
        Args:
            file_path: Path to Excel file containing UI texts
        """
        try:
            # Convert to absolute path using FileUtils if needed
            if not file_path.is_absolute():
                file_path = self.file_utils.get_data_path("config") / file_path.name
            
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found at {file_path}")
            
            # Load Excel file using pandas
            df = pd.read_excel(file_path)
            
            # Validate DataFrame structure
            required_columns = {'category', 'key', 'en', 'fi'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            
            # Get default configuration to ensure all required texts exist
            default_config = self._get_default_config()
            
            # Start with default config
            updated_config = default_config.copy()
            
            # Convert DataFrame to nested structure
            for _, row in df.iterrows():
                category = row['category']
                key = row['key']
                
                # Skip empty or invalid entries
                if not category or not key:
                    continue
                
                # Ensure category exists
                if category not in updated_config['texts']:
                    updated_config['texts'][category] = {}
                    if category not in updated_config['categories']:
                        updated_config['categories'][category] = f"{category.replace('_', ' ').title()} texts"
                
                # Update translations
                updated_config['texts'][category][key] = {
                    'en': row['en'] if pd.notna(row['en']) else default_config['texts'].get(category, {}).get(key, {}).get('en', ''),
                    'fi': row['fi'] if pd.notna(row['fi']) else default_config['texts'].get(category, {}).get(key, {}).get('fi', '')
                }
            
            # Ensure all default texts are included
            for category, items in default_config['texts'].items():
                if category not in updated_config['texts']:
                    updated_config['texts'][category] = items.copy()
                else:
                    for key, translations in items.items():
                        if key not in updated_config['texts'][category]:
                            updated_config['texts'][category][key] = translations.copy()
                        else:
                            # Ensure both languages exist
                            for lang in ['en', 'fi']:
                                if lang not in updated_config['texts'][category][key]:
                                    updated_config['texts'][category][key][lang] = translations.get(lang, '')
            
            # Update the instance config
            self.config = updated_config
            
            # Save updated configuration using FileUtils
            config_path = self.file_utils.get_data_path("config") / "ui_texts.yaml"
            self.file_utils.save_yaml(config_path, updated_config)
            
            # Verify the written file
            written_config = self.file_utils.load_yaml(config_path)
            
            # Verify all categories and texts
            for category, items in default_config['texts'].items():
                if category not in written_config.get('texts', {}):
                    raise ValueError(f"Missing category in written file: {category}")
                for key in items:
                    if key not in written_config['texts'][category]:
                        raise ValueError(f"Missing key in category {category}: {key}")
                    for lang in ['en', 'fi']:
                        if lang not in written_config['texts'][category][key]:
                            raise ValueError(f"Missing {lang} translation for {category}.{key}")
            
            logger.info(f"Updated UI text configuration at {config_path}")
            
        except Exception as e:
            logger.error(f"Error importing UI texts from Excel: {str(e)}")
            raise

def get_ui_text_manager(config_path: Optional[Path] = None, file_utils: Optional[FileUtils] = None) -> UITextManager:
    """Factory function to create UITextManager instance"""
    return UITextManager(config_path, file_utils) 