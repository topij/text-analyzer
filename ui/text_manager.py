import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class UITextManager:
    """Manages UI text content and translations"""
    
    def __init__(self):
        """Initialize with default UI texts"""
        self.config = self._create_default_config()
        self.default_language = 'en'
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default UI text configuration"""
        return {
            "default_language": "en",
            "categories": {
                "titles": "Main UI titles and headers",
                "messages": "Status and information messages",
                "buttons": "Button labels",
                "help_texts": "Help and documentation texts",
                "labels": "Input field labels",
                "placeholders": "Input field placeholder texts"
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
                    "file_upload": {
                        "en": "Upload Files",
                        "fi": "Tiedostojen lataus"
                    },
                    "parameters": {
                        "en": "Parameter Settings",
                        "fi": "Parametrien asetukset"
                    },
                    "results": {
                        "en": "Analysis Results",
                        "fi": "Analyysin tulokset"
                    },
                    "language_settings": {
                        "en": "Language Settings",
                        "fi": "Kieliasetukset"
                    }
                },
                "messages": {
                    "texts_loaded": {
                        "en": "✅ Texts loaded successfully!",
                        "fi": "✅ Tekstit ladattu onnistuneesti!"
                    },
                    "parameters_loaded": {
                        "en": "✅ Parameters loaded successfully!",
                        "fi": "✅ Parametrit ladattu onnistuneesti!"
                    },
                    "error_load": {
                        "en": "❌ Error loading file: {error}",
                        "fi": "❌ Virhe tiedoston latauksessa: {error}"
                    },
                    "no_texts": {
                        "en": "⚠️ Please upload texts first!",
                        "fi": "⚠️ Ole hyvä ja lataa tekstit ensin!"
                    },
                    "invalid_column": {
                        "en": "❌ Selected text column not found in the file!",
                        "fi": "❌ Valittua tekstisaraketta ei löydy tiedostosta!"
                    },
                    "analyzing": {
                        "en": "Analyzing texts...",
                        "fi": "Analysoidaan tekstejä..."
                    },
                    "analysis_complete": {
                        "en": "✅ Analysis complete!",
                        "fi": "✅ Analyysi valmis!"
                    }
                },
                "buttons": {
                    "analyze": {
                        "en": "🔍 Analyze Texts",
                        "fi": "🔍 Analysoi tekstit"
                    },
                    "download": {
                        "en": "📥 Download Results",
                        "fi": "📥 Lataa tulokset"
                    }
                },
                "help_texts": {
                    "what_is_title": {
                        "en": "What is Text Analyzer?",
                        "fi": "Mikä on Tekstianalysaattori?"
                    },
                    "what_is": {
                        "en": """Text Analyzer is a powerful tool that helps you analyze text content by:
- Extracting key terms and phrases
- Identifying main themes
- Categorizing content
- Using advanced AI for semantic analysis""",
                        "fi": """Tekstianalysaattori on tehokas työkalu, joka auttaa sinua analysoimaan tekstisisältöä:
- Poimimalla avainsanoja ja -fraaseja
- Tunnistamalla pääteemat
- Luokittelemalla sisältöä
- Käyttämällä edistynyttä tekoälyä semanttiseen analyysiin"""
                    },
                    "file_requirements_title": {
                        "en": "File Requirements",
                        "fi": "Tiedostovaatimukset"
                    },
                    "file_requirements": {
                        "en": """**Text File Format:**
- Supported formats: XLSX, CSV
- Must contain a column with text to analyze
- Maximum file size: 200MB

**Parameter File Format:**
- Excel file (.xlsx)
- Contains analysis settings""",
                        "fi": """**Tekstitiedoston muoto:**
- Tuetut muodot: XLSX, CSV
- Täytyy sisältää tekstisarake analyysiä varten
- Maksimikoko: 200MB

**Parametritiedoston muoto:**
- Excel-tiedosto (.xlsx)
- Sisältää analyysin asetukset"""
                    }
                },
                "labels": {
                    "interface_language": {
                        "en": "Interface Language",
                        "fi": "Käyttöliittymän kieli"
                    },
                    "upload_texts": {
                        "en": "Upload Text File",
                        "fi": "Lataa tekstitiedosto"
                    },
                    "choose_texts": {
                        "en": "Choose a text file (XLSX, CSV)",
                        "fi": "Valitse tekstitiedosto (XLSX, CSV)"
                    },
                    "upload_parameters": {
                        "en": "Upload Parameters",
                        "fi": "Lataa parametrit"
                    },
                    "choose_parameters": {
                        "en": "Choose a parameter file (XLSX)",
                        "fi": "Valitse parametritiedosto (XLSX)"
                    },
                    "max_keywords": {
                        "en": "Maximum Keywords",
                        "fi": "Avainsanojen maksimimäärä"
                    },
                    "max_themes": {
                        "en": "Maximum Themes",
                        "fi": "Teemojen maksimimäärä"
                    },
                    "focus": {
                        "en": "Analysis Focus",
                        "fi": "Analyysin fokus"
                    }
                }
            }
        }
    
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
            # Try fallback to default language
            try:
                text = self.config['texts'][category][key][self.default_language]
                return text.format(**kwargs) if kwargs else text
            except KeyError:
                return f"[{category}.{key}]"

def get_ui_text_manager() -> UITextManager:
    """Get or create a UITextManager instance"""
    return UITextManager() 