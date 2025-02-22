import sys
from pathlib import Path
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ui.text_manager import UITextManager
from FileUtils import FileUtils

def check_ui_texts():
    """Check UI texts configuration file."""
    try:
        # Initialize FileUtils
        file_utils = FileUtils()
        
        # Get default configuration
        text_manager = UITextManager(file_utils=file_utils)
        default_config = text_manager._get_default_config()
        
        # Read current YAML file using FileUtils
        config_path = file_utils.get_data_path("config") / "ui_texts.yaml"
        current_config = file_utils.load_yaml(config_path)
        
        # Compare categories
        default_categories = set(default_config['texts'].keys())
        current_categories = set(current_config['texts'].keys())
        
        print("Categories comparison:")
        print("\nPresent categories:")
        total_entries = 0
        for category in sorted(current_categories):
            items = current_config['texts'][category]
            entry_count = len(items)
            total_entries += entry_count
            print(f"- {category}: {entry_count} entries")
        
        print(f"\nTotal present entries: {total_entries}")
        
        if missing_categories := default_categories - current_categories:
            print("\nMissing categories:")
            for category in sorted(missing_categories):
                items = default_config['texts'][category]
                print(f"- {category}: {len(items)} entries in default config")
        
        if extra_categories := current_categories - default_categories:
            print("\nExtra categories (not in default config):")
            for category in sorted(extra_categories):
                print(f"- {category}")
        
        # Check for missing translations
        print("\nChecking translations...")
        missing_translations = []
        for category, items in current_config['texts'].items():
            for key, translations in items.items():
                for lang in ['en', 'fi']:
                    if lang not in translations:
                        missing_translations.append(f"{category}.{key}.{lang}")
        
        if missing_translations:
            print("\nMissing translations:")
            for item in missing_translations:
                print(f"- {item}")
        else:
            print("All present texts have both English and Finnish translations.")
        
        # Compare text entries with default config
        print("\nComparing with default configuration:")
        for category in current_categories & default_categories:
            default_keys = set(default_config['texts'][category].keys())
            current_keys = set(current_config['texts'][category].keys())
            
            if missing_keys := default_keys - current_keys:
                print(f"\nMissing keys in category '{category}':")
                for key in sorted(missing_keys):
                    print(f"- {key}")
            
            if extra_keys := current_keys - default_keys:
                print(f"\nExtra keys in category '{category}':")
                for key in sorted(extra_keys):
                    print(f"- {key}")
        
    except Exception as e:
        print(f"Error checking UI texts: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(check_ui_texts()) 