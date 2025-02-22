import click
from pathlib import Path
import logging
from text_manager import get_ui_text_manager
from FileUtils import FileUtils
from FileUtils import OutputFileType
import yaml

logger = logging.getLogger(__name__)

def setup_ui_texts(force: bool = False):
    """Initialize UI text configuration and export to Excel."""
    try:
        file_utils = FileUtils()
        
        # Create UI text manager (this will create default config if it doesn't exist)
        ui_manager = get_ui_text_manager(file_utils=file_utils)
        
        # Get default configuration
        default_config = ui_manager._get_default_config()
        
        # Ensure config file exists with all default texts
        config_path = file_utils.get_data_path("config") / "ui_texts.yaml"
        if not config_path.exists() or force:
            # Create config directory
            config_dir = file_utils.get_data_path("config")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Save default configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(default_config, f, allow_unicode=True, sort_keys=False)
            logger.info(f"Created default UI text configuration at {config_path}")
        
        # Verify configuration has all required texts
        with open(config_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)
        
        # Check for missing categories and texts
        for category, items in default_config['texts'].items():
            if category not in current_config['texts']:
                current_config['texts'][category] = items.copy()
            else:
                for key, translations in items.items():
                    if key not in current_config['texts'][category]:
                        current_config['texts'][category][key] = translations.copy()
                    else:
                        # Ensure both languages exist
                        for lang in ['en', 'fi']:
                            if lang not in current_config['texts'][category][key]:
                                current_config['texts'][category][key][lang] = translations[lang]
        
        # Update categories descriptions
        for category in current_config['texts']:
            if category not in current_config['categories']:
                current_config['categories'][category] = f"{category.replace('_', ' ').title()} texts"
        
        # Save updated configuration
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(current_config, f, allow_unicode=True, sort_keys=False)
        
        # Export to Excel for easy editing
        excel_path = ui_manager.export_to_excel()
        
        return True, f"UI texts initialized successfully. Excel file created at: {excel_path}"
    except Exception as e:
        return False, f"Failed to initialize UI texts: {str(e)}"

def setup_ui_directories():
    """Create necessary UI-related directories."""
    try:
        file_utils = FileUtils()
        
        # Create required directories
        config_dir = file_utils.get_data_path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        return True, "UI directories created successfully"
    except Exception as e:
        return False, f"Failed to create UI directories: {str(e)}"

@click.command()
@click.option('--force', is_flag=True, help='Force recreation of UI files even if they exist')
def setup(force):
    """Set up UI configuration and files."""
    click.echo("Setting up UI components...")
    
    # Setup directories
    success, message = setup_ui_directories()
    click.echo(message)
    if not success:
        click.echo("Failed to set up UI directories", err=True)
        return
    
    # Setup UI texts
    success, message = setup_ui_texts(force)
    click.echo(message)
    if not success:
        click.echo("Failed to set up UI texts", err=True)
        return
    
    click.echo("UI setup completed successfully!")

if __name__ == '__main__':
    setup() 