import click
from pathlib import Path
from text_manager import get_ui_text_manager
from FileUtils import FileUtils

@click.group()
def cli():
    """UI Text Management Tools"""
    pass

@cli.command()
@click.argument('output_file', type=click.Path(), required=False)
def export(output_file):
    """Export UI texts to Excel for editing"""
    try:
        file_utils = FileUtils()
        ui_manager = get_ui_text_manager(file_utils=file_utils)
        
        if output_file:
            # If output file is specified, ensure it's in the config directory
            output_path = file_utils.get_data_path("config") / Path(output_file).name
        else:
            output_path = None
            
        result_path = ui_manager.export_to_excel(output_path)
        click.echo(f"Successfully exported UI texts to: {result_path}")
    except Exception as e:
        click.echo(f"Error exporting UI texts: {e}", err=True)

@cli.command(name='import')
@click.argument('input_file', type=click.Path(exists=True))
def import_texts(input_file):
    """Import UI texts from Excel file"""
    try:
        file_utils = FileUtils()
        ui_manager = get_ui_text_manager(file_utils=file_utils)
        
        # Convert input path to absolute path if it's relative to project root
        input_path = Path(input_file)
        if not input_path.is_absolute():
            project_root = Path(file_utils.config.get("project_root", "."))
            input_path = project_root / input_path
            
        ui_manager.import_from_excel(input_path)
        click.echo(f"Successfully imported UI texts from: {input_path}")
    except Exception as e:
        click.echo(f"Error importing UI texts: {e}", err=True)

if __name__ == '__main__':
    cli() 