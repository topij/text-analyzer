import os
import sys
import argparse
from pathlib import Path
import subprocess

def setup_environment():
    """Setup the Python path and working directory."""
    project_root = Path(__file__).parent.parent.absolute()
    src_path = project_root / 'src'
    ui_path = project_root / 'ui'
    
    # Add paths to PYTHONPATH environment variable
    python_path = os.environ.get('PYTHONPATH', '')
    new_paths = [str(project_root), str(src_path), str(ui_path)]
    
    if python_path:
        new_paths.extend(python_path.split(os.pathsep))
    
    os.environ['PYTHONPATH'] = os.pathsep.join(new_paths)
    
    # Also add to sys.path for the current session
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(ui_path))
    
    os.chdir(project_root)
    return project_root, src_path, ui_path

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import streamlit
        import pandas as pd
        import openai
        import langdetect
        return True
    except ImportError as e:
        print(f"Error: Missing required package - {e.name}")
        print("Please install required packages using: pip install -r requirements.txt")
        return False

def run_app(port=None, browser=True):
    """Run the Streamlit app with specified options."""
    project_root, src_path, ui_path = setup_environment()
    
    if not check_requirements():
        sys.exit(1)

    # Build Streamlit command
    cmd = ["streamlit", "run", str(ui_path / "app.py")]
    if port:
        cmd.extend(["--server.port", str(port)])
    if not browser:
        cmd.extend(["--server.headless", "true"])

    print("Starting Text Analyzer...")
    print(f"Project root: {project_root}")
    print(f"Python path: {os.environ['PYTHONPATH']}")
    print("Use Ctrl+C to stop the application")
    
    # Run the Streamlit command with the updated environment
    try:
        subprocess.run(cmd, env=os.environ)
    except KeyboardInterrupt:
        print("\nStopping the application...")
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Text Analyzer Streamlit app')
    parser.add_argument('--port', type=int, help='Port to run the app on')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open the browser automatically')
    
    args = parser.parse_args()
    run_app(port=args.port, browser=not args.no_browser) 