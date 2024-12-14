# scripts/azure/setup_helpers.py

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    """Find the project root directory from current location."""
    current = Path.cwd()
    
    # If we're in notebooks dir, go up one level
    if current.name == "notebooks":
        return current.parent
        
    # If we're already at root (has src, scripts dirs)
    if (current / "src").exists() and (current / "scripts").exists():
        return current
        
    # Go up one level if needed
    parent = current.parent
    if (parent / "src").exists() and (parent / "scripts").exists():
        return parent
        
    raise ValueError(f"Cannot find project root from {current}")

def setup_voikko(project_root: Optional[Path] = None) -> bool:
    """Set up Voikko in Azure ML environment.
    
    Args:
        project_root: Optional path to project root. If None,
                     will attempt to find it automatically.
    
    Returns:
        bool: True if setup was successful, False otherwise.
    """
    try:
        # Find project root if not provided
        if project_root is None:
            project_root = find_project_root()
            logger.debug(f"Found project root: {project_root}")

        # Construct path to setup script
        setup_script = project_root / "scripts" / "azure" / "setup_voikko.sh"
        
        if not setup_script.exists():
            logger.error(f"Setup script not found at {setup_script}")
            return False

        logger.info("Running Voikko setup script...")
        result = subprocess.run(
            ['bash', str(setup_script)],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log the output
        for line in result.stdout.splitlines():
            logger.info(line)
            
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(line)

        # Verify installation
        lib_path = Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "libvoikko.so.1"
        if lib_path.exists() or lib_path.with_suffix("").exists():
            logger.info("Voikko library found at expected location")
            return True
        else:
            logger.warning("Voikko library not found at expected location")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running setup script: {e}")
        logger.error(f"Script output: {e.output}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Voikko setup: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run setup
    if setup_voikko():
        print("Voikko setup completed successfully")
    else:
        print("Voikko setup failed")