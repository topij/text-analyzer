# src/nb_helpers/logging.py
import logging
from typing import Optional

# src/nb_helpers/logging.py

def configure_logging(
    level: str = "WARNING",
    format_string: Optional[str] = None
) -> None:
    """Configure logging with proper handler cleanup."""
    root_logger = logging.getLogger()
    
    # First remove all handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(level)
    
    # Configure handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        format_string or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set all existing loggers to the specified level
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(level)
    
    # Extra step: ensure these specific loggers remain silent unless DEBUG
    if level != "DEBUG":
        verbose_loggers = [
            "src.core.language_processing.factory",
            "src.core.language_processing.english",
            "src.utils.FileUtils.file_utils",
            "httpx",
            "libvoikko"
        ]
        
        for logger_name in verbose_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

def setup_debug_logging(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Add detailed formatter for debug output
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Replace existing handlers
    logger.handlers.clear()
    logger.addHandler(handler)

def silence_logger(logger_name: str) -> None:
    logging.getLogger(logger_name).setLevel(logging.WARNING)