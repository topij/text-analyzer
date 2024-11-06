# src/core/config_and_logging.py

from .config import config_manager, logger

# Export main components
__all__ = ["config_manager", "logger", "get_logger"]


def get_logger(name: str):
    """Get a logger instance."""
    return config_manager.get_logger(name)
