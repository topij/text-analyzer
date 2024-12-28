import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    """Configuration for a logger."""
    name: str
    level: Union[str, int]
    propagate: bool = True
    format_string: Optional[str] = None


class LoggingManager:
    """Centralized manager for all logging-related functionality."""

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LOGGERS = [
        # Core loggers
        "src.nb_helpers.analyzers",
        "src.analyzers.keyword_analyzer",
        "src.analyzers.theme_analyzer",
        "src.analyzers.category_analyzer",
        "src.utils.FileUtils.file_utils",
        "src.core.language_processing",
        # Integration loggers
        "httpx",
        "httpcore",
        "openai",
        "anthropic",
    ]

    def __init__(self):
        """Initialize logging manager."""
        self._logger = logging.getLogger(__name__)

    def configure_logging(
        self,
        level: str = "WARNING",
        format_string: Optional[str] = None,
        loggers: Optional[List[str]] = None,
    ) -> None:
        """Configure logging with proper handler cleanup.
        
        Args:
            level: Base logging level for all loggers
            format_string: Optional custom format string
            loggers: Optional list of logger names to configure
        """
        numeric_level = self._get_numeric_level(level)
        formatter = logging.Formatter(format_string or self.DEFAULT_FORMAT)
        
        # Configure root logger
        self._configure_root_logger(numeric_level, formatter)
        
        # Configure specific loggers
        loggers_to_configure = loggers or self.DEFAULT_LOGGERS
        self._configure_loggers(loggers_to_configure, level, numeric_level)

    def setup_debug_logging(self, logger_name: str) -> None:
        """Set up debug logging for a specific logger."""
        logger = logging.getLogger(logger_name)
        self._configure_logger(
            LoggerConfig(
                name=logger_name,
                level=logging.DEBUG,
                format_string=self.DEFAULT_FORMAT
            )
        )

    def silence_logger(self, logger_name: str) -> None:
        """Silence a specific logger by setting it to WARNING level."""
        self._configure_logger(
            LoggerConfig(name=logger_name, level=logging.WARNING)
        )

    def verify_logging_setup(self, show_hierarchy: bool = False) -> None:
        """Verify logging configuration with optional hierarchy information."""
        loggers_to_check = [""] + (self.DEFAULT_LOGGERS)  # Root + default loggers
        
        print("\nLogging Configuration:")
        print("-" * 50)
        
        for name in loggers_to_check:
            self._display_logger_info(name, show_hierarchy)

    def reset_all_loggers(self, level: Union[str, int] = logging.DEBUG) -> None:
        """Reset all loggers to specified level."""
        numeric_level = self._get_numeric_level(level)
        root = logging.getLogger()
        
        # Reset root logger
        root.setLevel(numeric_level)
        for handler in root.handlers:
            handler.setLevel(numeric_level)
            
        # Reset all existing loggers
        for logger_name in self.DEFAULT_LOGGERS:
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)

    def _get_numeric_level(self, level: Union[str, int]) -> int:
        """Convert string level to numeric if needed."""
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.WARNING)
        return level

    def _configure_root_logger(
        self, level: int, formatter: logging.Formatter
    ) -> None:
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root_logger.addHandler(handler)

    def _configure_loggers(
        self, logger_names: List[str], level: str, numeric_level: int
    ) -> None:
        """Configure multiple loggers."""
        for logger_name in logger_names:
            is_http_logger = any(
                name in logger_name.lower()
                for name in ["httpx", "httpcore", "openai", "anthropic"]
            )
            
            logger_level = (
                logging.INFO if is_http_logger and level == "DEBUG"
                else numeric_level
            )
            
            self._configure_logger(
                LoggerConfig(name=logger_name, level=logger_level)
            )

    def _configure_logger(self, config: LoggerConfig) -> None:
        """Configure a single logger."""
        logger = logging.getLogger(config.name)
        logger.handlers.clear()
        logger.propagate = config.propagate
        logger.setLevel(self._get_numeric_level(config.level))
        
        if config.format_string:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(config.format_string))
            handler.setLevel(logger.level)
            logger.addHandler(handler)

    def _get_logger_hierarchy(self, logger_name: str) -> List[str]:
        """Get the hierarchy of logger names from root to specified logger."""
        if not logger_name:
            return []
        parts = logger_name.split(".")
        return [".".join(parts[: i + 1]) for i in range(len(parts))]

    def _display_logger_info(self, name: str, show_hierarchy: bool) -> None:
        """Display information about a specific logger."""
        logger = logging.getLogger(name)
        logger_name = "root" if name == "" else name
        print(f"\nLogger: {logger_name}")
        
        if show_hierarchy and name:
            print("Hierarchy:")
            for ancestor in self._get_logger_hierarchy(name):
                ancestor_logger = logging.getLogger(ancestor)
                print(
                    f"  {ancestor}: "
                    f"{logging.getLevelName(ancestor_logger.level)}"
                )
        
        print(f"Set Level: {logging.getLevelName(logger.level)}")
        print(
            f"Effective Level: "
            f"{logging.getLevelName(logger.getEffectiveLevel())}"
        )
        print(f"Propagates to root: {logger.propagate}")
        
        if logger.handlers:
            print("Handlers:")
            for i, handler in enumerate(logger.handlers):
                print(
                    f"  Handler {i+1} level: "
                    f"{logging.getLevelName(handler.level)}"
                )
        else:
            print("No handlers (uses root handlers)")
