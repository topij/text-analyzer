# src/nb_helpers/logging.py
import logging
from typing import Optional, List


def configure_logging(level: str = "WARNING", format_string: Optional[str] = None) -> None:
    """Configure logging with proper handler cleanup."""
    # Convert string level to numeric
    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    # First configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove all existing handlers
    root_logger.handlers.clear()

    # Add new handler to root logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)
    root_logger.addHandler(handler)

    # Explicitly configure all loggers we use
    loggers_to_configure = [
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

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        # Clear any existing handlers
        logger.handlers.clear()
        # Ensure propagation to root
        logger.propagate = True

        if level == "DEBUG":
            if logger_name in ["httpx", "httpcore", "openai", "anthropic"]:
                # Keep HTTP-related logging at INFO
                logger.setLevel(logging.INFO)
            else:
                # Set everything else to DEBUG
                logger.setLevel(numeric_level)
        else:
            # In non-debug mode, set all to specified level
            logger.setLevel(numeric_level)

    # Verify configuration
    test_logger = logging.getLogger("src.nb_helpers.logging")
    test_logger.debug("Logging configured at %s level", level)


# In src/nb_helpers/logging.py


def verify_logging_setup() -> None:
    """Verify logging configuration is correct.

    Shows both the actual logger level and effective level (inherited from parent).
    """
    loggers_to_check = [
        "",  # Root logger
        "src.nb_helpers.analyzers",
        "src.analyzers.keyword_analyzer",
        "src.analyzers.theme_analyzer",
        "src.analyzers.category_analyzer",
        "src.utils.FileUtils.file_utils",
        "httpx",
    ]

    print("\nLogging Configuration:")
    print("-" * 50)

    for name in loggers_to_check:
        logger = logging.getLogger(name)
        logger_name = "root" if name == "" else name
        print(f"\nLogger: {logger_name}")
        print(f"Set Level: {logging.getLevelName(logger.level)}")
        print(f"Effective Level: {logging.getLevelName(logger.getEffectiveLevel())}")
        print(f"Handlers: {len(logger.handlers)}")
        print(f"Propagate: {logger.propagate}")

        if logger.handlers:
            for i, handler in enumerate(logger.handlers):
                print(f"Handler {i+1} level: {logging.getLevelName(handler.level)}")


def _get_logger_hierarchy(logger_name: str) -> List[str]:
    """Get the hierarchy of logger names from root to the specified logger."""
    parts = logger_name.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def verify_logging_setup_with_hierarchy() -> None:
    """Verify logging configuration with detailed hierarchy information."""
    loggers_to_check = [
        "",  # Root logger
        "src.nb_helpers.analyzers",
        "src.analyzers.keyword_analyzer",
        "src.analyzers.theme_analyzer",
        "src.analyzers.category_analyzer",
        "src.utils.FileUtils.file_utils",
        "httpx",
    ]

    print("\nLogging Configuration:")
    print("-" * 50)

    for name in loggers_to_check:
        logger = logging.getLogger(name)
        logger_name = "root" if name == "" else name
        print(f"\nLogger: {logger_name}")

        # Show hierarchy
        if name:
            print("Hierarchy:")
            hierarchy = _get_logger_hierarchy(name)
            for ancestor in hierarchy:
                ancestor_logger = logging.getLogger(ancestor)
                print(f"  {ancestor}: {logging.getLevelName(ancestor_logger.level)}")

        print(f"Set Level: {logging.getLevelName(logger.level)}")
        print(f"Effective Level: {logging.getLevelName(logger.getEffectiveLevel())}")
        print(f"Propagates to root: {logger.propagate}")

        if logger.handlers:
            print("Handlers:")
            for i, handler in enumerate(logger.handlers):
                print(f"  Handler {i+1} level: {logging.getLevelName(handler.level)}")
        else:
            print("No handlers (uses root handlers)")


def setup_debug_logging(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Add detailed formatter for debug output
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Replace existing handlers
    logger.handlers.clear()
    logger.addHandler(handler)


def silence_logger(logger_name: str) -> None:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def reset_debug_logging():
    """Reset all loggers to DEBUG level."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in root.handlers:
        handler.setLevel(logging.DEBUG)

    # Reset specific loggers
    loggers_to_debug = [
        "src.nb_helpers.analyzers",
        "src.analyzers.keyword_analyzer",
        "src.analyzers.theme_analyzer",
        "src.analyzers.category_analyzer",
        "src.utils.FileUtils.file_utils",
        "src.loaders.parameter_handler",
    ]

    for logger_name in loggers_to_debug:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
