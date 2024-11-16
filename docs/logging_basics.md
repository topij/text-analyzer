# Logging in Python Projects

## Overview

Python's logging system is hierarchical, with loggers organized in a tree structure based on their names (e.g., "src.utils" is a child of "src"). This allows for flexible configuration and efficient log management.

## Basic Concepts

### Logger Levels (from least to most severe)

- DEBUG (10): Detailed information for diagnosing problems
- INFO (20): Confirmation that things are working as expected
- WARNING (30): Indication that something unexpected happened
- ERROR (40): A more serious problem occurred
- CRITICAL (50): Program may not be able to continue

### Logger Hierarchy

- Root Logger: The base of the hierarchy (empty string name)
- Child Loggers: Named using dot notation (e.g., "src.utils.file_utils")
- Level Inheritance: Loggers with NOTSET level inherit from their parent
- Propagation: Messages propagate up the hierarchy unless stopped

## Configuration Example

```python
import logging

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure specific logger
logger = logging.getLogger('src.utils')
logger.setLevel(logging.INFO)  # Override for this logger

# Usage
logger.debug('Debug message')  # Won't show (INFO level)
logger.info('Info message')    # Will show
```

## Best Practices

1. **Configuration**
   - Configure logging early in application startup
   - Use root logger for default settings
   - Override specific loggers only when needed

2. **Level Usage**
   - DEBUG: Development/troubleshooting info
   - INFO: Progress and status info
   - WARNING: Unexpected but handleable situations
   - ERROR: Functionality is affected
   - CRITICAL: Application cannot continue

3. **Message Guidelines**
   - Make messages clear and actionable
   - Include relevant context
   - Use appropriate log levels
   - Consider performance impact of debug messages

## Verification and Troubleshooting

```python
# Check logger configuration
def verify_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    print(f"Logger: {logger.name}")
    print(f"Level set: {logging.getLevelName(logger.level)}")
    print(f"Effective level: {logging.getLevelName(logger.getEffectiveLevel())}")
    print(f"Propagates: {logger.propagate}")
```

## Common Issues

- Logger showing NOTSET but still logging: Inheriting from parent
- Messages not appearing: Check effective level and handler levels
- Multiple messages: Check for duplicate handlers
- Performance issues: Too many debug messages in production

## Tips

- Use `__name__` as logger name in modules
- Configure handlers on root logger for centralized control
- Test logging configuration in different environments
- Consider using a logging configuration file
- Remember to handle logging in asyncio contexts properly

## Additional Tips for Project Use

### Testing with Logging

- Use temporary log levels during tests
- Reset logging state between tests
- Capture and verify log messages in tests
- Use `caplog` fixture in pytest for log verification

### Development Workflow

- Set DEBUG level in development
- Use INFO for production by default
- Consider environment-specific log configuration
- Look into structlog for structured logging if needed

### Example Configuration for Different Environments

```python
# development.py
configure_logging(
    level="DEBUG",
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
)

# production.py
configure_logging(
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
```

For more information, refer to:

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)