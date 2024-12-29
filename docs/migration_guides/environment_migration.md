# Migrating to EnvironmentManager

This guide helps you migrate from the legacy `AnalysisEnvironment` to the new `EnvironmentManager`.

## Why Migrate?

The new `EnvironmentManager` provides several improvements:
- Better separation of concerns between logging and environment management
- More robust configuration handling
- Improved type safety
- Better support for different environment types
- Centralized logging configuration

## Migration Steps

### 1. Update Import Statements

Replace:
```python
from src.nb_helpers.environment import AnalysisEnvironment
```

With:
```python
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig
```

### 2. Replace Environment Initialization

Replace:
```python
env = AnalysisEnvironment(
    env_type="local",
    project_root=Path().resolve(),
    log_level="INFO"
)
```

With:
```python
env_config = EnvironmentConfig(
    env_type="local",
    project_root=Path().resolve(),
    log_level="INFO"
)
environment = EnvironmentManager(env_config)
```

### 3. Update Component Access

The method for accessing components remains the same:
```python
components = environment.get_components()
```

### 4. Handle Deprecation Warnings

You may see deprecation warnings when using `AnalysisEnvironment`. These warnings indicate that you should migrate to `EnvironmentManager`. The legacy class will be removed in a future version.

## Example Migration

Here's a complete example of migrating a script:

```python
# Before
from src.nb_helpers.environment import AnalysisEnvironment

env = AnalysisEnvironment(log_level="INFO")
components = env.get_components()
runner = MyRunner(components)
```

```python
# After
from src.nb_helpers.environment_manager import EnvironmentManager, EnvironmentConfig

env_config = EnvironmentConfig(log_level="INFO")
environment = EnvironmentManager(env_config)
components = environment.get_components()
runner = MyRunner(components)
```

## Need Help?

If you encounter any issues during migration, please:
1. Check that all required components are properly initialized
2. Verify that your logging configuration is correctly set up
3. Ensure all environment variables are properly loaded

For more detailed information, refer to the `EnvironmentManager` documentation.
