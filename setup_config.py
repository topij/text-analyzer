# setup_config.py

import os
import shutil
from pathlib import Path

from FileUtils import FileUtils


def setup_test_config():
    """Set up test configuration files."""
    print("Setting up test configuration...")
    print(f"Current directory: {os.getcwd()}")

    # Initialize FileUtils
    utils = FileUtils()
    print(f"Project root: {utils.project_root}")

    # Create config directories
    config_dir = utils.get_data_path("configurations")
    stop_words_dir = config_dir / "stop_words"
    stop_words_dir.mkdir(exist_ok=True)
    print(f"Created directory: {stop_words_dir}")

    # Create semantic analyzer config
    semantic_config = {
        "semantic_analyzer": {
            "default_language": "en",
            "content_column": "content",
            "analysis": {
                "keywords": {
                    "max_keywords": 5,
                    "min_keyword_length": 3,
                    "include_compounds": True,
                },
                "themes": {
                    "max_themes": 3,
                    "min_confidence": 0.5,
                    "include_hierarchy": True,
                },
                "categories": {
                    "max_categories": 3,
                    "min_confidence": 0.3,
                    "require_evidence": True,
                },
            },
            "models": {
                "default_provider": "openai",
                "default_model": "gpt-4o-mini",
                "parameters": {"temperature": 0.0, "max_tokens": 1000},
            },
        }
    }

    # Update FileUtils config
    utils.config.update(semantic_config)
    print("Updated FileUtils configuration")

    # Save updated config
    config_file = config_dir / "config.yaml"
    utils.save_yaml(utils.config, config_file, include_timestamp=False)
    print(f"Saved configuration to: {config_file}")

    # Copy stop words files
    en_stop_words = stop_words_dir / "en.txt"
    fi_stop_words = stop_words_dir / "fi.txt"

    # Create stop words content
    en_content = """
a
an
and
are
as
at
be
by
for
from
has
he
in
is
it
its
of
on
that
the
to
was
were
will
with
    """.strip()

    fi_content = """
ja
on
ei
se
että
olla
joka
sen
hän
ne
sitä
tämä
kun
oli
myös
jos
sekä
niin
vain
mutta
    """.strip()

    # Write stop words files
    en_stop_words.write_text(en_content, encoding="utf-8")
    fi_stop_words.write_text(fi_content, encoding="utf-8")
    print(f"Created stop words files in: {stop_words_dir}")

    # Verify configuration
    print("\nVerifying configuration...")
    try:
        loaded_config = utils.load_yaml(config_file)
        assert "semantic_analyzer" in loaded_config
        print("✓ Configuration loaded successfully")

        assert en_stop_words.exists() and fi_stop_words.exists()
        print("✓ Stop words files created successfully")

    except Exception as e:
        print(f"Error verifying configuration: {e}")
        raise

    print("\nTest configuration setup complete!")
    print("Directory structure:")
    print(f"  {config_dir}/")
    print(f"  ├── config.yaml")
    print(f"  └── stop_words/")
    print(f"      ├── en.txt")
    print(f"      └── fi.txt")


if __name__ == "__main__":
    setup_test_config()
