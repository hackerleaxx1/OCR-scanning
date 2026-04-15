"""
Configuration loader for the Invoice OCR API.

Loads application settings from config.yaml including:
- App settings (host, port, directories)
- OCR configuration
- ML model hyperparameters
- Storage paths for vendors and invoices
"""

import yaml
import os
from pathlib import Path

# Path to the YAML configuration file (located in same directory as this module)
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    """
    Load configuration from config.yaml file.

    Returns:
        dict: Parsed YAML configuration as a nested dictionary

    Raises:
        FileNotFoundError: If config.yaml is missing
        yaml.YAMLError: If config.yaml is malformed
    """
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# Global configuration object - loaded once at module import
# Use this throughout the application: from config import config
config = load_config()
