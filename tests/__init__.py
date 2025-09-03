"""
Test suite for GammaFit package.

This module contains unit tests and integration tests for the
gamma-peak-identifier package.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Clean up test output directory before tests
def cleanup_test_output():
    """Remove all files from test output directory."""
    if TEST_OUTPUT_DIR.exists():
        for file in TEST_OUTPUT_DIR.glob("*"):
            if file.is_file():
                file.unlink()

# Test fixtures and utilities
def get_test_spectrum_path():
    """Get path to test spectrum file."""
    return TEST_DATA_DIR / "test_spectrum.csv"

def get_test_config_path():
    """Get path to test configuration file."""
    return TEST_DATA_DIR / "test_config.json"
