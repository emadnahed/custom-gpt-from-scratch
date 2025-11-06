"""
Python Utilities - Helper functions for Python executable management
"""

import os
import sys
from pathlib import Path


def get_venv_python():
    """
    Get the Python executable from venv if it exists

    Returns:
        str: Path to venv Python, or sys.executable if no venv
    """
    # Check common venv locations
    venv_paths = [
        'venv/bin/python',      # Unix/Mac
        'venv/Scripts/python.exe',  # Windows
        '.venv/bin/python',     # Alternative venv name
        '.venv/Scripts/python.exe',
    ]

    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            return os.path.abspath(venv_path)

    # If no venv found, return current executable
    return sys.executable


def is_in_virtualenv():
    """
    Check if we're running inside a virtual environment

    Returns:
        bool: True if in a virtual environment
    """
    # Check for VIRTUAL_ENV environment variable
    if os.environ.get('VIRTUAL_ENV'):
        return True

    # Check if sys.prefix differs from sys.base_prefix
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True

    return False


def get_python_for_scripts():
    """
    Get the best Python executable to use for running scripts

    Priority:
    1. If in venv, use sys.executable (current Python)
    2. If venv exists but not activated, use venv Python
    3. Otherwise use sys.executable with warning

    Returns:
        tuple: (python_path, warning_message or None)
    """
    # If we're in a venv, use current Python
    if is_in_virtualenv():
        return sys.executable, None

    # If venv exists but not activated, use it
    venv_python = get_venv_python()
    if venv_python != sys.executable:
        return venv_python, None

    # No venv available - warn user
    warning = (
        "⚠️  Virtual environment not detected!\n"
        "   For best results, activate the virtual environment:\n"
        "   source venv/bin/activate  (Mac/Linux)\n"
        "   venv\\Scripts\\activate     (Windows)"
    )

    return sys.executable, warning


def check_dependencies():
    """
    Check if required dependencies are installed

    Returns:
        tuple: (all_installed: bool, missing: list)
    """
    required = ['torch', 'numpy', 'tqdm']
    missing = []

    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    return len(missing) == 0, missing
