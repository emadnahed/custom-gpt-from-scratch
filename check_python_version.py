#!/usr/bin/env python3
"""
Python Version Checker

Ensures the correct Python version is being used.
Recommended: Python 3.11
Minimum: Python 3.8
"""

import shutil
import subprocess
import sys


def check_python_version(min_version=(3, 8), recommended_version=(3, 11)):
    """
    Check Python version and provide guidance

    Args:
        min_version: Minimum supported version tuple (major, minor)
        recommended_version: Recommended version tuple (major, minor)

    Returns:
        bool: True if version is acceptable, False otherwise
    """
    current = sys.version_info[:2]

    print(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"Python executable: {sys.executable}")
    print()

    if current < min_version:
        print(f"❌ ERROR: Python {min_version[0]}.{min_version[1]}+ is required")
        print(f"   You have: Python {current[0]}.{current[1]}")
        print()
        print("Please upgrade Python:")
        print("  - Download from: https://www.python.org/downloads/")
        print(f"  - Or use: python{recommended_version[0]}.{recommended_version[1]} command if available")
        print()
        return False

    if current < recommended_version:
        print(f"⚠️  WARNING: Python {recommended_version[0]}.{recommended_version[1]} is recommended")
        print(f"   You have: Python {current[0]}.{current[1]}")
        print(f"   Minimum required: Python {min_version[0]}.{min_version[1]}")
        print()
        print("Your version will work, but for best compatibility:")
        print(f"  - Consider upgrading to Python {recommended_version[0]}.{recommended_version[1]}")
        print()
        return True

    if current == recommended_version:
        print(f"✅ Perfect! You're using the recommended Python {recommended_version[0]}.{recommended_version[1]}")
        print()
        return True

    if current > recommended_version:
        print(f"✅ Good! Python {current[0]}.{current[1]} (newer than recommended {recommended_version[0]}.{recommended_version[1]})")
        print("   This should work fine!")
        print()
        return True

    return True


def get_python_command_suggestion():
    """
    Suggest the right Python command to use
    
    Returns:
        tuple or None: (command, version) of the best Python version found, or None if none found
    """
    print("Available Python commands on your system:")
    print()

    # Check for different Python versions in order of preference
    python_commands = [
        'python3.11',
        'python3.10',
        'python3.9',
        'python3.8',
        'python3',
        'python',
    ]

    available = []
    for cmd in python_commands:
        if shutil.which(cmd):
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                version = result.stdout.strip() or result.stderr.strip()
                available.append((cmd, version))
                print(f"  ✓ {cmd}: {version}")
            except (subprocess.SubprocessError, OSError):
                # Skip if command fails
                continue

    if not available:
        print("  ❌ No Python found!")
        return None

    print()
    print("Recommendation:")
    print(f"  Use: {available[0][0]}")
    print()
    print("To create virtual environment:")
    print(f"  {available[0][0]} -m venv venv")
    print()

    return available[0][0]


if __name__ == '__main__':
    print("="*70)
    print("PYTHON VERSION CHECK")
    print("="*70)
    print()

    is_ok = check_python_version()

    if not is_ok:
        print()
        get_python_command_suggestion()
        sys.exit(1)

    print("="*70)
    print("✅ Python version check passed!")
    print("="*70)
    sys.exit(0)
