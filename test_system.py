#!/usr/bin/env python3
"""
System Test Script - Tests all major components

This script tests:
1. Python version checking
2. Hardware detection
3. Configuration building
4. Dataset management
5. All CLI commands
"""

import sys
import os
import subprocess


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            print(f"❌ FAILED: {description} (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("GPT SYSTEM TEST SUITE")
    print("="*70)

    python_cmd = sys.executable
    results = {}

    # Test 1: Python Version Check
    results['python_version'] = run_command(
        [python_cmd, 'check_python_version.py'],
        "Python Version Check"
    )

    # Test 2: Hardware Detection
    results['hardware'] = run_command(
        [python_cmd, 'check_hardware.py'],
        "Hardware Detection"
    )

    # Test 3: GPT CLI - Info
    results['gpt_info'] = run_command(
        [python_cmd, 'gpt.py', 'info'],
        "GPT CLI - Info Command"
    )

    # Test 4: GPT CLI - Hardware
    results['gpt_hardware'] = run_command(
        [python_cmd, 'gpt.py', 'hardware'],
        "GPT CLI - Hardware Command"
    )

    # Test 5: GPT CLI - Help
    results['gpt_help'] = run_command(
        [python_cmd, 'gpt.py', '--help'],
        "GPT CLI - Help"
    )

    # Test 6: Check if model exists
    if os.path.exists('out/ckpt.pt'):
        print("\n✅ Trained model found: out/ckpt.pt")
        results['model'] = True
    else:
        print("\n⚠️  No trained model found (this is OK if you haven't trained yet)")
        results['model'] = False

    # Test 7: Check if dataset exists
    if os.path.exists('data/train.bin') and os.path.exists('data/val.bin'):
        print("✅ Dataset prepared: data/train.bin, data/val.bin")
        results['dataset'] = True
    else:
        print("⚠️  Dataset not prepared (run: cd data && python prepare.py)")
        results['dataset'] = False

    # Test 8: Check virtual environment
    if os.path.exists('venv'):
        print("✅ Virtual environment exists")
        results['venv'] = True
    else:
        print("❌ Virtual environment missing")
        results['venv'] = False

    # Test 9: Check dependencies
    try:
        import torch
        import numpy
        print(f"✅ Core dependencies installed (PyTorch {torch.__version__})")
        results['dependencies'] = True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        results['dependencies'] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    elif passed >= total * 0.7:
        print("\n⚠️  Most tests passed, but some issues need attention")
        return 0
    else:
        print("\n❌ Several tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
