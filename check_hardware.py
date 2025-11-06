#!/usr/bin/env python3
"""
Hardware Detection CLI Tool

This script detects and displays all available hardware accelerators on your system.

Usage:
    python check_hardware.py              # Show hardware summary
    python check_hardware.py --interactive # Interactively select hardware
    python check_hardware.py --json        # Output as JSON
"""

import argparse
import json
import sys

from utils.hardware_detector import HardwareDetector, interactive_device_selection


def main():
    parser = argparse.ArgumentParser(
        description='Detect and display available hardware for GPT training'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactively select hardware'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--recommended',
        action='store_true',
        help='Only show the recommended device'
    )

    args = parser.parse_args()

    if args.interactive:
        # Interactive mode
        device, dtype = interactive_device_selection()
        print(f"\nYou can now start training with:")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")
        return

    # Create detector
    detector = HardwareDetector()

    if args.json:
        # JSON output
        devices_data = []
        for device in detector.get_all_devices():
            device_dict = {
                'type': device.type.value,
                'name': device.name,
                'available': device.available,
                'device_count': device.device_count,
                'memory_gb': device.memory_gb,
                'compute_capability': device.compute_capability,
                'supports_bf16': device.supports_bf16,
                'supports_fp16': device.supports_fp16,
                'details': device.details
            }
            devices_data.append(device_dict)

        best = detector.get_best_device()
        output = {
            'devices': devices_data,
            'recommended': {
                'type': best.type.value,
                'device_string': detector.get_device_string(best),
                'dtype': detector.get_optimal_dtype(best)
            }
        }
        print(json.dumps(output, indent=2))

    elif args.recommended:
        # Only show recommended device
        best = detector.get_best_device()
        print(f"Recommended Device: {best.type.value.upper()}")
        print(f"  Name: {best.name}")
        print(f"  Device String: {detector.get_device_string(best)}")
        print(f"  Optimal Dtype: {detector.get_optimal_dtype(best)}")
        if best.details:
            print(f"  Details: {best.details}")

    else:
        # Standard output - show all hardware
        detector.print_hardware_summary()


if __name__ == '__main__':
    main()
