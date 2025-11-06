"""
Hardware Detection Utility for GPT Training

This module automatically detects available hardware accelerators and provides
a user-friendly interface for hardware selection.

Supported platforms:
- NVIDIA CUDA (GPU)
- AMD ROCm (GPU)
- Apple Metal Performance Shaders (MPS)
- Intel XPU
- CPU (with optimizations)
"""

import torch
import platform
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HardwareType(Enum):
    """Enumeration of supported hardware types"""
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    XPU = "xpu"
    CPU = "cpu"


@dataclass
class HardwareDevice:
    """Represents a detected hardware device"""
    type: HardwareType
    name: str
    available: bool
    device_count: int = 1
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    supports_bf16: bool = False
    supports_fp16: bool = False
    details: Optional[str] = None


class HardwareDetector:
    """Detects and manages available hardware for training"""

    def __init__(self):
        self.devices: List[HardwareDevice] = []
        self._detect_all_hardware()

    def _detect_cuda(self) -> HardwareDevice:
        """Detect NVIDIA CUDA devices"""
        available = torch.cuda.is_available()

        if available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)

            # Get memory info
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Check compute capability
            compute_cap = torch.cuda.get_device_capability(0)
            compute_capability = f"{compute_cap[0]}.{compute_cap[1]}"

            # Check precision support
            supports_bf16 = torch.cuda.is_bf16_supported()
            supports_fp16 = True  # CUDA always supports fp16

            details = f"{device_count} device(s), {memory_gb:.1f} GB memory"

            return HardwareDevice(
                type=HardwareType.CUDA,
                name=device_name,
                available=True,
                device_count=device_count,
                memory_gb=memory_gb,
                compute_capability=compute_capability,
                supports_bf16=supports_bf16,
                supports_fp16=supports_fp16,
                details=details
            )
        else:
            return HardwareDevice(
                type=HardwareType.CUDA,
                name="NVIDIA CUDA",
                available=False,
                details="CUDA not available or no NVIDIA GPU detected"
            )

    def _detect_rocm(self) -> HardwareDevice:
        """Detect AMD ROCm devices"""
        try:
            # Check if ROCm is available
            # PyTorch with ROCm uses 'cuda' backend but on AMD hardware
            if torch.cuda.is_available() and platform.system() == "Linux":
                # Try to detect if it's actually ROCm
                try:
                    rocm_version = subprocess.check_output(
                        ["rocm-smi", "--showproductname"],
                        stderr=subprocess.DEVNULL
                    ).decode().strip()

                    return HardwareDevice(
                        type=HardwareType.ROCM,
                        name="AMD ROCm GPU",
                        available=True,
                        device_count=torch.cuda.device_count(),
                        supports_fp16=True,
                        details=f"ROCm detected: {rocm_version}"
                    )
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

            return HardwareDevice(
                type=HardwareType.ROCM,
                name="AMD ROCm",
                available=False,
                details="ROCm not available or no AMD GPU detected"
            )
        except Exception:
            return HardwareDevice(
                type=HardwareType.ROCM,
                name="AMD ROCm",
                available=False,
                details="ROCm detection failed"
            )

    def _detect_mps(self) -> HardwareDevice:
        """Detect Apple Metal Performance Shaders (Apple Silicon)"""
        try:
            # MPS is only available on macOS with Apple Silicon
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Get system info
                system_info = platform.processor()

                return HardwareDevice(
                    type=HardwareType.MPS,
                    name="Apple Metal (MPS)",
                    available=True,
                    supports_fp16=True,
                    supports_bf16=False,  # MPS doesn't support bfloat16 yet
                    details=f"Apple Silicon: {system_info}"
                )
            else:
                if platform.system() == "Darwin":
                    details = "MPS not available (requires macOS 12.3+ and Apple Silicon)"
                else:
                    details = "Not on macOS"

                return HardwareDevice(
                    type=HardwareType.MPS,
                    name="Apple Metal (MPS)",
                    available=False,
                    details=details
                )
        except Exception:
            return HardwareDevice(
                type=HardwareType.MPS,
                name="Apple Metal (MPS)",
                available=False,
                details="MPS detection failed"
            )

    def _detect_xpu(self) -> HardwareDevice:
        """Detect Intel XPU devices"""
        try:
            # Check for Intel Extension for PyTorch
            try:
                import intel_extension_for_pytorch as ipex
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    device_count = torch.xpu.device_count()
                    device_name = torch.xpu.get_device_name(0)

                    return HardwareDevice(
                        type=HardwareType.XPU,
                        name=device_name,
                        available=True,
                        device_count=device_count,
                        supports_fp16=True,
                        details=f"{device_count} Intel XPU device(s) with IPEX"
                    )
            except ImportError:
                pass

            return HardwareDevice(
                type=HardwareType.XPU,
                name="Intel XPU",
                available=False,
                details="Intel Extension for PyTorch not installed"
            )
        except Exception:
            return HardwareDevice(
                type=HardwareType.XPU,
                name="Intel XPU",
                available=False,
                details="XPU detection failed"
            )

    def _detect_cpu(self) -> HardwareDevice:
        """Detect CPU and its capabilities"""
        cpu_name = platform.processor() or platform.machine()
        system = platform.system()

        # CPU is always available
        details = f"{system} - {cpu_name}"

        return HardwareDevice(
            type=HardwareType.CPU,
            name=f"CPU ({cpu_name})",
            available=True,
            supports_fp16=True,
            supports_bf16=False,
            details=details
        )

    def _detect_all_hardware(self):
        """Detect all available hardware"""
        self.devices = [
            self._detect_cuda(),
            self._detect_rocm(),
            self._detect_mps(),
            self._detect_xpu(),
            self._detect_cpu(),
        ]

    def get_available_devices(self) -> List[HardwareDevice]:
        """Get list of available hardware devices"""
        return [device for device in self.devices if device.available]

    def get_all_devices(self) -> List[HardwareDevice]:
        """Get list of all devices (including unavailable)"""
        return self.devices

    def get_best_device(self) -> HardwareDevice:
        """Get the best available device for training"""
        # Priority order: CUDA > ROCm > MPS > XPU > CPU
        priority = [HardwareType.CUDA, HardwareType.ROCM, HardwareType.MPS,
                   HardwareType.XPU, HardwareType.CPU]

        for hw_type in priority:
            for device in self.devices:
                if device.type == hw_type and device.available:
                    return device

        # Fallback to CPU (should always be available)
        return self.devices[-1]

    def get_device_string(self, device: HardwareDevice) -> str:
        """Get PyTorch device string for a hardware device"""
        if device.type == HardwareType.CUDA:
            return "cuda"
        elif device.type == HardwareType.ROCM:
            return "cuda"  # ROCm uses cuda backend in PyTorch
        elif device.type == HardwareType.MPS:
            return "mps"
        elif device.type == HardwareType.XPU:
            return "xpu"
        else:
            return "cpu"

    def get_optimal_dtype(self, device: HardwareDevice) -> str:
        """Get optimal dtype for a device"""
        if device.supports_bf16:
            return "bfloat16"
        elif device.supports_fp16:
            return "float16"
        else:
            return "float32"

    def print_hardware_summary(self):
        """Print a formatted summary of detected hardware"""
        print("\n" + "="*70)
        print("HARDWARE DETECTION SUMMARY")
        print("="*70)

        for i, device in enumerate(self.devices):
            status = "✓ AVAILABLE" if device.available else "✗ UNAVAILABLE"
            status_color = "\033[92m" if device.available else "\033[90m"  # Green or gray
            reset_color = "\033[0m"

            print(f"\n{status_color}[{i+1}] {device.type.value.upper()}: {status}{reset_color}")
            print(f"    Name: {device.name}")

            if device.device_count > 1:
                print(f"    Device Count: {device.device_count}")

            if device.memory_gb:
                print(f"    Memory: {device.memory_gb:.1f} GB")

            if device.compute_capability:
                print(f"    Compute Capability: {device.compute_capability}")

            if device.available:
                precisions = []
                if device.supports_bf16:
                    precisions.append("bfloat16")
                if device.supports_fp16:
                    precisions.append("float16")
                precisions.append("float32")
                print(f"    Supported Precisions: {', '.join(precisions)}")

            if device.details:
                print(f"    Details: {device.details}")

        print("\n" + "="*70)
        best = self.get_best_device()
        print(f"RECOMMENDED: {best.type.value.upper()} - {best.name}")
        print(f"Device String: {self.get_device_string(best)}")
        print(f"Optimal Dtype: {self.get_optimal_dtype(best)}")
        print("="*70 + "\n")


def interactive_device_selection() -> Tuple[str, str]:
    """
    Interactive CLI for device selection
    Returns: (device_string, dtype_string)
    """
    detector = HardwareDetector()
    detector.print_hardware_summary()

    available_devices = detector.get_available_devices()

    if len(available_devices) == 1:
        print("Only one device available. Using it automatically.\n")
        device = available_devices[0]
    else:
        print("Multiple devices available. Please select one:")
        for i, device in enumerate(available_devices):
            print(f"  [{i+1}] {device.type.value.upper()} - {device.name}")

        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(available_devices)}) or press Enter for recommended: ").strip()
                if choice == "":
                    device = detector.get_best_device()
                    print(f"Using recommended: {device.type.value.upper()}")
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(available_devices):
                    device = available_devices[idx]
                    break
                else:
                    print(f"Invalid choice. Please enter 1-{len(available_devices)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nUsing recommended device.")
                device = detector.get_best_device()
                break

    device_string = detector.get_device_string(device)
    dtype_string = detector.get_optimal_dtype(device)

    print(f"\nSelected: {device.type.value.upper()} - {device.name}")
    print(f"Device: {device_string}, Dtype: {dtype_string}\n")

    return device_string, dtype_string


def auto_detect_device() -> Tuple[str, str]:
    """
    Automatically detect and return best device
    Returns: (device_string, dtype_string)
    """
    detector = HardwareDetector()
    device = detector.get_best_device()
    device_string = detector.get_device_string(device)
    dtype_string = detector.get_optimal_dtype(device)

    return device_string, dtype_string


if __name__ == "__main__":
    # Demo the hardware detection
    interactive_device_selection()
