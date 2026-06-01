"""GPU detection and hardware validation utilities.

Provides runtime detection of CUDA/GPU availability for model routing
and performance optimization decisions. When GPU is available, enables
speculative decoding and multi-model inference strategies.

Usage:
    from common.gpu_utils import has_cuda, get_gpu_info, should_use_speculative_decoding

    if has_cuda():
        # Use GPU-optimized models
        pass
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """GPU hardware information."""
    available: bool
    cuda_version: str = ""
    device_count: int = 0
    device_name: str = ""
    total_memory_gb: float = 0.0
    driver_version: str = ""


def has_cuda() -> bool:
    """Check if CUDA is available.

    Returns True if NVIDIA GPU with CUDA drivers is detected.
    """
    # Check environment variable override
    if os.getenv("FORCE_CPU", "").lower() == "true":
        return False

    try:
        # Try nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try PyTorch CUDA detection if available
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    return False


def get_gpu_info() -> GPUInfo:
    """Get detailed GPU hardware information.

    Returns GPUInfo with all fields populated if GPU is available,
    otherwise returns GPUInfo(available=False).
    """
    if not has_cuda():
        return GPUInfo(available=False)

    try:
        # Query nvidia-smi for details
        queries = [
            "name",
            "memory.total",
            "driver_version",
            "cuda_version"
        ]
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(queries)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode != 0:
            return GPUInfo(available=True)

        lines = result.stdout.strip().split("\n")
        if not lines:
            return GPUInfo(available=True)

        # Parse first GPU
        parts = lines[0].split(", ")
        if len(parts) >= 4:
            name = parts[0].strip()
            memory_mb = float(parts[1].strip())
            driver = parts[2].strip()
            cuda = parts[3].strip()

            return GPUInfo(
                available=True,
                cuda_version=cuda,
                device_count=len(lines),
                device_name=name,
                total_memory_gb=round(memory_mb / 1024, 1),
                driver_version=driver
            )
    except Exception:
        pass

    return GPUInfo(available=True)


def should_use_speculative_decoding() -> bool:
    """Determine if speculative decoding should be enabled.

    Returns True if:
    1. GPU is available
    2. SPECULATIVE_DRAFT_MODEL and SPECULATIVE_VERIFY_MODEL are configured
    3. ENABLE_SPECULATIVE_DECODING is not explicitly disabled
    """
    if not has_cuda():
        return False

    if os.getenv("ENABLE_SPECULATIVE_DECODING", "").lower() == "false":
        return False

    draft_model = os.getenv("SPECULATIVE_DRAFT_MODEL", "")
    verify_model = os.getenv("SPECULATIVE_VERIFY_MODEL", "")

    return bool(draft_model) and bool(verify_model)


def get_speculative_config() -> Optional[dict]:
    """Get speculative decoding configuration.

    Returns dict with draft_model and verify_model if enabled,
    otherwise None.
    """
    if not should_use_speculative_decoding():
        return None

    return {
        "draft_model": os.getenv("SPECULATIVE_DRAFT_MODEL", "qwen2:0.5b"),
        "verify_model": os.getenv("SPECULATIVE_VERIFY_MODEL", "qwen2.5:7b"),
        "draft_tokens": int(os.getenv("SPECULATIVE_DRAFT_TOKENS", "5")),
    }


def get_recommended_model() -> str:
    """Get recommended default model based on available hardware.

    Returns:
        - qwen2.5:7b if GPU is available
        - qwen2:0.5b for CPU-only environments
    """
    if has_cuda():
        gpu = get_gpu_info()
        # Recommend model based on available VRAM
        if gpu.total_memory_gb >= 40:
            return "llama3.3:70b"  # 70B model for high-end GPUs
        elif gpu.total_memory_gb >= 16:
            return "qwen2.5:14b"   # 14B model for RTX 5080 class
        elif gpu.total_memory_gb >= 8:
            return "qwen2.5:7b"    # 7B model for mid-range GPUs
        else:
            return "qwen2:1.5b"    # Small model for low VRAM

    return "qwen2:0.5b"  # CPU fallback
