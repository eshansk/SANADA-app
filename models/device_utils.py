#!/usr/bin/env python3
"""
models/device_utils.py
======================
Hardware acceleration and compute device utilities for SANADA.
Supports NVIDIA CUDA, AMD ROCm (via CUDA/HIP interface), Apple Silicon MPS,
Intel Arc/Core Ultra XPU, and CPU (x86_64 & ARM64).
"""

import os
import gc
import logging
from typing import Optional, Dict, Any, Union
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

logger = logging.getLogger("SANADA-DeviceUtils")


def _xpu_available() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def select_compute_device() -> torch.device:
    """
    Selects the optimal compute device according to priority:
    1. SANADA_DEVICE environment variable if set (e.g. 'cpu', 'cuda:0', 'mps', 'xpu:0').
    2. NVIDIA CUDA or AMD ROCm (cuda:0) if available.
    3. Apple Silicon Metal Performance Shaders (mps) if available.
       Sets PYTORCH_ENABLE_MPS_FALLBACK=1 so unsupported complex STFT/iSTFT ops
       fall back gracefully to CPU without crashing macOS.
    4. Intel Arc / Core Ultra XPU (xpu:0) if available.
    5. CPU fallback (Fast-CPU mode).
    """
    env_device = os.getenv("SANADA_DEVICE")
    if env_device and env_device.strip():
        dev_str = env_device.strip()
        device = torch.device(dev_str)
        if device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    if _xpu_available():
        return torch.device("xpu:0")

    return torch.device("cpu")


def is_low_vram_device(device: Optional[Union[str, torch.device]] = None) -> bool:
    """
    Returns True if SANADA_LOW_VRAM=1 or if GPU total VRAM <= 6.5 GB (e.g. 4GB/6GB laptop GPU).
    """
    if os.getenv("SANADA_LOW_VRAM", "").strip() == "1":
        return True

    if device is None:
        device = select_compute_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            dev_idx = device.index if device.index is not None else 0
            total_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
            if (total_bytes / (1024 ** 3)) <= 6.5:
                return True
        except Exception as e:
            logger.debug(f"Unable to query CUDA memory properties: {e}")
    elif device.type == "xpu" and _xpu_available():
        try:
            dev_idx = device.index if device.index is not None else 0
            total_bytes = torch.xpu.get_device_properties(dev_idx).total_memory
            if (total_bytes / (1024 ** 3)) <= 6.5:
                return True
        except Exception as e:
            logger.debug(f"Unable to query XPU memory properties: {e}")

    return False


def is_fast_cpu_mode(device: Optional[Union[str, torch.device]] = None) -> bool:
    """
    Returns True if device.type == "cpu" or SANADA_FAST_CPU=1.
    """
    if os.getenv("SANADA_FAST_CPU", "").strip() == "1":
        return True

    if device is None:
        device = select_compute_device()
    elif isinstance(device, str):
        device = torch.device(device)

    return device.type == "cpu"


def clear_device_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Runs gc.collect() and empties device cache for CUDA/ROCm, MPS, or XPU.
    """
    gc.collect()

    if device is None:
        device = select_compute_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == "xpu" and _xpu_available() and hasattr(torch.xpu, "empty_cache"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def get_device_telemetry(device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """
    Returns a telemetry dictionary containing compute device & accelerator details.
    """
    if device is None:
        device = select_compute_device()
    elif isinstance(device, str):
        device = torch.device(device)

    cuda_avail = torch.cuda.is_available()
    mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    xpu_avail = _xpu_available()
    rocm_hip = getattr(torch.version, "hip", None)

    gpu_name = None
    total_vram_gb = None

    if device.type == "cuda" and cuda_avail:
        try:
            dev_idx = device.index if device.index is not None else 0
            gpu_name = torch.cuda.get_device_name(dev_idx)
            total_vram_gb = round(torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3), 2)
        except Exception:
            gpu_name = "AMD ROCm GPU" if rocm_hip else "NVIDIA CUDA Device"
    elif device.type == "mps" and mps_avail:
        gpu_name = "Apple Silicon MPS"
    elif device.type == "xpu" and xpu_avail:
        try:
            dev_idx = device.index if device.index is not None else 0
            gpu_name = torch.xpu.get_device_name(dev_idx)
            total_vram_gb = round(torch.xpu.get_device_properties(dev_idx).total_memory / (1024 ** 3), 2)
        except Exception:
            gpu_name = "Intel XPU Device"

    return {
        "device": str(device),
        "cuda_available": cuda_avail,
        "rocm_available": bool(cuda_avail and rocm_hip),
        "mps_available": mps_avail,
        "xpu_available": xpu_avail,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "low_vram_mode": is_low_vram_device(device),
        "fast_cpu_mode": is_fast_cpu_mode(device),
    }


if __name__ == "__main__":
    dev = select_compute_device()
    telemetry = get_device_telemetry(dev)
    print("SANADA Compute Device Telemetry:")
    for k, v in telemetry.items():
        print(f"  {k}: {v}")
