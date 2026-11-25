#!/usr/bin/env python3
"""
setup_sanada.py
===============
Universal cross-platform setup & launcher for SANADA SOTA Wave 1-4.
Works across Windows, macOS (Apple Silicon & Intel), and Linux/WSL
on NVIDIA CUDA, Apple Silicon MPS, AMD ROCm (Linux), Intel XPU, and CPU-only machines.

Uses ONLY the Python standard library so teammates can run it right after `git pull`.
"""

import os
import re
import sys
import stat
import shutil
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "models" / "checkpoints"
VENV_DIR = BASE_DIR / ".venv"
REQ_FILE = BASE_DIR / "requirements.txt"

# Exact filename -> target relative path inside models/checkpoints/
EXPECTED_FILES: Dict[str, str] = {
    # Wave 1: Session E1 & Session A+
    "mossformer2_speech_vs_env_16k.pt": "session_e1/mossformer2_speech_vs_env_16k.pt",
    "sessionAplus_melati_eat_large.pt": "session_a_plus/sessionAplus_melati_eat_large.pt",
    "sessionAplus_sslam_as2m.pt": "session_a_plus/sessionAplus_sslam_as2m.pt",
    "sessionAplus_wavlm_large.pt": "session_a_plus/sessionAplus_wavlm_large.pt",
    "sessionAplus_meta_calibrator.json": "session_a_plus/sessionAplus_meta_calibrator.json",
    # Wave 2: Session B
    "sessionB_melati_eat_large_fsd32.pt": "session_b/sessionB_melati_eat_large_fsd32.pt",
    "sessionB_sslam_as2m_fsd32.pt": "session_b/sessionB_sslam_as2m_fsd32.pt",
    "sessionB_ast13_fsd32.pt": "session_b/sessionB_ast13_fsd32.pt",
    "sessionB_ced_base_fsd32.pt": "session_b/sessionB_ced_base_fsd32.pt",
    "sessionB_meta_calibrator.json": "session_b/sessionB_meta_calibrator.json",
    "sessionB_count_meta_models.pkl": "session_b/sessionB_count_meta_models.pkl",
    # Wave 3: Session C1, C2, D, E2
    "mossformer2_2spk_spkid_16k.pt": "session_c1/mossformer2_2spk_spkid_16k.pt",
    "mossformer2_3spk_spkid_16k.pt": "session_c1/mossformer2_3spk_spkid_16k.pt",
    "mossformer2_4spk_spkid_16k.pt": "session_c2/mossformer2_4spk_spkid_16k.pt",
    "mossformer2_orpit_5to9spk_16k.pt": "session_d/mossformer2_orpit_5to9spk_16k.pt",
    "clapsep_lora_32class_env_16k.pt": "session_e2/clapsep_lora_32class_env_16k.pt",
    "sessionE2_clapsep_config.json": "session_e2/sessionE2_clapsep_config.json",
    "sessionE2_class_prompts.json": "session_e2/sessionE2_class_prompts.json",
    # Wave 4: Session F
    "sessionF_pipeline_config.json": "session_f/sessionF_pipeline_config.json",
    "sessionF_master_benchmarks.json": "session_f/sessionF_master_benchmarks.json",
}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None, check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env)
    if check and proc.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return proc.returncode


def detect_nvidia_cuda_version() -> Optional[Tuple[int, int]]:
    """Checks nvidia-smi and returns max driver-supported (major, minor) CUDA version."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi and os.name == "nt":
        win_cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if win_cand.is_file():
            nvsmi = str(win_cand)
    if not nvsmi:
        return None
    try:
        res = subprocess.run([nvsmi], capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return None
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", res.stdout)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (12, 6)
    except Exception:
        return None


def detect_amd_rocm() -> Optional[str]:
    """Detects AMD ROCm on Linux and returns version string if available."""
    if platform.system() != "Linux":
        return None
    ver_file = Path("/opt/rocm/.info/version")
    if ver_file.is_file():
        return ver_file.read_text(encoding="utf-8", errors="ignore").strip()
    if shutil.which("rocm-smi") or Path("/opt/rocm").is_dir():
        return "detected"
    return None


def _cuda_index_cascade(cuda_ver: Tuple[int, int]) -> List[Optional[str]]:
    if cuda_ver >= (12, 8):
        return [
            "https://download.pytorch.org/whl/cu128",
            "https://download.pytorch.org/whl/cu126",
            None,
        ]
    if cuda_ver >= (12, 6):
        return [
            "https://download.pytorch.org/whl/cu126",
            "https://download.pytorch.org/whl/cu124",
            None,
        ]
    if cuda_ver >= (12, 4):
        return [
            "https://download.pytorch.org/whl/cu124",
            "https://download.pytorch.org/whl/cu121",
            None,
        ]
    return [
        "https://download.pytorch.org/whl/cu118",
        None,
    ]


def detect_hardware_backend(requested: str = "auto") -> Tuple[str, str, List[Optional[str]]]:
    """
    Returns (backend_name, description, candidate_pytorch_index_urls).
    None in candidate_pytorch_index_urls means standard PyPI default index.
    """
    sys_name = platform.system()
    arch = platform.machine().lower()
    req = requested.strip().lower()

    if req == "cpu":
        if sys_name == "Darwin":
            return ("cpu", "macOS CPU (forced)", [None])
        return ("cpu", f"{sys_name} CPU-only wheel (lightweight)", ["https://download.pytorch.org/whl/cpu", None])

    if req == "mps":
        return ("mps", "Apple Silicon MPS (PyPI macOS wheel)", [None])

    if req == "xpu":
        return ("xpu", "Intel Arc / Core Ultra XPU", ["https://download.pytorch.org/whl/xpu", None])

    if req == "rocm":
        return (
            "rocm",
            "AMD ROCm GPU (Linux)",
            ["https://download.pytorch.org/whl/rocm7.2", "https://download.pytorch.org/whl/rocm6.4", None],
        )

    if req == "cuda":
        cuda_ver = detect_nvidia_cuda_version() or (12, 8)
        return ("cuda", f"NVIDIA CUDA (driver supports <= {cuda_ver[0]}.{cuda_ver[1]})", _cuda_index_cascade(cuda_ver))

    # Auto-detection across OS / CPU / GPU
    if sys_name == "Darwin":
        if arch in ("arm64", "aarch64"):
            return ("mps", "macOS Apple Silicon (Metal MPS GPU acceleration)", [None])
        return ("cpu", "macOS Intel x86_64 (CPU Fast-CPU mode)", [None])

    if sys_name in ("Linux", "Windows"):
        cuda_ver = detect_nvidia_cuda_version()
        if cuda_ver is not None:
            return ("cuda", f"{sys_name} NVIDIA GPU (CUDA <= {cuda_ver[0]}.{cuda_ver[1]})", _cuda_index_cascade(cuda_ver))

        rocm_ver = detect_amd_rocm()
        if rocm_ver is not None:
            return (
                "rocm",
                f"Linux AMD ROCm GPU ({rocm_ver})",
                ["https://download.pytorch.org/whl/rocm7.2", "https://download.pytorch.org/whl/rocm6.4", "https://download.pytorch.org/whl/cpu"],
            )

        if arch in ("x86_64", "amd64"):
            return ("cpu", f"{sys_name} {arch} CPU (lightweight CPU wheel, Fast-CPU mode)", ["https://download.pytorch.org/whl/cpu", None])
        return ("cpu", f"{sys_name} {arch} CPU (PyPI wheel, Fast-CPU mode)", [None])

    return ("cpu", f"{sys_name} {arch} CPU (default PyPI wheel)", [None])


def check_prerequisites(backend_req: str) -> Tuple[str, str, List[Optional[str]]]:
    banner("STEP 1/5: Checking OS, CPU, GPU & System Prerequisites")
    v = sys.version_info
    backend, desc, urls = detect_hardware_backend(backend_req)
    print(f"  OS       : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python   : {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    print(f"  Hardware : {desc} [backend={backend}]")

    if (v.major, v.minor) < (3, 10):
        raise SystemExit("[ERROR] Python 3.10+ is required. Please install Python 3.10, 3.11, or 3.12.")
    if (v.major, v.minor) >= (3, 13):
        print("  [WARN] Python 3.13+ detected; if any audio dependency lacks wheels, use Python 3.11 or 3.12.")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  FFmpeg   : OK ({ffmpeg_path})")
    else:
        print("  [WARN] FFmpeg not found in PATH (WAV works, but mobile MP3/M4A decoding requires FFmpeg).")
        sys_name = platform.system()
        if sys_name == "Darwin":
            print("         Install on macOS   : brew install ffmpeg")
        elif sys_name == "Windows":
            print("         Install on Windows : winget install Gyan.FFmpeg")
        else:
            print("         Install on Linux   : sudo apt update && sudo apt install -y ffmpeg libsndfile1")

    return backend, desc, urls


def place_checkpoints(models_dir: Optional[str], move_files: bool = False) -> None:
    banner("STEP 2/5: Organizing Model Checkpoints into models/checkpoints/")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    search_roots: List[Path] = []
    if models_dir:
        p = Path(models_dir).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(f"[ERROR] --models-dir not found: {p}")
        search_roots.append(p)
    search_roots.append(CKPT_DIR)

    found_map: Dict[str, Path] = {}
    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn in EXPECTED_FILES and fn not in found_map:
                    cand = Path(dirpath) / fn
                    target = CKPT_DIR / EXPECTED_FILES[fn]
                    if cand.resolve() != target.resolve() and cand.stat().st_size > 0:
                        found_map[fn] = cand

    placed = 0
    already_ok = 0
    missing: List[str] = []

    for fname, rel_target in EXPECTED_FILES.items():
        target = CKPT_DIR / rel_target
        if target.is_file() and target.stat().st_size > 50:
            already_ok += 1
            continue
        src = found_map.get(fname)
        if src and src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            is_inside_ckpt = str(src.resolve()).startswith(str(CKPT_DIR.resolve()))
            if move_files or is_inside_ckpt:
                shutil.move(str(src), str(target))
                action = "MOVED"
            else:
                shutil.copy2(str(src), str(target))
                action = "COPIED"
            print(f"  [{action:<6}] {fname} -> models/checkpoints/{rel_target}")
            placed += 1
        else:
            missing.append(rel_target)

    print(f"  Summary: {already_ok} already in place | {placed} newly placed | {len(missing)} missing")
    if missing:
        print("  [WARN] Missing checkpoint files:")
        for m in missing:
            print(f"         - models/checkpoints/{m}")
        print("         Tip: re-run with --models-dir /path/to/downloaded_hf_models")


def install_pytorch_for_hardware(venv_py: Path, index_urls: List[Optional[str]]) -> None:
    """Installs matching torch + torchaudio wheel first, with automatic fallback cascade."""
    for idx_url in index_urls:
        cmd = [str(venv_py), "-m", "pip", "install", "torch>=2.0.0", "torchaudio>=2.0.0"]
        if idx_url:
            cmd.extend(["--index-url", idx_url])
        rc = run_cmd(cmd, check=False)
        if rc == 0:
            return
        print(f"  [WARN] PyTorch install failed for index={idx_url or 'PyPI'}; trying next fallback...")
    raise SystemExit("[ERROR] Could not install compatible torch & torchaudio wheels for this Python/OS.")


def read_non_torch_requirements() -> List[str]:
    reqs: List[str] = []
    for raw in REQ_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip().lower()
        if pkg_name in ("torch", "torchaudio", "torchvision"):
            continue
        reqs.append(line)
    return reqs


def setup_virtualenv(index_urls: List[Optional[str]], skip_install: bool = False) -> Path:
    banner("STEP 3/5: Creating Virtual Environment & Installing Hardware-Matched Dependencies")
    venv_py = get_venv_python(VENV_DIR)
    if not venv_py.is_file():
        print(f"  Creating virtual environment at {VENV_DIR} ...")
        run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"  Reusing existing virtual environment: {VENV_DIR}")

    if skip_install:
        print("  --skip-install set: skipping pip install.")
        return venv_py

    if not REQ_FILE.is_file():
        raise SystemExit(f"[ERROR] requirements.txt not found at {REQ_FILE}")

    run_cmd([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    install_pytorch_for_hardware(venv_py, index_urls)

    other_reqs = read_non_torch_requirements()
    if other_reqs:
        run_cmd([str(venv_py), "-m", "pip", "install", *other_reqs])

    run_cmd([str(venv_py), "-m", "pip", "install", "--no-deps", "clearvoice"])
    return venv_py


def verify_environment(venv_py: Path) -> None:
    banner("STEP 4/5: Verifying Imports, Accelerator Telemetry & Offline Checkpoints")
    check_snippet = (
        "import torch, torchaudio, fastapi, uvicorn, transformers, clearvoice, timm, librosa, soundfile; "
        "from models.device_utils import select_compute_device, get_device_telemetry; "
        "dev = select_compute_device(); tel = get_device_telemetry(dev); "
        "print(f'  [OK] Core imports verified | PyTorch {torch.__version__}'); "
        "print(f'  [OK] Active Compute Device: {tel}')"
    )
    run_cmd([str(venv_py), "-c", check_snippet])

    env = os.environ.copy()
    env["OFFLINE_MODE"] = "1"
    run_cmd([str(venv_py), str(BASE_DIR / "scripts" / "download_checkpoints.py")], env=env)


def launch_server(venv_py: Path, port: int = 8000) -> None:
    banner(f"Launching SANADA Unified Server on http://localhost:{port}")
    env = os.environ.copy()
    env["SANADA_DEPLOY_MODE"] = "local"
    env["HF_HOME"] = str(CKPT_DIR / "hf_cache")
    if platform.system() == "Darwin":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    run_cmd([str(venv_py), "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", str(port)], env=env)


def finalize_launcher(venv_py: Path, auto_start: bool = False, port: int = 8000) -> None:
    banner("STEP 5/5: Finalizing Cross-Platform Launcher")
    start_sh = BASE_DIR / "start_system.sh"
    if os.name != "nt" and start_sh.is_file():
        mode = start_sh.stat().st_mode
        start_sh.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("  [OK] Marked start_system.sh as executable.")

    print("\n  SETUP COMPLETE! How to run SANADA on any OS:")
    print(f"    Universal (Win/Mac/Linux) : python3 setup_sanada.py --run" + (f" --port {port}" if port != 8000 else ""))
    if os.name == "nt":
        print(f"    Windows PowerShell        : python setup_sanada.py --run" + (f" --port {port}" if port != 8000 else ""))
    else:
        print(f"    macOS / Linux / WSL       : ./start_system.sh" + (f" --port {port}" if port != 8000 else ""))
    print(f"    Web Studio                : http://localhost:{port}")
    print(f"    API Docs                  : http://localhost:{port}/docs")
    print(f"    Health Check              : http://localhost:{port}/api/health")

    if auto_start:
        launch_server(venv_py, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal OS/CPU/GPU setup & launcher for SANADA SOTA Wave 1-4")
    parser.add_argument("--models-dir", default=None, help="Folder where Hugging Face models were downloaded")
    parser.add_argument("--move", action="store_true", help="Move files from --models-dir instead of copying (saves disk space)")
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "mps", "rocm", "xpu", "cpu"],
        default="auto",
        help="Compute backend for PyTorch wheel installation (default: auto)",
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install (only organize & verify checkpoints)")
    parser.add_argument("--check-only", action="store_true", help="Verify hardware, checkpoints, and current Python env without modifying .venv")
    parser.add_argument("--start", action="store_true", help="Start the SANADA server immediately after setup")
    parser.add_argument("--run", action="store_true", help="Skip setup and launch the SANADA server directly (cross-platform alternative to ./start_system.sh)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    if args.run:
        venv_py = get_venv_python(VENV_DIR)
        active_py = venv_py if venv_py.is_file() else Path(sys.executable)
        launch_server(active_py, port=args.port)
        return

    _, _, index_urls = check_prerequisites(args.backend)
    place_checkpoints(args.models_dir, move_files=args.move)

    if args.check_only:
        venv_py = get_venv_python(VENV_DIR)
        active_py = venv_py if venv_py.is_file() else Path(sys.executable)
        verify_environment(active_py)
        finalize_launcher(active_py, auto_start=False, port=args.port)
        return

    venv_py = setup_virtualenv(index_urls=index_urls, skip_install=args.skip_install)
    verify_environment(venv_py)
    finalize_launcher(venv_py, auto_start=args.start, port=args.port)


if __name__ == "__main__":
    main()
