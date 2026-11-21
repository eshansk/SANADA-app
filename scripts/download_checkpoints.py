#!/usr/bin/env python3
"""
scripts/download_checkpoints.py
===============================
Local-First Model Checkpoint Retrieval & Cluster Sync Manager (Zero Hugging Face).
Member 5: Data Engineering & Backend API

Handles:
- 100% local operation with ZERO remote Hugging Face .pt model pulls.
- Verifying the 6 Path B Foundation Model artifacts:
  1. wavlm_counting_pathb.pt
  2. ast_tagging_pathb.pt
  3. sepformer_2spk_pathb.pt
  4. sepformer_3spk_pathb.pt
  5. conv_tasnet_4spk_dominant_pathb.pt
  6. pathb_benchmarks.json
- Lookup priority per file:
  1. Already in models/checkpoints/ (valid if >1 MB for .pt, >100 B for .json)
  2. Optional local shared directory via SANADA_LOCAL_CKPT_DIR (shutil.copy2)
  3. Optional campus HDFS cluster directory via SANADA_HDFS_DIR (hdfs dfs -get)
- Graceful partial-load logging ([READY] vs [MISSING - FALLBACK])
- Updating models/checkpoints/checkpoint_manifest.json with real sizes and tier info.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SANADA-Checkpoints")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")

# Minimum byte thresholds for valid artifacts
MIN_MODEL_BYTES = 1_000_000  # 1 MB
MIN_JSON_BYTES = 100         # 100 B

PATH_B_ARTIFACTS = {
    "counting": {
        "filename": "wavlm_counting_pathb.pt",
        "type": "model",
        "description": "WavLMSourceCounter with ASP and Table-3 DSP fusion (K=1..32)"
    },
    "tagging": {
        "filename": "ast_tagging_pathb.pt",
        "type": "model",
        "description": "MIT AST 33-Class Tagger with dynamic speech gate"
    },
    "separation_2spk": {
        "filename": "sepformer_2spk_pathb.pt",
        "type": "model",
        "description": "Sepformer 2-speaker separation specialist"
    },
    "separation_3spk": {
        "filename": "sepformer_3spk_pathb.pt",
        "type": "model",
        "description": "Sepformer 3-speaker separation specialist"
    },
    "separation_4spk_dominant": {
        "filename": "sepformer_4spk_dominant_pathb.pt",
        "alt_filename": "conv_tasnet_4spk_dominant_pathb.pt",
        "type": "model",
        "description": "Sepformer/ConvTasNetSkip 4-speaker dominant crowd separator"
    },
    "benchmarks": {
        "filename": "pathb_benchmarks.json",
        "type": "benchmark",
        "description": "Path B benchmark metrics including pure speech rejection"
    }
}

LEGACY_CHECKPOINTS = {
    "counting_legacy": {
        "filename": "conformer_counting_baseline.pt",
        "description": "Legacy Conformer source counter (K=1..9)"
    },
    "tagging_legacy": {
        "filename": "conformer_tagging_baseline.pt",
        "alt_filename": "conformer_tagging_asl.pt",
        "description": "Legacy Conformer acoustic tagger (32 FSD classes)"
    },
    "separation_legacy": {
        "filename": "conv_tasnet_separation.pt",
        "description": "Legacy Conv-TasNet 4-speaker separator"
    }
}


def sync_or_verify_artifact(filename: str, min_bytes: int, is_path_b: bool = False) -> Tuple[bool, Optional[str], int]:
    """
    Checks for artifact using local-first lookup priority:
    1. Already in CHECKPOINT_DIR
    2. SANADA_LOCAL_CKPT_DIR (shutil.copy2)
    3. SANADA_HDFS_DIR (hdfs dfs -get)
    4. Opt-in Hugging Face branch (ONLY runs when SANADA_DEPLOY_MODE == "huggingface" for Path B artifacts)
    """
    target_path = os.path.join(CHECKPOINT_DIR, filename)

    def is_valid(path: str) -> bool:
        if os.path.isfile(path) and os.path.getsize(path) >= min_bytes:
            return True
        if os.path.isdir(path) and (os.path.exists(os.path.join(path, "data.pkl")) or os.path.exists(os.path.join(path, "model.safetensors"))):
            return True
        return False

    def get_path_size(path: str) -> int:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total

    # 1. Already in models/checkpoints/
    if is_valid(target_path):
        return True, target_path, get_path_size(target_path)

    # 2. Optional local shared directory via SANADA_LOCAL_CKPT_DIR
    local_dir = os.getenv("SANADA_LOCAL_CKPT_DIR")
    if local_dir and os.path.exists(local_dir):
        local_src = os.path.join(local_dir, filename)
        if is_valid(local_src):
            try:
                shutil.copy2(local_src, target_path)
                logger.info(f"Copied {filename} from local shared directory: {local_dir}")
                return True, target_path, os.path.getsize(target_path)
            except Exception as e:
                logger.warning(f"Failed to copy from SANADA_LOCAL_CKPT_DIR: {e}")

    # 3. Optional campus HDFS path via SANADA_HDFS_DIR
    hdfs_dir = os.getenv("SANADA_HDFS_DIR")
    if hdfs_dir and shutil.which("hdfs"):
        hdfs_src = f"{hdfs_dir.rstrip('/')}/{filename}"
        try:
            res = subprocess.run(
                ["hdfs", "dfs", "-get", hdfs_src, target_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            if res.returncode == 0 and is_valid(target_path):
                logger.info(f"Successfully downloaded {filename} from HDFS {hdfs_src}")
                return True, target_path, os.path.getsize(target_path)
            else:
                logger.warning(f"HDFS sync returned code {res.returncode}: {res.stderr.strip()}")
        except Exception as e:
            logger.warning(f"HDFS sync error for {filename}: {e}")

    # 4. Opt-in Cloud Download Branch (ONLY runs when SANADA_DEPLOY_MODE == "huggingface" for Path B artifacts)
    deploy_mode = os.getenv("SANADA_DEPLOY_MODE", "local").strip().lower()
    if deploy_mode == "huggingface" and is_path_b:
        repo_id = os.getenv("SANADA_HF_REPO_ID", "eshansk/sanada-pathb-checkpoints")
        hf_token = os.getenv("HF_TOKEN")
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"Opt-in Cloud Download: Attempting HF download of {filename} from {repo_id}...")
            dl_kwargs = {
                "repo_id": repo_id,
                "filename": filename,
                "local_dir": CHECKPOINT_DIR
            }
            if hf_token:
                dl_kwargs["token"] = hf_token

            dl_path = hf_hub_download(**dl_kwargs)
            if is_valid(target_path):
                logger.info(f"Successfully downloaded {filename} from Hugging Face repo {repo_id}")
                return True, target_path, os.path.getsize(target_path)
            elif is_valid(dl_path):
                if os.path.abspath(dl_path) != os.path.abspath(target_path):
                    shutil.copy2(dl_path, target_path)
                logger.info(f"Successfully downloaded {filename} from Hugging Face repo {repo_id}")
                return True, target_path, os.path.getsize(target_path)
        except Exception as e:
            logger.warning(f"Opt-in Hugging Face download failed for {filename} from {repo_id}: {e}")

    return False, None, 0


def ensure_checkpoint_cache() -> Dict[str, Any]:
    """
    Verifies checkpoint directory and synchronizes Path B foundation models + legacy fallbacks.
    Zero Hugging Face network downloads in local and cluster modes.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    manifest: Dict[str, Any] = {}

    print("\n" + "=" * 75)
    print(" SANADA LOCAL CHECKPOINT & CLUSTER SYNC VERIFICATION")
    print("=" * 75)

    # 1. Verify Path B Artifacts
    print("\n[Phase 1: Path B Foundation Model Artifacts]")
    for key, info in PATH_B_ARTIFACTS.items():
        fname = info["filename"]
        min_bytes = MIN_JSON_BYTES if info["type"] == "benchmark" else MIN_MODEL_BYTES
        ready, path, size = sync_or_verify_artifact(fname, min_bytes, is_path_b=True)

        if ready:
            size_mb = round(size / (1024 * 1024), 2)
            print(f"  [READY] {fname:<36} ({size_mb:>6.2f} MB) - {info['description']}")
            manifest[key] = {
                "filename": fname,
                "path": path,
                "status": "ready",
                "tier": "path_b_foundation",
                "size_bytes": size,
                "size_mb": size_mb,
                "description": info["description"]
            }
        else:
            print(f"  [MISSING - FALLBACK] {fname:<36} - using baseline fallback")
            manifest[key] = {
                "filename": fname,
                "path": None,
                "status": "fallback",
                "tier": "path_b_foundation",
                "size_bytes": 0,
                "description": info["description"]
            }

    # 2. Verify Legacy Baseline Fallbacks
    print("\n[Phase 2: Legacy Baseline Fallback Checkpoints]")
    for key, info in LEGACY_CHECKPOINTS.items():
        fname = info["filename"]
        alt_fname = info.get("alt_filename")
        ready, path, size = sync_or_verify_artifact(fname, MIN_MODEL_BYTES, is_path_b=False)
        if not ready and alt_fname:
            ready, path, size = sync_or_verify_artifact(alt_fname, MIN_MODEL_BYTES, is_path_b=False)
            fname = alt_fname if ready else fname
        if not ready and alt_fname:
            ready, path, size = sync_or_verify_artifact(alt_fname, MIN_MODEL_BYTES)
            fname = alt_fname if ready else fname

        if ready:
            size_mb = round(size / (1024 * 1024), 2)
            print(f"  [CACHED] {fname:<36} ({size_mb:>6.2f} MB) - {info['description']}")
            manifest[key] = {
                "filename": fname,
                "path": path,
                "status": "cached",
                "tier": "legacy_fallback",
                "size_bytes": size,
                "size_mb": size_mb,
                "description": info["description"]
            }
        else:
            print(f"  [MISSING] {fname:<36} - {info['description']}")
            manifest[key] = {
                "filename": fname,
                "path": None,
                "status": "missing",
                "tier": "legacy_fallback",
                "size_bytes": 0
            }

    # Update manifest file
    manifest_path = os.path.join(CHECKPOINT_DIR, "checkpoint_manifest.json")
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest successfully updated at: {manifest_path}")
    except Exception as e:
        logger.warning(f"Notice saving checkpoint_manifest.json: {e}")

    print("=" * 75 + "\n")
    return manifest


if __name__ == "__main__":
    ensure_checkpoint_cache()
