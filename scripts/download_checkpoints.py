#!/usr/bin/env python3
"""
scripts/download_checkpoints.py
===============================
Local-First Model Checkpoint Retrieval & Cluster Sync Manager.
Member 5: Data Engineering & Backend API

Handles:
- Phase 0: Scanning and verifying SOTA Wave 1-3 Checkpoints & Meta-Calibrators:
  * Wave 1 (Session E1): mossformer2_speech_vs_env_16k.pt
  * Wave 1 (Session A+): sessionAplus_melati_eat_large.pt, sessionAplus_sslam_as2m.pt,
                         sessionAplus_wavlm_large.pt, sessionAplus_meta_calibrator.json
  * Wave 2 (Session B):  sessionB_melati_eat_large_fsd32.pt, sessionB_sslam_as2m_fsd32.pt,
                         sessionB_ast13_fsd32.pt, sessionB_ced_base_fsd32.pt,
                         sessionB_meta_calibrator.json, sessionB_count_meta_models.pkl
  * Wave 3 (Session C1): mossformer2_2spk_spkid_16k.pt, mossformer2_3spk_spkid_16k.pt
  * Wave 3 (Session C2): mossformer2_4spk_spkid_16k.pt
  * Wave 3 (Session D):  mossformer2_orpit_5to9spk_16k.pt
  * Wave 3 Plug-and-Play (Session E2): clapsep_lora_32class_env_16k.pt, sessionE2_clapsep_config.json,
                                       sessionE2_class_prompts.json (optional)
  * Session F Pipeline Config: sessionF_pipeline_config.json (optional)
- Phase 1: Verifying the 6 Path B Foundation Model artifacts (fallback tier)
- Phase 2: Verifying Legacy Baseline Fallbacks
- Updating models/checkpoints/checkpoint_manifest.json with real sizes, tier info, and SOTA inventory.
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
MIN_JSON_BYTES = 50          # 50 B
MIN_PKL_BYTES = 1_000        # 1 KB

# Exact SOTA Wave 1-3 Checkpoints Inventory
SOTA_WAVE_CHECKPOINTS = {
    # Wave 1: Domain Classification & Speech Source Counting Foundation
    "session_e1/mossformer2_speech_vs_env_16k.pt": {
        "session": "session_e1",
        "wave": "Wave 1",
        "type": "model",
        "required": True,
        "description": "MossFormer2 Speech vs Environmental binary classifier (16kHz)"
    },
    "session_a_plus/sessionAplus_melati_eat_large.pt": {
        "session": "session_a_plus",
        "wave": "Wave 1",
        "type": "model",
        "required": True,
        "description": "EAT-Large speech source counting backbone (Session A+)"
    },
    "session_a_plus/sessionAplus_sslam_as2m.pt": {
        "session": "session_a_plus",
        "wave": "Wave 1",
        "type": "model",
        "required": True,
        "description": "SS-LAM AS2M speech source counting backbone (Session A+)"
    },
    "session_a_plus/sessionAplus_wavlm_large.pt": {
        "session": "session_a_plus",
        "wave": "Wave 1",
        "type": "model",
        "required": True,
        "description": "WavLM-Large speech source counting backbone (Session A+)"
    },
    "session_a_plus/sessionAplus_meta_calibrator.json": {
        "session": "session_a_plus",
        "wave": "Wave 1",
        "type": "meta",
        "required": True,
        "description": "Session A+ speech counting meta-calibrator configuration"
    },

    # Wave 2: Environmental Tagging & Counting (32 FSD Classes Quad-Ensemble)
    "session_b/sessionB_melati_eat_large_fsd32.pt": {
        "session": "session_b",
        "wave": "Wave 2",
        "type": "model",
        "required": True,
        "description": "EAT-Large 32-class environmental acoustic classifier (Session B)"
    },
    "session_b/sessionB_sslam_as2m_fsd32.pt": {
        "session": "session_b",
        "wave": "Wave 2",
        "type": "model",
        "required": True,
        "description": "SS-LAM 32-class environmental acoustic classifier (Session B)"
    },
    "session_b/sessionB_ast13_fsd32.pt": {
        "session": "session_b",
        "wave": "Wave 2",
        "type": "model",
        "required": True,
        "description": "AST-13 32-class environmental acoustic classifier (Session B)"
    },
    "session_b/sessionB_ced_base_fsd32.pt": {
        "session": "session_b",
        "wave": "Wave 2",
        "type": "model",
        "required": True,
        "description": "CED-Base 32-class environmental acoustic classifier (Session B)"
    },
    "session_b/sessionB_meta_calibrator.json": {
        "session": "session_b",
        "wave": "Wave 2",
        "type": "meta",
        "required": True,
        "description": "Session B environmental meta-calibrator configuration"
    },
    "session_b/sessionB_count_meta_models.pkl": {
        "session": "session_b",
        "wave": "Wave 2",
        "type": "meta_pkl",
        "required": True,
        "description": "Session B counting meta-models ensemble weights"
    },

    # Wave 3: SOTA Speech Separation Specialists & ORPIT
    "session_c1/mossformer2_2spk_spkid_16k.pt": {
        "session": "session_c1",
        "wave": "Wave 3",
        "type": "model",
        "required": True,
        "description": "MossFormer2 2-speaker separation specialist with SpkID (16kHz)"
    },
    "session_c1/mossformer2_3spk_spkid_16k.pt": {
        "session": "session_c1",
        "wave": "Wave 3",
        "type": "model",
        "required": True,
        "description": "MossFormer2 3-speaker separation specialist with SpkID (16kHz)"
    },
    "session_c2/mossformer2_4spk_spkid_16k.pt": {
        "session": "session_c2",
        "wave": "Wave 3",
        "type": "model",
        "required": True,
        "description": "MossFormer2 4-speaker separation specialist with SpkID (16kHz)"
    },
    "session_d/mossformer2_orpit_5to9spk_16k.pt": {
        "session": "session_d",
        "wave": "Wave 3",
        "type": "model",
        "required": True,
        "description": "MossFormer2 ORPIT 5-9 speaker dominant separation specialist (16kHz)"
    },

    # Optional Plug-and-Play Checkpoints
    "session_e2/clapsep_lora_32class_env_16k.pt": {
        "session": "session_e2",
        "wave": "Wave 3 (Optional)",
        "type": "model",
        "required": False,
        "description": "CLAPSep LoRA 32-class environmental audio separator (16kHz)"
    },
    "session_e2/sessionE2_clapsep_config.json": {
        "session": "session_e2",
        "wave": "Wave 3 (Optional)",
        "type": "config",
        "required": False,
        "description": "Session E2 CLAPSep model architecture configuration"
    },
    "session_e2/sessionE2_class_prompts.json": {
        "session": "session_e2",
        "wave": "Wave 3 (Optional)",
        "type": "config",
        "required": False,
        "description": "Session E2 text prompt embeddings for 32 FSD classes"
    },
    "session_f/sessionF_pipeline_config.json": {
        "session": "session_f",
        "wave": "Wave 4",
        "type": "config",
        "required": False,
        "description": "Session F v6 leak-free calibrated cutpoints, gates & thresholds"
    },
    "session_f/sessionF_master_benchmarks.json": {
        "session": "session_f",
        "wave": "Wave 4",
        "type": "config",
        "required": False,
        "description": "Session F v6 strictly blind 8-suite defense benchmark telemetry"
    }
}

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


def sync_or_verify_artifact(filename: str, min_bytes: int, is_path_b: bool = False, allow_sota_hf: bool = False) -> Tuple[bool, Optional[str], int]:
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
        if os.path.isdir(path) and (
            os.path.exists(os.path.join(path, "data.pkl")) or os.path.exists(os.path.join(path, "model.safetensors"))
        ):
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
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
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
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
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

    # 4. Opt-in Cloud Download Branch (skipped in OFFLINE_MODE=1 / HF_HUB_OFFLINE=1 cluster mode)
    offline_mode = os.getenv("OFFLINE_MODE", "0").strip() == "1" or os.getenv("HF_HUB_OFFLINE", "0").strip() == "1"
    deploy_mode = os.getenv("SANADA_DEPLOY_MODE", "local").strip().lower()
    hf_sota_repo = os.getenv("HF_CHECKPOINT_REPO", "").strip()
    use_sota_hf = bool(allow_sota_hf and hf_sota_repo and not offline_mode)
    use_pathb_hf = bool(deploy_mode == "huggingface" and is_path_b and not offline_mode)
    if use_sota_hf or use_pathb_hf:
        repo_id = hf_sota_repo if use_sota_hf else os.getenv("SANADA_HF_REPO_ID", "eshansk/sanada-pathb-checkpoints")
        hf_token = os.getenv("HF_TOKEN", "").strip() or None
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"Opt-in Cloud Download: Attempting HF download of {filename} from {repo_id}...")
            dl_kwargs = {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": "model",
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
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(dl_path, target_path)
                logger.info(f"Successfully downloaded {filename} from Hugging Face repo {repo_id}")
                return True, target_path, os.path.getsize(target_path)
        except Exception as e:
            logger.warning(f"Opt-in Hugging Face download failed for {filename} from {repo_id}: {e}")

    return False, None, 0


def get_sota_inventory() -> Dict[str, Any]:
    """
    Returns structured inventory of SOTA Wave 1-3 checkpoints,
    grouped by session, with verification status and file sizes.
    """
    inventory: Dict[str, Any] = {}
    for rel_path, info in SOTA_WAVE_CHECKPOINTS.items():
        session = info["session"]
        if session not in inventory:
            inventory[session] = {
                "session": session,
                "wave": info["wave"],
                "models": [],
                "ready_count": 0,
                "required_count": 0,
                "fully_ready": False
            }

        full_path = os.path.join(CHECKPOINT_DIR, rel_path)
        if info["type"] in ["meta", "config"]:
            min_bytes = MIN_JSON_BYTES
        elif info["type"] == "meta_pkl":
            min_bytes = MIN_PKL_BYTES
        else:
            min_bytes = MIN_MODEL_BYTES

        exists = os.path.isfile(full_path) and os.path.getsize(full_path) >= min_bytes
        size_bytes = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
        size_mb = round(size_bytes / (1024 * 1024), 2)

        if info["required"]:
            inventory[session]["required_count"] += 1
            if exists:
                inventory[session]["ready_count"] += 1

        inventory[session]["models"].append({
            "rel_path": rel_path,
            "filename": os.path.basename(rel_path),
            "type": info["type"],
            "required": info["required"],
            "ready": exists,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "description": info["description"]
        })

    for session, data in inventory.items():
        total_files = len(data["models"])
        ready_files = sum(1 for m in data["models"] if m["ready"])
        req_count = data["required_count"]
        req_ready = data["ready_count"]

        if req_count > 0:
            data["fully_ready"] = (req_ready == req_count)
        else:
            # Optional session (e.g. session_e2, session_f):
            # Only mark as READY / true when ALL of its files actually exist on disk
            data["fully_ready"] = (ready_files == total_files and total_files > 0)

    return inventory


def ensure_checkpoint_cache() -> Dict[str, Any]:
    """
    Verifies checkpoint directory and synchronizes SOTA Wave 1-3 checkpoints,
    Path B foundation models (fallback), and legacy baseline fallbacks.
    Zero remote downloads in local and cluster modes unless explicitly requested.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    manifest: Dict[str, Any] = {}

    print("\n" + "=" * 78)
    print(" SANADA LOCAL CHECKPOINT & CLUSTER SYNC VERIFICATION")
    print("=" * 78)

    # 0. Scan & Verify SOTA Wave 1-3 Checkpoints
    print("\n[Phase 0: SOTA Wave 1-3 Model Checkpoints & Meta-Calibrators]")
    sota_ready_count = 0
    sota_total_required = sum(1 for v in SOTA_WAVE_CHECKPOINTS.values() if v["required"])
    sota_inventory = get_sota_inventory()

    current_session = None
    for rel_path, info in SOTA_WAVE_CHECKPOINTS.items():
        session = info["session"]
        if session != current_session:
            current_session = session
            if sota_inventory[session]["fully_ready"]:
                session_status = "READY"
            elif sota_inventory[session]["required_count"] == 0:
                session_status = "NOT INSTALLED"
            else:
                session_status = "PARTIAL"
            print(f"\n  -- {session.upper()} ({info['wave']}) [{session_status}] --")

        if info["type"] in ["meta", "config"]:
            min_bytes = MIN_JSON_BYTES
        elif info["type"] == "meta_pkl":
            min_bytes = MIN_PKL_BYTES
        else:
            min_bytes = MIN_MODEL_BYTES

        ready, path, size = sync_or_verify_artifact(rel_path, min_bytes, is_path_b=False, allow_sota_hf=True)
        size_mb = round(size / (1024 * 1024), 2)
        base_name = os.path.basename(rel_path)

        manifest_key = f"sota_{session}_{base_name.replace('.', '_')}"

        if ready:
            if info["required"]:
                sota_ready_count += 1
            status_tag = "[READY]"
            print(f"  {status_tag:<10} {rel_path:<48} ({size_mb:>7.2f} MB) - {info['description']}")
            manifest[manifest_key] = {
                "rel_path": rel_path,
                "filename": base_name,
                "session": session,
                "wave": info["wave"],
                "path": path,
                "status": "ready",
                "tier": "sota_wave_1_3",
                "required": info["required"],
                "size_bytes": size,
                "size_mb": size_mb,
                "description": info["description"]
            }
        else:
            if info["required"]:
                status_tag = "[MISSING]"
                print(f"  {status_tag:<10} {rel_path:<48} (Fallback: Path B/Legacy) - {info['description']}")
                manifest[manifest_key] = {
                    "rel_path": rel_path,
                    "filename": base_name,
                    "session": session,
                    "wave": info["wave"],
                    "path": None,
                    "status": "missing",
                    "tier": "sota_wave_1_3",
                    "required": info["required"],
                    "size_bytes": 0,
                    "description": info["description"]
                }
            else:
                status_tag = "[OPTIONAL]"
                print(f"  {status_tag:<10} {rel_path:<48} (Not installed) - {info['description']}")
                manifest[manifest_key] = {
                    "rel_path": rel_path,
                    "filename": base_name,
                    "session": session,
                    "wave": info["wave"],
                    "path": None,
                    "status": "optional_not_installed",
                    "tier": "sota_wave_1_3",
                    "required": info["required"],
                    "size_bytes": 0,
                    "description": info["description"]
                }

    sota_inventory = get_sota_inventory()
    print(f"\n  Phase 0 SOTA Wave 1-4 Summary: {sota_ready_count}/{sota_total_required} required models verified and ready.")

    # 1. Verify Path B Artifacts (Fallback Tier)
    print("\n[Phase 1: Path B Foundation Model Artifacts (Fallback Tier)]")
    for key, info in PATH_B_ARTIFACTS.items():
        fname = info["filename"]
        min_bytes = MIN_JSON_BYTES if info["type"] == "benchmark" else MIN_MODEL_BYTES
        ready, path, size = sync_or_verify_artifact(fname, min_bytes, is_path_b=True)

        if ready:
            size_mb = round(size / (1024 * 1024), 2)
            print(f"  [READY]    {fname:<36} ({size_mb:>7.2f} MB) - {info['description']}")
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
            print(f"  [MISSING]  {fname:<36} - using baseline fallback")
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

        if ready:
            size_mb = round(size / (1024 * 1024), 2)
            print(f"  [CACHED]   {fname:<36} ({size_mb:>7.2f} MB) - {info['description']}")
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
            print(f"  [MISSING]  {fname:<36} - {info['description']}")
            manifest[key] = {
                "filename": fname,
                "path": None,
                "status": "missing",
                "tier": "legacy_fallback",
                "size_bytes": 0
            }

    # Top-level manifest metadata & SOTA inventory
    manifest["_meta"] = {
        "sota_required_count": sota_total_required,
        "sota_ready_count": sota_ready_count,
        "sota_fully_available": (sota_ready_count == sota_total_required),
        "primary_tier": ("sota_wave4" if sota_inventory.get("session_f", {}).get("fully_ready") else "sota_wave3") if sota_ready_count == sota_total_required else "path_b_foundation"
    }
    manifest["sota_wave_inventory"] = sota_inventory

    # Update manifest file
    manifest_path = os.path.join(CHECKPOINT_DIR, "checkpoint_manifest.json")
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest successfully updated at: {manifest_path}")
    except Exception as e:
        logger.warning(f"Notice saving checkpoint_manifest.json: {e}")

    print("=" * 78 + "\n")
    return manifest


if __name__ == "__main__":
    ensure_checkpoint_cache()
