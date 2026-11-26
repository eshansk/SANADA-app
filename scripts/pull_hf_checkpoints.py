#!/usr/bin/env python3
"""
scripts/pull_hf_checkpoints.py
==============================
Pulls all SANADA model weights, meta-calibrators, and session checkpoints
from Hugging Face into models/checkpoints/ using snapshot_download.

Usage:
    python3 scripts/pull_hf_checkpoints.py
    python3 scripts/pull_hf_checkpoints.py --repo your-org/sanada-checkpoints
    python3 scripts/pull_hf_checkpoints.py --dest /custom/checkpoint/dir --token hf_xxx
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SANADA-HFPull")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_DEST = os.path.join(BASE_DIR, "models", "checkpoints")
DEFAULT_REPO = os.getenv("HF_CHECKPOINT_REPO", "eshansk/sanada-pathb-checkpoints")


def pull_hf_checkpoints(
    repo_id: str,
    dest_dir: str = DEFAULT_DEST,
    token: str = None,
    allow_patterns: list = None,
    ignore_patterns: list = None
) -> bool:
    """
    Downloads full repository snapshot from Hugging Face into dest_dir.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub package is not installed. Run: pip install huggingface_hub")
        return False

    os.makedirs(dest_dir, exist_ok=True)

    token_val = token or os.getenv("HF_TOKEN")
    logger.info(f"Target repository : {repo_id}")
    logger.info(f"Target directory  : {dest_dir}")
    logger.info(f"Authentication    : {'Token provided' if token_val else 'Anonymous / Public'}")

    dl_kwargs = {
        "repo_id": repo_id,
        "local_dir": dest_dir,
        "local_dir_use_symlinks": False,
    }
    if token_val:
        dl_kwargs["token"] = token_val
    if allow_patterns:
        dl_kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        dl_kwargs["ignore_patterns"] = ignore_patterns

    try:
        logger.info("Initiating Hugging Face snapshot download (this may take a few minutes for multi-gigabyte models)...")
        downloaded_path = snapshot_download(**dl_kwargs)
        logger.info(f"Snapshot successfully pulled to: {downloaded_path}")

        # Post-download verification
        from scripts.download_checkpoints import ensure_checkpoint_cache
        logger.info("Verifying checkpoint integrity and updating manifest...")
        ensure_checkpoint_cache()
        return True
    except Exception as e:
        logger.error(f"Failed to download repository snapshot from {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pull SANADA model checkpoints from Hugging Face Hub using snapshot_download."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO} or $HF_CHECKPOINT_REPO)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DEST,
        help=f"Destination directory for checkpoints (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set $HF_TOKEN)"
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Optional pattern filters (e.g. *.pt *.json *.pkl)"
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="+",
        default=None,
        help="Optional ignore patterns"
    )

    args = parser.parse_args()
    success = pull_hf_checkpoints(
        repo_id=args.repo,
        dest_dir=args.dest,
        token=args.token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
