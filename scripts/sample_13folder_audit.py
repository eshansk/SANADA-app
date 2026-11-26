#!/usr/bin/env python3
"""
scripts/sample_13folder_audit.py
================================
Sampled 13-Folder Final Audit (3-4 Held-Out Test Files per Folder)
Target Root: ../sanada_project/SANADA/Org_Fsd_Mix
"""

import os
import sys
import ast
import json
import wave
import time
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.sota_pipeline import SANADASOTAPipeline
from models.device_utils import select_compute_device

EXPECTED_FOLDERS = [
    "FSD_Mix_2_Sounds",
    "FSD_Mix_3_Sounds",
    "FSD_Mix_15_Sounds",
    "FSD_Mix_20_Sounds",
    "FSD_Mix_25_Sounds",
    "FSD_Mix_Variable_Sounds",
    "Org_Mix_2_Sounds",
    "Org_Mix_3_Sounds",
    "Org_Mix_4_Sounds",
    "Org_Mix_4_Sounds_Variable",
    "Org_Mix_6_Sounds",
    "Org_Mix_9_Sounds",
    "Org_Mix_9_Variable",
]


def safe_parse_list(val) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        out = json.loads(s)
        return [str(x).strip() for x in out] if isinstance(out, list) else [str(out).strip()]
    except Exception:
        try:
            out = ast.literal_eval(s)
            return [str(x).strip() for x in out] if isinstance(out, list) else [str(out).strip()]
        except Exception:
            return [s]


def norm_label(lbl: str) -> str:
    s = str(lbl).strip()
    if s.startswith("FSD_"):
        s = s[4:]
    return s.replace(" ", "_")


def check_wav_integrity(wav_path: str) -> Dict[str, Any]:
    with wave.open(wav_path, "rb") as wf:
        ch, sw, sr, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        pcm = wf.readframes(frames)
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    issues = []
    if sr != 16000:
        issues.append(f"sr={sr}")
    if ch != 1:
        issues.append(f"ch={ch}")
    if sw != 2:
        issues.append(f"sw={sw}")
    if frames != 128000:
        issues.append(f"frames={frames}")
    if not np.isfinite(arr).all():
        issues.append("nan_inf")
    if rms < 1e-6:
        issues.append("silent")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "rms_dbfs": round(float(20.0 * np.log10(max(rms, 1e-12))), 2),
        "peak_dbfs": round(float(20.0 * np.log10(max(peak, 1e-12))), 2),
    }


def select_test_samples(dataset_root: str, folder: str, per_folder: int) -> List[Dict[str, Any]]:
    test_dir = os.path.join(dataset_root, folder, "test")
    csv_path = os.path.join(test_dir, "metadata.csv")
    if not os.path.isdir(test_dir) or not os.path.isfile(csv_path):
        return []

    wav_map = {}
    for root, _, files in os.walk(test_dir):
        for fn in files:
            if fn.lower().endswith(".wav"):
                wav_map[fn] = os.path.join(root, fn)

    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        mix_col = row["mixture_path"] if "mixture_path" in df.columns else row.get("filename", "")
        bn = os.path.basename(str(mix_col))
        if bn not in wav_map:
            continue
        lbls = [norm_label(x) for x in safe_parse_list(row["speaker_labels"])] if "speaker_labels" in df.columns else []
        k_val = int(row["num_speakers"]) if ("num_speakers" in df.columns and not pd.isna(row["num_speakers"])) else len(lbls)
        records.append({"wav_path": wav_map[bn], "filename": bn, "k_gt": k_val, "labels_gt": lbls})

    if not records:
        return []

    # Sort Variable folders by cardinality so sampled files span low -> mid -> high K
    if "Variable" in folder:
        records.sort(key=lambda r: (r["k_gt"], r["filename"]))

    n = len(records)
    if n <= per_folder:
        return records
    idxs = np.linspace(0, n - 1, per_folder, dtype=int)
    return [records[int(i)] for i in idxs]


def main():
    parser = argparse.ArgumentParser(description="Sampled 3-4 file/folder SOTA audit across all 13 Org_Fsd_Mix folders")
    parser.add_argument(
        "--dataset-root",
        default=os.path.abspath(os.path.join(BASE_DIR, "..", "sanada_project", "SANADA", "Org_Fsd_Mix")),
    )
    parser.add_argument("--per-folder", type=int, default=3, help="Number of held-out test files to check per folder (default: 3)")
    parser.add_argument("--full-sep", action="store_true", help="Also run full stem separation (slower on CPU)")
    parser.add_argument("--out-json", default=os.path.join(BASE_DIR, "results", "sample_13folder_audit.json"))
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        print(f"ERROR: Dataset root not found: {dataset_root}")
        sys.exit(1)

    t0_all = time.time()
    device = select_compute_device()
    pipe = SANADASOTAPipeline(device=device)
    assert pipe.is_ready(), f"SOTA pipeline not ready: {pipe.get_readiness_dict()}"

    task_mode = "all" if args.full_sep else "analyze"
    print("=" * 112)
    print(f" SANADA SAMPLED 13-FOLDER AUDIT | {args.per_folder} test files/folder | task={task_mode} | device={device}")
    print(f" Dataset Root : {dataset_root}")
    print(f" Readiness    : {pipe.get_readiness_dict()}")
    print("=" * 112)

    all_file_results = []
    folder_summaries = []

    for folder in EXPECTED_FOLDERS:
        is_fsd = folder.upper().startswith("FSD")
        domain = "environmental" if is_fsd else "speech"
        samples = select_test_samples(dataset_root, folder, args.per_folder)
        if not samples:
            print(f"\n[MISSING] {folder} — test/ or metadata.csv not found")
            folder_summaries.append({"folder": folder, "domain": domain, "status": "MISSING", "checked": 0})
            continue

        print(f"\n--- {folder} ({domain.upper()} | {len(samples)} files) ---")
        wav_ok_cnt = 0
        pass_cnt = 0
        warn_cnt = 0
        fail_cnt = 0
        k_exact_cnt = 0
        k_pm1_cnt = 0
        fa_total = 0
        hit_total = 0
        pred_tag_total = 0

        for item in samples:
            wav_chk = check_wav_integrity(item["wav_path"])
            if wav_chk["ok"]:
                wav_ok_cnt += 1

            t0 = time.time()
            with torch.no_grad():
                out = pipe.analyze_and_separate(
                    audio_input=item["wav_path"],
                    orig_sr=16000,
                    mode="auto",
                    output_dir="/tmp/sanada_sample_audit_stems",
                    tag_threshold=None,
                    file_prefix=f"{folder}_{item['filename']}",
                    all_4_stems=False,
                    task=task_mode,
                )
            lat_s = round(time.time() - t0, 1)
            pred_k = int(out["predicted_k"])
            raw_k = float(out["raw_k"])
            pred_tags = [norm_label(ev["label"]) for ev in out.get("active_events", [])]

            if not is_fsd:
                k_gt = int(item["k_gt"])
                err = abs(pred_k - k_gt)
                if err == 0:
                    k_exact_cnt += 1
                if err <= 1:
                    k_pm1_cnt += 1
                fa = len(pred_tags)
                fa_total += fa
                verdict = "PASS" if (wav_chk["ok"] and err <= 1 and fa == 0) else ("WARN" if (wav_chk["ok"] and err <= 1) else "FAIL")
                print(
                    f"  [{verdict:<4}] {item['filename']:<15} | WAV={'OK' if wav_chk['ok'] else wav_chk['issues']} (pk={wav_chk['peak_dbfs']} dB) "
                    f"| GT K={k_gt} -> Pred K={pred_k} (cont={raw_k:.2f}) | EnvFA={fa} | {lat_s}s"
                )
                rec = {
                    "folder": folder, "domain": domain, "file": item["filename"],
                    "wav_ok": wav_chk["ok"], "peak_dbfs": wav_chk["peak_dbfs"],
                    "k_gt": k_gt, "k_pred": pred_k, "k_cont": round(raw_k, 3), "k_err": err,
                    "env_fa": fa, "pred_tags": pred_tags, "latency_s": lat_s, "verdict": verdict,
                }
            else:
                gt_set = set(item["labels_gt"])
                hits = [t for t in pred_tags if t in gt_set]
                hit_total += len(hits)
                pred_tag_total += len(pred_tags)
                if pred_k == 0:
                    k_exact_cnt += 1
                verdict = "PASS" if (wav_chk["ok"] and pred_k == 0 and len(hits) >= 1) else ("WARN" if (wav_chk["ok"] and pred_k == 0) else "FAIL")
                print(
                    f"  [{verdict:<4}] {item['filename']:<15} | WAV={'OK' if wav_chk['ok'] else wav_chk['issues']} (pk={wav_chk['peak_dbfs']} dB) "
                    f"| Speech K={pred_k} (exp 0) | M={len(gt_set):<2} | Hits={len(hits)}/{len(pred_tags)} {hits} | {lat_s}s"
                )
                rec = {
                    "folder": folder, "domain": domain, "file": item["filename"],
                    "wav_ok": wav_chk["ok"], "peak_dbfs": wav_chk["peak_dbfs"],
                    "m_gt": len(gt_set), "k_pred": pred_k, "k_cont": round(raw_k, 3),
                    "pred_tags": pred_tags, "gt_hits": hits, "latency_s": lat_s, "verdict": verdict,
                }

            if verdict == "PASS":
                pass_cnt += 1
            elif verdict == "WARN":
                warn_cnt += 1
            else:
                fail_cnt += 1
            all_file_results.append(rec)

        folder_summaries.append({
            "folder": folder,
            "domain": domain,
            "checked": len(samples),
            "wav_ok": wav_ok_cnt,
            "pass": pass_cnt,
            "warn": warn_cnt,
            "fail": fail_cnt,
            "exact_k_or_k0": k_exact_cnt,
            "pm1_k": k_pm1_cnt if not is_fsd else None,
            "env_fa_total": fa_total if not is_fsd else None,
            "tag_precision_pct": round((hit_total / pred_tag_total) * 100.0, 1) if (is_fsd and pred_tag_total > 0) else None,
            "status": "PASS" if fail_cnt == 0 else "REVIEW",
        })

    print("\n" + "=" * 112)
    print(f"{'Folder':<28} | {'Domain':<6} | {'Files':>5} | {'WAV':>5} | {'Exact/K0':>8} | {'±1 / TagPrec':>12} | {'EnvFA':>5} | {'Status'}")
    print("-" * 112)
    for s in folder_summaries:
        if s["status"] == "MISSING":
            print(f"{s['folder']:<28} | {s['domain'][:6]:<6} |     0 |   0/0 |      N/A |          N/A |   N/A | MISSING")
            continue
        n = s["checked"]
        col2 = f"{s['pm1_k']}/{n}" if s["domain"] == "speech" else (f"{s['tag_precision_pct']}%" if s["tag_precision_pct"] is not None else "0.0%")
        fa_col = str(s["env_fa_total"]) if s["domain"] == "speech" else "-"
        print(
            f"{s['folder']:<28} | {s['domain'][:6]:<6} | {n:>5} | {s['wav_ok']}/{n:<3} | "
            f"{s['exact_k_or_k0']}/{n:<6} | {col2:>12} | {fa_col:>5} | {s['status']}"
        )
    print("=" * 112)
    print(f"Total Checked: {len(all_file_results)} files across {len(folder_summaries)} folders in {time.time() - t0_all:.1f}s")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"dataset_root": dataset_root, "folder_summaries": folder_summaries, "files": all_file_results}, f, indent=2)
    print(f"Saved JSON report to: {args.out_json}")


if __name__ == "__main__":
    main()
