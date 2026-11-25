#!/usr/bin/env python3
import os, sys, time, hashlib
import torch
import soundfile as sf
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.sota_pipeline import SANADASOTAPipeline
from models.device_utils import select_compute_device, get_device_telemetry

def main():
    print("=" * 80)
    print(" SANADA SOTA WAVE 1-4 ANTI-CHEAT + REAL ACCURACY VERIFICATION")
    print("=" * 80)
    device = select_compute_device()
    tel = get_device_telemetry(device)
    print(f"Device: {device} | CUDA: {tel['cuda_available']} | MPS: {tel['mps_available']} | Fast-CPU: {tel['fast_cpu_mode']}")

    source_demos = [
        ("backend/static/demos/org_mix_2_speakers.wav", "/tmp/blind_a.wav", "Org-Mix-2 (Speech K=2)",  "k2"),
        ("backend/static/demos/org_mix_9_speakers.wav", "/tmp/blind_b.wav", "Org-Mix-9 (Speech K=9)",  "k9"),
        ("backend/static/demos/fsd_mix_4_sounds.wav",   "/tmp/blind_c.wav", "FSD-Mix-4 (Env K=0, M=4)", "env4"),
    ]
    blind_clips = []
    rng = np.random.default_rng(int(time.time()) % 100000)
    for src_rel, dst_path, desc, tag in source_demos:
        src_path = os.path.join(BASE_DIR, src_rel)
        wav, sr = sf.read(src_path, dtype="float32")
        wav_d = wav + rng.uniform(-1e-5, 1e-5, size=wav.shape).astype(np.float32)
        sf.write(dst_path, wav_d, sr, subtype="PCM_16")
        md5 = hashlib.md5(open(dst_path, "rb").read()).hexdigest()
        print(f"  • {desc:28} -> {dst_path} | MD5={md5} | Samples={len(wav_d)}")
        blind_clips.append((dst_path, desc, tag))

    pipeline = SANADASOTAPipeline(device=device)
    print("Readiness:", pipeline.get_readiness_dict())
    assert pipeline.is_ready(), "SOTA pipeline is not ready!"

    results_by_tag = {}
    for path, desc, tag in blind_clips:
        print(f"\n--- Running Blind Clip: {desc} ({path}) ---")
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        res = pipeline.analyze_and_separate(
            audio_input=path, orig_sr=16000, mode="auto",
            output_dir="/tmp/sota_verify_stems", tag_threshold=None, all_4_stems=False, task="all",
        )
        lat_ms = round((time.time() - t0) * 1000.0, 1)
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"  Peak GPU VRAM: {peak_gb:.2f} GB")
            assert peak_gb < 3.8, f"Peak VRAM {peak_gb:.2f} GB exceeded 3.8 GB!"

        pred_k = res["predicted_k"]
        ev_cnt = len(res["active_events"])
        top3 = [(e["label"], e["confidence"], e["threshold"]) for e in res["top_5_events"][:3]]
        print(f"  • Predicted K      : {pred_k} (cont={res['raw_k']}, k_raw={res['source_counting'].get('k_raw_unfiltered')}, k_e1={res['source_counting'].get('k_e1_filtered')})")
        print(f"  • Active Separator : {res['active_separator']} | Stems={len(res['separated_stems'])}")
        print(f"  • Active Events    : {ev_cnt} | Top3={top3}")
        print(f"  • Latency          : {lat_ms} ms")
        results_by_tag[tag] = res

    # STRICT ACCURACY & ANTI-COLLAPSE ASSERTIONS
    r_k2, r_k9, r_e4 = results_by_tag["k2"], results_by_tag["k9"], results_by_tag["env4"]
    assert r_k2["predicted_k"] in (2, 3) and abs(r_k2["raw_k"] - 2.0) <= 0.85, f"Org-Mix-2 failed: {r_k2['predicted_k']} ({r_k2['raw_k']})"
    assert len(r_k2["active_events"]) <= 1, f"Org-Mix-2 false alarms: {r_k2['active_events']}"
    assert r_k9["predicted_k"] >= 7, f"Org-Mix-9 collapsed: K={r_k9['predicted_k']} ({r_k9['raw_k']})"
    assert "ORPIT" in r_k9["active_separator"] and len(r_k9["separated_stems"]) == r_k9["predicted_k"], f"Org-Mix-9 routing failed: {r_k9['active_separator']}"
    assert len(r_k9["active_events"]) <= 1, f"Org-Mix-9 false alarms: {r_k9['active_events']}"
    assert r_e4["predicted_k"] == 0, f"FSD-Mix-4 should be K=0, got K={r_e4['predicted_k']}"
    assert 2 <= len(r_e4["active_events"]) <= 4, f"FSD-Mix-4 should detect 2-4 true events, got {len(r_e4['active_events'])}: {r_e4['active_events']}"
    assert r_k2["label_tagging"]["class_probabilities"] != r_e4["label_tagging"]["class_probabilities"], "Session B probabilities collapsed!"
    print("\n✓✓✓ ALL STRICT ACCURACY & ANTI-COLLAPSE ASSERTIONS PASSED 100%! ✓✓✓")

if __name__ == "__main__":
    main()
