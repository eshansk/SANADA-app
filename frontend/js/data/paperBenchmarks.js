/**
 * frontend/js/data/paperBenchmarks.js
 * ==================================
 * Single Source of Truth for paper-reported metrics and empirical bounds.
 * Corresponds to Murthy & Prakash (2025) IEEE DataPort / SSRN benchmark release.
 */

export const PAPER_BENCHMARKS = {
    citation: {
        title: "SANADA: A Reproducible Synthetic Acoustic Dataset for Intractable Mixtures, Evaluated with Baseline Models in Separation, Counting, and Tagging",
        authors: "P. V. N. Balarama Murthy and R. Vijaya Prakash",
        institutions: "SR University / Neil Gogte Institute of Technology",
        doi: "10.21227/z7hv-e014",
        repository: "IEEE DataPort",
        ssrnPreprint: "SSRN preprint 7199188, not peer reviewed"
    },

    datasetSummary: {
        totalStems: 27112,
        totalStemsDurationHours: 60.25,
        totalStemsSizeGb: 6.47,
        totalMixtures: 15600,
        totalMixturesDurationHours: 34.66,
        totalMixturesSizeGb: 3.72,
        subDatasetsCount: 13,
        sampleRateHz: 16000,
        bitDepth: "16-bit PCM",
        channels: 1, // Monaural
        clipDurationSeconds: 8.0,
        clipSamples: 128000
    },

    counting: {
        model: "Conformer Regressor (MSE Loss)",
        results: [
            {
                corpus: "Org-Mix-4",
                kMax: 4,
                mae: 0.3243,
                rmse: 0.4302,
                exactAccuracyPct: 74.75,
                plusMinusOneAccuracyPct: 100.0
            },
            {
                corpus: "Org-Mix-9",
                kMax: 9,
                mae: 0.8766,
                rmse: 1.0925,
                exactAccuracyPct: 36.46,
                plusMinusOneAccuracyPct: 84.38
            },
            {
                corpus: "FSD-Mix-32",
                kMax: 32,
                mae: 6.7262,
                rmse: 8.0885,
                exactAccuracyPct: 5.56,
                plusMinusOneAccuracyPct: 14.44
            }
        ]
    },

    tagging: {
        model: "Conformer Classifier (Asymmetric Loss - ASL)",
        results: [
            {
                corpus: "Org-Mix-4",
                mAP: 0.9902,
                macroF1: 0.9449,
                rocAuc: 0.9765
            },
            {
                corpus: "Org-Mix-9",
                mAP: 0.9122,
                macroF1: 0.8305,
                rocAuc: 0.8744
            },
            {
                corpus: "FSD-Mix",
                mAP: 0.6265,
                macroF1: 0.6849,
                rocAuc: 0.6161
            }
        ]
    },

    separationSisdr: {
        model: "Conv-TasNet (uPIT Loss)",
        orgMixFixed: [
            { k: 2, sisdrDb: 10.59, status: "Clean (+10.6 dB)" },
            { k: 3, sisdrDb: 5.08, status: "Usable (+5.1 dB)" },
            { k: 4, sisdrDb: 4.24, status: "Usable Ceiling (+4.2 dB)" },
            { k: 6, sisdrDb: -3.40, status: "Degraded (-3.4 dB)" },
            { k: 9, sisdrDb: -8.57, status: "Intractable (-8.6 dB)" }
        ],
        fsdMixFixed: [
            { k: 2, sisdrDb: 5.05, status: "Usable (+5.1 dB)" },
            { k: 3, sisdrDb: 0.73, status: "Near 0 dB Threshold (+0.7 dB)" }
        ],
        orgMixVariable: [
            { condition: "K <= 4 (Org-Mix-4 Variable)", sisdrDb: 1.98, status: "Usable (+2.0 dB)" },
            { condition: "K <= 9 (Org-Mix-9 Variable)", sisdrDb: -9.61, status: "Degraded (< 0 dB)" }
        ]
    }
};

/**
 * Path B Foundation Model benchmarks (Verified Empirical Baselines)
 * Seeded from pathb_benchmarks.json for offline fallback and zero-latency hydration.
 */
export const PATH_B_FALLBACK_BENCHMARKS = {
    updated_at: "2026-11-20T08:20:10Z",
    tier: "path_b_foundation",
    trained_on: "Kaggle CUDA (2X GPU)",
    models: {
        counting: "WavLM-Base+ + Table-3 DSP",
        tagging: "MIT AST (33-Class Speech-Proof ASL)",
        separation_2spk: "SepFormer-Libri2Mix (Adapted)",
        separation_3spk: "SepFormer-Libri3Mix (Adapted)",
        separation_4spk: "SepFormer-4Spk (Dominant-4)"
    },
    pure_speech_rejection: {
        zero_false_env_tag_clips_pct: 93.41,
        env_true_negative_acc_pct: 99.79,
        human_speech_detection_pct: 100.0,
        // Compatibility aliases for UI cards
        speech_detection_accuracy_pct: 100.0,
        speech_false_positive_rate_pct: 0.21,
        speech_gate_bonus_applied: 0.05,
        speech_prob_threshold: 0.60
    },
    counting: {
        model: "WavLM-Base+ + Table-3 DSP",
        results: [
            {
                domain: "Org-Mix-4 (Max K=4)",
                mae: 0.2828,
                rmse: 0.3578,
                exact_acc_pct: 84.85,
                relaxed_acc_pct: 100.0
            },
            {
                domain: "Org-Mix-9 (Max K=9)",
                mae: 0.5567,
                rmse: 0.708,
                exact_acc_pct: 55.21,
                relaxed_acc_pct: 96.88
            },
            {
                domain: "FSD-Mix-32 (Max K=32)",
                mae: 2.4385,
                rmse: 3.0842,
                exact_acc_pct: 11.11,
                relaxed_acc_pct: 40.0,
                plus_minus_2_acc_pct: 64.44
            }
        ]
    },
    tagging: {
        model: "MIT AST (33-Class Speech-Proof ASL)",
        results: [
            {
                domain: "FSD-Mix-2 (32 Classes)",
                classes: "32 Classes",
                map_macro: 1.0,
                f1_macro: 1.0,
                roc_auc: 1.0,
                per_class_acc_pct: 99.53
            },
            {
                domain: "FSD-Mix-3 (32 Classes)",
                classes: "32 Classes",
                map_macro: 1.0,
                f1_macro: 0.9587,
                roc_auc: 0.9964,
                per_class_acc_pct: 98.56
            },
            {
                domain: "FSD-Mix-Variable (32 Classes)",
                classes: "32 Classes",
                map_macro: 0.8443,
                f1_macro: 0.789,
                roc_auc: 0.8492,
                per_class_acc_pct: 75.59
            }
        ]
    },
    separationSisdr: {
        model: "Adaptive Specialist Router (SepFormer / ConvTasNetSkip)",
        results: [
            {
                density_k: 2,
                domain: "Org-Mix (SepFormer-Libri2Mix)",
                si_sdr_db: 13.36,
                paper_baseline_db: 10.59,
                unseen_9spk_db: 13.51,
                status: "Studio Clean"
            },
            {
                density_k: 3,
                domain: "Org-Mix (SepFormer-Libri3Mix)",
                si_sdr_db: 10.81,
                paper_baseline_db: 5.08,
                unseen_9spk_db: 8.15,
                status: "Clean"
            },
            {
                density_k: 4,
                domain: "Org-Mix (SepFormer-4Spk)",
                si_sdr_db: 4.29,
                paper_baseline_db: 4.24,
                unseen_9spk_db: 4.74,
                status: "Enhanced Ceiling"
            },
            {
                density_k: 6,
                domain: "Crowded Live-Mic (Top-4 Dominant)",
                si_sdr_db: -0.8,
                paper_baseline_db: -3.4,
                status: "Dominant-4 Extracted"
            }
        ]
    },
    // Direct mirror schemas matching pathb_benchmarks.json exactly
    source_counting_mae: [
        {
            domain: "Org-Mix-4 (Max K=4)",
            mae: 0.2828,
            rmse: 0.3578,
            exact_acc_pct: 84.85,
            relaxed_acc_pct: 100.0
        },
        {
            domain: "Org-Mix-9 (Max K=9)",
            mae: 0.5567,
            rmse: 0.708,
            exact_acc_pct: 55.21,
            relaxed_acc_pct: 96.88
        },
        {
            domain: "FSD-Mix-32 (Max K=32)",
            mae: 2.4385,
            rmse: 3.0842,
            exact_acc_pct: 11.11,
            relaxed_acc_pct: 40.0,
            plus_minus_2_acc_pct: 64.44
        }
    ],
    multi_label_tagging_asl: [
        {
            domain: "FSD-Mix-2 (32 Classes)",
            classes: 32,
            map_macro: 1.0,
            f1_macro: 1.0,
            roc_auc: 1.0,
            per_class_acc_pct: 99.53
        },
        {
            domain: "FSD-Mix-3 (32 Classes)",
            classes: 32,
            map_macro: 1.0,
            f1_macro: 0.9587,
            roc_auc: 0.9964,
            per_class_acc_pct: 98.56
        },
        {
            domain: "FSD-Mix-Variable (32 Classes)",
            classes: 32,
            map_macro: 0.8443,
            f1_macro: 0.789,
            roc_auc: 0.8492,
            per_class_acc_pct: 75.59
        }
    ],
    source_separation_sisdr_db: [
        {
            density_k: 2,
            domain: "Org-Mix (SepFormer-Libri2Mix)",
            si_sdr_db: 13.36,
            paper_baseline_db: 10.59,
            unseen_9spk_db: 13.51,
            status: "Studio Clean"
        },
        {
            density_k: 3,
            domain: "Org-Mix (SepFormer-Libri3Mix)",
            si_sdr_db: 10.81,
            paper_baseline_db: 5.08,
            unseen_9spk_db: 8.15,
            status: "Clean"
        },
        {
            density_k: 4,
            domain: "Org-Mix (SepFormer-4Spk)",
            si_sdr_db: 4.29,
            paper_baseline_db: 4.24,
            unseen_9spk_db: 4.74,
            status: "Enhanced Ceiling"
        },
        {
            density_k: 6,
            domain: "Crowded Live-Mic (Top-4 Dominant)",
            si_sdr_db: -0.8,
            paper_baseline_db: -3.4,
            status: "Dominant-4 Extracted"
        }
    ]
};
