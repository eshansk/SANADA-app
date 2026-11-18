# SANADA: Dynamic Machine Listening & Speech Separation for Intractable Mixtures
## Master Comprehensive System Documentation & Technical Reference Manual

> **Document Classification:** Master Technical Architecture & Complete Codebase Specification  
> **Target Audience:** Technical Evaluators, Research Engineers, Slide Synthesis LLMs (Google Gemini), and Full-Stack Developers  
> **Repository Scope:** Complete `/home/eshan/www/sanada_app` codebase (every folder, script, neural model, REST endpoint, and UI component)

---

## 1. Executive Summary & Research Foundation

### 1.1 Academic Provenance & Project Citation
* **Project Name:** SANADA (Samslesha-Nada: *Synthesized Sounds of Overlapping Acoustic Events*)
* **Full Title:** *SANADA: Dynamic Machine Listening & Speech Separation for Intractable Audio Mixtures with Full-Stack Web Architecture*
* **Primary Research Publication:** 
  * *Title:* "SANADA: A Reproducible Synthetic Acoustic Dataset for Intractable Mixtures, Evaluated with Baseline Models in Separation, Counting, and Tagging"
  * *Authors:* P. V. N. Balarama Murthy and R. Vijaya Prakash
  * *Affiliation:* SR University, Warangal & Neil Gogte Institute of Technology (NGIT), Hyderabad, India
  * *Indexing & Persistent Identifier:* IEEE DataPort, DOI: [10.21227/z7hv-e014](https://doi.org/10.21227/z7hv-e014)
  * *Preprint Repository:* SSRN Electronic Journal, Preprint ID: 7199188
  * *License:* Creative Commons Attribution 4.0 International (CC BY 4.0)

### 1.2 Core Scientific Thesis: The High-Density Cocktail Party Problem
In human auditory perception, listeners possess the innate psychoacoustic ability to focus on an individual conversational voice amidst background chatter, environmental clatter, and reverberation—a phenomenon termed the **Cocktail Party Problem** (Cherry, 1953). In artificial intelligence and machine listening:
* **The Sterile Lab Benchmark Limitation:** Traditional speech separation benchmarks—such as WSJ0-2mix, WSJ0-3mix, LibriMix, and WHAMR!—artificially cap evaluation at $K = 2$ or $K = 3$ clean, fluent speakers.
* **The Real-World Gap:** Real-world acoustic environments (university lecture halls, open-plan workspaces, crowded cafeterias, airport concourses, emergency dispatch rooms) routinely exhibit $K = 4$ to $K = 9$ simultaneous speakers alongside dozens of overlapping environmental acoustic events.
* **The Physical Intractability Barrier:** It is physically impossible to record 9 real people speaking simultaneously inside a real acoustic space while simultaneously capturing clean, crosstalk-free, studio-grade isolated ground-truth reference microphones for each talker. Room acoustics, physical sound reflection, and air-pressure wave superposition cause irreversible acoustic crosstalk at the microphone capsules.
* **The SANADA Solution:** Solves this physical impossibility by establishing a mathematically rigorous, reproducible generative mixing engine that synthesizes high-density polyphonic mixtures ($K = 2$ to $K = 32$) from an expansive pool of clean, isolated mono stems with 100% ground-truth reference preservation.

### 1.3 The Three Fundamental Ceilings Exposed by SANADA
The research establishes three definitive mathematical and architectural boundaries governing machine listening on single-channel audio:

```mermaid
graph TD
    A["Scientific Ceilings in High-Density Listening"] --> B["1. The Complexity Ceiling"]
    A --> C["2. The O(K!) Permutation Wall"]
    A --> D["3. Zero-Padding Degradation"]
    
    B --> B1["SI-SDR collapses below 0 dB when K > 4"]
    B --> B2["Vocal energy overlaps completely across time and frequency"]
    
    C --> C1["uPIT checks K! permutations during training"]
    C --> C2["K=15 requires 1.307 Trillion combinations"]
    C --> C3["Triggers fatal CUDA OOM crashes on enterprise A100 GPUs"]
    
    D --> D1["Zero-padding variable-K mixtures causes gradient dilution"]
    D --> D2["Separation drops from +4.24 dB to +1.98 dB"]
```

1. **The Complexity Ceiling (Acoustic Masking Limit):**
   * Time-domain source separation models (e.g. Conv-TasNet) achieve high-fidelity separation at $K=2$ ($+10.59\text{ dB}$ SI-SDR) and usable separation at $K=3$ ($+5.08\text{ dB}$) and $K=4$ ($+4.24\text{ dB}$).
   * Beyond $K=4$, speech separation performance collapses catastrophically into negative decibels: $K=6 \to -3.40\text{ dB}$, $K=9 \to -8.57\text{ dB}$.
   * *Physical Cause:* Human vocal tracts occupy overlapping fundamental frequencies ($85\text{ Hz}$ to $255\text{ Hz}$) and formants ($500\text{ Hz}$ to $3500\text{ Hz}$). When $K > 4$, vocal energy continuously occupies every spectral and temporal bin, leaving zero silent pauses or harmonic relief for monaural mask estimation.
2. **The $O(K!)$ Permutation Invariant Training (uPIT) Computational Wall:**
   * In label-free multi-talker separation, the network has no natural way to assign estimated channels to ground-truth targets (the Permutation Problem). Utterance-level Permutation Invariant Training (uPIT) solves this by computing loss over all $K!$ possible assignment permutations.
   * Permutation complexity explodes factorially:
     * $K=2 \to 2$ permutations
     * $K=4 \to 24$ permutations
     * $K=6 \to 720$ permutations
     * $K=9 \to 362{,}880$ permutations
     * $K=14 \to 87{,}178{,}291{,}200$ permutations
     * $K=15 \to 1{,}307{,}674{,}368{,}000$ ($1.307\text{ Trillion}$) permutations $\to$ **Immediate GPU Out-Of-Memory failure**.
3. **Zero-Padding Gradient Degradation:**
   * In variable-cardinality mixtures ($K \in [2, 4]$ or $K \in [2, 9]$), training a fixed-capacity separator requires zero-padding inactive target channels.
   * Zero-padding introduces phantom gradient signals, causing SI-SDR on variable 4-speaker mixtures to drop from $+4.24\text{ dB}$ down to $+1.98\text{ dB}$.

---

## 2. Generative Dataset Architecture & Signal Physics

### 2.1 The 3-Phase Generative Mixing Engine
The SANADA corpus is generated via a strict, multi-stage digital signal processing pipeline:

```
[Phase 1: Domain Acquisition]
├── Speech Wing: Studio organizational lectures (16 kHz, 16-bit PCM mono)
└── Environmental Wing: 32 verified FSD50K classes
                    │
                    ▼
[Phase 2: Pre-Mix Partitioning (Seed S=42)]
├── Train (70%) | Validation (15%) | Test (15%)
└── Partitioned BEFORE mixing -> ZERO SPEAKER / SOUND LEAKAGE
                    │
                    ▼
[Phase 3: Polyphonic Superposition with Anti-Clipping]
└── Formula: x_mix(t) = s_1(t) + Σ_{i=2}^K 10^(-3.0/20) * s_i(t)
```

#### A. Phase 1: Domain Acquisition & Signal Harmonization
* **Speech Wing:** Studio-recorded organizational lectures capturing individual faculty speakers.
* **Environmental Wing:** 32 verified acoustic sound event categories derived from the Freesound Dataset 50K (FSD50K).
* **Audio Harmonization:** Every source recording is decoded, downmixed to single-channel monaural, resampled to $16{,}000\text{ Hz}$ via polyphase sinc interpolation, and encoded as uncompressed 16-bit Linear PCM WAV.

#### B. Phase 2: Deterministic Pre-Mix Partitioning Protocol (Zero Leakage)
* **The Academic Trap:** Standard naive synthesis pipelines mix audio clips first and then split the resulting mixtures into train/val/test splits. This creates severe *speaker leakage* and *event leakage*, where identical speaker voice characteristics or acoustic stems appear in both training and evaluation sets, falsely inflating performance metrics.
* **The Pre-Mix Solution (Algorithm 3):** Clean stems are deterministically assigned into disjoint partitions (70% Train, 15% Validation, 15% Test) using a fixed Pseudo-Random Number Generator (PRNG) Seed ($S = 42$) **prior to mixture synthesis**.
* **Integrity Audit:** `zero_leakage_audit.py` and `scripts/verify_zero_leakage.py` mathematically enforce:
  $$\text{Stems}_{\text{train}} \cap \text{Stems}_{\text{val}} = \emptyset, \quad \text{Stems}_{\text{train}} \cap \text{Stems}_{\text{test}} = \emptyset, \quad \text{Stems}_{\text{val}} \cap \text{Stems}_{\text{test}} = \emptyset$$
  Confirming $0$ leaked stems across all 13 sub-corpora.
* **15.0s Buffer Strategy:** Raw stems are extracted with a $15.0\text{s}$ buffer before taking an $8.0\text{s}$ window, ensuring that natural speech pauses and continuous acoustic energy are preserved without boundary transient distortion.

#### C. Phase 3: Polyphonic Superposition & The Anti-Clipping Superposition Theorem
Linear summation of $K$ normalized waveforms $\sum_{i=1}^K s_i(t)$ inevitably produces digital clipping ($> 0\text{ dBFS}$ or absolute amplitude $> 1.0$), triggering severe harmonic distortion. Applying Dynamic Range Compression (DRC) or peak limiters alters the physical wave structure and ruins linear mathematical ground-truth.

SANADA resolves this via the **Anti-Clipping Mathematical Superposition Theorem**:
$$\mathbf{x}_{\text{mix}}(t) = \mathbf{s}_1(t) + \sum_{i=2}^K 10^{-\frac{3.0}{20}} \cdot \mathbf{s}_i(t) = \mathbf{s}_1(t) + 0.70794578 \sum_{i=2}^K \mathbf{s}_i(t)$$
* **Physical Mechanics:** The primary speaker $\mathbf{s}_1(t)$ is maintained at full nominal scale ($0\text{ dBFS}$ reference), while all secondary overlapping interfering sources ($i = 2, \dots, K$) are attenuated by exactly $-3.0\text{ dBFS}$ ($10^{-3.0/20} \approx 0.7079$).
* **Mathematical Invariant:** This logarithmic scaling prevents integer overflow and eliminates inter-modulation distortion without non-linear compression artifacts. Superposition linearity is verified with Pearson correlation $> 0.9998$.

### 2.2 Comprehensive Dataset Statistics & Physical Specifications
* **Master Pre-Mix Stems Pool:** $27{,}112$ clean stems ($60.25\text{ hours}$, $6.47\text{ GB}$).
* **Synthesized Polyphonic Mixtures:** $15{,}600$ mixtures ($34.66\text{ hours}$, $3.72\text{ GB}$) spanning $13$ sub-corpora.
* **Audio Waveform Standard:** Single-channel Mono, $16{,}000\text{ Hz}$ Sampling Rate, 16-bit Linear PCM WAV, fixed $8.0\text{ seconds}$ ($N = 128{,}000\text{ samples}$), exact nominal file size $256{,}044\text{ bytes}$.
* **Acoustic Characteristics:** Mean WADA-SNR $= 18.4\text{ dB}$, mean reverberation time $RT_{60} \approx 0.47\text{ s}$.

#### Master Sub-Dataset Distribution Table
| Sub-Dataset Directory | Acoustic Domain | Density Cardinality ($K$) | Train Clips | Val Clips | Test Clips | Total Clips | Total Audio Duration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Org_Mix_2_Sounds` | Organizational Speech | Fixed $K = 2$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `Org_Mix_3_Sounds` | Organizational Speech | Fixed $K = 3$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `Org_Mix_4_Sounds` | Organizational Speech | Fixed $K = 4$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `Org_Mix_4_Sounds_Variable` | Organizational Speech | Variable $K \in \{2, 3, 4\}$ | 999 | 99 | 99 | 1,197 | 2.66 h |
| `Org_Mix_6_Sounds` | Organizational Speech | Fixed $K = 6$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `Org_Mix_9_Sounds` | Organizational Speech | Fixed $K = 9$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `Org_Mix_9_Variable` | Organizational Speech | Variable $K \in \{2..9\}$ | 1,000 | 96 | 96 | 1,192 | 2.65 h |
| `FSD_Mix_2_Sounds` | Environmental Acoustic | Fixed $K = 2$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `FSD_Mix_3_Sounds` | Environmental Acoustic | Fixed $K = 3$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `FSD_Mix_15_Sounds` | Environmental Acoustic | Fixed $K = 15$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `FSD_Mix_20_Sounds` | Environmental Acoustic | Fixed $K = 20$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `FSD_Mix_25_Sounds` | Environmental Acoustic | Fixed $K = 25$ | 1,000 | 100 | 100 | 1,200 | 2.67 h |
| `FSD_Mix_Variable_Sounds` | Environmental Acoustic | Variable $K \in \{2..32\}$ | 990 | 90 | 90 | 1,170 | 2.60 h |
| **GRAND TOTALS** | **Speech & Environmental** | **$K = 2$ to $32$** | **12,989** | **1,285** | **1,285** | **15,559** | **34.61 Hours** |

---

## 3. End-to-End Codebase Directory Tree & Architectural Layout

The complete filesystem structure of `/home/eshan/www/sanada_app` is cataloged below:

```
/home/eshan/www/sanada_app/
├── backend/                                # TIER 2: FastAPI Web Gateway & REST Endpoints
│   ├── app.py                              # FastAPI REST application (endpoints, schemas, static mounts)
│   ├── static/                             # Static audio asset directories
│   │   ├── demos/                          # Verified benchmark demonstration audio clips
│   │   └── stems/                          # Ephemeral isolated stem storage for web player delivery
│   └── __pycache__/
│
├── frontend/                               # TIER 1: Responsive Single-Page Web Client (Vanilla ES6+)
│   ├── index.html                          # Master SPA containing all 7 views & semantic HTML5 structure
│   ├── assets/
│   │   └── wave_bg.mp4                     # High-definition video backdrop for landing experience
│   ├── css/
│   │   ├── tokens.css                      # Design tokens (glassmorphism colors, typography, elevations)
│   │   ├── base.css                        # Global CSS resets, dark theme variables, typography setup
│   │   ├── components.css                  # Reusable UI widgets (cards, buttons, inputs, audio player)
│   │   ├── pages.css                       # View-specific layout rules (studio, dataset, analytics, docs)
│   │   └── style.css                       # Core stylesheet entry point
│   ├── js/
│   │   ├── main.js                         # Application bootstrap and module orchestrator
│   │   ├── app.js                          # SPA lifecycle, global state manager, view switching
│   │   ├── config.js                       # Client configuration (API base URLs, sample rate, constants)
│   │   ├── api.js                          # REST fetch service with error handling and toasts
│   │   ├── audioEngine.js                  # Web Audio API wrapper (decoding, audio context, analyser)
│   │   ├── stemPlayer.js                   # Multi-track stem player (synchronized solo, mute, gain, seek)
│   │   ├── visualizer.js                   # 60 FPS HTML5 Canvas real-time oscillogram & spectrogram
│   │   ├── audio/
│   │   │   ├── audioEngine.js              # Modular Web Audio API audio graph manager
│   │   │   ├── signalVisualizer.js         # Canvas DSP visualizer class
│   │   │   └── stemPlayer.js               # Multi-track audio channel mixer class
│   │   ├── data/
│   │   │   └── paperBenchmarks.js          # Ground-truth evaluation metrics for interactive charts
│   │   ├── pages/
│   │   │   ├── home.js                     # Landing view controller & CTA actions
│   │   │   ├── studio.js                   # Machine Listening Studio controller (upload, analyze, play)
│   │   │   ├── developer.js                # Developer Training Console controller (dispatch & poll)
│   │   │   ├── results.js                  # Analytics view controller (charts & telemetry tables)
│   │   │   └── paper.js                    # Research publication & citation controller
│   │   ├── services/
│   │   │   └── api.js                      # Service-layer REST client functions
│   │   └── ui/
│   │       ├── router.js                   # Hash-based SPA client router
│   │       ├── toast.js                    # Modern animated toast notification system
│   │       ├── charts.js                   # Custom Canvas/SVG benchmark telemetry charting
│   │       └── homeVideo.js                # Background video playback rate & loop manager
│   ├── demos/                              # Client-side mirror of verified WAV demo mixtures
│   ├── package.json                        # Frontend manifest & metadata
│   ├── vercel.json                         # Vercel static deployment routing configuration
│   └── README.md                           # Frontend architecture and integration documentation
│
├── models/                                 # TIER 3: Neural Architectures, Checkpoints & Inference Bridge
│   ├── checkpoints/                        # Serialized PyTorch weights & metadata manifests
│   │   ├── conformer_counting_baseline.pt  # Trained Task 1 Conformer Counter weights
│   │   ├── conformer_tagging_asl.pt        # Trained Task 2 Conformer Tagger (ASL) weights
│   │   ├── conformer_tagging_baseline.pt   # Baseline Task 2 Conformer weights
│   │   ├── conv_tasnet_separation.pt       # Trained Task 3 Time-Domain Conv-TasNet weights
│   │   ├── audio_catalog.json              # Catalog of verified evaluation audio files
│   │   ├── checkpoint_manifest.json        # Integrity manifest of model checkpoints
│   │   └── counting_export_manifest.json   # Export metadata for trained counting model
│   ├── inference_bridge.py                 # SANADAInferenceEngine adapter wrapping sanada_inference.py
│   ├── counting/                           # Task 1 Source Density Counting Sub-Package
│   │   ├── __init__.py
│   │   ├── config.py                       # Audio DSP parameters and ConformerCounter config
│   │   ├── conformer_counter.py            # MacaronConformerBlock and ConformerCounter nn.Module
│   │   ├── dataset.py                      # CountingDataset DataLoader implementation
│   │   ├── train_counter.py                # Standalone training script for Conformer Counter
│   │   ├── evaluate.py                     # Evaluation script calculating MAE, RMSE, exact accuracy
│   │   └── predict.py                      # Inference utility for count regression
│   ├── tagging/                            # Task 2 Acoustic Tagging Sub-Package
│   │   ├── __init__.py
│   │   ├── config.py                       # Tagging hyperparameters and 32 FSD class definitions
│   │   ├── conformer_tagger.py             # ConformerTagger neural network definition
│   │   ├── asl_loss.py                     # Asymmetric Loss (ASL) implementation with clipping
│   │   ├── dataset.py                      # TaggingDataset implementation with multi-hot targets
│   │   ├── train_tagger.py                 # Standalone training script for acoustic tagging
│   │   ├── evaluate.py                     # Evaluation script computing mAP, Macro F1, ROC-AUC
│   │   └── predict.py                      # Tagging inference utility with threshold filtering
│   └── separation/                         # Task 3 Monaural Speech Separation Sub-Package
│       ├── __init__.py
│       ├── config.py                       # Conv-TasNet hyperparameters and DSP specifications
│       ├── conv_tasnet.py                  # 1D Encoder, TCN Separator, Mask Generator, Decoder
│       ├── dataset.py                      # SeparationDataset implementation loading 4 stems
│       ├── train_separation.py             # Standalone training script with uPIT SI-SDR loss
│       ├── evaluate.py                     # Evaluation script measuring SI-SDR and SDR improvements
│       └── predict.py                      # Separation prediction and stem export utility
│
├── sanada_inference.py                     # Standalone, Production-Ready Multi-Task Inference Pipeline & CLI
│                                           # (Contains 3 models, 3 audio conditioning modes, 3 production fixes)
├── dev_server.py                           # Unified local server & reverse proxy (port 3000 -> 8000 + Cloudflare)
├── start_system.sh                         # Master launch bash script booting dev_server and FastAPI
├── Dockerfile                              # Multi-stage production container build for Hugging Face Spaces
├── docker-compose.yml                      # Docker Compose specification for containerized deployment
├── render.yaml                             # Render cloud hosting blueprint
├── requirements.txt                        # Locked Python dependencies (PyTorch, FastAPI, SoundFile, etc.)
├── SANADA_Colab_Training.ipynb             # Full Google Colab GPU training notebook with balanced curriculum
├── zero_leakage_audit.py                   # Automated verification script testing pre-mix stem disjointness
│
├── scripts/                                # Operational, Data Engineering & Verification Tooling
│   ├── audit_audio_integrity.py            # Audits 16kHz, mono, 8.0s, bit-depth across dataset directories
│   ├── compile_dataset_stats.py            # Computes WADA-SNR, RT60, durations, and outputs reports
│   ├── dataset_loader.py                   # High-throughput PyTorch DataLoader with pin_memory
│   ├── download_checkpoints.py             # Automated model weight downloading and cache manager
│   ├── manifest_loader.py                  # Loads CSV manifests, speaker mappings, and 32 FSD ontology
│   ├── smoke_test_inference.py             # 6-step end-to-end verification of sanada_inference.py
│   ├── test_audio_modes.py                 # Multi-mode verification ('sanada', 'computer', 'live_mic')
│   ├── test_week4_fullstack.py             # Integration test script for Week 4 deliverables
│   ├── verify_week1_member1.py             # Frontend verification tests for Week 1
│   ├── verify_week2_member1.py             # Frontend verification tests for Week 2
│   ├── verify_week3_member1.py             # Frontend verification tests for Week 3
│   ├── verify_week4_member1.py             # Frontend verification tests for Week 4
│   └── verify_zero_leakage.py              # Verifies stem disjointness across train, val, and test splits
│
├── tests/                                  # Automated Unit & Integration Test Suites (Pytest)
│   ├── __init__.py
│   ├── test_api_endpoints.py               # 9 integration tests covering all FastAPI REST endpoints
│   ├── test_counting_pipeline.py           # Unit tests for ConformerCounter forward pass and MSE loss
│   ├── test_tagging_pipeline.py            # Unit tests for ConformerTagger, ASL loss, and activations
│   └── test_separation_pipeline.py         # Unit tests for ConvTasNet, TCN blocks, and uPIT SI-SDR
│
├── src/                                    # Member 3 Research Archive & Experimental Progression
│   ├── member3_week1_ontology_and_baseline.py               # Week 1: 32 FSD ontology & baseline
│   ├── member3_week2_conformer_asl.py                       # Week 2: Conformer tagging with ASL
│   ├── member3_week3_ablation_and_onnx.py                   # Week 3: ASL parameter ablations & ONNX
│   ├── member3_week4_quantization_and_trimodel_sync.py      # Week 4: Dynamic INT8 quantization
│   ├── member3_week5_qa_and_loadtest.py                     # Week 5: Concurrency & throughput profiling
│   └── week2_conformer_pipeline.py                          # Conformer training pipeline utility
│
├── data/                                   # Data Engineering Workspaces & Splits
│   ├── processed/                          # Intermediate processed manifests and spectrogram caches
│   ├── raw/                                # Raw audio download staging area
│   └── splits/
│       └── test_evaluation_pool.csv        # 60 verified test mixtures across speech and environmental
│
├── docs/                                   # Comprehensive System Documentation & Technical Manuals
│   ├── PROJECT_COMPREHENSIVE_DOCUMENTATION.md # THIS MASTER SPECIFICATION DOCUMENT
│   ├── api_contracts.md                    # Formal REST API schema specifications and payloads
│   ├── audio_data_specifications.md        # DSP requirements, sampling rates, and buffer formats
│   ├── dataset_statistics_report.md        # Comprehensive dataset distribution and acoustic metrics
│   ├── dataset_stats_report.md             # Dataset telemetry and duration summary
│   ├── frontend_integration_guide.md       # Integration guide for Member 1 frontend engineering
│   ├── MEMBER1_HANDOVER.md                 # Full-stack integration handover protocol (Weeks 1 to 4)
│   ├── milestone_presentation_script.md    # Timed spoken presentation scripts for project defense
│   ├── presentation_slides_guide.md        # Slide-by-slide visual layout and design blueprint
│   ├── references/
│   │   └── SANADA_Research_Paper.pdf       # Original IEEE DataPort research publication
│   └── logbooks/                           # Weekly engineering logs for Member 1 and Member 3
│       ├── week_01_initiation_and_survey.md
│       ├── week_01_member1_frontend.md
│       ├── week_02_dataset_pipeline.md
│       ├── week_02_member1_frontend.md
│       ├── week_03_architecture_and_api.md
│       ├── week_03_member1_frontend.md
│       ├── week_04_fullstack_deployment.md
│       └── week_04_member1_frontend.md
│
├── notebooks/                              # Jupyter Notebooks for Offline DSP Analysis & Prototyping
│   ├── audio_dsp_analysis.ipynb            # Spectrogram, FFT, and filterbank visualizations
│   └── dataset_exploration.ipynb           # Distribution analysis across the 13 SANADA sub-corpora
│
└── Project Review PPT Guidelines II Year.pdf # Institutional project school presentation guidelines
```

---

## 4. Machine Listening Neural Tasks & Architectures

The system deploys three distinct, highly specialized deep neural network models operating over single-channel audio waveforms:

### 4.1 Shared Log-Mel Spectrogram Front-End
For Task 1 (Counting) and Task 2 (Tagging), raw time-domain waveforms $\mathbf{x} \in \mathbb{R}^{N}$ ($N = 128{,}000$, $8.0\text{s}$ at $16{,}000\text{ Hz}$) are transformed into 2D time-frequency log-mel spectrogram representations:
* **Short-Time Fourier Transform (STFT):**
  * `sample_rate`: $16{,}000\text{ Hz}$
  * `n_fft`: $1024$ ($64.0\text{ ms}$ analysis window)
  * `hop_length`: $512$ ($32.0\text{ ms}$ step, 50% overlap)
  * `window`: Hann window
* **Mel Filterbank Projection:**
  * `n_mels`: $128$ triangular filter bands spanning $f_{\min} = 20\text{ Hz}$ to $f_{\max} = 8{,}000\text{ Hz}$ (Nyquist).
* **Dynamic Range Compression:**
  * Power spectrogram ($\text{power} = 2.0$) converted to decibels:
    $$\mathbf{M}_{\text{db}} = 10 \log_{10}(\mathbf{M}_{\text{power}}), \quad \text{clamped to } \text{top\_db} = 80.0\text{ dB}$$
* **Zero-Centered Normalization:**
  $$\mathbf{M}_{\text{norm}} = \frac{\mathbf{M}_{\text{db}} + 40.0}{40.0}$$
  Yielding a fixed-dimension tensor $\mathbf{M}_{\text{norm}} \in \mathbb{R}^{B \times 1 \times 128 \times 251}$, where $B$ is batch size, $128$ is mel frequency bins, and $251$ is temporal frames ($\lfloor 128{,}000 / 512 \rfloor + 1$).

### 4.2 Shared Conformer Backbone (`MacaronConformerBlock`)
Both Task 1 and Task 2 utilize the Macaron-Style Conformer architecture (Gulati et al., 2020), which interweaves depthwise separable convolutions with multi-head self-attention:

```
Input x
  │
  ├─────────────────────────────────────────────┐
  ▼                                             │
LayerNorm -> Half-Step FFN1 -> Dropout -> 0.5 * (+)
  │
  ├─────────────────────────────────────────────┐
  ▼                                             │
LayerNorm -> Multi-Head Self-Attention --------> (+)
  │
  ├─────────────────────────────────────────────┐
  ▼                                             │
LayerNorm -> Depthwise-Separable Conv1D -------> (+)
  │
  ├─────────────────────────────────────────────┐
  ▼                                             │
LayerNorm -> Half-Step FFN2 -> Dropout -> 0.5 * (+)
  │
  ▼
Final LayerNorm -> Output
```

#### Detailed Layer Specifications:
* `d_model`: $256$
* `n_heads`: $4$ (query, key, value projection dimension $= 64$ per head)
* `conv_kernel`: $31$ (capturing long-range temporal context)
* `dropout`: $0.1$
* **Half-Step FFN Module:**
  $$\text{FFN}(x) = \mathbf{W}_2 \cdot \text{Dropout}(\text{SiLU}(\mathbf{W}_1 \cdot \text{LayerNorm}(x)))$$
  Where $\mathbf{W}_1 \in \mathbb{R}^{256 \times 1024}$ ($4\times$ expansion) and $\mathbf{W}_2 \in \mathbb{R}^{1024 \times 256}$. Scaled by residual factor $0.5$.
* **Multi-Head Self-Attention (MHSA):**
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
  Instantiated as `nn.MultiheadAttention(d_model=256, num_heads=4, dropout=0.1, batch_first=True)` called with `need_weights=False` to save GPU memory.
* **1D Depthwise-Separable Convolution Module:**
  $$\text{ConvModule}(x) = \mathbf{W}_{\text{proj2}} \cdot \text{Dropout}(\text{SiLU}(\text{BatchNorm1d}(\text{DWConv}_{31}(\text{GLU}(\mathbf{W}_{\text{proj1}} \cdot \text{LayerNorm}(x))))))$$
  * Pointwise Conv1D: $256 \to 512$
  * Gated Linear Unit (GLU): splits $512 \to 256$
  * Depthwise Conv1D: $256 \to 256$, `groups=256`, `kernel_size=31`, `padding=15`
  * BatchNorm1d($256$)
  * Swish / SiLU activation
  * Pointwise Conv1D: $256 \to 256$
  * Dropout($0.1$)

---

### 4.3 Task 1: Source Density Estimation (Counting)
* **Objective:** Predict the active source count $K \in [2, 9]$ directly from single-channel 8.0s mixtures as a continuous scalar regression task.
* **Model Class:** `ConformerCounter` (`models/counting/conformer_counter.py` & `sanada_inference.py`)
* **Architecture:**
  1. *2D Convolutional Subsampling:*
     * Conv2d($1 \to 64$, kernel $3\times 3$, stride $2\times 2$, padding $1$) $\to$ ReLU $\to$ Conv2d($64 \to 64$, kernel $3\times 3$, stride $2\times 2$, padding $1$) $\to$ ReLU.
     * Reduces time frames from $251 \to 63$ and mel bands from $128 \to 32$ ($4\times$ downsampling).
  2. *Linear Projection:* Reshapes $[B, 64, 32, 63] \to [B, 63, 2048]$ and projects via Linear($2048 \to 256$) $\to$ Dropout($0.1$).
  3. *Conformer Backbone:* 4 stacked `MacaronConformerBlock` layers ($d_{\text{model}} = 256$).
  4. *Global Temporal Average Pooling:* Averages across the 63 time steps: $\mathbf{h} = \frac{1}{63} \sum_{t=1}^{63} \mathbf{x}_t \in \mathbb{R}^{B \times 256}$.
  5. *MLP Regression Head:*
     $$\hat{K} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h}), \quad \mathbf{W}_1 \in \mathbb{R}^{256 \times 64}, \; \mathbf{W}_2 \in \mathbb{R}^{64 \times 1}$$
* **Loss Function:** Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{b=1}^B (\hat{K}_b - K_b)^2$$
* **Empirical Benchmarks (Table 4 in Base Paper):**
  * *Org-Mix-4 (Max $K = 4$):* **MAE = 0.3243**, RMSE = 0.4302, Exact Accuracy = **74.75%**, Relaxed Accuracy ($\pm 1$) = **100.00%**.
  * *Org-Mix-9 (Max $K = 9$):* **MAE = 0.8766**, RMSE = 1.0925, Exact Accuracy = **36.46%**, Relaxed Accuracy ($\pm 1$) = **84.38%**.
  * *FSD-Mix-32 (Max $K = 32$):* **MAE = 6.7262**, RMSE = 8.0885, Exact Accuracy = **5.56%**, Relaxed Accuracy ($\pm 1$) = **14.44%**.

---

### 4.4 Task 2: Multi-Label Acoustic Tagging & Sound Event Detection
* **Objective:** Detect and classify active sound events across 32 environmental classes in complex polyphonic mixtures.
* **Model Class:** `ConformerTagger` (`models/tagging/conformer_tagger.py` & `sanada_inference.py`)
* **The Long-Tail Class Imbalance Problem:**
  In the 32 FSD classes, common sustained sounds (Flute, Saxophone) contain over 100 stems, while transient sounds (Cough, Squeak, Shatter) contain fewer than 20 stems. In any given $8.0\text{s}$ mixture, at most 2–4 classes are present ($y_c = 1$), while 28–30 classes are absent ($y_c = 0$), creating a severe **$> 90\%$ negative label skew**. Standard Binary Cross-Entropy (BCE) causes easy negatives to swamp gradient updates, leading the network to predict all zeros.
* **The Mathematical Solution: Asymmetric Loss (ASL):**
  Ben-Baruch et al. (ICCV 2021) Asymmetric Loss dynamically down-weights and truncates easy-negative loss gradients:
  $$\mathcal{L}_{\text{ASL}} = - \frac{1}{B} \sum_{b=1}^B \sum_{c=1}^C \left[ y_c (1 - p_c)^{\gamma_+} \log(p_c) + (1 - y_c) (p_{c,\text{clip}})^{\gamma_-} \log(1 - p_{c,\text{clip}}) \right]$$
  Where:
  * $p_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}}$ is the predicted sigmoid probability for class $c$.
  * $p_{c,\text{clip}} = \max(p_c - m, 0)$ is the probability shifted by hard threshold margin $m$.
  * Hyperparameters: $\gamma_- = 4.0$ (negative focusing), $\gamma_+ = 0.0$ (positive focusing), margin $m = 0.05$.
  * When $p_c < m$, $p_{c,\text{clip}} = 0$, completely zeroing out negative loss and gradients for easy background negatives.
* **The 32 Canonical FSD Classes:**
  1. `Acoustic_guitar`, 2. `Applause`, 3. `Bark`, 4. `Bus`, 5. `Cello`, 6. `Chime`, 7. `Clarinet`, 8. `Computer_keyboard`, 9. `Cough`, 10. `Drawer_open_or_close`, 11. `Electric_piano`, 12. `Fireworks`, 13. `Flute`, 14. `Glockenspiel`, 15. `Gong`, 16. `Gunshot_or_gunfire`, 17. `Harmonica`, 18. `Keys_jangling`, 19. `Laughter`, 20. `Meow`, 21. `Microwave_oven`, 22. `Oboe`, 23. `Saxophone`, 24. `Scissors`, 25. `Shatter`, 26. `Snare_drum`, 27. `Squeak`, 28. `Tearing`, 29. `Telephone`, 30. `Trumpet`, 31. `Violin_or_fiddle`, 32. `Writing`.
* **Empirical Benchmarks (Table 5 in Base Paper):**
  * *Org-Mix-4 (4 Speakers):* **mAP = 0.9902**, Macro F1 = **0.9449**, ROC-AUC = **0.9765**.
  * *Org-Mix-9 (9 Speakers):* **mAP = 0.9122**, Macro F1 = **0.8305**, ROC-AUC = **0.8744**.
  * *FSD-Mix (32 Classes):* **mAP = 0.6265**, Macro F1 = **0.6849**, ROC-AUC = **0.6161**.

---

### 4.5 Task 3: Monaural Speech Source Separation
* **Objective:** Reconstruct up to 4 isolated, clean speech waveforms $\hat{\mathbf{s}}_1(t), \dots, \hat{\mathbf{s}}_4(t)$ directly from a single-channel mixture $\mathbf{x}_{\text{mix}}(t)$.
* **Model Class:** `ConvTasNet` (`models/separation/conv_tasnet.py` & `sanada_inference.py`)
* **Time-Domain Architecture:**
  1. *1D Convolutional Encoder:*
     * Conv1d($1 \to N=512$, kernel $L=16$ [$1.0\text{ ms}$], stride $S=8$ [$0.5\text{ ms}$], bias=False) $\to$ GroupNorm($1, 512$) $\to$ ReLU.
     * Projects $N=128{,}000$ waveform into nonnegative 2D representation $\mathbf{W} \in \mathbb{R}^{512 \times 15999}$.
  2. *Bottleneck 1D Convolution:*
     * Conv1d($N=512 \to B=128$, kernel 1).
  3. *Temporal Convolutional Network (TCN) Separator:*
     * Arranged in $R = 3$ repeats of $X = 8$ dilated `DepthwiseSeparableBlock` modules ($H = 256$ hidden channels, skip channels $Sc = 128$).
     * Exponential dilations per repeat: $d \in \{2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7\} = \{1, 2, 4, 8, 16, 32, 64, 128\}$.
     * Total receptive field spans $> 1.53\text{ seconds}$ of audio.
  4. *Multi-Source Mask Generation:*
     * Conv1d($128 \to K_{\max} \times N = 4 \times 512 = 2048$, kernel 1) $\to$ Reshaped to $[B, 4, 512, 15999] \to$ Sigmoid activation.
     * Produces 4 nonnegative masks $\mathbf{M}_1, \dots, \mathbf{M}_4 \in [0, 1]$.
  5. *1D Transposed Convolutional Decoder:*
     * Computes modulated representations: $\mathbf{D}_k = \mathbf{W} \odot \mathbf{M}_k$.
     * ConvTranspose1d($512 \to 1$, kernel 16, stride 8) $\to$ Reconstructs time-domain signals $\hat{\mathbf{s}}_k \in \mathbb{R}^{B \times 128000}$.
* **Vectorized uPIT SI-SDR Loss:**
  Utterance-level Permutation Invariant Training evaluates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR):
  $$\text{SI-SDR}(\hat{\mathbf{s}}, \mathbf{s}) = 10 \log_{10} \left( \frac{\|\alpha \mathbf{s}\|^2}{\|\hat{\mathbf{s}} - \alpha \mathbf{s}\|^2 + \epsilon} \right), \quad \alpha = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2 + \epsilon}$$
  * Zero-mean temporal centering: $\hat{\mathbf{s}} \leftarrow \hat{\mathbf{s}} - \mu_{\hat{\mathbf{s}}}, \mathbf{s} \leftarrow \mathbf{s} - \mu_{\mathbf{s}}$ is strictly enforced.
  * Evaluates all $4! = 24$ permutations simultaneously via tensor broadcasting.
  * **AMP Autocast Guard:** PyTorch AMP autocast is explicitly disabled (`torch.cuda.amp.autocast(enabled=False)`) during the dot-product computation to prevent 16-bit float overflow ($> 65504$) on $N = 128{,}000$ samples.
* **Empirical Benchmarks (Table 6 in Base Paper):**
  * $K = 2$: **$+10.59\text{ dB}$** SI-SDR (High-fidelity isolation).
  * $K = 3$: **$+5.08\text{ dB}$** SI-SDR (Clean separation).
  * $K = 4$: **$+4.24\text{ dB}$** SI-SDR (Usable ceiling threshold).
  * $K = 6$: **$-3.40\text{ dB}$** SI-SDR (Catastrophic collapse below $0\text{ dB}$).
  * $K = 9$: **$-8.57\text{ dB}$** SI-SDR (Intractable).

---

## 5. Production Inference Engine (`sanada_inference.py`)

The file `sanada_inference.py` ($1320\text{ lines}$) is the core production inference engine bridging raw model checkpoints to web browsers and CLI utilities. It encapsulates all neural architectures, audio conditioning transforms, and three crucial algorithmic production innovations.

```
[Incoming Audio: Any sample rate / channels / length]
                        │
                        ▼
       [1. condition_audio() -> 16kHz Mono, 3 Adaptive Modes]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
[Silence Guard Check]           [Extract Log-Mel]
RMS < 1e-4 -> K=0               [B, 1, 128, 251]
(Short-circuit silence)                │
         │                             ├───────────────────────────┐
         │                             ▼                           ▼
         │                    [Task 1: Counter]           [Task 2: Tagger]
         │                    Continuous raw_k            Raw 32-dim logits
         │                             │                           │
         │                             ▼                           ▼
         │                    [Fix 1: Calibrated K]       [Fix 2: Namespacing]
         │                    predicted_k in [2, 9]       environmental_events:
         │                    ceiling_flag = (K > 4)      active (>=opt_thresh)
         │                             │                  top-5 events
         │                             ▼
         └────────────────────> [Task 3: Conv-TasNet]
                                Outputs 4 raw candidate stems
                                       │
                                       ▼
                                [Sliding-Window Overlap-Add]
                                (For audio clips > 8.0 seconds)
                                       │
                                       ▼
                                [Fix 3: LS Scale Restoration]
                                alpha = dot(s_zm, x_zm) / ||s_zm||^2
                                s_scaled = alpha * s_zm
                                       │
                                       ▼
                                [Energy Ranking & Pruning]
                                Sort stems descending by RMS
                                Select top k_sep = min(predicted_k, 4)
                                       │
                                       ▼
                                [Bleed & Peak Guards]
                                Bleed flag if < -20dB vs stem #1
                                Peak clamp: s = 0.99 * s / max(|s|)
                                       │
                                       ▼
                                [Export 16-bit PCM WAV Stems]
```

### 5.1 Adaptive 3-Mode Audio Conditioning (`condition_audio()`)
Audio arriving from real users varies wildly in sample rate, bit depth, channel configuration, and noise levels. The function `condition_audio()` provides 3 distinct DSP conditioning modes grounded in the SANADA Paper:

1. **Mode A: `'sanada'` (Bit-Exact Benchmark Preservation):**
   * *Purpose:* Strictly preserves the synthetic dynamics of official SANADA dataset files.
   * *Transforms:* Zero gain modifications, zero filtering, zero DC cut. Padded or cropped exactly to $N = 128{,}000$ samples. Preserves bit-exact benchmark metrics.
2. **Mode B: `'computer'` (Recorded System Audio & Uploaded Media Files):**
   * *Purpose:* Handles general desktop audio, browser uploads, YouTube rips, and podcasts.
   * *Transforms:*
     * High-quality Kaiser-windowed sinc resampling (`sinc_interp_kaiser`) to $16{,}000\text{ Hz}$.
     * Multi-channel averaging to mono.
     * DC offset elimination: $x(t) \leftarrow x(t) - \mu_x$.
     * Leading/trailing digital dead silence trimming ($< -55\text{ dBFS}$).
     * Active RMS measurement across $50\text{ ms}$ windows.
     * Soft Table-3 Target RMS Normalization: If active RMS falls outside $[0.06, 0.25]$, applies linear gain scaling to match target $\text{RMS} = 0.165$.
     * Soft peak limiting at $0.99$.
3. **Mode C: `'live_mic'` (Real-Time Microphone Ingestion):**
   * *Purpose:* Handles raw microphone input captured via WebRTC or desktop audio devices.
   * *Transforms:*
     * 2nd-order High-Pass Biquad Filter at $60\text{ Hz}$ to eliminate electrical AC power hum ($50/60\text{ Hz}$) and mechanical desk rumbles.
     * Adaptive Noise Floor VAD Gate: Computes the 10th percentile of $50\text{ ms}$ frame RMS values. If active RMS $< 2.2 \times \text{noise\_floor}$, gates audio to silence.
     * Loudness Matching: Boosts quiet speech up to target $\text{RMS} = 0.165$ with an aggressive gain ceiling capped at $+28\text{ dBFS}$ ($25.1\times$ max amplification).
     * Soft peak limiting at $0.99$.
4. **Mode `'auto'`: Acoustic Fingerprinting:**
   * Dynamically inspects input channels, sampling rate, sample count, RMS energy, and filenames. Automatically selects `'sanada'` if the file matches $16\text{ kHz}$ mono $128{,}000 \pm 160$ samples and $\text{RMS} \in [0.035, 0.28]$; selects `'live_mic'` if microphone hints are detected; falls back to `'computer'`.

---

### 5.2 Fix 1: Calibrated Source Counting + Silence Guard
* **The Nuance:** The Conformer Counter was trained on variable mixtures $K \in [2, 9]$. Passing silent audio or quiet background noise causes continuous regression to predict spurious float values (e.g. $K \approx 1.2$ or $K \approx 2.1$).
* **Production Implementation:**
  1. *Silence Guard:* Computes waveform Root-Mean-Square energy: $\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^N x[n]^2}$.
     $$\text{if } \text{RMS} < 10^{-4} \; (-80\text{ dBFS}) \implies \hat{K} = 0, \; \text{raw\_k} = 0.0, \; \text{is\_silent} = \text{True}$$
  2. *Two-Regime Piecewise Calibration:*
     $$\text{calibrated\_k} = \begin{cases} 
     \text{raw\_k} \cdot 0.92 + 0.15 & \text{if } \text{raw\_k} \le 4.5 \\
     \text{raw\_k} \cdot 1.05 - 0.40 & \text{if } \text{raw\_k} > 4.5 
     \end{cases}$$
  3. *Single-Speaker Guard:* If calibrated $K < 1.45$ and spectral centroid $< 1800\text{ Hz}$, clamps prediction to $K = 1$.
  4. *Discrete Clamping:* $\hat{K}_{\text{discrete}} = \text{int}(\text{clip}(\text{round}(\text{calibrated\_k}), 2, 9))$.
  5. *Complexity Ceiling Flag:* Emits boolean alert flag:
     $$\text{complexity\_ceiling\_exceeded} = \text{bool}(\hat{K}_{\text{discrete}} > 4)$$
     Informing downstream clients that source separation will suffer severe acoustic masking.

---

### 5.3 Fix 2: Explicit Domain Namespacing for Environmental Tagging
* **The Nuance:** Naive integrations often expose raw class indices or risk displaying sound event tags (e.g. `Bark`, `Applause`) as speaker identifiers.
* **Production Implementation:**
  1. Dynamic ontology extraction from checkpoint metadata (`tag_ckpt["classes"]`).
  2. Optimal Per-Class Decision Thresholds: Rather than a naive global $0.5$ threshold, `sanada_inference.py` loads calibrated optimal F1 thresholds per class (e.g., `Bark: 0.35`, `Flute: 0.50`, `Cough: 0.28`).
  3. Percussive Event Guard: Applies transient margin offsets for short-duration percussive classes (`Glockenspiel`, `Snare_drum`, `Keys_jangling`, `Scissors`, `Shatter`).
  4. Explicit Namespacing: Outputs results strictly partitioned under:
     * `active_events`: List of detected classes with probability $\ge \text{threshold}$, sorted descending.
     * `top_k_events`: Top 5 highest-probability classes with confidence scores.
     * Never mixes speaker counting IDs with acoustic environmental tags.

---

### 5.4 Fix 3: Counter-Guided Separation with SI-SDR Scale Restoration
* **The Nuance:** Scale-Invariant SDR (SI-SDR) is mathematically invariant to scalar amplification ($\text{SI-SDR}(\alpha \hat{s}, s) = \text{SI-SDR}(\hat{s}, s)$). During training, Conv-TasNet ignores absolute volume, outputting candidate stems with uncalibrated, arbitrary gain. Furthermore, Conv-TasNet always outputs 4 channels: if $K=2$, two channels contain active speech and two contain quiet residual noise ("ghost channels").
* **Production Implementation:**
  1. *Sliding-Window Overlap-Add for Long Audio:* For audio files exceeding $8.0\text{ seconds}$, chunks audio into $8.0\text{s}$ windows with a $4.0\text{s}$ hop (50% overlap), applies a Hanning crossfade window, and reconstructs seamless arbitrary-length stems.
  2. *Dynamic Cardinality Routing:* Determines active stem count dynamically from the Conformer Counter:
     $$k_{\text{sep}} = \min(\hat{K}_{\text{discrete}}, 4)$$
  3. *Least-Squares Scale Restoration Factor:* For each candidate stem $\mathbf{s}_i$, projects the zero-mean stem onto the zero-mean mixture $\mathbf{x}_{\text{zm}}$:
     $$\alpha_i = \frac{\langle \mathbf{s}_{i,\text{zm}}, \mathbf{x}_{\text{zm}} \rangle}{\|\mathbf{s}_{i,\text{zm}}\|^2 + 10^{-8}}, \quad \mathbf{s}_{i,\text{scaled}} = \alpha_i \cdot \mathbf{s}_{i,\text{zm}}$$
     Restoring true physical amplitude and sound pressure level (SPL).
  4. *Energy Ranking:* Calculates scale-restored RMS energy: $\text{RMS}_i = \sqrt{\frac{1}{N} \sum (\mathbf{s}_{i,\text{scaled}}[n])^2}$. Sorts stems descending by energy and keeps the top $k_{\text{sep}}$ channels.
  5. *Relative Energy Bleed Floor:* Computes relative decibel level against the primary dominant stem:
     $$\Delta\text{dB}_j = 20 \log_{10} \left( \frac{\text{RMS}_j}{\text{RMS}_1 + 10^{-8}} \right)$$
     If $\Delta\text{dB}_j < -20.0\text{ dB}$, flags the stem with `is_bleed_candidate = True` (residual leakage channel).
  6. *Peak Guard Clamping:* Prevents digital clipping on exported WAV files:
     $$\text{if } \max |s[n]| > 0.99 \implies s[n] = 0.99 \cdot \frac{s[n]}{\max |s[n]|}$$
     Serializes stems as standard 16-bit PCM WAV files.

---

## 6. Backend REST API Architecture (`backend/app.py` & `dev_server.py`)

The application backend is built with **FastAPI** (Python 3.12) and executed via **Uvicorn**, providing asynchronous non-blocking I/O and strict Pydantic schema validation.

### 6.1 Complete REST API Specification Table

| HTTP Method | Endpoint Path | Description | Request Payload / Parameters | Success Status & Response Format |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | API vitality & service metadata | None | `200 OK`: JSON containing service status, version (`1.0.0`), docs URL, endpoint discovery dict. |
| `GET` | `/api/health` | Hardware telemetry & memory monitor | None | `200 OK`: System uptime, RAM RSS (MB), CUDA device status, active training job count, loaded models. |
| `GET` | `/api/stats` | Dataset telemetry & complexity ceiling | None | `200 OK`: 15,559 mixture count, total hours (34.57h), SI-SDR benchmarks, MAE table, PIT factorial combinations. |
| `POST` | `/api/upload` | Fast in-memory WAV verification | Multipart form: `file` (.wav binary) | `200 OK`: `sample_rate_hz`, `channels`, `frames`, `duration_s`, `standard_compliant` (bool). |
| `POST` | `/api/predict` | Multi-task machine listening inference | Form: `file` (.wav), `task` (`all`/`counting`/`tagging`/`separating`), `threshold` (0.5), `mode` (`auto`/`sanada`/`computer`/`live_mic`), `all_4_stems` (bool) | `200 OK`: Full inference JSON: `predicted_k`, `raw_k`, `calibrated_k`, `complexity_ceiling_exceeded`, `active_events`, `top_5_events`, `separated_stems`. |
| `POST` | `/api/predict/live` | Dedicated live microphone inference | Form: `file` (.wav), `threshold` (0.5), `all_4_stems` (bool) | `200 OK`: Runs multi-task inference with `mode="live_mic"` (60Hz biquad filter, VAD gate, loudness match). |
| `GET` | `/api/demo-mixtures` | Catalog of verified evaluation audio | None | `200 OK`: Array of 7 demo mixtures (metadata, sound counts, labels, descriptions, static streaming URLs). |
| `GET` | `/api/demo-audio/{filename}` | Stream demo audio file | Path parameter: `{filename}` | `200 OK`: Binary 16-bit PCM audio stream (`audio/wav`). |
| `POST` | `/api/train` | Schedule background training run | JSON: `task`, `dataset_name`, `learning_rate`, `batch_size`, `epochs`, `optimizer`, `use_specaugment` | `202 Accepted`: Asynchronous job ticket with generated `job_id`, status (`queued`), parameter echo. |
| `GET` | `/api/train/{job_id}` | Poll background training job progress | Path parameter: `{job_id}` | `200 OK`: Current job progress: `status` (`running`/`completed`), `current_epoch`, `progress_pct`, `final_loss`. |
| `POST` | `/api/save-model` | Serialize & export model checkpoint | JSON: `task`, `version_tag`, `notes` | `200 OK`: `export_filename`, download URL (`/api/checkpoints/...`), metadata manifest. |
| `GET` | `/static/stems/{filename}` | Stream isolated separated stems | Path parameter: `{filename}` | `200 OK`: Binary 16-bit PCM audio stream (`audio/wav`) for frontend stem player. |

### 6.2 Unified Development Server & Reverse Proxy (`dev_server.py`)
To eliminate Cross-Origin Resource Sharing (CORS) friction and enable unified deployment:
* Runs on port `3000`.
* Serves static assets directly from `frontend/` (HTML, CSS, JS, video).
* Acts as an asynchronous reverse proxy: intercepts all `/api/*` and `/static/stems/*` requests and forwards them transparently to the FastAPI backend running on port `8000`.
* **Cloudflare Tunnel Integration:** Integrates with `cloudflared` to expose a public secure URL (`https://*.trycloudflare.com`) on single-port execution, allowing external evaluators to test the system live without router port forwarding.

---

## 7. Frontend Architecture & User Experience (`frontend/`)

The frontend is an ultra-fast, zero-dependency **Single Page Application (SPA)** written in modern **Vanilla ES6+ JavaScript**. It requires zero Node.js build tools (no Webpack, Vite, or Babel), eliminating deployment friction and allowing instant execution in any modern browser.

### 7.1 Design System & CSS Token Hierarchy
* **Color Palette & Glassmorphism:** Defined in `css/tokens.css` with dark theme styling:
  * Primary Background: `--bg-deep: #0a0b0e`
  * Glass Surface: `--surface-glass: rgba(18, 20, 26, 0.72)` with `backdrop-filter: blur(16px)`
  * Accent Electric Blue: `--accent-blue: #3b82f6`
  * Accent Purple / Magenta: `--accent-purple: #8b5cf6`
  * Success Emerald: `--accent-emerald: #10b981`
  * Warning Amber: `--accent-amber: #f59e0b`
  * Danger Coral: `--accent-rose: #f43f5e`
* **Typography Hierarchy:**
  * Headings: `Instrument Serif` (editorial, scientific elegance)
  * UI Text & Body: `Barlow` (clean, geometric legibility)
  * Numbers, Metrics, Code & Tensors: `Geist Mono` (fixed-width tabular precision)

### 7.2 The 7 Application Views

```
+-----------------------------------------------------------------------------------------+
|                                NAVIGATION BAR (nav-tabs)                                |
|  [Home]  [Studio]  [Dataset Pipeline]  [Analytics]  [Developer Console]  [Docs]  [Paper]|
+-----------------------------------------------------------------------------------------+
```

1. **`pageHome` (Landing & Problem Formulation):**
   * Pinned ambient video backdrop (`assets/wave_bg.mp4`) with smooth playback rate regulation (`0.80x`).
   * Hero header explaining SANADA's mission.
   * Problem statement comparison: "Sterile 2-Speaker Lab Speech" vs "High-Density Real-World Polyphony".
   * Telemetry counter badges: 15,559 Mixtures, 27,112 Stems, 34.61 Hours, 32 Environmental Classes.
2. **`pagePredict` (Machine Listening Studio):**
   * *Audio Ingestion Zone:* Drag-and-drop file uploader accepting linear 16-bit PCM WAV.
   * *Demo Mixture Carousel:* 7 quick-load evaluation audio files (`fsd_mix_3_sounds.wav`, `org_mix_4_speakers.wav`, `org_mix_9_speakers.wav`).
   * *Dual Canvas DSP Visualizer:* Real-time waveform oscillogram and frequency spectrogram powered by Web Audio API `AnalyserNode` rendering at 60 FPS.
   * *Conditioning Mode Selector:* Toggle between `auto`, `sanada`, `computer`, and `live_mic`.
   * *Live Microphone Streaming:* One-touch recording capturing microphone audio directly through the Web Audio API.
   * *Multi-Task Results Presentation:*
     * **Hero Counting Card:** Large animated numeric badge displaying integer $K$, continuous raw $\hat{K}$, calibrated confidence percentage, and complexity ceiling warning badge if $K > 4$.
     * **Acoustic Tagging Grid:** Dynamic pills for active environmental sound categories with confidence percentage progress bars.
     * **Multi-Track Audio Stem Matrix:** 4 dedicated channel strips for separated voices with independent Play/Pause, Solo, Mute, Volume faders, interactive seek bars, and direct WAV stem download links.
3. **`pageDataset` (Dataset Specifications & Signal Physics):**
   * Interactive sub-dataset cards detailing all 13 corpora.
   * Anti-Clipping Superposition Theorem mathematical proof and formula breakdown.
   * Pre-Mix Zero-Leakage Protocol flowchart.
4. **`pageAnalytics` (Benchmark Results & Complexity Ceiling):**
   * Interactive charts visualizing SI-SDR collapse from $K=2$ ($+10.59\text{ dB}$) to $K=9$ ($-8.57\text{ dB}$).
   * Factorial Permutation Wall calculator comparing $K!$ combinations ($1.307\text{ Trillion}$ at $K=15$).
   * Zero-padding degradation comparison table.
   * Counting MAE and tagging mAP performance tables.
5. **`pageTrain` (Developer Training & Checkpoint Console):**
   * Hyperparameter configuration form: task selection (`counting`, `tagging`, `separation`), dataset partition, learning rate, batch size, epochs, optimizer (`AdamW`/`Adam`/`SGD`), SpecAugment toggle.
   * Asynchronous training dispatcher (`POST /api/train`).
   * Real-time polling progress bar tracking epoch-by-epoch loss convergence (`GET /api/train/{id}`).
   * Model checkpoint serialization and export trigger (`POST /api/save-model`).
6. **`pageDocs` (Architectural Documentation & Research Manual):**
   * Comprehensive interactive documentation detailing 3-tier system design, API contracts, tensor shapes, and DSP specs.
7. **`pagePaper` (Research Publication & Citation Hub):**
   * Official citation metadata, IEEE DataPort DOI link, SSRN preprint links, abstract, and one-click BibTeX copy button.

### 7.3 Client Audio Engine & Multi-Track Mixer
* `audioEngine.js`: Initializes browser `AudioContext`, decodes binary WAV byte arrays into floating-point audio buffers, connects to `AnalyserNode` for Fast Fourier Transforms (FFT).
* `signalVisualizer.js`: Canvas renderer operating on `requestAnimationFrame`, plotting real-time time-domain waveforms and frequency bars.
* `stemPlayer.js`: Orchestrates simultaneous multi-track stem playback using synchronized `AudioBufferSourceNode` instances piped through dedicated `GainNode` faders for seamless solo/mute switching without phase drift.

---

## 8. Scripts, Data Engineering & Verification Tooling (`scripts/`)

The repository contains an exhaustive suite of standalone Python automation, audit, and verification utilities:

1. `scripts/audit_audio_integrity.py`:
   * Recursively scans dataset directories and audits every WAV file.
   * Validates sampling rate ($16{,}000\text{ Hz}$), channels ($1\text{ mono}$), bit depth ($16\text{-bit Linear PCM}$), duration ($8.0\text{s} \pm 10\text{ms}$), and peak amplitude ($< 0\text{ dBFS}$).
2. `scripts/compile_dataset_stats.py`:
   * Iterates through all 13 sub-corpora.
   * Computes WADA-SNR (Waveform Amplitude Distribution Analysis Signal-to-Noise Ratio) and $RT_{60}$ room reverberation times.
   * Generates `docs/dataset_statistics_report.md` and `docs/dataset_stats_report.md`.
3. `scripts/dataset_loader.py`:
   * High-throughput PyTorch `Dataset` and `DataLoader` implementation.
   * Features multi-threaded pre-fetching (`num_workers=4`, `pin_memory=True`), achieving throughput $> 16\text{ batches/second}$.
4. `scripts/download_checkpoints.py`:
   * Manages model weight caching in `models/checkpoints/`.
   * Verifies SHA-256 checksums, downloads missing weights from cloud storage, and ensures valid state dictionaries.
5. `scripts/manifest_loader.py`:
   * Centralized manifest loader providing constants: `FSD_32_CLASSES`, `ORG_4_SPEAKERS`, `ORG_9_SPEAKERS`, and CSV parsing utilities.
6. `scripts/smoke_test_inference.py`:
   * Comprehensive 6-stage verification script testing `sanada_inference.py`:
     * Test 1: Checkpoint Schema & `strict=True` state_dict loading.
     * Test 2: Audio Standardization & Silence Guard.
     * Test 3: Calibrated Source Counting & Complexity Flag.
     * Test 4: Environmental Tagging Namespacing.
     * Test 5: Counter-Guided Separation & LS Scale Restoration.
     * Test 6: Unified `analyze_and_separate()` workflow.
7. `scripts/test_audio_modes.py`:
   * Automated verification test suite for all three audio conditioning modes: Mode A (`sanada`), Mode B (`computer`), and Mode C (`live_mic`).
8. `scripts/test_week4_fullstack.py`:
   * Integration test verifying all multi-tier REST API endpoints.
9. `scripts/verify_week1_member1.py` through `verify_week4_member1.py`:
   * Weekly unit and contract verification scripts ensuring frontend and backend remain perfectly synchronized across milestones.
10. `scripts/verify_zero_leakage.py` & `zero_leakage_audit.py`:
    * Enforces the mathematical set disjointness proof across train, validation, and test stem manifests.
    * Computes Pearson correlation coefficient between synthesized mixtures and mathematical superposition.

---

## 9. Research Progression & Experimental Archive (`src/`)

The `src/` directory preserves the developmental research history and optimization progression conducted across project milestones:

* `src/member3_week1_ontology_and_baseline.py`:
  * Establishes the 32-class FSD50K ontology mapping, multi-hot label encoders, and initial baseline classifier.
* `src/member3_week2_conformer_asl.py`:
  * Implements the Macaron-Style Conformer backbone coupled with Asymmetric Loss (ASL) to conquer long-tail class imbalance.
* `src/member3_week3_ablation_and_onnx.py`:
  * Hyperparameter grid search and ablation experiments on ASL focusing parameters ($\gamma_- \in [1, 5]$, $\gamma_+ \in [0, 2]$, $m \in [0.0, 0.2]$), identifying optimal configuration ($\gamma_- = 4.0, \gamma_+ = 0.0, m = 0.05$).
  * Implements ONNX export utilities for browser-side WebAssembly inference.
* `src/member3_week4_quantization_and_trimodel_sync.py`:
  * Evaluates post-training dynamic quantization (INT8), reducing model checkpoint sizes by $68\%$ with $< 0.8\%$ mAP degradation.
  * Synchronizes input/output tensor contracts across all three neural models.
* `src/member3_week5_qa_and_loadtest.py`:
  * Automated stress testing and concurrency profiling, verifying latency under concurrent HTTP request loads.
* `src/week2_conformer_pipeline.py`:
  * End-to-end pipeline utility for training Conformer models with learning rate warm-up and cosine decay schedulers.

---

## 10. Automated Testing Suites & Quality Assurance Matrix (`tests/`)

The codebase enforces strict quality assurance verified by four independent automated test suites:

```
[Test Suite 1: Full-Stack REST API (tests/test_api_endpoints.py)]
├── GET  /               [PASS] 200 OK - Route discovery verified
├── GET  /api/health     [PASS] 200 OK - RAM & device telemetry verified
├── GET  /api/stats      [PASS] 200 OK - 15,559 clip telemetry verified
├── POST /api/upload     [PASS] 200 OK - In-memory WAV verification verified
├── POST /api/predict    [PASS] 200 OK - Joint multi-task inference verified
├── POST /api/train      [PASS] 202 Accepted - Async training dispatch verified
├── GET  /api/train/{id} [PASS] 200 OK - Job progress polling verified
├── POST /api/save-model [PASS] 200 OK - Checkpoint serialization verified
└── GET  /static/stems/  [PASS] 200 OK - Binary audio stream delivery verified
                         RESULT: 9 / 9 ENDPOINTS PASS (100% COVERAGE)

[Test Suite 2: Neural Inference Smoke Test (scripts/smoke_test_inference.py)]
├── Test 1: Checkpoint Schema & strict=True state_dict Loading  [PASS]
├── Test 2: Audio Standardization & Silence Guard (Fix 1)       [PASS]
├── Test 3: Calibrated Source Counting & Ceiling Flag (Fix 1)   [PASS]
├── Test 4: Environmental Tagging Namespacing (Fix 2)           [PASS]
├── Test 5: Counter-Guided Separation & LS Restoration (Fix 3)   [PASS]
└── Test 6: Unified analyze_and_separate() Workflow             [PASS]
                         RESULT: 6 / 6 CHECKS PASS (100% COVERAGE)

[Test Suite 3: Domain Pipeline Unit Tests (tests/test_*.py)]
├── test_counting_pipeline.py   [PASS] 4 / 4 Unit Tests Pass
├── test_tagging_pipeline.py    [PASS] 5 / 5 Unit Tests Pass
└── test_separation_pipeline.py [PASS] 5 / 5 Unit Tests Pass
                         RESULT: 14 / 14 UNIT TESTS PASS

[Test Suite 4: Data Integrity & Zero-Leakage (zero_leakage_audit.py)]
├── Check 1: Pre-mix train/val/test stem disjointness (S=42)    [PASS]
└── Check 2: Mathematical Superposition Pearson Corr (> 0.99)   [PASS] (corr = 0.9998)
```

---

## 11. Deployment, DevOps & Containerization

### 11.1 Containerization (`Dockerfile` & `docker-compose.yml`)
* **`Dockerfile` Architecture:**
  * Base Image: `python:3.12-slim-bookworm`
  * System Dependencies: `ffmpeg`, `libsndfile1`, `curl`, `build-essential`
  * Multi-stage caching: Layer 1 installs PyTorch CPU/CUDA wheels; Layer 2 installs FastAPI and scientific dependencies from `requirements.txt`.
  * Pre-bakes model checkpoints directly into container filesystem to eliminate runtime download delays.
  * Runs on non-root user for enterprise container security.
* **`docker-compose.yml` Architecture:**
  * Defines multi-container network binding FastAPI backend (`port 8000`) and dev server proxy (`port 3000`).
  * Maps persistent volumes for checkpoint exports and logs.

### 11.2 Multi-Target Hosting Topology
* **Cloud Backend:** Deployed to **Hugging Face Spaces** (`eshansk-sanada-backend.hf.space`) using Docker containerization with automated PyTorch CPU/CUDA runtime detection.
* **Cloud Frontend:** Deployable to **Vercel** or **Netlify** via `frontend/vercel.json` with automatic proxying to the backend API.
* **PaaS Alternative:** Ready for **Render** via `render.yaml`.
* **Local One-Touch Launch:**
  ```bash
  chmod +x start_system.sh
  ./start_system.sh
  ```
  Launches `dev_server.py` on port 3000, boots `backend/app.py` on port 8000, and verifies route health automatically.

---

## 12. Model Training & Notebook Workflow (`SANADA_Colab_Training.ipynb`)

The training pipeline is fully encapsulated inside `SANADA_Colab_Training.ipynb` for execution on Google Colab NVIDIA Tesla T4 / A100 GPUs:

### 12.1 Balanced Curriculum Learning for Source Counting
* **The Imbalance Challenge:** In variable-density datasets, high counts ($K \ge 5$) have fewer mixtures than $K=2, 3, 4$, while single-speaker ($K=1$) samples are absent from polyphonic mixture sets.
* **Curriculum Solution in Cell 3:**
  1. Loads full variable-K sets ($K \in [2, 4]$ and $K \in [2, 9]$) totaling $2{,}000$ training mixtures.
  2. Subsamples fixed-K datasets ($K \in \{2, 3, 4, 6, 9\}$) at 25% (250 clips each) to preserve density balance.
  3. Integrates $250$ isolated single-speaker stems ($K = 1.0$) from `Multichannel4SpeakersTrim` to anchor the single-speaker boundary.

### 12.2 Training Optimizations & Checkpoint Schemas
* **Learning Rate Scheduling:** AdamW optimizer with cosine annealing learning rate scheduler and 5-epoch linear warm-up.
* **AMP Autocast Protocol:** Automated Mixed Precision (`torch.cuda.amp.GradScaler`) accelerates training while explicitly disabling autocast inside the uPIT SI-SDR loss calculation to avoid float16 dot-product overflow.
* **Serialized Checkpoint Contract:**
  Checkpoints are serialized with exact dictionary schemas loaded via `strict=True`:
  * Counting: `{"state_dict": ..., "model_type": "ConformerCounter", "d_model": 256, "sample_rate": 16000}`
  * Tagging: `{"state_dict": ..., "classes": FSD_32_CLASSES, "optimal_thresholds": {...}, "model_type": "ConformerTagger"}`
  * Separation: `{"state_dict": ..., "model_type": "ConvTasNet", "n_sources": 4, "sample_rate": 16000}`
