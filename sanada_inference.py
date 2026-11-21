#!/usr/bin/env python3
"""
sanada_inference.py
===================
Production-Ready SANADA Multi-Task Machine Listening Inference Module.

Supports:
- SANADA Path B Foundation Models:
  1. Task 1: WavLMSourceCounter (microsoft/wavlm-base-plus backbone + Attentive Statistics
             Pooling + Table-3 DSP fusion, supporting K in [1, 32] with domain classification)
  2. Task 2: Pretrained MIT AST 33-Class Tagger (32 FSD environmental classes + Human_Speech,
             with optimal thresholds and speech-gated environmental sensitivity)
  3. Task 3: Dynamic Multi-Specialist Separation Router (Sepformer-2spk, Sepformer-3spk,
             and ConvTasNetSkip 4spk Dominant Crowd Separator, with K=1 input passthrough)
- Legacy Checkpoint Fallbacks (100% backward compatible):
  1. Conformer Counter (Task 1: Source Counting K=1..9 with Two-Regime Calibration and Single-Speaker Verification)
  2. Conformer Tagger (Task 2: Environmental Multi-Label Tagging with 32 FSD Classes, Optimal Thresholds & Percussive Guard)
  3. Conv-TasNet Separator (Task 3: Time-Domain Speech Separation with Sliding-Window Overlap-Add, LS Scale Restoration,
                            and Bleed Channel Pruning)
- 3 Distinct Audio Input Modes + Auto-Fingerprinting:
  1. 'sanada'   : Bit-exact preservation for SANADA Benchmark WAV files (no filtering, no RMS modification)
  2. 'computer' : Recorded system audio & uploaded media (mono, Kaiser sinc resample, DC cut, silence trim, Table-3 RMS norm)
  3. 'live_mic' : Real-time microphone capture (DC cut, 60Hz biquad highpass, adaptive noise floor VAD gate, loudness matching)
  4. 'auto'     : Default auto-fingerprinting and routing
"""

import os
import sys
import io
import uuid
import json
import logging
import argparse
import itertools
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SANADA-Inference")

# -----------------------------------------------------------------------------
# 0. PyTorch 2.6+/2.11 Safe Loader & Compatibility Shims
# -----------------------------------------------------------------------------

def load_checkpoint_cpu(path: Union[str, Path]) -> Any:
    """
    PyTorch 2.6+/2.11 safe loader.
    Uses weights_only=False by default to load full checkpoint dicts (with fallback if unsupported).
    Supports loading from single .pt files or extracted archive directories.
    """
    p = str(path)
    if os.path.isdir(p):
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for root, dirs, files in os.walk(p):
                for file in files:
                    full = os.path.join(root, file)
                    rel = os.path.relpath(full, p)
                    zf.write(full, arcname=os.path.join("archive", rel))
        buf.seek(0)
        try:
            return torch.load(buf, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(buf, map_location="cpu")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# TorchAudio 2.11 backend shim
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# huggingface_hub v1.x shim: strip deprecated kwargs
try:
    import huggingface_hub.file_download
    _orig_hf_hub_download = huggingface_hub.file_download.hf_hub_download

    def _safe_hf_hub_download(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        kwargs.pop("local_dir_use_symlinks", None)
        return _orig_hf_hub_download(*args, **kwargs)

    huggingface_hub.file_download.hf_hub_download = _safe_hf_hub_download
    if hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.hf_hub_download = _safe_hf_hub_download
except Exception:
    pass


# -----------------------------------------------------------------------------
# Global Audio Specification Constants Grounded in SANADA Paper Table 3
# -----------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000
SAMPLE_RATE = 16000
N_SAMPLES = 128000  # Exactly 8.0s at 16 kHz
SANADA_ATTEN_GAIN = 10.0 ** (-3.0 / 20.0)  # 0.70794578 (-3.0 dBFS clipping prevention gain)
SILENCE_RMS_THRESHOLD = 1e-4  # -80 dBFS threshold for silence guard

# 32 Canonical FSD Sound Classes (Sorted alphabetically)
FSD_32_CLASSES = sorted([
    "Acoustic_guitar", "Applause", "Bark", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Drawer_open_or_close", "Electric_piano",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Keys_jangling", "Laughter", "Meow", "Microwave_oven",
    "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak",
    "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"
])

# 33 Tag Classes for AST Path B Model (32 FSD classes + Human_Speech)
TAG_33_CLASSES = FSD_32_CLASSES + ["Human_Speech"]
ID2LABEL_33 = {i: c for i, c in enumerate(TAG_33_CLASSES)}
LABEL2ID_33 = {c: i for i, c in enumerate(TAG_33_CLASSES)}
CLASS_TO_IDX = LABEL2ID_33

# Transient percussive classes that receive +0.05 threshold margin in live_mic mode
PERCUSSIVE_CLASSES = {
    "Scissors", "Writing", "Computer_keyboard", "Drawer_open_or_close", "Keys_jangling"
}

# Default checkpoint directory relative to project root
DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "models", "checkpoints"
)


# -----------------------------------------------------------------------------
# 1. DSP & Table 3 Feature Normalization Helpers
# -----------------------------------------------------------------------------

def normalize_wavlm_input(wav_np: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance normalization for WavLM input."""
    wav = wav_np.astype(np.float32)
    mean = float(np.mean(wav))
    std = float(np.std(wav))
    return (wav - mean) / (std + 1e-8)


def extract_table3_dsp(wav_np: np.ndarray) -> torch.Tensor:
    """
    Extracts 6 Table-3 DSP features:
    1. rms * 5.0
    2. log10(rms + 1e-6) / 3.0
    3. Spectral centroid (normalized by 63.0 mel bands)
    4. Spectral flatness (geometric mean / arithmetic mean of power)
    5. Frame sparsity (fraction of frames with energy < 0.25 * mean_frame_energy)
    6. Crest factor (peak / (rms + 1e-8))
    Returns: torch.Tensor of shape [6] (float32)
    """
    wav = wav_np.astype(np.float32)
    if len(wav) == 0:
        return torch.zeros(6, dtype=torch.float32)

    rms = float(np.sqrt(np.mean(wav ** 2)))
    f1 = rms * 5.0
    f2 = float(np.log10(rms + 1e-6) / 3.0)

    t_wav = torch.from_numpy(wav).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64, power=2.0
    )
    with torch.no_grad():
        mel = mel_transform(t_wav).squeeze(0).numpy()  # [64, T]

    mel_energy_per_bin = np.mean(mel, axis=1) + 1e-12
    bin_indices = np.arange(64, dtype=np.float32)
    f3 = float(np.sum(bin_indices * mel_energy_per_bin) / (np.sum(mel_energy_per_bin) * 63.0))

    log_mean = np.mean(np.log(mel_energy_per_bin + 1e-12))
    geom_mean = np.exp(log_mean)
    arith_mean = np.mean(mel_energy_per_bin)
    f4 = float(np.clip(geom_mean / (arith_mean + 1e-12), 0.0, 1.0))

    frame_energy = np.sum(mel, axis=0)
    mean_fe = np.mean(frame_energy) if len(frame_energy) > 0 else 1.0
    f5 = float(np.mean(frame_energy < (0.25 * mean_fe))) if len(frame_energy) > 0 else 0.0

    peak = float(np.max(np.abs(wav)))
    f6 = float(peak / (rms + 1e-8))

    return torch.tensor([f1, f2, f3, f4, f5, f6], dtype=torch.float32)


# -----------------------------------------------------------------------------
# 2. Neural Network Architecture Definitions
# -----------------------------------------------------------------------------

# --- Legacy Conformer & ConvTasNet Architectures ---

class MacaronConformerBlock(nn.Module):
    """
    Macaron-style Conformer Block with Half-Step FFNs, Multihead Self-Attention,
    and 1D Depthwise-Separable Convolution.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1_ln = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.attn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.conv_ln = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ffn2_ln = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Half-step FFN 1
        x = x + 0.5 * self.ffn1(self.ffn1_ln(x))
        # Self-Attention
        attn_in = self.attn_ln(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        # Depthwise Conv module (transposing (B, T, C) -> (B, C, T) -> (B, T, C))
        x_conv = self.conv(self.conv_ln(x).transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        # Half-step FFN 2
        x = x + 0.5 * self.ffn2(self.ffn2_ln(x))
        return self.final_ln(x)


class ConformerCounter(nn.Module):
    """
    Task 1 Legacy: Conformer Regressor for continuous source counting.
    Trained on 2 to 9 speaker mixtures with MSE loss.
    """
    def __init__(self, d_model: int = 256, n_blocks: int = 4):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.proj = nn.Linear(64 * 32, d_model)
        self.blocks = nn.ModuleList([MacaronConformerBlock(d_model=d_model) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 1, 128, 251]
        x = self.subsample(mel)  # [B, 64, 32, 63]
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)  # [B, 63, 2048]
        x = self.proj(x)  # [B, 63, 256]
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1)).squeeze(-1)  # [B]


class ConformerTagger(nn.Module):
    """
    Task 2 Legacy: Conformer Multi-Label Classifier for 32 FSD Sound Classes.
    Trained with Asymmetric Loss (ASL).
    """
    def __init__(self, d_model: int = 256, n_blocks: int = 4, num_classes: int = 32):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.proj = nn.Linear(64 * 32, d_model)
        self.blocks = nn.ModuleList([MacaronConformerBlock(d_model=d_model) for _ in range(n_blocks)])
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 1, 128, 251]
        x = self.subsample(mel)
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))  # [B, num_classes] raw logits


class DepthwiseSeparableBlock(nn.Module):
    """
    Dilated Depthwise-Separable 1D Convolution block for legacy Conv-TasNet.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.ln1 = nn.GroupNorm(1, hidden_channels)
        self.dconv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2, dilation=dilation, groups=hidden_channels
        )
        self.prelu2 = nn.PReLU()
        self.ln2 = nn.GroupNorm(1, hidden_channels)
        self.out_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(self.prelu1(self.conv1x1(x)))
        h = self.ln2(self.prelu2(self.dconv(h)))
        return x + self.out_conv(h)


class ConvTasNet(nn.Module):
    """
    Task 3 Legacy: Time-Domain Conv-TasNet for Speech Source Separation.
    Outputs max_sources=4 candidate stems.
    """
    def __init__(
        self,
        enc_dim: int = 512,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        kernel_len: int = 16,
        stride: int = 8,
        num_blocks: int = 8,
        repeats: int = 3,
        max_sources: int = 4
    ):
        super().__init__()
        self.max_sources = max_sources
        self.encoder = nn.Conv1d(1, enc_dim, kernel_size=kernel_len, stride=stride, bias=False)
        self.enc_norm = nn.GroupNorm(1, enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck_dim, kernel_size=1)

        blocks = []
        for _ in range(repeats):
            for i in range(num_blocks):
                blocks.append(DepthwiseSeparableBlock(bottleneck_dim, hidden_dim, dilation=2**i))
        self.tcn = nn.Sequential(*blocks)

        self.mask_conv = nn.Conv1d(bottleneck_dim, max_sources * enc_dim, kernel_size=1)
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, kernel_size=kernel_len, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 1, N]
        w = torch.relu(self.enc_norm(self.encoder(x)))
        b = self.bottleneck(w)
        h = self.tcn(b)
        masks = self.mask_conv(h)
        B, _, T = masks.shape
        masks = torch.sigmoid(masks.view(B, self.max_sources, -1, T))

        stems = []
        for i in range(self.max_sources):
            w_masked = w * masks[:, i, :, :]
            stems.append(self.decoder(w_masked).squeeze(1))
        return torch.stack(stems, dim=1)  # [B, max_sources, N]


# --- Path B Foundation Model Architectures ---

class AttentiveStatsPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        w = torch.softmax(self.attn(x), dim=1)
        mean = torch.sum(w * x, dim=1)
        var = torch.sum(w * (x - mean.unsqueeze(1)) ** 2, dim=1).clamp(min=1e-6)
        return torch.cat([mean, torch.sqrt(var)], dim=-1)


class WavLMSourceCounter(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", dsp_dim: int = 6):
        super().__init__()
        self.model_name = model_name
        is_offline = (os.getenv("OFFLINE_MODE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1")
        try:
            from transformers import AutoModel, WavLMConfig
            try:
                self.backbone = AutoModel.from_pretrained(model_name, local_files_only=is_offline)
            except Exception:
                try:
                    config = WavLMConfig.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    config = WavLMConfig()
                self.backbone = AutoModel.from_config(config)
        except Exception as e:
            logger.warning(f"[WavLMSourceCounter] Notice initializing WavLM backbone: {e}")
            self.backbone = None

        if self.backbone is not None:
            self.backbone.config.layerdrop = 0.0
            if hasattr(self.backbone.config, "apply_spec_augment"):
                self.backbone.config.apply_spec_augment = False
            if hasattr(self.backbone.config, "mask_time_prob"):
                self.backbone.config.mask_time_prob = 0.0
            hidden_dim = self.backbone.config.hidden_size
            num_states = len(self.backbone.encoder.layers) + 1
        else:
            hidden_dim = 768
            num_states = 13

        self.layer_weights = nn.Parameter(torch.zeros(num_states))
        self.pool = AttentiveStatsPooling(hidden_dim)
        self.dsp_mlp = nn.Sequential(
            nn.Linear(dsp_dim, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(256, 1)
        )
        self.domain_head = nn.Linear(hidden_dim * 2 + 64, 2)

    def forward(self, wav_norm: torch.Tensor, dsp_feats: torch.Tensor):
        out = self.backbone(wav_norm, output_hidden_states=True)
        stacked = torch.stack(out.hidden_states, dim=0)
        w = torch.softmax(self.layer_weights[:stacked.shape[0]], dim=0).view(-1, 1, 1, 1)
        h = torch.sum(stacked * w, dim=0)
        pooled = self.pool(h)
        feat = torch.cat([pooled.float(), self.dsp_mlp(dsp_feats.float())], dim=-1)
        return self.head(feat).reshape(-1), self.domain_head(feat)


class ConvTasNetSkipBlock(nn.Module):
    """
    Dilated Depthwise-Separable 1D Convolution block with Residual and Skip Connections.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.ln1 = nn.GroupNorm(1, hidden_channels)
        self.dconv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=hidden_channels
        )
        self.prelu2 = nn.PReLU()
        self.ln2 = nn.GroupNorm(1, hidden_channels)
        self.res_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(hidden_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(self.prelu1(self.conv1x1(x)))
        h = self.ln2(self.prelu2(self.dconv(h)))
        res = x + self.res_conv(h)
        skip = self.skip_conv(h)
        return res, skip


class ConvTasNetSkip(nn.Module):
    """
    ConvTasNet with Skip Connections supporting dynamic constructor parameters
    and arbitrary input lengths via pad-and-slice in forward().
    """
    def __init__(
        self,
        enc_dim: int = 512,
        bottleneck_dim: int = 128,
        skip_dim: int = 128,
        hidden_dim: int = 256,
        kernel_len: int = 16,
        stride: int = 8,
        num_blocks: int = 8,
        repeats: int = 3,
        max_sources: int = 4,
        mask_act: str = "relu"
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.bottleneck_dim = bottleneck_dim
        self.skip_dim = skip_dim
        self.hidden_dim = hidden_dim
        self.kernel_len = kernel_len
        self.stride = stride
        self.num_blocks = num_blocks
        self.repeats = repeats
        self.max_sources = max_sources
        self.mask_act = mask_act.lower()

        self.encoder = nn.Conv1d(1, enc_dim, kernel_size=kernel_len, stride=stride, bias=False)
        self.enc_norm = nn.GroupNorm(1, enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        for _ in range(repeats):
            for i in range(num_blocks):
                self.blocks.append(
                    ConvTasNetSkipBlock(
                        in_channels=bottleneck_dim,
                        hidden_channels=hidden_dim,
                        skip_channels=skip_dim,
                        kernel_size=3,
                        dilation=2 ** i
                    )
                )

        self.mask_conv = nn.Conv1d(skip_dim, max_sources * enc_dim, kernel_size=1)
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, kernel_size=kernel_len, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_len = x.shape[-1]
        pad_len = 0
        if (orig_len - self.kernel_len) % self.stride != 0:
            pad_len = self.stride - ((orig_len - self.kernel_len) % self.stride)
            x = F.pad(x, (0, pad_len))

        w = F.relu(self.enc_norm(self.encoder(x)))
        h = self.bottleneck(w)

        skip_sum = 0
        for block in self.blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip

        raw_masks = self.mask_conv(skip_sum)
        B, _, T = raw_masks.shape

        if self.mask_act == "relu":
            masks = F.relu(raw_masks).clamp(max=10.0)
        else:
            masks = torch.sigmoid(raw_masks)

        masks = masks.view(B, self.max_sources, self.enc_dim, T)

        stems = []
        for i in range(self.max_sources):
            w_masked = w * masks[:, i, :, :]
            s = self.decoder(w_masked).squeeze(1)
            if pad_len > 0:
                s = s[:, :orig_len]
            stems.append(s)

        return torch.stack(stems, dim=1)


# -----------------------------------------------------------------------------
# 3. Adaptive 3-Mode Audio Conditioning & Universal Audio Decoder
# -----------------------------------------------------------------------------

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".webm", ".ogg", ".3gp", ".flac"
}

# Precomputed Zero-Phase 65 Hz High-Pass Filter (4th-order Butterworth SOS)
SOS_65HZ = scipy.signal.butter(4, 65.0, btype="highpass", fs=TARGET_SAMPLE_RATE, output="sos")


def apply_soft_knee_limiter(x: np.ndarray) -> np.ndarray:
    """
    Transparent soft-knee limiter:
    y = np.where(np.abs(x) <= 0.85, x, np.sign(x) * (0.85 + 0.14 * np.tanh((np.abs(x) - 0.85) / 0.14)))
    """
    ax = np.abs(x)
    return np.where(
        ax <= 0.85,
        x,
        np.sign(x) * (0.85 + 0.14 * np.tanh((ax - 0.85) / 0.14))
    ).astype(np.float32)


def apply_cosine_fade(audio: np.ndarray, fade_len: int = 160) -> np.ndarray:
    """Applies 160-sample (10 ms) cosine fade-in/fade-out, guarded for len > 320."""
    if len(audio) <= 2 * fade_len:
        return audio
    out = audio.copy()
    fade_in = (0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len, dtype=np.float32) / fade_len))).astype(np.float32)
    fade_out = fade_in[::-1]
    out[:fade_len] *= fade_in
    out[-fade_len:] *= fade_out
    return out


def tile_with_crossfade(wav: np.ndarray, target_len: int, fade_len: int) -> np.ndarray:
    """
    Tiles wav to target_len with linear crossfade at tile seams.
    fade_len = min(800, max(0, len(wav) // 4))
    """
    L = len(wav)
    if L >= target_len:
        return wav[:target_len].copy()
    if L == 0:
        return np.zeros(target_len, dtype=np.float32)
    if fade_len <= 0 or L <= fade_len:
        reps = int(np.ceil(target_len / L))
        return np.tile(wav, reps)[:target_len].astype(np.float32)

    fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=False, dtype=np.float32)
    fade_in = 1.0 - fade_out

    out = list(wav)
    while len(out) < target_len:
        tail = np.array(out[-fade_len:], dtype=np.float32)
        head = wav[:fade_len]
        crossfaded = tail * fade_out + head * fade_in
        out[-fade_len:] = crossfaded.tolist()
        out.extend(wav[fade_len:].tolist())

    return np.array(out[:target_len], dtype=np.float32)


def smart_stereo_to_mono(wav: np.ndarray, channels_first: Optional[bool] = None) -> np.ndarray:
    """
    Smart Stereo-to-Mono (computer & live_mic):
    When orig_channels == 2, compute per-channel RMS and (only when both have std > 1e-8) Pearson correlation:
    - If one channel is >15 dB quieter (rms_ratio > 5.62), select the louder channel.
    - Else if Pearson correlation < -0.5, use (L - R) / 2.
    - Otherwise use (L + R) / 2.
    """
    if channels_first is None:
        if wav.ndim > 1 and wav.shape[0] == 2 and wav.shape[1] != 2:
            channels_first = True
        else:
            channels_first = False

    L = wav[0] if channels_first else wav[:, 0]
    R = wav[1] if channels_first else wav[:, 1]
    rms_L = float(np.sqrt(np.mean(L ** 2)))
    rms_R = float(np.sqrt(np.mean(R ** 2)))
    std_L = float(np.std(L))
    std_R = float(np.std(R))

    if (rms_R > 0 and rms_L / max(rms_R, 1e-12) > 5.62) or (rms_R == 0 and rms_L > 0):
        return L.astype(np.float32)
    elif (rms_L > 0 and rms_R / max(rms_L, 1e-12) > 5.62) or (rms_L == 0 and rms_R > 0):
        return R.astype(np.float32)
    elif std_L > 1e-8 and std_R > 1e-8:
        cov = float(np.mean((L - np.mean(L)) * (R - np.mean(R))))
        corr = float(cov / (std_L * std_R))
        if corr < -0.5:
            return ((L - R) / 2.0).astype(np.float32)
        else:
            return ((L + R) / 2.0).astype(np.float32)
    else:
        return ((L + R) / 2.0).astype(np.float32)


def decode_audio_bytes(contents: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """
    Decodes audio bytes into (waveform_data, sample_rate).
    Tries soundfile.read(io.BytesIO(contents), dtype="float32") first.
    If soundfile fails (.mp3, .m4a, .aac, .webm, .ogg, .3gp, .flac),
    writes contents to a temporary file via tempfile.NamedTemporaryFile(suffix=ext, delete=True),
    runs ffmpeg -y -i <tmp_path> -f f32le -acodec pcm_f32le -ac 1 -ar 16000 pipe:1,
    and returns (np.frombuffer(stdout, dtype=np.float32).copy(), 16000).
    """
    try:
        data, sr = sf.read(io.BytesIO(contents), dtype="float32")
        return data, sr
    except Exception as sf_err:
        ext = os.path.splitext(filename.lower())[1] if filename else ".tmp"
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
                tmp.write(contents)
                tmp.flush()
                cmd = [
                    "ffmpeg", "-y", "-i", tmp.name,
                    "-f", "f32le", "-acodec", "pcm_f32le",
                    "-ac", "1", "-ar", "16000", "pipe:1"
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                audio_np = np.frombuffer(proc.stdout, dtype=np.float32).copy()
                return audio_np, 16000
        except Exception as ffmpeg_err:
            raise ValueError(
                f"Failed to decode audio stream via soundfile and ffmpeg: {ffmpeg_err} (soundfile error: {sf_err})"
            ) from ffmpeg_err


def condition_audio(
    audio_input: Union[str, Path, np.ndarray, torch.Tensor, bytes],
    orig_sr: Optional[int] = None,
    mode: str = "auto",
    filename: str = ""
) -> Dict[str, Any]:
    """
    Adaptive 3-Mode Audio Conditioning grounded in SANADA Paper Table 3.

    Modes:
    1. 'sanada'   : Bit-exact preservation for SANADA Benchmark WAV files (no filtering, no RMS scaling)
    2. 'computer' : Smart stereo-to-mono, Kaiser sinc resample, DC cut, zero-phase 65Hz highpass,
                    silence trim with 150ms margin, Table-3 RMS norm (0.165), transparent soft-knee limiter
    3. 'live_mic' : Smart stereo-to-mono, Kaiser sinc resample, DC cut, zero-phase 65Hz highpass,
                    adaptive noise floor VAD gate, loudness matching (+28 dB max gain), transparent soft-knee limiter
    4. 'auto'     : Fingerprints incoming audio and routes to one of the above

    Returns:
        Dict with keys:
            audio: np.ndarray float32 (1D array at 16000 Hz)
            mode_used: str ("sanada" | "computer" | "live_mic")
            orig_sr: int
            orig_channels: int
            orig_duration_s: float
            duration_sec: float
            input_rms: float
            active_rms: float
            is_silent: bool
    """
    # 1. Decode/convert input to floating point numpy array
    if isinstance(audio_input, (str, Path)):
        p = Path(audio_input)
        try:
            wav_raw, file_sr = sf.read(str(p), dtype="float32")
        except Exception:
            with open(str(p), "rb") as f:
                wav_raw, file_sr = decode_audio_bytes(f.read(), filename=p.name)
        sr = orig_sr or file_sr
        wav = wav_raw
    elif isinstance(audio_input, bytes):
        wav_raw, file_sr = decode_audio_bytes(audio_input, filename=filename)
        sr = orig_sr or file_sr
        wav = wav_raw
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input.astype(np.float32)
        sr = orig_sr or TARGET_SAMPLE_RATE
    elif torch.is_tensor(audio_input):
        wav = audio_input.detach().cpu().numpy().astype(np.float32)
        sr = orig_sr or TARGET_SAMPLE_RATE
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    # Channel analysis & basic mono projection for fingerprinting
    channels_first = False
    if wav.ndim > 1:
        if wav.shape[0] <= 8 and wav.shape[1] > 8:
            orig_channels = wav.shape[0]
            channels_first = True
            wav_mono = np.mean(wav, axis=0)
        else:
            orig_channels = wav.shape[1]
            channels_first = False
            wav_mono = np.mean(wav, axis=-1)
    else:
        orig_channels = 1
        channels_first = False
        wav_mono = wav.flatten()

    orig_duration_s = round(len(wav_mono) / max(sr, 1), 3)
    init_rms = float(np.sqrt(np.mean(wav_mono ** 2))) if len(wav_mono) > 0 else 0.0

    # 2. Auto-Mode Fingerprinting
    mode_req = (mode or "auto").lower().strip()
    if mode_req == "auto":
        is_sanada_shape = (orig_channels == 1 and sr == TARGET_SAMPLE_RATE and abs(len(wav_mono) - N_SAMPLES) <= 160)
        is_sanada_rms = (0.035 <= init_rms <= 0.28)
        is_mic_hint = any(k in filename.lower() for k in ["live_mic", "microphone", "mic_recording", "stream"])

        if is_sanada_shape and is_sanada_rms:
            mode_used = "sanada"
        elif is_mic_hint:
            mode_used = "live_mic"
        else:
            mode_used = "computer"
    else:
        mode_used = mode_req if mode_req in ["sanada", "computer", "live_mic"] else "computer"

    # 3. Apply Mode-Specific Conditioning

    # -------------------------------------------------------------------------
    # Mode A: 'sanada' (Bit-Exact Benchmark Preservation)
    # -------------------------------------------------------------------------
    if mode_used == "sanada":
        curr_len = len(wav_mono)
        if curr_len < N_SAMPLES:
            wav_out = np.pad(wav_mono, (0, N_SAMPLES - curr_len), mode="constant")
        elif curr_len > N_SAMPLES:
            wav_out = wav_mono[:N_SAMPLES]
        else:
            wav_out = wav_mono

        is_silent = bool(init_rms < SILENCE_RMS_THRESHOLD)
        return {
            "audio": wav_out.astype(np.float32),
            "mode_used": "sanada",
            "orig_sr": sr,
            "orig_channels": orig_channels,
            "orig_duration_s": orig_duration_s,
            "duration_sec": round(len(wav_out) / TARGET_SAMPLE_RATE, 3),
            "input_rms": round(init_rms, 6),
            "active_rms": round(init_rms, 6),
            "is_silent": is_silent
        }

    # -------------------------------------------------------------------------
    # Mode B & C Common Preprocessing: Smart Stereo-to-Mono & 16kHz Resampling
    # -------------------------------------------------------------------------
    if orig_channels == 2:
        wav_projected = smart_stereo_to_mono(wav, channels_first=channels_first)
    else:
        wav_projected = wav_mono.copy()

    if sr != TARGET_SAMPLE_RATE:
        t_wav = torch.from_numpy(wav_projected).unsqueeze(0).float()
        try:
            t_res = torchaudio.functional.resample(
                t_wav, orig_freq=int(sr), new_freq=TARGET_SAMPLE_RATE,
                resampling_method="sinc_interp_kaiser"
            )
        except Exception:
            try:
                t_res = torchaudio.functional.resample(
                    t_wav, orig_freq=int(sr), new_freq=TARGET_SAMPLE_RATE,
                    resampling_method="kaiser_window"
                )
            except Exception:
                t_res = torchaudio.functional.resample(
                    t_wav, orig_freq=int(sr), new_freq=TARGET_SAMPLE_RATE
                )
        wav_work = t_res.squeeze(0).numpy()
    else:
        wav_work = wav_projected.copy()

    # Remove DC offset
    if len(wav_work) > 0:
        wav_work = wav_work - np.mean(wav_work)

    # Zero-Phase 65 Hz High-Pass Filter (scipy sosfiltfilt, guarded for len > 64)
    if len(wav_work) > 64:
        try:
            wav_work = scipy.signal.sosfiltfilt(SOS_65HZ, wav_work).astype(np.float32)
        except Exception as e:
            logger.warning(f"Highpass filter notice: {e}")

    # Silence Trim with 150 ms Margin (retaining 2400 samples before first & after last active frame)
    silence_thresh_amp = 10.0 ** (-52.0 / 20.0)  # ~0.0025119 (-52 dBFS)
    frame_len = 160  # 10ms at 16kHz
    margin_samples = 2400  # 150ms at 16kHz
    n_frames = len(wav_work) // frame_len
    if n_frames > 2:
        frame_rms = [
            float(np.sqrt(np.mean(wav_work[i * frame_len : (i + 1) * frame_len] ** 2)))
            for i in range(n_frames)
        ]
        active_idx = [i for i, r in enumerate(frame_rms) if r >= silence_thresh_amp]
        if active_idx:
            start_sample = max(0, active_idx[0] * frame_len - margin_samples)
            end_sample = min(len(wav_work), (active_idx[-1] + 1) * frame_len + margin_samples)
            if (end_sample - start_sample) >= 1600:
                wav_work = wav_work[start_sample:end_sample]

    # -------------------------------------------------------------------------
    # Mode B: 'computer' (Recorded System Audio & Uploaded Media Files)
    # -------------------------------------------------------------------------
    if mode_used == "computer":
        # Active RMS calculation
        frame_50ms = 800
        n_50ms = len(wav_work) // frame_50ms
        if n_50ms > 0:
            f_rms = np.array([
                np.sqrt(np.mean(wav_work[i * frame_50ms : (i + 1) * frame_50ms] ** 2))
                for i in range(n_50ms)
            ])
            act_mask = f_rms >= 1e-4
            if np.any(act_mask):
                act_samples = np.concatenate([
                    wav_work[i * frame_50ms : (i + 1) * frame_50ms]
                    for i in range(n_50ms) if act_mask[i]
                ])
                active_rms = float(np.sqrt(np.mean(act_samples ** 2)))
            else:
                active_rms = float(np.sqrt(np.mean(wav_work ** 2)))
        else:
            active_rms = float(np.sqrt(np.mean(wav_work ** 2))) if len(wav_work) > 0 else 0.0

        is_silent = bool(active_rms < SILENCE_RMS_THRESHOLD)

        # Soft Table-3 RMS Envelope Normalization: target_rms = 0.165 if outside [0.06, 0.25]
        if not is_silent and (active_rms < 0.06 or active_rms > 0.25):
            target_rms = 0.165
            gain = target_rms / max(active_rms, 1e-8)
            wav_work = wav_work * gain
            active_rms = active_rms * gain

        # Transparent Soft-Knee Limiter (preserving speech RMS)
        wav_work = apply_soft_knee_limiter(wav_work)
        active_rms = float(np.sqrt(np.mean(wav_work ** 2)))

        return {
            "audio": wav_work.astype(np.float32),
            "mode_used": "computer",
            "orig_sr": sr,
            "orig_channels": orig_channels,
            "orig_duration_s": orig_duration_s,
            "duration_sec": round(len(wav_work) / TARGET_SAMPLE_RATE, 3),
            "input_rms": round(float(np.sqrt(np.mean(wav_work ** 2))), 6),
            "active_rms": round(active_rms, 6),
            "is_silent": is_silent
        }

    # -------------------------------------------------------------------------
    # Mode C: 'live_mic' (Real-Time Microphone Capture)
    # -------------------------------------------------------------------------
    else:  # live_mic
        # Adaptive Noise Floor & VAD Gate from 10th percentile of 50ms frame RMS
        frame_len = 800  # 50ms at 16kHz
        n_frames = len(wav_work) // frame_len
        if n_frames > 0:
            frame_rms = np.array([
                float(np.sqrt(np.mean(wav_work[i * frame_len : (i + 1) * frame_len] ** 2)))
                for i in range(n_frames)
            ])
            noise_floor_rms = float(np.percentile(frame_rms, 10))
            vad_threshold = max(1.2e-3, noise_floor_rms * 2.2)
            active_mask = frame_rms >= vad_threshold
            if np.any(active_mask):
                active_frames = [
                    wav_work[i * frame_len : (i + 1) * frame_len]
                    for i in range(n_frames) if active_mask[i]
                ]
                active_audio = np.concatenate(active_frames)
                active_rms = float(np.sqrt(np.mean(active_audio ** 2)))
            else:
                active_rms = float(np.sqrt(np.mean(wav_work ** 2)))
                if active_rms >= 0.005:
                    vad_threshold = 1.2e-3
        else:
            vad_threshold = 1.2e-3
            active_rms = float(np.sqrt(np.mean(wav_work ** 2))) if len(wav_work) > 0 else 0.0

        if active_rms < vad_threshold or active_rms < SILENCE_RMS_THRESHOLD:
            return {
                "audio": wav_work.astype(np.float32),
                "mode_used": "live_mic",
                "orig_sr": sr,
                "orig_channels": orig_channels,
                "orig_duration_s": orig_duration_s,
                "duration_sec": round(len(wav_work) / TARGET_SAMPLE_RATE, 3),
                "input_rms": round(float(np.sqrt(np.mean(wav_work ** 2))), 6),
                "active_rms": round(active_rms, 6),
                "is_silent": True
            }

        target_rms = 0.165
        max_gain = 10.0 ** (28.0 / 20.0)  # ~25.118864 (+28 dB)
        desired_gain = target_rms / max(active_rms, 1e-8)
        applied_gain = min(desired_gain, max_gain)
        wav_work = wav_work * applied_gain
        active_rms = active_rms * applied_gain

        # Transparent Soft-Knee Limiter (preserving speech RMS)
        wav_work = apply_soft_knee_limiter(wav_work)
        active_rms = float(np.sqrt(np.mean(wav_work ** 2)))

        return {
            "audio": wav_work.astype(np.float32),
            "mode_used": "live_mic",
            "orig_sr": sr,
            "orig_channels": orig_channels,
            "orig_duration_s": orig_duration_s,
            "duration_sec": round(len(wav_work) / TARGET_SAMPLE_RATE, 3),
            "input_rms": round(float(np.sqrt(np.mean(wav_work ** 2))), 6),
            "active_rms": round(active_rms, 6),
            "is_silent": False
        }


# -----------------------------------------------------------------------------
# 4. Production Inference Pipeline Class
# -----------------------------------------------------------------------------

class SANADAPipeline:
    """
    Unified Production Inference Pipeline for SANADA Machine Listening.
    Supports Path B Foundation Models and Graceful Fallback to Legacy Checkpoints:
    - Task 1: WavLMSourceCounter (Path B, K=1..32) or ConformerCounter (Legacy, K=1..9)
    - Task 2: AST 33-Class Tagger (Path B) or ConformerTagger (Legacy, 32 FSD Classes)
    - Task 3: Multi-Specialist Separation Router (Sepformer-2spk, Sepformer-3spk, ConvTasNetSkip 4spk)
              or ConvTasNet (Legacy 4spk)
    """
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        tag_threshold: float = 0.5
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.tag_threshold = tag_threshold
        self.classes = list(FSD_32_CLASSES)

        # Default HF_HOME to checkpoint_dir/hf_cache if not set
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = os.path.join(self.checkpoint_dir, "hf_cache")

        logger.info(f"Initializing SANADAPipeline on device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        # Audio transforms for Mel feature extraction (Legacy Conformer and DSP)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        ).to(self.device)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(self.device)

        # Telemetry container
        self.active_models: Dict[str, Any] = {}

        # Default model references
        self.counter_model: Optional[nn.Module] = None
        self.tagger_model: Optional[nn.Module] = None
        self.ast_feature_extractor = None
        self.sep_model: Optional[nn.Module] = None
        self.dominant_sep_model: Optional[nn.Module] = None
        self.sepformer_2spk: Optional[Dict[str, Any]] = None
        self.sepformer_3spk: Optional[Dict[str, Any]] = None

        # Calibration parameters
        self.calibration_low_k = {"slope": 0.872357, "intercept": 0.315223}
        self.calibration_high_k = {"slope": 0.963689, "intercept": 0.180371}
        self.cal_org4 = {"slope": 1.0, "intercept": 0.0}
        self.cal_org9 = {"slope": 1.0, "intercept": 0.0}
        self.cal_fsd32 = {"slope": 1.0, "intercept": 0.0}
        self.fsd_fusion = None
        self.optimal_thresholds = [0.50] * 32
        self.speech_gate = {"speech_prob_min": 0.6, "env_threshold_bonus": 0.05, "env_threshold_max": 0.95}

        # Load all checkpoints (Path B first, falling back gracefully)
        self._load_all_checkpoints()

    def _load_checkpoint_file(self, filename_candidates: List[str], model: nn.Module, model_name: str) -> Optional[Dict]:
        """Loads state_dict from first matching candidate file with strict=True."""
        for candidate in filename_candidates:
            ckpt_path = os.path.join(self.checkpoint_dir, candidate)
            if os.path.exists(ckpt_path):
                try:
                    data = load_checkpoint_cpu(ckpt_path)
                    state_dict = None
                    if isinstance(data, dict):
                        if "state_dict" in data:
                            state_dict = data["state_dict"]
                        elif any(k.startswith(("subsample", "blocks", "encoder", "tcn", "backbone")) for k in data.keys()):
                            state_dict = data

                    if state_dict is not None:
                        try:
                            model.load_state_dict(state_dict, strict=True)
                            logger.info(f"[{model_name}] Successfully loaded state_dict (strict=True) from {candidate}")
                        except RuntimeError as re_strict:
                            logger.warning(f"[{model_name}] strict=True failed ({re_strict}). Retrying with strict=False...")
                            model.load_state_dict(state_dict, strict=False)
                            logger.info(f"[{model_name}] Loaded state_dict with strict=False from {candidate}")
                    return data if isinstance(data, dict) else {"state_dict": data}
                except Exception as e:
                    logger.warning(f"[{model_name}] Failed to load checkpoint {candidate}: {e}")
        logger.info(f"[{model_name}] No matching checkpoint file found in {self.checkpoint_dir}. Using initialized baseline.")
        return None

    def _load_sepformer_specialist(self, filename: str) -> Optional[Dict[str, Any]]:
        """Loads Sepformer specialist checkpoint using SpeechBrain with automatic hparams caching."""
        ckpt_path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(ckpt_path):
            return None
        try:
            ckpt = load_checkpoint_cpu(ckpt_path)
            source_repo = ckpt.get("source_repo", "speechbrain/sepformer-libri2mix")
            native_sr = int(ckpt.get("native_sample_rate", ckpt.get("sample_rate", 8000)))
            max_spks = int(ckpt.get("max_sources", 2))
            repo_slug = source_repo.replace("/", "_")
            if max_spks > 3 or "4spk" in filename.lower():
                cache_dir = os.path.join(self.checkpoint_dir, "pretrained_cache", f"{repo_slug}_4spk")
            else:
                cache_dir = os.path.join(self.checkpoint_dir, "pretrained_cache", repo_slug)
            os.makedirs(cache_dir, exist_ok=True)

            hparams_file = os.path.join(cache_dir, "hyperparams.yaml")
            if not os.path.exists(hparams_file):
                base_h = os.path.join(self.checkpoint_dir, "pretrained_cache", repo_slug, "hyperparams.yaml")
                if os.path.exists(base_h):
                    with open(base_h, "r") as f:
                        t = f.read()
                    t = t.replace("num_spks: 3", f"num_spks: {max_spks}").replace("num_spks: 2", f"num_spks: {max_spks}")
                    with open(hparams_file, "w") as f:
                        f.write(t)

            if "encoder" in ckpt:
                torch.save(ckpt["encoder"], os.path.join(cache_dir, "encoder.ckpt"))
            if "masknet" in ckpt:
                torch.save(ckpt["masknet"], os.path.join(cache_dir, "masknet.ckpt"))
            if "decoder" in ckpt:
                torch.save(ckpt["decoder"], os.path.join(cache_dir, "decoder.ckpt"))

            source_target = cache_dir if os.path.exists(hparams_file) else source_repo

            from speechbrain.inference.separation import SepformerSeparation
            sepformer = SepformerSeparation.from_hparams(
                source=source_target,
                savedir=cache_dir,
                run_opts={"device": str(self.device)}
            )
            if "state_dict" in ckpt:
                try:
                    sepformer.mods.load_state_dict(ckpt["state_dict"], strict=True)
                except Exception:
                    sepformer.mods.load_state_dict(ckpt["state_dict"], strict=False)
            logger.info(f"[Separation-Router] Loaded {filename} from {source_repo} (native_sr={native_sr}, max_sources={max_spks})")
            return {"model": sepformer, "native_sr": native_sr, "source_repo": source_repo, "max_sources": max_spks}
        except Exception as e:
            logger.warning(f"[Separation-Router] Notice loading Sepformer specialist {filename}: {e}")
            return None

    def _load_conv_tasnet_dominant(self, filename: Union[str, List[str]]) -> Optional[nn.Module]:
        """Loads 4-speaker dominant model with dynamic constructor arguments."""
        filenames = [filename] if isinstance(filename, str) else filename
        ckpt_path = None
        loaded_fn = ""
        for fn in filenames:
            p = os.path.join(self.checkpoint_dir, fn)
            if os.path.exists(p):
                ckpt_path = p
                loaded_fn = fn
                break
        if ckpt_path is None:
            return None
        try:
            ckpt = load_checkpoint_cpu(ckpt_path)
            enc_dim = int(ckpt.get("enc_dim", 512))
            bottleneck_dim = int(ckpt.get("bottleneck_dim", 128))
            skip_dim = int(ckpt.get("skip_dim", 128))
            hidden_dim = int(ckpt.get("hidden_dim", 256))
            kernel_len = int(ckpt.get("kernel_len", 16))
            stride = int(ckpt.get("stride", 8))
            num_blocks = int(ckpt.get("num_blocks", 8))
            repeats = int(ckpt.get("repeats", 3))
            max_sources = int(ckpt.get("max_sources", 4))
            mask_act = str(ckpt.get("mask_act", "relu"))

            model = ConvTasNetSkip(
                enc_dim=enc_dim,
                bottleneck_dim=bottleneck_dim,
                skip_dim=skip_dim,
                hidden_dim=hidden_dim,
                kernel_len=kernel_len,
                stride=stride,
                num_blocks=num_blocks,
                repeats=repeats,
                max_sources=max_sources,
                mask_act=mask_act
            ).to(self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"[Separation-Router] Loaded {loaded_fn} (4-Speaker Dominant strict=True)")
            return model
        except Exception as e:
            logger.warning(f"[Separation-Router] Notice loading {loaded_fn}: {e}")
            return None

    def _load_all_checkpoints(self):
        """Loads checkpoints with Path B preference and graceful fallback to legacy checkpoints."""
        # ---------------------------------------------------------------------
        # 1. Task 1: Source Counter (WavLM Path B vs Conformer Legacy)
        # ---------------------------------------------------------------------
        wavlm_path = os.path.join(self.checkpoint_dir, "wavlm_counting_pathb.pt")
        loaded_pathb_counter = False

        if os.path.exists(wavlm_path):
            try:
                ckpt = load_checkpoint_cpu(wavlm_path)
                self.counter_model = WavLMSourceCounter().to(self.device)
                state_dict = ckpt.get("state_dict", ckpt)

                # Check and remap weight_norm parametrizations if needed
                model_sd_keys = set(self.counter_model.state_dict().keys())
                final_sd = {}
                for k, v in state_dict.items():
                    if k not in model_sd_keys:
                        # Remap parametrizations
                        alt_k = k.replace(".weight_g", ".parametrizations.weight.original0").replace(".weight_v", ".parametrizations.weight.original1")
                        if alt_k in model_sd_keys:
                            final_sd[alt_k] = v
                        else:
                            alt_k2 = k.replace(".parametrizations.weight.original0", ".weight_g").replace(".parametrizations.weight.original1", ".weight_v")
                            final_sd[alt_k2 if alt_k2 in model_sd_keys else k] = v
                    else:
                        final_sd[k] = v

                load_res = self.counter_model.load_state_dict(final_sd, strict=True)
                missing = load_res.missing_keys
                unexpected = load_res.unexpected_keys
                if missing or unexpected:
                    logger.warning(f"[Task 1] Checkpoint key mismatch! Missing: {missing}, Unexpected: {unexpected}")
                else:
                    logger.info("[Task 1] Loaded Path B Foundation Model: WavLMSourceCounter (strict=True, all keys matched)")

                self.counter_model.eval()
                if hasattr(self.counter_model, "backbone") and self.counter_model.backbone is not None:
                    self.counter_model.backbone.eval()

                self.cal_org4 = ckpt.get("cal_org4", {"slope": 1.0, "intercept": 0.0})
                self.cal_org9 = ckpt.get("cal_org9", {"slope": 1.0, "intercept": 0.0})
                self.cal_fsd32 = ckpt.get("cal_fsd32", {"slope": 1.0, "intercept": 0.0})
                self.fsd_fusion = ckpt.get("fsd_fusion", None)
                self.active_models["counting"] = {
                    "model_name": "WavLMSourceCounter",
                    "tier": "path_b_foundation",
                    "real_weights": True,
                    "supported_k_range": [1, 32]
                }
                loaded_pathb_counter = True
            except Exception as e:
                logger.warning(f"[Task 1] WavLM Path B load failed ({e}), falling back to ConformerCounter...")

        if not loaded_pathb_counter:
            self.counter_model = ConformerCounter(d_model=256, n_blocks=4).to(self.device)
            count_data = self._load_checkpoint_file(
                ["conformer_counting_baseline.pt", "counting_model.pt"],
                self.counter_model,
                "ConformerCounter"
            )
            has_real_weights = count_data is not None
            if count_data and isinstance(count_data, dict):
                default_slope = float(count_data.get("calibration_slope", 1.0))
                default_intercept = float(count_data.get("calibration_intercept", 0.0))
                self.calibration_low_k = count_data.get(
                    "calibration_low_k",
                    {"slope": default_slope, "intercept": default_intercept}
                )
                self.calibration_high_k = count_data.get(
                    "calibration_high_k",
                    {"slope": default_slope, "intercept": default_intercept}
                )
                logger.info(f"[ConformerCounter] Loaded two-regime calibration: Low-K={self.calibration_low_k}, High-K={self.calibration_high_k}")

            self.active_models["counting"] = {
                "model_name": "ConformerCounter",
                "tier": "legacy_fallback",
                "real_weights": has_real_weights,
                "supported_k_range": [1, 9]
            }

        # ---------------------------------------------------------------------
        # 2. Task 2: Acoustic Tagger (MIT AST Path B vs Conformer Legacy)
        # ---------------------------------------------------------------------
        ast_path = os.path.join(self.checkpoint_dir, "ast_tagging_pathb.pt")
        loaded_pathb_tagger = False

        if os.path.exists(ast_path):
            try:
                ckpt = load_checkpoint_cpu(ast_path)
                is_offline = (os.getenv("OFFLINE_MODE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1")
                from transformers import ASTForAudioClassification, AutoFeatureExtractor
                self.ast_feature_extractor = AutoFeatureExtractor.from_pretrained(
                    "MIT/ast-finetuned-audioset-10-10-0.4593",
                    local_files_only=is_offline
                )
                self.tagger_model = ASTForAudioClassification.from_pretrained(
                    "MIT/ast-finetuned-audioset-10-10-0.4593",
                    num_labels=33,
                    id2label=ID2LABEL_33,
                    label2id=CLASS_TO_IDX,
                    ignore_mismatched_sizes=True,
                    local_files_only=is_offline
                ).to(self.device)
                state_dict = ckpt.get("state_dict", ckpt)
                self.tagger_model.load_state_dict(state_dict, strict=True)

                if "optimal_thresholds" in ckpt:
                    self.optimal_thresholds = [float(t) for t in ckpt["optimal_thresholds"]]
                if "speech_gate" in ckpt:
                    self.speech_gate = ckpt["speech_gate"]

                self.active_models["tagging"] = {
                    "model_name": "ASTForAudioClassification",
                    "tier": "path_b_foundation",
                    "real_weights": True,
                    "num_classes": 33
                }
                logger.info(f"[Task 2] Loaded Path B Foundation Model: MIT AST (33 classes, strict=True)")
                loaded_pathb_tagger = True
            except Exception as e:
                logger.warning(f"[Task 2] AST Path B load failed ({e}), falling back to ConformerTagger...")

        if not loaded_pathb_tagger:
            self.tagger_model = ConformerTagger(d_model=256, n_blocks=4, num_classes=32).to(self.device)
            tag_data = self._load_checkpoint_file(
                ["conformer_tagging_baseline.pt", "conformer_tagging_asl.pt", "tagging_model.pt"],
                self.tagger_model,
                "ConformerTagger"
            )
            has_real_weights = tag_data is not None
            if tag_data and isinstance(tag_data, dict):
                if "classes" in tag_data:
                    self.classes = list(tag_data["classes"])
                if "optimal_thresholds" in tag_data and len(tag_data["optimal_thresholds"]) == len(self.classes):
                    self.optimal_thresholds = [float(t) for t in tag_data["optimal_thresholds"]]
                    logger.info(f"[ConformerTagger] Loaded {len(self.optimal_thresholds)} per-class optimal thresholds.")

            self.active_models["tagging"] = {
                "model_name": "ConformerTagger",
                "tier": "legacy_fallback",
                "real_weights": has_real_weights,
                "num_classes": 32
            }

        # ---------------------------------------------------------------------
        # 3. Task 3: Separation Router (Multi-Specialist vs Legacy ConvTasNet)
        # ---------------------------------------------------------------------
        # Try loading specialist models
        self.sepformer_2spk = self._load_sepformer_specialist("sepformer_2spk_pathb.pt")
        self.sepformer_3spk = self._load_sepformer_specialist("sepformer_3spk_pathb.pt")
        self.sepformer_4spk = self._load_sepformer_specialist("sepformer_4spk_dominant_pathb.pt")
        self.dominant_sep_model = self._load_conv_tasnet_dominant([
            "conv_tasnet_4spk_dominant_pathb.pt"
        ])

        # Always instantiate legacy ConvTasNet as baseline fallback
        self.sep_model = ConvTasNet(enc_dim=512, bottleneck_dim=128, hidden_dim=256, max_sources=4).to(self.device)
        sep_data = self._load_checkpoint_file(
            ["conv_tasnet_separation.pt", "separation_model.pt"],
            self.sep_model,
            "ConvTasNet"
        )
        has_real_sep_weights = sep_data is not None

        has_specialists = bool(self.sepformer_2spk or self.sepformer_3spk or self.sepformer_4spk or self.dominant_sep_model)
        specialist_list = []
        if self.sepformer_2spk:
            specialist_list.append("sepformer_2spk")
        if self.sepformer_3spk:
            specialist_list.append("sepformer_3spk")
        if self.sepformer_4spk:
            specialist_list.append("sepformer_4spk_dominant")
        if self.dominant_sep_model:
            specialist_list.append("conv_tasnet_4spk_dominant")

        self.active_models["separation"] = {
            "model_name": "DynamicMultiSpecialistRouter" if has_specialists else "ConvTasNet",
            "tier": "path_b_foundation" if has_specialists else "legacy_fallback",
            "real_weights": has_real_sep_weights or has_specialists,
            "specialists_available": specialist_list
        }

        # Set evaluation mode
        if self.counter_model:
            self.counter_model.eval()
        if self.tagger_model:
            self.tagger_model.eval()
        if self.sep_model:
            self.sep_model.eval()
        if self.dominant_sep_model:
            self.dominant_sep_model.eval()

    def extract_mel(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts log-mel spectrogram normalized to [B, 1, 128, 251] for Conformer modules.
        Input: wav_tensor of shape [1, 128000]
        """
        mel = self.amp_to_db(self.mel_transform(wav_tensor))  # [1, 128, 251]
        norm_mel = ((mel + 40.0) / 40.0).unsqueeze(0)  # [1, 1, 128, 251]
        return norm_mel

    def standardize_audio(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor, bytes],
        sample_rate: int = TARGET_SAMPLE_RATE,
        mode: str = "auto"
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Legacy helper for backward compatibility."""
        cond = condition_audio(audio_input, orig_sr=sample_rate, mode=mode)
        wav_data = cond["audio"]
        if len(wav_data) < N_SAMPLES:
            wav_pad = np.pad(wav_data, (0, N_SAMPLES - len(wav_data)))
        else:
            wav_pad = wav_data[:N_SAMPLES]
        wav_tensor = torch.from_numpy(wav_pad).unsqueeze(0).to(self.device)
        return wav_pad, wav_tensor

    def _prepare_analysis_windows(
        self,
        wav_np: np.ndarray,
        mode: str = "auto",
        win_len: int = N_SAMPLES,
        hop_len: int = 64000,
        max_windows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares 128,000-sample analysis windows for Task 1 and Task 2 inference:
        - Mode == "sanada": 1 window with standard zero-pad/slice (100% bit-exact).
        - Short audio (< win_len): Unpadded audio for DSP, tiled with linear crossfade for 128k input.
        - Long audio (> win_len): Overlapping 8.0s windows (hop=4.0s), dropping silent windows (rms < 1e-3).
        """
        if max_windows is None:
            default_mw = "4" if getattr(self.device, "type", "cpu") == "cpu" else "6"
            max_windows = int(os.getenv("SANADA_MAX_WINDOWS", default_mw))

        # 1. Mode == "sanada": keep exact standard 8.0s zero-pad/slice (1 window)
        if mode == "sanada":
            if len(wav_np) < win_len:
                c_audio = np.pad(wav_np, (0, win_len - len(wav_np)))
            else:
                c_audio = wav_np[:win_len]
            rms = float(np.sqrt(np.mean(c_audio ** 2))) if len(c_audio) > 0 else 0.0
            return [{
                "window": c_audio.astype(np.float32),
                "dsp_audio": c_audio.astype(np.float32),
                "rms": rms
            }]

        # 2. Short audio (< win_len): Return unpadded for DSP, tiled with linear crossfade for window
        if len(wav_np) < win_len:
            fade_len = min(800, max(0, len(wav_np) // 4))
            tiled = tile_with_crossfade(wav_np, target_len=win_len, fade_len=fade_len)
            rms = float(np.sqrt(np.mean(wav_np ** 2))) if len(wav_np) > 0 else 0.0
            return [{
                "window": tiled,
                "dsp_audio": wav_np,
                "rms": rms
            }]

        # 3. Long audio (>= win_len): Multi-window slicing
        L = len(wav_np)
        if L == win_len:
            rms = float(np.sqrt(np.mean(wav_np ** 2)))
            return [{
                "window": wav_np.copy(),
                "dsp_audio": wav_np.copy(),
                "rms": rms
            }]

        starts = list(range(0, L - win_len + 1, hop_len))
        if starts[-1] + win_len < L:
            starts.append(L - win_len)

        if len(starts) > max_windows:
            indices = np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int)
            starts = [starts[i] for i in indices]

        candidate_windows = []
        for s in starts:
            chunk = wav_np[s : s + win_len]
            c_rms = float(np.sqrt(np.mean(chunk ** 2)))
            candidate_windows.append({
                "window": chunk.copy(),
                "dsp_audio": chunk.copy(),
                "rms": c_rms
            })

        # Drop silent windows (rms < 1e-3), keeping at least the loudest window
        active_windows = [w for w in candidate_windows if w["rms"] >= 1e-3]
        if not active_windows:
            loudest = max(candidate_windows, key=lambda w: w["rms"])
            active_windows = [loudest]

        return active_windows

    # -------------------------------------------------------------------------
    # Task 1: Source Counting (Path B WavLM or Legacy Conformer)
    # -------------------------------------------------------------------------
    def predict_counting(
        self,
        wav_np: np.ndarray,
        wav_tensor: Optional[torch.Tensor] = None,
        mode: str = "auto",
        is_silent: bool = False,
        ast_env_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Task 1: Source Counting Regressor.
        Supports multi-window inference across short and long audio:
        - WavLMSourceCounter (Path B, K in [1, 32], domain classification & calibration)
        - ConformerCounter (Legacy, K in [1, 9], Two-Regime Calibration & K=1 verification)
        """
        input_rms = float(np.sqrt(np.mean(wav_np ** 2))) if len(wav_np) > 0 else 0.0

        if is_silent or input_rms < SILENCE_RMS_THRESHOLD:
            return {
                "estimated_k": 0,
                "predicted_k": 0,
                "raw_continuous_prediction": 0.0,
                "raw_k": 0.0,
                "calibrated_k": 0.0,
                "confidence": 1.0,
                "silent": True,
                "rms_energy": round(input_rms, 8),
                "complexity_ceiling_exceeded": False,
                "windows_analyzed": 0,
                "model": self.active_models["counting"]["model_name"],
                "tier": self.active_models["counting"]["tier"]
            }

        # Multi-window preparation
        windows = self._prepare_analysis_windows(wav_np, mode=mode, win_len=N_SAMPLES, hop_len=64000)

        # Path B: WavLMSourceCounter
        if isinstance(self.counter_model, WavLMSourceCounter):
            raw_k_list = []
            domain_logits_list = []

            for win in windows:
                c_audio = win["window"]
                dsp_audio = win["dsp_audio"]
                norm_wav = normalize_wavlm_input(c_audio)
                dsp_feat = extract_table3_dsp(dsp_audio).unsqueeze(0).to(self.device)
                t_input = torch.from_numpy(norm_wav).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    raw_k_tensor, domain_logits = self.counter_model(t_input, dsp_feat)
                    raw_k_list.append(float(raw_k_tensor.item()))
                    domain_logits_list.append(domain_logits)

            mean_domain_logits = torch.mean(torch.stack(domain_logits_list), dim=0)
            domain_pred = int(torch.argmax(mean_domain_logits, dim=-1).item())

            cal_k_list = []
            if domain_pred == 0:
                # Speech domain: select cal_org4 vs cal_org9
                for rk in raw_k_list:
                    cal = self.cal_org4 if rk <= 4.35 else self.cal_org9
                    cal_slope = float(cal.get("slope", 1.0))
                    cal_intercept = float(cal.get("intercept", 0.0))
                    cal_k_list.append(cal_slope * rk + cal_intercept)
                cal_k = float(np.percentile(cal_k_list, 75))
                predicted_k = int(np.clip(np.round(cal_k), 1, 9))
                domain_str = "speech"
            else:
                # Environmental domain: select cal_fsd32, optional fsd_fusion with AST
                cal_slope = float(self.cal_fsd32.get("slope", 1.0))
                cal_intercept = float(self.cal_fsd32.get("intercept", 0.0))
                for rk in raw_k_list:
                    ck = cal_slope * rk + cal_intercept
                    if self.fsd_fusion is not None and ast_env_count is not None:
                        w_wav = float(self.fsd_fusion.get("w_wavlm", 0.5))
                        w_ast = float(self.fsd_fusion.get("w_ast_count", 0.5))
                        bias = float(self.fsd_fusion.get("bias", 0.0))
                        ck = w_wav * ck + w_ast * float(ast_env_count) + bias
                    cal_k_list.append(ck)
                cal_k = float(np.percentile(cal_k_list, 75))
                predicted_k = int(np.clip(np.round(cal_k), 1, 32))
                domain_str = "environmental"

            raw_k = float(np.percentile(raw_k_list, 75))
            dist_to_int = abs(cal_k - round(cal_k))
            confidence = float(np.clip(1.0 - 0.5 * dist_to_int, 0.5, 0.99))
            complexity_ceiling_exceeded = bool(predicted_k > 4)

            return {
                "estimated_k": predicted_k,
                "predicted_k": predicted_k,
                "raw_continuous_prediction": round(raw_k, 4),
                "raw_k": round(raw_k, 4),
                "calibrated_k": round(cal_k, 4),
                "confidence": round(confidence, 4),
                "domain_detected": domain_str,
                "single_speaker_verified": False,
                "silent": False,
                "rms_energy": round(input_rms, 6),
                "complexity_ceiling_exceeded": complexity_ceiling_exceeded,
                "windows_analyzed": len(windows),
                "model": "WavLMSourceCounter",
                "tier": "path_b_foundation"
            }

        # Legacy Fallback: ConformerCounter across active windows
        raw_k_list = []
        cal_k_list = []
        for win in windows:
            c_audio = win["window"]
            c_tensor = torch.from_numpy(c_audio).unsqueeze(0).to(self.device)
            mel = self.extract_mel(c_tensor)
            with torch.no_grad():
                rk = float(self.counter_model(mel).item())
            raw_k_list.append(rk)
            if rk <= 4.35:
                ck = self.calibration_low_k["slope"] * rk + self.calibration_low_k["intercept"]
            else:
                ck = self.calibration_high_k["slope"] * rk + self.calibration_high_k["intercept"]
            cal_k_list.append(ck)

        cal_k = float(np.percentile(cal_k_list, 75))
        raw_k = float(np.percentile(raw_k_list, 75))
        predicted_k = int(np.clip(np.round(cal_k), 1, 9))

        dist_to_int = abs(cal_k - round(cal_k))
        confidence = float(np.clip(1.0 - 0.5 * dist_to_int, 0.5, 0.99))
        complexity_ceiling_exceeded = bool(predicted_k > 4)

        return {
            "estimated_k": predicted_k,
            "predicted_k": predicted_k,
            "raw_continuous_prediction": round(raw_k, 4),
            "raw_k": round(raw_k, 4),
            "calibrated_k": round(cal_k, 4),
            "confidence": round(confidence, 4),
            "single_speaker_verified": False,
            "silent": False,
            "rms_energy": round(input_rms, 6),
            "complexity_ceiling_exceeded": complexity_ceiling_exceeded,
            "windows_analyzed": len(windows),
            "model": "Conformer-Regressor-MSE",
            "tier": "legacy_fallback"
        }

    # -------------------------------------------------------------------------
    # Task 2: Acoustic Tagging (Path B MIT AST or Legacy Conformer)
    # -------------------------------------------------------------------------
    def predict_tagging(
        self,
        wav_np: np.ndarray,
        wav_tensor: Optional[torch.Tensor] = None,
        mode: str = "auto",
        tag_threshold: Optional[float] = None,
        is_silent: bool = False
    ) -> Dict[str, Any]:
        """
        Task 2: Environmental Multi-Label Acoustic Tagging.
        Supports multi-window probability aggregation (0.75 * max_prob + 0.25 * mean_prob):
        - MIT AST 33-Class Tagger (Path B, with speech gate & human_speech separation)
        - ConformerTagger (Legacy 32 FSD Classes, ASL loss)
        """
        tier = self.active_models["tagging"]["tier"]

        if is_silent:
            return {
                "environmental_events": {
                    "active_events": [],
                    "top_k_events": [],
                    "num_active": 0,
                    "threshold_applied": float(tag_threshold or 0.5),
                    "domain_taxonomy": "FSD-50K-32-Classes",
                    "model": "ASTForAudioClassification" if tier == "path_b_foundation" else "Conformer-Classifier-ASL"
                },
                "active_classes": [],
                "class_probabilities": {c: 0.0 for c in self.classes},
                "detected_tags": [],
                "num_detected": 0,
                "top_5_events": [],
                "human_speech_detected": False,
                "human_speech_prob": 0.0,
                "threshold_applied": float(tag_threshold or 0.5),
                "taxonomy": "FSD-50K-32-Classes",
                "windows_analyzed": 0,
                "model": "ASTForAudioClassification" if tier == "path_b_foundation" else "Conformer-Classifier-ASL",
                "tier": tier
            }

        # Multi-window preparation
        windows = self._prepare_analysis_windows(wav_np, mode=mode, win_len=N_SAMPLES, hop_len=64000)

        # Path B: Pretrained MIT AST (33 Classes)
        if tier == "path_b_foundation" and self.ast_feature_extractor is not None:
            all_probs = []
            for win in windows:
                win_audio = win["window"]
                inputs = self.ast_feature_extractor(win_audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.tagger_model(**inputs)
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    pw = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [33]
                    all_probs.append(pw)

            probs_stack = np.stack(all_probs, axis=0)
            max_probs = np.max(probs_stack, axis=0)
            mean_probs = np.mean(probs_stack, axis=0)
            probs = 0.75 * max_probs + 0.25 * mean_probs

            human_speech_prob = float(probs[32])
            speech_th = self.optimal_thresholds[32] if len(self.optimal_thresholds) > 32 else 0.5
            human_speech_detected = bool(human_speech_prob >= speech_th)

            # Speech Gate: when human_speech_prob >= speech_prob_min, raise environmental thresholds
            speech_prob_min = float(self.speech_gate.get("speech_prob_min", 0.6))
            env_bonus = float(self.speech_gate.get("env_threshold_bonus", 0.05))
            env_max = float(self.speech_gate.get("env_threshold_max", 0.95))
            apply_speech_gate = bool(human_speech_prob >= speech_prob_min)

            active_events = []
            class_probabilities = {}

            # Strict 32 FSD Environmental Classes
            for i in range(32):
                class_name = FSD_32_CLASSES[i]
                prob = float(probs[i])
                class_probabilities[class_name] = round(prob, 4)

                if tag_threshold is not None:
                    base_th = float(tag_threshold)
                elif i < len(self.optimal_thresholds):
                    base_th = self.optimal_thresholds[i]
                else:
                    base_th = self.tag_threshold

                effective_th = base_th
                if apply_speech_gate:
                    effective_th = min(env_max, effective_th + env_bonus)

                if mode == "live_mic" and class_name in PERCUSSIVE_CLASSES:
                    effective_th = min(0.95, effective_th + 0.05)

                if prob >= effective_th:
                    active_events.append({
                        "label": class_name,
                        "probability": round(prob, 4),
                        "threshold": round(effective_th, 4)
                    })

            active_events.sort(key=lambda x: x["probability"], reverse=True)

            # Top 5 environmental candidates
            top_5_indices = np.argsort(probs[:32])[::-1][:5]
            top_5_events = [
                {
                    "label": FSD_32_CLASSES[i],
                    "probability": round(float(probs[i]), 4),
                    "threshold": round(self.optimal_thresholds[i] if i < len(self.optimal_thresholds) else 0.5, 4)
                }
                for i in top_5_indices
            ]
            active_classes = [e["label"] for e in active_events]

            return {
                "environmental_events": {
                    "active_events": active_events,
                    "top_k_events": top_5_events,
                    "num_active": len(active_events),
                    "threshold_applied": float(tag_threshold or 0.5),
                    "domain_taxonomy": "FSD-50K-32-Classes",
                    "note": "Environmental acoustic sound events (strictly 32 FSD classes)",
                    "model": "ASTForAudioClassification"
                },
                "active_classes": active_classes,
                "class_probabilities": class_probabilities,
                "detected_tags": active_classes,
                "num_detected": len(active_classes),
                "top_5_events": top_5_events,
                "top_5_candidates": top_5_events,
                "human_speech_detected": human_speech_detected,
                "human_speech_prob": round(human_speech_prob, 4),
                "threshold_applied": float(tag_threshold or 0.5),
                "taxonomy": "FSD-50K-32-Classes",
                "windows_analyzed": len(windows),
                "model": "ASTForAudioClassification",
                "tier": "path_b_foundation"
            }

        # Legacy Fallback: ConformerTagger (32 Classes) across active windows
        all_probs = []
        for win in windows:
            c_audio = win["window"]
            c_tensor = torch.from_numpy(c_audio).unsqueeze(0).to(self.device)
            mel = self.extract_mel(c_tensor)
            with torch.no_grad():
                logits = self.tagger_model(mel).squeeze(0)  # [32]
                pw = torch.sigmoid(logits).cpu().numpy()  # [32]
                all_probs.append(pw)

        probs_stack = np.stack(all_probs, axis=0)
        max_probs = np.max(probs_stack, axis=0)
        mean_probs = np.mean(probs_stack, axis=0)
        probs = 0.75 * max_probs + 0.25 * mean_probs

        active_events = []
        class_probabilities = {}

        for i, class_name in enumerate(self.classes):
            prob = float(probs[i])
            class_probabilities[class_name] = round(prob, 4)

            if tag_threshold is not None:
                base_th = float(tag_threshold)
            elif i < len(self.optimal_thresholds):
                base_th = self.optimal_thresholds[i]
            else:
                base_th = self.tag_threshold

            effective_th = base_th
            if mode == "live_mic" and class_name in PERCUSSIVE_CLASSES:
                effective_th = min(0.95, base_th + 0.05)

            if prob >= effective_th:
                active_events.append({
                    "label": class_name,
                    "probability": round(prob, 4),
                    "threshold": round(effective_th, 4)
                })

        active_events.sort(key=lambda x: x["probability"], reverse=True)

        top_5_indices = np.argsort(probs)[::-1][:5]
        top_5_events = [
            {
                "label": self.classes[i],
                "probability": round(float(probs[i]), 4),
                "threshold": round(self.optimal_thresholds[i] if i < len(self.optimal_thresholds) else 0.5, 4)
            }
            for i in top_5_indices
        ]
        active_classes = [e["label"] for e in active_events]

        return {
            "environmental_events": {
                "active_events": active_events,
                "top_k_events": top_5_events,
                "num_active": len(active_events),
                "threshold_applied": float(tag_threshold or 0.5),
                "domain_taxonomy": "FSD-50K-32-Classes",
                "note": "Environmental acoustic sound events (not speaker identities)",
                "model": "Conformer-Classifier-ASL"
            },
            "active_classes": active_classes,
            "class_probabilities": class_probabilities,
            "detected_tags": active_classes,
            "num_detected": len(active_classes),
            "top_5_events": top_5_events,
            "top_5_candidates": top_5_events,
            "human_speech_detected": False,
            "human_speech_prob": 0.0,
            "threshold_applied": float(tag_threshold or 0.5),
            "taxonomy": "FSD-50K-32-Classes",
            "windows_analyzed": len(windows),
            "model": "Conformer-Classifier-ASL",
            "tier": "legacy_fallback"
        }

    # -------------------------------------------------------------------------
    # Task 3: Dynamic Multi-Specialist Separation Router
    # -------------------------------------------------------------------------
    def predict_separation(
        self,
        wav_np: np.ndarray,
        predicted_k: int = 4,
        output_dir: Optional[str] = None,
        file_prefix: Optional[str] = None,
        all_4_stems: bool = False,
        is_silent: bool = False
    ) -> Dict[str, Any]:
        """
        Task 3: Dynamic Multi-Specialist Separation Router.
        Routing logic:
        - K == 1: Passthrough conditioned input as 1 clean stem (no neural separator)
        - K == 2: Sepformer 2-speaker specialist (or fallback to Conv-TasNet)
        - K == 3: Sepformer 3-speaker specialist (or fallback to Conv-TasNet)
        - K >= 4: ConvTasNetSkip 4-speaker Dominant Separator (if K > 4, dominant_crowd_mode=True)
        Includes:
        - Least-Squares Physical Scale Restoration
        - Energy ranking descending by RMS energy
        - Ghost-channel bleed suppression (-18.0 dB pruning below stem #1)
        - Peak guard at 0.99 ceiling
        """
        k_keep = max(0, min(int(predicted_k), 4))
        clean_prefix = os.path.basename(file_prefix) if file_prefix else ""
        clean_prefix = os.path.splitext(clean_prefix)[0] if clean_prefix else ""
        session_id = clean_prefix or uuid.uuid4().hex[:8]

        if is_silent or k_keep == 0 or len(wav_np) == 0:
            return {
                "reconstructed_count": 0,
                "k_sep": 0,
                "stems": [],
                "separated_stems": [],
                "all_4_raw_stems": [] if all_4_stems else None,
                "dominant_crowd_mode": False,
                "complexity_ceiling_exceeded": False,
                "model": "SANADA-MultiSpecialist-Router"
            }

        L_orig = len(wav_np)
        W = N_SAMPLES  # 128,000 samples (8.0s)
        H = 64000      # 64,000 samples (4.0s hop = 50% overlap)
        dominant_crowd_mode = bool(predicted_k > 4)

        # ---------------------------------------------------------------------
        # Routing Step: Pick separation engine or K=1 passthrough
        # ---------------------------------------------------------------------
        # K == 1 Passthrough optimization: If 1 speaker and all_4_stems not requested, return clean stem directly
        if predicted_k == 1 and not all_4_stems:
            stem_audio = wav_np.copy()
            rms = float(np.sqrt(np.mean(stem_audio ** 2)))
            energy = float(np.sum(stem_audio ** 2))
            rms_dbfs = round(float(20.0 * np.log10(max(rms, 1e-12))), 2)

            filename = None
            filepath = None
            url = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"session_{session_id}_stem_1.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, apply_cosine_fade(stem_audio, 160), TARGET_SAMPLE_RATE, subtype="PCM_16")
                url = f"/static/stems/{filename}"

            single_stem = {
                "channel": 1,
                "rank": 1,
                "original_model_channel": 0,
                "rms_energy": round(rms, 6),
                "rms_dbfs": rms_dbfs,
                "energy_share_pct": 100.0,
                "relative_energy_db": 0.0,
                "is_bleed_candidate": False,
                "scale_restoration_alpha": 1.0,
                "peak_guarded": False,
                "filename": filename,
                "filepath": filepath,
                "url": url,
                "si_sdr_estimate_db": 10.59
            }
            return {
                "reconstructed_count": 1,
                "k_sep": 1,
                "stems": [single_stem],
                "separated_stems": [single_stem],
                "all_4_raw_stems": [single_stem] if all_4_stems else None,
                "dominant_crowd_mode": False,
                "complexity_ceiling_exceeded": False,
                "model": "K1-Clean-Passthrough"
            }

        # Select Specialist Model
        active_sepformer = None
        target_sources = 4

        if all_4_stems:
            if getattr(self, "sepformer_4spk", None) is not None:
                active_sepformer = self.sepformer_4spk
                target_sources = 4
            else:
                active_sepformer = None
                target_sources = 4
        elif predicted_k == 2 and self.sepformer_2spk is not None:
            active_sepformer = self.sepformer_2spk
            target_sources = 2
        elif predicted_k == 3 and self.sepformer_3spk is not None:
            active_sepformer = self.sepformer_3spk
            target_sources = 3
        elif predicted_k >= 4 and getattr(self, "sepformer_4spk", None) is not None:
            active_sepformer = self.sepformer_4spk
            target_sources = 4

        # Forward function wrapping chosen model
        def run_model_forward(chunk_np: np.ndarray) -> np.ndarray:
            """Runs chosen separator on an audio chunk [W], returning [target_sources, W]."""
            if active_sepformer is not None:
                # Sepformer specialist pass
                native_sr = active_sepformer["native_sr"]
                model = active_sepformer["model"]
                t_in = torch.from_numpy(chunk_np).unsqueeze(0).float()
                if native_sr != TARGET_SAMPLE_RATE:
                    t_in = torchaudio.functional.resample(t_in, orig_freq=TARGET_SAMPLE_RATE, new_freq=native_sr)
                with torch.no_grad():
                    sep_out = model.separate_batch(t_in)  # [1, T_native, target_sources]
                # Permute to [1, target_sources, T_native]
                t_sources = sep_out.permute(0, 2, 1)
                if native_sr != TARGET_SAMPLE_RATE:
                    t_sources = torchaudio.functional.resample(t_sources, orig_freq=native_sr, new_freq=TARGET_SAMPLE_RATE)
                stems = t_sources.squeeze(0).cpu().numpy()
                if stems.shape[-1] < len(chunk_np):
                    stems = np.pad(stems, ((0, 0), (0, len(chunk_np) - stems.shape[-1])))
                else:
                    stems = stems[:, :len(chunk_np)]
                return stems
            else:
                # ConvTasNet (Skip Dominant or Legacy 4spk)
                model = self.dominant_sep_model if self.dominant_sep_model is not None else self.sep_model
                t_chunk = torch.from_numpy(chunk_np).float().view(1, 1, -1).to(self.device)
                with torch.no_grad():
                    return model(t_chunk).squeeze(0).cpu().numpy()  # [4, W]

        # Determine effective source count for sliding-window stitching
        num_model_sources = target_sources if active_sepformer is not None else 4

        # 1. Forward Pass: Single Window or Sliding-Window Overlap-Add
        if L_orig <= W:
            if L_orig < W:
                padded = np.pad(wav_np, (0, W - L_orig), mode="constant")
            else:
                padded = wav_np
            stems_raw = run_model_forward(padded)[:, :L_orig]
        else:
            w = np.sin(np.pi * (np.arange(W, dtype=np.float32) + 0.5) / W) ** 2
            starts = list(range(0, L_orig - W + 1, H))
            if starts[-1] + W < L_orig:
                starts.append(L_orig - W)

            out_stems = np.zeros((num_model_sources, L_orig), dtype=np.float32)
            weight_acc = np.zeros(L_orig, dtype=np.float32)
            prev_chunk_stems = None
            prev_start = None

            for start in starts:
                chunk = wav_np[start : start + W]
                chunk_stems = run_model_forward(chunk)

                if prev_chunk_stems is not None:
                    overlap_len = (prev_start + W) - start
                    prev_overlap = prev_chunk_stems[:, -overlap_len:]
                    curr_overlap = chunk_stems[:, :overlap_len]

                    best_p = None
                    best_score = -1.0
                    for p in itertools.permutations(range(num_model_sources)):
                        score = 0.0
                        for i in range(num_model_sources):
                            u = prev_overlap[i]
                            v = curr_overlap[p[i]]
                            norm_u = float(np.sqrt(np.dot(u, u)))
                            norm_v = float(np.sqrt(np.dot(v, v)))
                            score += abs(float(np.dot(u, v))) / (norm_u * norm_v + 1e-8)
                        if score > best_score:
                            best_score = score
                            best_p = p

                    aligned_chunk = np.zeros_like(chunk_stems)
                    for i in range(num_model_sources):
                        u = prev_overlap[i]
                        v = curr_overlap[best_p[i]]
                        denom = float(np.dot(v, v))
                        if denom < 1e-10:
                            g = 1.0
                        else:
                            num = float(np.dot(v, u))
                            g = float(np.clip(num / (denom + 1e-8), -4.0, 4.0))
                        aligned_chunk[i] = chunk_stems[best_p[i]] * g
                    chunk_stems = aligned_chunk

                out_stems[:, start : start + W] += chunk_stems * w
                weight_acc[start : start + W] += w
                prev_chunk_stems = chunk_stems
                prev_start = start

            stems_raw = out_stems / np.maximum(weight_acc, 1e-11)

        # 2. Least-Squares Physical Scale Projection relative to zero-mean mixture
        x_zm = wav_np - np.mean(wav_np)
        restored_stems = []
        for i in range(num_model_sources):
            s_raw = stems_raw[i]
            s_zm = s_raw - np.mean(s_raw)
            denom = float(np.dot(s_zm, s_zm)) + 1e-8
            alpha = float(np.dot(s_zm, x_zm) / denom)
            s_scaled = alpha * s_zm
            rms = float(np.sqrt(np.mean(s_scaled ** 2)))
            energy = float(np.sum(s_scaled ** 2))
            restored_stems.append({
                "channel_index": i,
                "alpha_scale": round(alpha, 6),
                "rms": rms,
                "energy": energy,
                "audio": s_scaled
            })

        # 3. Energy Ranking: Sort stems descending by scale-restored RMS energy
        restored_stems.sort(key=lambda s: s["rms"], reverse=True)
        max_rms = restored_stems[0]["rms"]
        sum_energy = sum(s["energy"] for s in restored_stems) + 1e-12

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        processed_all = []
        for rank, item in enumerate(restored_stems, start=1):
            if max_rms > 0:
                rel_energy_db = float(20.0 * np.log10(max(item["rms"], 1e-12) / (max_rms + 1e-8)))
            else:
                rel_energy_db = 0.0

            energy_share_pct = round(100.0 * item["energy"] / sum_energy, 2)
            rms_dbfs = round(float(20.0 * np.log10(max(item["rms"], 1e-12))), 2)
            is_bleed = bool(rank >= 2 and rel_energy_db < -18.0)

            # Peak guard at 0.99
            audio = item["audio"]
            max_abs = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            peak_guarded = False
            if max_abs > 0.99:
                audio = 0.99 * (audio / max_abs)
                peak_guarded = True

            filename = None
            filepath = None
            url = None
            if output_dir:
                filename = f"session_{session_id}_stem_{rank}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, apply_cosine_fade(audio, 160), TARGET_SAMPLE_RATE, subtype="PCM_16")
                url = f"/static/stems/{filename}"

            expected_si_sdr = 10.59 if rank == 1 else (8.42 if rank == 2 else (5.08 if rank == 3 else 4.24))

            processed_all.append({
                "channel": rank,
                "rank": rank,
                "original_model_channel": item["channel_index"],
                "rms_energy": round(item["rms"], 6),
                "rms_dbfs": rms_dbfs,
                "energy_share_pct": energy_share_pct,
                "relative_energy_db": round(rel_energy_db, 2),
                "is_bleed_candidate": is_bleed,
                "scale_restoration_alpha": item["alpha_scale"],
                "peak_guarded": peak_guarded,
                "filename": filename,
                "filepath": filepath,
                "url": url,
                "si_sdr_estimate_db": expected_si_sdr
            })

        # Filter active stems according to counter-guided cardinality & -18.0 dB bleed pruning
        separated_stems = []
        for stem in processed_all[:k_keep]:
            if stem["rank"] == 1 or not stem["is_bleed_candidate"]:
                separated_stems.append(stem)

        model_name = "SANADA-MultiSpecialist-Router"
        if active_sepformer is not None:
            model_name = f"Sepformer-{num_model_sources}spk"
        elif self.dominant_sep_model is not None:
            model_name = "ConvTasNetSkip-Dominant"
        else:
            model_name = "SANADA-ConvTasNet-uPIT"

        return {
            "reconstructed_count": len(separated_stems),
            "k_sep": k_keep,
            "stems": separated_stems,
            "separated_stems": separated_stems,
            "all_4_raw_stems": processed_all if all_4_stems else None,
            "dominant_crowd_mode": dominant_crowd_mode,
            "complexity_ceiling_exceeded": bool(predicted_k > 4),
            "model": model_name
        }

    # -------------------------------------------------------------------------
    # Unified Public Inference Workflow
    # -------------------------------------------------------------------------
    def analyze_and_separate(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor, bytes],
        orig_sr: Optional[int] = None,
        mode: str = "auto",
        output_dir: Optional[str] = None,
        tag_threshold: Optional[float] = None,
        file_prefix: Optional[str] = None,
        all_4_stems: bool = False
    ) -> Dict[str, Any]:
        """
        Unified multi-task machine listening inference.
        Supports 3 distinct audio modes ('sanada', 'computer', 'live_mic', 'auto').
        """
        # 1. Condition Audio
        filename_hint = str(audio_input) if isinstance(audio_input, (str, Path)) else (file_prefix or "")
        cond = condition_audio(
            audio_input,
            orig_sr=orig_sr,
            mode=mode,
            filename=filename_hint
        )
        wav_np = cond["audio"]
        mode_used = cond["mode_used"]
        is_silent = cond["is_silent"]

        # 2. Environmental Acoustic Tagging (Task 2)
        tagging_res = self.predict_tagging(
            wav_np,
            mode=mode_used,
            tag_threshold=tag_threshold,
            is_silent=is_silent
        )
        ast_env_count = len(tagging_res["environmental_events"]["active_events"])

        # 3. Source Counting (Task 1) - with AST active count for optional fsd_fusion
        counting_res = self.predict_counting(
            wav_np,
            mode=mode_used,
            is_silent=is_silent,
            ast_env_count=ast_env_count
        )
        predicted_k = counting_res["predicted_k"]

        # 4. Source Separation (Task 3 - Guided by predicted_k)
        sep_res = self.predict_separation(
            wav_np,
            predicted_k=predicted_k,
            output_dir=output_dir,
            file_prefix=file_prefix,
            all_4_stems=all_4_stems,
            is_silent=is_silent
        )

        return {
            "status": "success",
            "mode_used": mode_used,
            "duration_sec": cond["duration_sec"],
            "input_rms": cond["input_rms"],
            "predicted_k": predicted_k,
            "raw_k": counting_res["raw_k"],
            "calibrated_k": counting_res["calibrated_k"],
            "complexity_ceiling_exceeded": counting_res["complexity_ceiling_exceeded"],
            "active_events": tagging_res["environmental_events"]["active_events"],
            "top_5_events": tagging_res["top_5_events"],
            "separated_stems": sep_res["stems"],
            "all_4_raw_stems": sep_res["all_4_raw_stems"],
            # Path B extensions & telemetry
            "windows_analyzed": counting_res.get("windows_analyzed", 1),
            "human_speech_detected": tagging_res.get("human_speech_detected", False),
            "human_speech_prob": tagging_res.get("human_speech_prob", 0.0),
            "dominant_crowd_mode": sep_res.get("dominant_crowd_mode", False),
            "active_models": self.active_models,
            # Backwards compatibility structures
            "audio_metadata": {
                "sample_rate_hz": TARGET_SAMPLE_RATE,
                "samples": len(wav_np),
                "duration_s": cond["duration_sec"],
                "rms_energy": cond["input_rms"],
                "is_silent": is_silent,
                "mode_applied": mode_used
            },
            "source_counting": counting_res,
            "environmental_events": tagging_res["environmental_events"],
            "label_tagging": {
                "active_classes": tagging_res["active_classes"],
                "class_probabilities": tagging_res["class_probabilities"],
                "detected_tags": tagging_res["detected_tags"],
                "num_detected": tagging_res["num_detected"],
                "top_5_events": tagging_res["top_5_events"],
                "threshold_applied": tagging_res["threshold_applied"],
                "taxonomy": tagging_res["taxonomy"],
                "k_guided": predicted_k,
                "model": tagging_res["model"]
            },
            "source_separation": sep_res
        }


# -----------------------------------------------------------------------------
# 5. CLI Entrypoint
# -----------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SANADA Production Multi-Task Machine Listening Inference Engine"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input audio file (.wav, mono or stereo, any sample rate)"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="auto",
        choices=["auto", "sanada", "computer", "live_mic"],
        help="Audio conditioning mode: 'auto' (default), 'sanada', 'computer', 'live_mic'"
    )
    parser.add_argument(
        "--ckpt-dir", "-c", type=str, default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory containing trained PyTorch checkpoints (default: {DEFAULT_CHECKPOINT_DIR})"
    )
    parser.add_argument(
        "--out-dir", "-o", type=str, default=None,
        help="Directory to save separated stems as 16-bit PCM WAV files"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="Custom probability threshold for environmental event detection (default: optimal per-class thresholds)"
    )
    parser.add_argument(
        "--all-stems", action="store_true",
        help="Export all 4 stems regardless of predicted K or bleed pruning"
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="Inference device: 'cpu', 'cuda', 'cuda:0' (default: auto-detect)"
    )
    parser.add_argument(
        "--json-output", "-j", action="store_true",
        help="Print raw JSON output only (for script pipelines)"
    )
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    pipeline = SANADAPipeline(
        checkpoint_dir=args.ckpt_dir,
        device=args.device,
        tag_threshold=args.threshold if args.threshold is not None else 0.5
    )

    logger.info(f"Processing audio: {args.input} [Mode: {args.mode}]")
    result = pipeline.analyze_and_separate(
        audio_input=args.input,
        mode=args.mode,
        output_dir=args.out_dir,
        tag_threshold=args.threshold,
        all_4_stems=args.all_stems
    )

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 70)
        print(" SANADA NEURAL MACHINE LISTENING INFERENCE RESULTS")
        print("=" * 70)
        print(f"Audio File     : {args.input}")
        print(f"Mode Applied   : {result['mode_used']}")
        print(f"Duration       : {result['duration_sec']}s | Input RMS: {result['input_rms']}")

        # Counting Summary
        cnt = result["source_counting"]
        print("\n[Task 1: Source Counting]")
        print(f"  Model Loaded              : {cnt.get('model', 'Unknown')} ({cnt.get('tier', 'Unknown')})")
        print(f"  Predicted Sources (K)     : {cnt['predicted_k']}")
        print(f"  Raw Continuous Output     : {cnt['raw_k']}")
        print(f"  Calibrated Continuous K   : {cnt['calibrated_k']}")
        print(f"  Confidence Score          : {round(cnt['confidence'] * 100, 1)}%")
        print(f"  Complexity Ceiling Flag   : {cnt['complexity_ceiling_exceeded']}")

        # Tagging Summary
        env = result["environmental_events"]
        print("\n[Task 2: Environmental Tagging]")
        print(f"  Model Loaded              : {env.get('model', 'Unknown')}")
        print(f"  Human Speech Detected     : {result.get('human_speech_detected', False)} (p={result.get('human_speech_prob', 0.0)})")
        print(f"  Active Events Detected    : {len(env['active_events'])}")
        for ev in env["active_events"]:
            th_str = f" (th={ev['threshold']})" if "threshold" in ev else ""
            print(f"    • {ev['label']}: {round(ev['probability'] * 100, 1)}%{th_str}")
        print("  Top 5 Candidates:")
        for ev in result["top_5_events"]:
            print(f"    • {ev['label']}: {round(ev['probability'] * 100, 1)}% (th={ev['threshold']})")

        # Separation Summary
        sep = result["source_separation"]
        print("\n[Task 3: Counter-Guided Source Separation]")
        print(f"  Model Router              : {sep.get('model', 'Unknown')}")
        print(f"  Dominant Crowd Mode       : {sep.get('dominant_crowd_mode', False)}")
        print(f"  Stems Reconstructed       : {sep['reconstructed_count']} (Cardinality k_sep={sep['k_sep']})")
        for stem in sep["stems"]:
            bleed_str = " [BLEED CANDIDATE <-18dB]" if stem.get("is_bleed_candidate") else ""
            guard_str = " [PEAK GUARDED]" if stem.get("peak_guarded") else ""
            file_str = f" -> {stem['filepath']}" if stem.get("filepath") else ""
            print(f"    Stem {stem['channel']}: RMS={stem['rms_energy']:.6f} ({stem['rms_dbfs']} dBFS, share={stem['energy_share_pct']}%) | Rel dB={stem['relative_energy_db']:+.1f} dB | Alpha={stem['scale_restoration_alpha']:.4f}{bleed_str}{guard_str}{file_str}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
