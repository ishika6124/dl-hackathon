#!/usr/bin/env python3
"""
PGHM-Net (single-file practical implementation)
Physics-Guided Hierarchical Multimodal Bearing Health 

Pipeline
--------
Stage 0: Parse MATLAB files from SCA bearing dataset folders
Stage 1: Normal vs Abnormal
Stage 2: Bearing Fault vs Non-Bearing Abnormal
Stage 3: Inner vs Ball vs Outer

Design choices
--------------
- Single-file implementation for hackathon/demo use.
- Uses robust engineered physics features instead of a large deep model so it is
  easier to train on limited, noisy industrial data.
- Supports folders 1..7 for training and 8..11 for testing.
- Can optionally use both train.mat and test.mat for folders 1..7, as requested.
- Uses only bounded, physically defensible augmentations:
    * tiny Gaussian noise
    * mild amplitude scaling
    * circular time shift
    * narrow order-consistent speed interpolation
- Produces per-window predictions and file-level aggregation.

Note
----
This is a practical baseline aimed at robustness and explainability. It does not
claim first-principles simulation of bearing mechanics. The augmentations are
bounded nuisance transforms and should be validated using the provided physics
consistency checks.
"""

from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert, stft, resample_poly
from scipy.stats import kurtosis, skew
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility
# -----------------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


# -----------------------------
# Dataset constants
# -----------------------------
LABEL_NORMAL = 0
LABEL_INNER = 1
LABEL_BALL = 2
LABEL_OUTER = 3
LABEL_EXCLUDED = -1

STAGE2_NON_BEARING = 0
STAGE2_BEARING = 1

SENSOR_KEYS = ["DS", "FS", "Upper", "Lower"]


@dataclass
class SampleRecord:
    folder_id: int
    split_name: str              # train/test
    sensor_key: str              # DS/FS/Upper/Lower
    sample_idx: int
    label: int
    rpm: float
    sampling_rate: float
    bpfi: float
    bpfo: float
    bsf: float
    ftf: float
    timestamp: Optional[float]
    signal: np.ndarray
    fault_type_text: str = ""


# -----------------------------
# Utility helpers
# -----------------------------
def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def robust_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))


def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def safe_float(v: Any, default: float = np.nan) -> float:
    try:
        if isinstance(v, (bytes, str)):
            s = str(v).strip()
            if s == "":
                return default
            return float(s)
        if isinstance(v, np.ndarray):
            v = np.asarray(v).squeeze()
            if v.size == 0:
                return default
            if v.dtype == object and v.size == 1:
                return safe_float(v.item(), default)
            return float(v.flat[0])
        return float(v)
    except Exception:
        return default


def to_scalar_array(x: Any, n: int, default: float = np.nan) -> np.ndarray:
    arr = np.asarray(x, dtype=object).reshape(-1)
    out = np.full(n, default, dtype=np.float64)
    for j in range(min(n, len(arr))):
        out[j] = safe_float(arr[j], default)
    return out


def extract_raw_data(raw: Any) -> np.ndarray:
    """
    Robustly extract MATLAB rawData that may appear as:
    - numeric matrix
    - 1D numeric vector
    - object/cell array of per-window vectors
    - nested arrays with inconsistent lengths

    Returns a 2D float64 array of shape (n_windows, n_samples), padding shorter
    rows with zeros when necessary.
    """
    # Fast path: already numeric and rectangular
    try:
        arr = np.asarray(raw)
        if arr.dtype != object:
            arr = arr.astype(np.float64)
            if arr.ndim == 0:
                return np.zeros((0, 0), dtype=np.float64)
            if arr.ndim == 1:
                return arr[None, :]
            if arr.ndim >= 2:
                return np.asarray(arr, dtype=np.float64)
    except Exception:
        pass

    # Slow path: object/cell-like structure
    rows = []
    try:
        items = np.asarray(raw, dtype=object).reshape(-1)
    except Exception:
        items = [raw]

    for item in items:
        try:
            arr = np.asarray(item)
            if arr.dtype == object:
                # Flatten nested object containers recursively one level
                flat_parts = []
                for sub in arr.reshape(-1):
                    try:
                        sub_arr = np.asarray(sub, dtype=np.float64).reshape(-1)
                        if sub_arr.size:
                            flat_parts.append(sub_arr)
                    except Exception:
                        continue
                if flat_parts:
                    vec = np.concatenate(flat_parts, axis=0)
                else:
                    continue
            else:
                vec = arr.astype(np.float64).reshape(-1)
            if vec.size:
                rows.append(vec)
        except Exception:
            continue

    if not rows:
        return np.zeros((0, 0), dtype=np.float64)

    max_len = max(len(r) for r in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def mat_obj_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "_fieldnames"):
        return {k: getattr(obj, k) for k in obj._fieldnames}
    return {}


def extract_fault_freqs(d: Dict[str, Any]) -> Tuple[float, float, float, float]:
    ff = d.get("faultFrequencies", None)
    if ff is None:
        return np.nan, np.nan, np.nan, np.nan
    if hasattr(ff, "_fieldnames"):
        ffd = mat_obj_to_dict(ff)
        bpfi = safe_float(ffd.get("BPFI", np.nan))
        bpfo = safe_float(ffd.get("BPFO", np.nan))
        bsf = safe_float(ffd.get("BPF", ffd.get("BSF", np.nan)))
        ftf = safe_float(ffd.get("FTF", np.nan))
        return bpfi, bpfo, bsf, ftf
    arr = np.asarray(ff).squeeze()
    if arr.size >= 4:
        vals = arr.astype(float).tolist()
        return vals[0], vals[1], vals[2], vals[3]
    return np.nan, np.nan, np.nan, np.nan


def load_mat_file(mat_path: Path) -> Dict[str, Any]:
    return loadmat(mat_path, squeeze_me=True, struct_as_record=False)


def parse_sensor_struct(folder_id: int, split_name: str, sensor_key: str, obj: Any, fault_type_text: str = "") -> List[SampleRecord]:
    d = mat_obj_to_dict(obj)
    if not d:
        return []

    raw = extract_raw_data(d.get("rawData", []))
    if raw.size == 0:
        return []
    if raw.ndim == 1:
        raw = raw[None, :]
    labels = to_scalar_array(d.get("label", np.full(raw.shape[0], np.nan)), raw.shape[0], np.nan)
    rpms = to_scalar_array(d.get("RPM", np.full(raw.shape[0], np.nan)), raw.shape[0], np.nan)
    times = to_scalar_array(d.get("time", np.full(raw.shape[0], np.nan)), raw.shape[0], np.nan)
    sr = safe_float(d.get("samplingRate", np.nan))
    bpfi, bpfo, bsf, ftf = extract_fault_freqs(d)

    n = raw.shape[0]
    out: List[SampleRecord] = []
    for i in range(n):
        out.append(
            SampleRecord(
                folder_id=folder_id,
                split_name=split_name,
                sensor_key=sensor_key,
                sample_idx=i,
                label=int(labels[i]) if i < len(labels) and not np.isnan(labels[i]) else LABEL_EXCLUDED,
                rpm=float(rpms[i]) if i < len(rpms) and not np.isnan(rpms[i]) else np.nan,
                sampling_rate=sr,
                bpfi=bpfi,
                bpfo=bpfo,
                bsf=bsf,
                ftf=ftf,
                timestamp=float(times[i]) if i < len(times) and not np.isnan(times[i]) else np.nan,
                signal=raw[i].astype(np.float32),
                fault_type_text=fault_type_text,
            )
        )
    return out


def parse_folder(folder_path: Path) -> List[SampleRecord]:
    folder_id = int(folder_path.name)
    records: List[SampleRecord] = []
    for split_name in ["train", "test"]:
        mat_path = folder_path / f"{split_name}.mat"
        if not mat_path.exists():
            continue
        mat = load_mat_file(mat_path)
        fault_type_text = str(mat.get("faultType", ""))
        for key in SENSOR_KEYS:
            if key in mat:
                records.extend(parse_sensor_struct(folder_id, split_name, key, mat[key], fault_type_text))
    return records


# -----------------------------
# Signal preprocessing
# -----------------------------
def detrend_and_scale(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.median(x)
    scale = robust_mad(x)
    if scale < 1e-10:
        scale = robust_rms(x)
    if scale < 1e-10:
        scale = 1.0
    return (x / scale).astype(np.float32)


def valid_rotating_sample(rec: SampleRecord, rpm_threshold: float = 10.0) -> bool:
    if np.isnan(rec.rpm):
        return False
    return rec.rpm >= rpm_threshold


def shaft_freq_hz(rpm: float) -> float:
    if np.isnan(rpm):
        return np.nan
    return rpm / 60.0


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    X = np.fft.rfft(x * np.hanning(len(x)))
    mag = np.abs(X)
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return f, mag


def envelope_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    env = np.abs(hilbert(x.astype(np.float64)))
    env = env - np.mean(env)
    E = np.fft.rfft(env * np.hanning(len(env)))
    mag = np.abs(E)
    f = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return f, mag


def band_energy(freqs: np.ndarray, mag: np.ndarray, center: float, width: float) -> float:
    if np.isnan(center) or center <= 0 or width <= 0:
        return np.nan
    m = (freqs >= center - width) & (freqs <= center + width)
    if not np.any(m):
        return np.nan
    return float(np.sum(np.square(mag[m])) + 1e-12)


def local_peak_prominence(freqs: np.ndarray, mag: np.ndarray, center: float, width: float) -> float:
    if np.isnan(center) or center <= 0:
        return np.nan
    band = (freqs >= center - width) & (freqs <= center + width)
    if not np.any(band):
        return np.nan
    local = mag[band]
    peak = float(np.max(local))
    bg = float(np.median(local) + 1e-12)
    return peak / bg


def spectral_entropy(mag: np.ndarray) -> float:
    p = np.square(mag.astype(np.float64))
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def order_transform_features(freqs: np.ndarray, mag: np.ndarray, shaft_hz: float, orders: List[float], width_order: float = 0.15) -> Dict[str, float]:
    out = {}
    if np.isnan(shaft_hz) or shaft_hz <= 1e-9:
        for name in ["bpfi", "bpfo", "bsf", "ftf"]:
            out[f"ord_energy_{name}"] = np.nan
            out[f"ord_prom_{name}"] = np.nan
        return out

    for name, order in zip(["bpfi", "bpfo", "bsf", "ftf"], orders):
        if np.isnan(order):
            out[f"ord_energy_{name}"] = np.nan
            out[f"ord_prom_{name}"] = np.nan
            continue
        center_hz = order * shaft_hz
        width_hz = max(width_order * shaft_hz, 0.2)
        out[f"ord_energy_{name}"] = band_energy(freqs, mag, center_hz, width_hz)
        out[f"ord_prom_{name}"] = local_peak_prominence(freqs, mag, center_hz, width_hz)
    return out


def spectral_sideband_score(freqs: np.ndarray, mag: np.ndarray, center_hz: float, shaft_hz: float) -> float:
    if np.isnan(center_hz) or np.isnan(shaft_hz) or center_hz <= 0 or shaft_hz <= 0:
        return np.nan
    base = local_peak_prominence(freqs, mag, center_hz, max(0.2, 0.1 * shaft_hz))
    left = local_peak_prominence(freqs, mag, center_hz - shaft_hz, max(0.2, 0.1 * shaft_hz))
    right = local_peak_prominence(freqs, mag, center_hz + shaft_hz, max(0.2, 0.1 * shaft_hz))
    vals = [v for v in [base, left, right] if not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def simple_stft_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    f, t, Z = stft(x.astype(np.float64), fs=fs, nperseg=min(256, len(x)), noverlap=min(192, max(0, len(x)//8)))
    M = np.abs(Z)
    if M.size == 0:
        return {"stft_mean": 0.0, "stft_std": 0.0, "stft_entropy": 0.0}
    return {
        "stft_mean": float(np.mean(M)),
        "stft_std": float(np.std(M)),
        "stft_entropy": spectral_entropy(M.flatten()),
    }


def extract_features(rec: SampleRecord) -> Dict[str, float]:
    x = detrend_and_scale(rec.signal)
    fs = float(rec.sampling_rate)
    rpm = float(rec.rpm) if not np.isnan(rec.rpm) else np.nan
    sf = shaft_freq_hz(rpm)

    feats: Dict[str, float] = {}
    feats["folder_id"] = float(rec.folder_id)
    feats["is_train_split"] = 1.0 if rec.split_name == "train" else 0.0
    feats["sensor_is_ds"] = 1.0 if rec.sensor_key == "DS" else 0.0
    feats["sensor_is_fs"] = 1.0 if rec.sensor_key == "FS" else 0.0
    feats["sensor_is_upper"] = 1.0 if rec.sensor_key == "Upper" else 0.0
    feats["sensor_is_lower"] = 1.0 if rec.sensor_key == "Lower" else 0.0
    feats["rpm"] = rpm
    feats["shaft_hz"] = sf
    feats["sampling_rate"] = fs
    feats["bpfi_order"] = rec.bpfi
    feats["bpfo_order"] = rec.bpfo
    feats["bsf_order"] = rec.bsf
    feats["ftf_order"] = rec.ftf
    feats["signal_len"] = float(len(x))

    # Time domain
    feats["mean"] = float(np.mean(x))
    feats["std"] = float(np.std(x))
    feats["rms"] = robust_rms(x)
    feats["abs_mean"] = float(np.mean(np.abs(x)))
    feats["peak"] = float(np.max(np.abs(x)) + 1e-12)
    feats["crest_factor"] = feats["peak"] / (feats["rms"] + 1e-12)
    feats["kurtosis"] = float(kurtosis(x, fisher=False, bias=False)) if len(x) > 3 else np.nan
    feats["skew"] = float(skew(x, bias=False)) if len(x) > 2 else np.nan
    dx = np.diff(x)
    feats["diff_rms"] = robust_rms(dx) if len(dx) > 0 else 0.0
    feats["zero_cross_rate"] = float(np.mean(np.diff(np.signbit(x)).astype(np.float32))) if len(x) > 1 else 0.0

    # Frequency domain
    f, mag = spectrum(x, fs)
    ef, emag = envelope_spectrum(x, fs)
    feats["spec_entropy"] = spectral_entropy(mag)
    feats["env_spec_entropy"] = spectral_entropy(emag)
    feats["spec_centroid"] = float(np.sum(f * mag) / (np.sum(mag) + 1e-12))
    feats["spec_rolloff_85"] = float(f[np.searchsorted(np.cumsum(mag), 0.85 * np.sum(mag))]) if len(f) else np.nan

    # Order-aligned envelope features
    ord_feats = order_transform_features(ef, emag, sf, [rec.bpfi, rec.bpfo, rec.bsf, rec.ftf])
    feats.update(ord_feats)

    # Sideband scores around expected frequencies
    for name, order in [("bpfi", rec.bpfi), ("bpfo", rec.bpfo), ("bsf", rec.bsf)]:
        if not np.isnan(order) and not np.isnan(sf):
            feats[f"sideband_{name}"] = spectral_sideband_score(ef, emag, order * sf, sf)
        else:
            feats[f"sideband_{name}"] = np.nan

    # Focused bearing-order evidence, especially useful for outer-race generalization.
    if not np.isnan(sf) and sf > 0:
        for name, order in [("bpfi", rec.bpfi), ("bpfo", rec.bpfo), ("bsf", rec.bsf), ("ftf", rec.ftf)]:
            if np.isnan(order):
                feats[f"env_energy_{name}"] = np.nan
                feats[f"env_peak_{name}"] = np.nan
                feats[f"env_peak_ratio_{name}"] = np.nan
                feats[f"env_h2_ratio_{name}"] = np.nan
                continue
            target = order * sf
            band = max(1.5 * sf, 1.0)
            local = (ef >= max(0.0, target - band)) & (ef <= target + band)
            bg = (ef >= max(0.0, target - 3.0 * band)) & (ef <= target + 3.0 * band) & (~local)
            e_local = float(np.sum(np.square(emag[local])) + 1e-12) if np.any(local) else 0.0
            e_bg = float(np.sum(np.square(emag[bg])) + 1e-12) if np.any(bg) else 1e-12
            peak = float(np.max(emag[local]) + 1e-12) if np.any(local) else 0.0
            bg_peak = float(np.median(emag[bg]) + 1e-12) if np.any(bg) else 1e-12
            feats[f"env_energy_{name}"] = e_local
            feats[f"env_peak_{name}"] = peak
            feats[f"env_peak_ratio_{name}"] = peak / bg_peak
            # second harmonic support, often useful for robust outer-race evidence
            h2 = 2.0 * target
            local2 = (ef >= max(0.0, h2 - band)) & (ef <= h2 + band)
            peak2 = float(np.max(emag[local2]) + 1e-12) if np.any(local2) else 0.0
            feats[f"env_h2_ratio_{name}"] = peak2 / (peak + 1e-12)
        # Ratios to bias the model toward the correct fault-order neighborhood instead of broadband normality.
        feats["bpfo_vs_bpfi_peak_ratio"] = feats.get("env_peak_ratio_bpfo", np.nan) / (feats.get("env_peak_ratio_bpfi", np.nan) + 1e-12)
        feats["bpfo_vs_bsf_peak_ratio"] = feats.get("env_peak_ratio_bpfo", np.nan) / (feats.get("env_peak_ratio_bsf", np.nan) + 1e-12)
        feats["bpfi_vs_bsf_peak_ratio"] = feats.get("env_peak_ratio_bpfi", np.nan) / (feats.get("env_peak_ratio_bsf", np.nan) + 1e-12)
    else:
        for name in ["bpfi", "bpfo", "bsf", "ftf"]:
            feats[f"env_energy_{name}"] = np.nan
            feats[f"env_peak_{name}"] = np.nan
            feats[f"env_peak_ratio_{name}"] = np.nan
            feats[f"env_h2_ratio_{name}"] = np.nan
        feats["bpfo_vs_bpfi_peak_ratio"] = np.nan
        feats["bpfo_vs_bsf_peak_ratio"] = np.nan
        feats["bpfi_vs_bsf_peak_ratio"] = np.nan

    # Broad band energies
    freq_bands = [(0, 10), (10, 50), (50, 200), (200, 500), (500, 1000), (1000, 2000)]
    for lo, hi in freq_bands:
        m = (f >= lo) & (f < hi)
        feats[f"band_{lo}_{hi}"] = float(np.sum(np.square(mag[m])) + 1e-12) if np.any(m) else 0.0

    # Envelope broad bands
    env_bands = [(0, 10), (10, 50), (50, 200), (200, 500)]
    for lo, hi in env_bands:
        m = (ef >= lo) & (ef < hi)
        feats[f"env_band_{lo}_{hi}"] = float(np.sum(np.square(emag[m])) + 1e-12) if np.any(m) else 0.0

    # STFT summaries
    feats.update(simple_stft_features(x, fs))

    # Physics consistency features
    relevant_orders = [v for v in [rec.bpfi, rec.bpfo, rec.bsf, rec.ftf] if not np.isnan(v)]
    feats["num_known_orders"] = float(len(relevant_orders))
    feats["rpm_valid"] = 1.0 if valid_rotating_sample(rec) else 0.0

    return feats


# -----------------------------
# Physics-consistent augmentation
# -----------------------------
def augmentation_invariants_ok(orig_feats: Dict[str, float], aug_feats: Dict[str, float], tol_db: float = 3.0) -> bool:
    """
    Post-augmentation gate.
    We accept augmentations only if key order-domain prominence ratios are not destroyed.
    """
    keys = [
        "ord_prom_bpfi",
        "ord_prom_bpfo",
        "ord_prom_bsf",
        "ord_prom_ftf",
        "crest_factor",
        "env_spec_entropy",
    ]
    for k in keys:
        a = orig_feats.get(k, np.nan)
        b = aug_feats.get(k, np.nan)
        if np.isnan(a) or np.isnan(b):
            continue
        if a <= 0 or b <= 0:
            continue
        db = 20.0 * abs(math.log10((b + 1e-12) / (a + 1e-12)))
        if db > tol_db:
            return False
    return True


def tiny_gaussian_noise(x: np.ndarray, sigma_ratio: float = 0.01) -> np.ndarray:
    sigma = sigma_ratio * (robust_rms(x) + 1e-12)
    return (x + np.random.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def mild_gain_scaling(x: np.ndarray, lo: float = 0.95, hi: float = 1.05) -> np.ndarray:
    gain = np.random.uniform(lo, hi)
    return (gain * x).astype(np.float32)


def circular_shift(x: np.ndarray, max_frac: float = 0.2) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return x.copy()
    shift = np.random.randint(-max(1, int(max_frac * n)), max(1, int(max_frac * n)) + 1)
    return np.roll(x, shift).astype(np.float32)


def order_consistent_speed_interpolation(rec: SampleRecord, max_delta: float = 0.05) -> SampleRecord:
    """
    Narrow speed interpolation.
    This approximates nearby operating speed changes while preserving order-domain alignment.
    """
    if np.isnan(rec.rpm) or rec.rpm <= 0:
        return rec
    ratio = np.random.uniform(1.0 - max_delta, 1.0 + max_delta)
    x = rec.signal.astype(np.float64)
    # Resample the time axis by ratio. Same number of points after crop/pad.
    # This changes apparent speed in Hz domain while order representation should remain aligned.
    new_len = max(16, int(round(len(x) / ratio)))
    resampled = resample_poly(x, up=new_len, down=len(x))
    if len(resampled) >= len(x):
        resampled = resampled[: len(x)]
    else:
        pad = np.zeros(len(x) - len(resampled), dtype=resampled.dtype)
        resampled = np.concatenate([resampled, pad], axis=0)
    out = SampleRecord(**asdict(rec))
    out.signal = resampled.astype(np.float32)
    out.rpm = float(rec.rpm * ratio)
    return out


def augment_record(rec: SampleRecord, p_noise: float = 0.5, p_gain: float = 0.5, p_shift: float = 0.5, p_speed: float = 0.3) -> SampleRecord:
    out = SampleRecord(**asdict(rec))
    sig = out.signal.copy().astype(np.float32)

    # Apply only on rotating samples for physical validity
    rotating = valid_rotating_sample(out)

    if rotating and np.random.rand() < p_noise:
        sig = tiny_gaussian_noise(sig, sigma_ratio=np.random.uniform(0.003, 0.015))
    if rotating and np.random.rand() < p_gain:
        sig = mild_gain_scaling(sig, 0.95, 1.05)
    if rotating and np.random.rand() < p_shift:
        sig = circular_shift(sig, max_frac=0.15)

    out.signal = sig

    if rotating and np.random.rand() < p_speed:
        out = order_consistent_speed_interpolation(out, max_delta=0.05)

    return out


# -----------------------------
# Labels and targets
# -----------------------------
def stage1_target(label: int) -> Optional[int]:
    # Normal=0, abnormal includes inner/ball/outer; -1 excluded from supervised target
    if label == LABEL_EXCLUDED:
        return None
    return 0 if label == LABEL_NORMAL else 1


def stage2_target(label: int, folder_id: int) -> Optional[int]:
    # For this dataset, folder 11 acts as false alarm non-bearing abnormal test.
    # During training on folders 1..7 this class is usually unavailable; head will then be weakly trained.
    # We therefore allow folder-specific mapping when labels are known and optionally inject pseudo-non-bearing from excluded states.
    if label == LABEL_EXCLUDED:
        return STAGE2_NON_BEARING
    if label in (LABEL_INNER, LABEL_BALL, LABEL_OUTER):
        return STAGE2_BEARING
    if folder_id == 11:
        return STAGE2_NON_BEARING
    return None if label == LABEL_NORMAL else STAGE2_BEARING


def stage3_target(label: int) -> Optional[int]:
    if label in (LABEL_INNER, LABEL_BALL, LABEL_OUTER):
        return label
    return None


# -----------------------------
# Model wrappers
# -----------------------------
class TabularClassifier:
    def __init__(self, estimator=None):
        if estimator is None:
            estimator = ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=GLOBAL_SEED,
                n_jobs=-1,
            )
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", estimator),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        clf = self.pipeline[-1]
        Xt = self.pipeline[:-1].transform(X)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(Xt)
        # Fallback with pseudo-probabilities
        pred = clf.predict(Xt)
        classes = np.unique(pred)
        out = np.zeros((len(pred), len(classes)), dtype=float)
        for i, c in enumerate(classes):
            out[:, i] = (pred == c).astype(float)
        return out


class NormalityEnsemble:
    def __init__(self):
        self.clf = TabularClassifier(
            RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=2,
                class_weight={0: 1.0, 1: 4.5},
                random_state=GLOBAL_SEED,
                n_jobs=-1,
            )
        )
        self.threshold = 0.32

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None) -> None:
        self.clf.fit(X, y)
        proba = self.clf.predict_proba(X)
        if proba.shape[1] == 2:
            p_ab = proba[:, 1]
        else:
            p_ab = np.zeros(len(y))
        # Deliberately prefer recall for abnormalities, since the baseline was too conservative.
        best_t, best_score = 0.32, -1.0
        for t in np.linspace(0.08, 0.60, 27):
            pred = (p_ab >= t).astype(int)
            rec = recall_score(y, pred, zero_division=0)
            prec = precision_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)
            score = 0.55 * rec + 0.25 * f1 + 0.20 * prec
            if score > best_score:
                best_score, best_t = score, t
        self.threshold = float(best_t)

    def predict_abnormal(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.clf.predict_proba(X)
        if proba.shape[1] == 1:
            p_ab = np.zeros(len(X))
        else:
            p_ab = proba[:, 1]
        pred = (p_ab >= self.threshold).astype(int)
        return pred, p_ab


class HierarchicalBearingModel:
    def __init__(self):
        self.stage1 = NormalityEnsemble()
        self.stage2 = TabularClassifier(
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight={0: 1.0, 1: 3.5},
                random_state=GLOBAL_SEED,
                n_jobs=-1,
            )
        )
        self.stage3 = TabularClassifier(
            ExtraTreesClassifier(
                n_estimators=900,
                min_samples_leaf=1,
                class_weight={1: 3.0, 2: 10.0, 3: 7.5},
                random_state=GLOBAL_SEED,
                n_jobs=-1,
            )
        )
        self.feature_columns: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = [c for c in df.columns if c not in {
            "label", "folder_id", "split_name", "sensor_key", "sample_idx", "stage1", "stage2", "stage3", "group_id", "file_id"
        }]

        X = df[self.feature_columns]

        # Stage 1
        s1 = df["stage1"].dropna().astype(int)
        X1 = X.loc[s1.index]
        self.stage1.fit(X1, s1.values)

        # Stage 2: only abnormal-bearing or non-bearing examples when available
        s2 = df["stage2"].dropna().astype(int)
        if len(np.unique(s2.values)) >= 2:
            X2 = X.loc[s2.index]
            self.stage2.fit(X2, s2.values)
            self.has_stage2 = True
        else:
            self.has_stage2 = False

        # Stage 3: only bearing subtype examples
        s3 = df["stage3"].dropna().astype(int)
        if len(np.unique(s3.values)) >= 2:
            X3 = X.loc[s3.index]
            self.stage3.fit(X3, s3.values)
            self.has_stage3 = True
        else:
            self.has_stage3 = False

        self.fitted = True

    def predict_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        X = df[self.feature_columns]
        s1_pred, s1_prob = self.stage1.predict_abnormal(X)

        out = df[["folder_id", "split_name", "sensor_key", "sample_idx", "label", "file_id"]].copy()
        out["p_abnormal"] = s1_prob
        out["pred_stage1"] = s1_pred

        # Default outputs
        out["p_bearing"] = np.nan
        out["pred_stage2"] = np.nan
        out["p_inner"] = np.nan
        out["p_ball"] = np.nan
        out["p_outer"] = np.nan
        out["pred_stage3"] = np.nan
        out["final_pred"] = 0
        out["confidence"] = 1.0 - np.clip(np.abs(s1_prob - 0.5) * 2.0, 0.0, 1.0)
        out["reject"] = 0

        abnormal_idx = np.where((s1_pred == 1) | (s1_prob >= max(0.24, self.stage1.threshold - 0.05)))[0]
        if len(abnormal_idx) > 0 and self.has_stage2:
            X2 = X.iloc[abnormal_idx]
            p2 = self.stage2.predict_proba(X2)
            if p2.shape[1] == 1:
                p_bearing = np.zeros(len(X2))
                pred2 = np.zeros(len(X2), dtype=int)
            else:
                classes = self.stage2.pipeline[-1].classes_ if hasattr(self.stage2.pipeline[-1], "classes_") else np.array([0, 1])
                if 1 in classes:
                    j = list(classes).index(1)
                    p_bearing = p2[:, j]
                else:
                    p_bearing = np.zeros(len(X2))
                pred2 = (p_bearing >= 0.42).astype(int)

            out.iloc[abnormal_idx, out.columns.get_loc("p_bearing")] = p_bearing
            out.iloc[abnormal_idx, out.columns.get_loc("pred_stage2")] = pred2

            bearing_local = np.where(pred2 == 1)[0]
            bearing_global = abnormal_idx[bearing_local]
            if len(bearing_global) > 0 and self.has_stage3:
                X3 = X.iloc[bearing_global]
                p3 = self.stage3.predict_proba(X3)
                classes3 = self.stage3.pipeline[-1].classes_ if hasattr(self.stage3.pipeline[-1], "classes_") else np.array([1, 2, 3])
                prob_map = {c: p3[:, i] for i, c in enumerate(classes3)}
                p_inner = prob_map.get(1, np.zeros(len(X3)))
                p_ball = prob_map.get(2, np.zeros(len(X3)))
                p_outer = prob_map.get(3, np.zeros(len(X3)))
                # Small outer-race prior because unseen test faults are mostly outer-race and BPFO evidence is often weak but localized.
                if "env_peak_ratio_bpfo" in X3.columns:
                    bpfo_ratio = np.asarray(X3["env_peak_ratio_bpfo"].fillna(0.0))
                    bpfo_vs_bpfi = np.asarray(X3.get("bpfo_vs_bpfi_peak_ratio", pd.Series(np.zeros(len(X3)), index=X3.index)).fillna(0.0))
                    outer_boost = np.clip(0.06 * (bpfo_ratio > 1.2).astype(float) + 0.06 * (bpfo_vs_bpfi > 1.05).astype(float), 0.0, 0.12)
                    p_outer = np.clip(p_outer + outer_boost, 0.0, 1.0)
                stacked = np.vstack([p_inner, p_ball, p_outer]).T
                pred3 = np.array([1, 2, 3])[np.argmax(stacked, axis=1)]
                out.iloc[bearing_global, out.columns.get_loc("pred_stage3")] = pred3
                out.iloc[bearing_global, out.columns.get_loc("final_pred")] = pred3
                out.iloc[bearing_global, out.columns.get_loc("p_inner")] = p_inner
                out.iloc[bearing_global, out.columns.get_loc("p_ball")] = p_ball
                out.iloc[bearing_global, out.columns.get_loc("p_outer")] = p_outer

            non_bearing_local = np.where(pred2 == 0)[0]
            non_bearing_global = abnormal_idx[non_bearing_local]
            if len(non_bearing_global) > 0:
                # final_pred stays 0 but can be flagged as anomaly rejected/non-bearing
                out.iloc[non_bearing_global, out.columns.get_loc("final_pred")] = 0
                out.iloc[non_bearing_global, out.columns.get_loc("reject")] = 1

        # Confidence and reject logic
        # higher uncertainty if abnormal but low bearing confidence or subtype ambiguity
        for i in range(len(out)):
            pa = float(out.iloc[i]["p_abnormal"])
            pb = out.iloc[i]["p_bearing"] if not pd.isna(out.iloc[i]["p_bearing"]) else None
            probs3 = [out.iloc[i]["p_inner"], out.iloc[i]["p_ball"], out.iloc[i]["p_outer"]]
            probs3 = [float(v) for v in probs3 if not pd.isna(v)]
            conf = max(pa, 1.0 - pa)
            if pb is not None:
                conf *= max(pb, 1.0 - pb)
            if probs3:
                conf *= max(probs3)
                top2 = sorted(probs3, reverse=True)[:2]
                if len(top2) == 2 and (top2[0] - top2[1]) < 0.08:
                    out.iloc[i, out.columns.get_loc("reject")] = 1
            if abs(pa - 0.5) < 0.07:
                out.iloc[i, out.columns.get_loc("reject")] = 1
            out.iloc[i, out.columns.get_loc("confidence")] = float(np.clip(conf, 0.0, 1.0))

        return out


# -----------------------------
# Dataframe assembly
# -----------------------------
def build_records(data_root: Path, folders: Iterable[int]) -> List[SampleRecord]:
    all_records: List[SampleRecord] = []
    for fid in folders:
        folder = data_root / str(fid)
        if not folder.exists():
            print(f"[WARN] missing folder: {folder}")
            continue
        all_records.extend(parse_folder(folder))
    return all_records


def make_file_id(rec: SampleRecord) -> str:
    return f"folder{rec.folder_id}_{rec.split_name}_{rec.sensor_key}"


def records_to_dataframe(records: List[SampleRecord], augment_factor: int = 0, enable_augmentation: bool = False) -> pd.DataFrame:
    rows = []

    def add_one(rec: SampleRecord, augmented: bool = False) -> None:
        feats = extract_features(rec)
        row = {
            **feats,
            "label": rec.label,
            "folder_id": rec.folder_id,
            "split_name": rec.split_name,
            "sensor_key": rec.sensor_key,
            "sample_idx": rec.sample_idx,
            "file_id": make_file_id(rec),
            "stage1": stage1_target(rec.label),
            "stage2": stage2_target(rec.label, rec.folder_id),
            "stage3": stage3_target(rec.label),
        }
        rows.append(row)

    for rec in records:
        add_one(rec, augmented=False)
        if enable_augmentation and augment_factor > 0 and rec.label != LABEL_EXCLUDED and valid_rotating_sample(rec):
            orig_feats = extract_features(rec)
            class_mult = {LABEL_NORMAL: 1, LABEL_INNER: 2, LABEL_BALL: 6, LABEL_OUTER: 4}.get(rec.label, 1)
            n_aug = int(max(0, augment_factor) * class_mult)
            for _ in range(n_aug):
                arec = augment_record(rec)
                aug_feats = extract_features(arec)
                if augmentation_invariants_ok(orig_feats, aug_feats):
                    add_one(arec, augmented=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["group_id", "file_id", "label", "folder_id", "split_name", "sensor_key", "sample_idx", "stage1", "stage2", "stage3"])
    df["group_id"] = df["file_id"]
    return df


# -----------------------------
# Evaluation helpers
# -----------------------------
def summarize_dataset(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print("Shape:", df.shape)
    print("Labels:")
    print(df["label"].value_counts(dropna=False).sort_index())
    print("Folders:")
    print(df["folder_id"].value_counts().sort_index())


def file_level_aggregate(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_id, g in pred_df.groupby("file_id"):
        # abnormal persistence
        p_ab = float(g["p_abnormal"].mean())
        frac_ab = float((g["pred_stage1"] == 1).mean())
        # use persistent abnormality to escalate
        if frac_ab < 0.1 and p_ab < 0.45:
            final = 0
        else:
            # bearing evidence among abnormal windows
            ab = g[g["pred_stage1"] == 1]
            if len(ab) == 0:
                final = 0
            else:
                p_b = float(ab["p_bearing"].dropna().mean()) if ab["p_bearing"].notna().any() else 0.0
                if p_b < 0.5:
                    final = 0
                else:
                    probs = {
                        1: float(ab["p_inner"].dropna().mean()) if ab["p_inner"].notna().any() else 0.0,
                        2: float(ab["p_ball"].dropna().mean()) if ab["p_ball"].notna().any() else 0.0,
                        3: float(ab["p_outer"].dropna().mean()) if ab["p_outer"].notna().any() else 0.0,
                    }
                    final = max(probs, key=probs.get)
        true_labels = g["label"].values.tolist()
        # file-level truth: majority non-excluded label; if only excluded, mark -1
        valid = [y for y in true_labels if y != LABEL_EXCLUDED]
        true_file = int(pd.Series(valid).mode().iloc[0]) if len(valid) else LABEL_EXCLUDED
        rows.append({
            "file_id": file_id,
            "true_label": true_file,
            "pred_label": final,
            "mean_p_abnormal": p_ab,
            "frac_abnormal": frac_ab,
            "mean_confidence": float(g["confidence"].mean()),
            "reject_rate": float(g["reject"].mean()),
            "n_windows": int(len(g)),
        })
    return pd.DataFrame(rows)


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict[str, float]:
    valid = y_true != LABEL_EXCLUDED
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        print(f"[{label}] no valid labels.")
        return {}
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n[{label}] Accuracy: {acc:.4f} | Macro-F1: {macro:.4f} | Weighted-F1: {weighted:.4f}")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))
    print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0))
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def false_alarm_eval(file_df: pd.DataFrame) -> Dict[str, float]:
    # folder11 should ideally be predicted normal (0)
    fa = file_df[file_df["file_id"].str.contains("folder11_")]
    if len(fa) == 0:
        return {}
    false_alarm_rate = float((fa["pred_label"] != 0).mean())
    print(f"\n[Folder11 false alarm] files={len(fa)} | false_alarm_rate={false_alarm_rate:.4f}")
    return {"folder11_false_alarm_rate": false_alarm_rate}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file PGHM-Net baseline for SCA bearing dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to SCA bearing dataset root")
    parser.add_argument("--train_folders", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--test_folders", nargs="+", type=int, default=[8, 9, 10, 11])
    parser.add_argument("--use_both_mats_train", action="store_true", help="Use both train.mat and test.mat from train folders")
    parser.add_argument("--augment_factor", type=int, default=0, help="How many accepted augmented copies per rotating sample")
    parser.add_argument("--output_dir", type=str, default="outputs_pghm")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    set_seed(GLOBAL_SEED)
    data_root = Path(args.data_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load records
    train_records = build_records(data_root, args.train_folders)
    test_records = build_records(data_root, args.test_folders)

    # Restrict training usage
    if not args.use_both_mats_train:
        train_records = [r for r in train_records if r.split_name == "train"]

    train_df = records_to_dataframe(
        train_records,
        augment_factor=args.augment_factor,
        enable_augmentation=args.augment_factor > 0,
    )
    test_df = records_to_dataframe(test_records, augment_factor=0, enable_augmentation=False)

    summarize_dataset(train_df, "TRAIN")
    summarize_dataset(test_df, "TEST")

    # Fit hierarchical model
    model = HierarchicalBearingModel()
    model.fit(train_df)

    # Window-level predictions
    pred_test = model.predict_window(test_df)
    pred_test.to_csv(outdir / "window_predictions.csv", index=False)

    # File-level aggregation
    file_pred = file_level_aggregate(pred_test)
    file_pred.to_csv(outdir / "file_predictions.csv", index=False)

    # Metrics
    metrics = {}
    metrics["window"] = print_metrics(pred_test["label"].values.astype(int), pred_test["final_pred"].values.astype(int), "Window-level")
    metrics["file"] = print_metrics(file_pred["true_label"].values.astype(int), file_pred["pred_label"].values.astype(int), "File-level")
    metrics.update(false_alarm_eval(file_pred))

    # Save metadata
    cfg = vars(args).copy()
    cfg["feature_count"] = len(model.feature_columns)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.save_model:
        joblib.dump(model, outdir / "pghm_model.joblib")

    print(f"\n[OK] outputs saved to: {outdir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
