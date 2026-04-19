"""
Train on Folders 1-7  →  Test on Folders 8-11
==============================================
• Loads raw .mat files directly (no pre-processed .npy needed)
• Training data  : folders 1-7  (both train.mat + test.mat)
• Test data      : folders 8-11 (both train.mat + test.mat)
• Architecture   : same PhysicsHybridNet from best-physics.py
• Outputs        : ./outputs_1to7/
"""

import os, sys, json, warnings
import numpy as np
import scipy.io as sio
import scipy.signal as sp_signal
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              accuracy_score, roc_curve, auc,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_dataset():
    """Try several candidate paths so the script works regardless of where it lives."""
    candidates = [
        os.path.join(_HERE, 'SCA_bearing_dataset'),                          # script beside dataset/
        os.path.join(_HERE, 'SCA_bearing_dataset', 'SCA_bearing_dataset'),   # nested
        os.path.join(_HERE, '..', 'SCA_bearing_dataset', 'SCA_bearing_dataset'),  # sibling dir
        os.path.join(_HERE, '..', 'SCA_bearing_dataset'),
        os.path.join(os.getcwd(), 'SCA_bearing_dataset'),
        os.path.join(os.getcwd(), 'SCA_bearing_dataset', 'SCA_bearing_dataset'),
    ]
    for p in candidates:
        if os.path.isdir(os.path.join(p, '1')):   # folder 1 must exist inside
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Cannot find SCA bearing dataset. Tried:\n" + "\n".join(candidates) +
        "\nSet DATASET_PATH manually at the top of this script."
    )

DATASET_PATH = _find_dataset()
print(f"Dataset path: {DATASET_PATH}")

OUTPUT_DIR = os.path.join(_HERE, 'outputs_1to7')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FOLDERS = list(range(1, 8))   # 1-7
TEST_FOLDERS  = list(range(8, 12))  # 8-11

TARGET_LEN    = 4096
EXCLUDE_LABEL = -1
ASSET_TYPES   = ['Roller', 'Engine', 'Pump', 'Agitator', 'Strainer']
SENSOR_KEYS   = ['DS', 'FS', 'Upper', 'Lower']

BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-3
DROPOUT       = 0.4
PROJ_DIM      = 48
N_CLASSES     = 4
SUPCON_TEMP   = 0.07
WARMUP_EPOCHS = 5
MC_PASSES     = 20
PATIENCE      = 20

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
COLORS      = ['steelblue', 'darkorange', 'green', 'red']
print(f"Device: {DEVICE}")

FAULT_FREQ_MAP = {1: 0, 3: 1, 2: 2}
FAULT_ORDERS   = {1: 0.749, 2: 0.213, 3: 0.524}
SR_NOMINAL     = 640


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (same logic as preprocess.py)
# ═══════════════════════════════════════════════════════════════════════════════

def time_domain_features(sig):
    rms            = np.sqrt(np.mean(sig ** 2))
    peak           = np.max(np.abs(sig))
    mean_abs       = np.mean(np.abs(sig)) + 1e-10
    std            = np.std(sig)
    crest_factor   = peak / (rms + 1e-10)
    shape_factor   = rms / mean_abs
    impulse_factor = peak / mean_abs
    kurt           = float(kurtosis(sig))
    skewness       = float(skew(sig))
    p2p            = np.max(sig) - np.min(sig)
    return np.array([rms, peak, mean_abs, std, crest_factor,
                     shape_factor, impulse_factor, kurt, skewness, p2p],
                    dtype=np.float32)


def frequency_domain_features(sig, sr):
    n       = len(sig)
    freqs   = np.fft.rfftfreq(n, d=1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(sig)) / n
    total_mag = np.sum(fft_mag) + 1e-10
    centroid  = np.sum(freqs * fft_mag) / total_mag
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / total_mag)
    psd_norm  = (fft_mag ** 2) / (np.sum(fft_mag ** 2) + 1e-10)
    entropy   = float(-np.sum(psd_norm * np.log(psd_norm + 1e-15)))
    spec_kurt = float(kurtosis(fft_mag))
    q = len(fft_mag) // 4
    total_energy = np.sum(fft_mag ** 2) + 1e-10
    band_ratios  = [np.sum(fft_mag[q*i: q*(i+1)] ** 2) / total_energy for i in range(4)]
    return np.array([centroid, bandwidth, entropy, spec_kurt] + band_ratios, dtype=np.float32)


def envelope_spectrum_features(sig, sr, fault_freqs, rpm):
    if rpm <= 10:
        return np.zeros(4, dtype=np.float32)
    shaft_hz = rpm / 60.0
    env      = np.abs(hilbert(sig)); env -= np.mean(env)
    n        = len(env)
    freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
    env_fft  = np.abs(np.fft.rfft(env)) / n

    def amp_at(mult):
        target = mult * shaft_hz
        if target <= 0 or target >= sr / 2.0:
            return 0.0
        idx = np.argmin(np.abs(freqs - target))
        w   = max(1, int(len(freqs) * 0.01))
        lo, hi = max(0, idx - w), min(len(env_fft), idx + w + 1)
        return float(np.max(env_fft[lo:hi]))

    return np.array([amp_at(fault_freqs['BPFIMultiple']),
                     amp_at(fault_freqs['BPFOMultiple']),
                     amp_at(fault_freqs['BPFMultiple']),
                     amp_at(fault_freqs['FTFMultiple'])], dtype=np.float32)


def raw_signal_normalized(sig):
    resampled = sp_signal.resample(sig, TARGET_LEN).astype(np.float32)
    mu, sigma = resampled.mean(), resampled.std()
    return (resampled - mu) / (sigma + 1e-10)


# ── MATLAB v7.3 (HDF5) support ────────────────────────────────────────────────

def _scalar(v):
    return float(np.array(v).flat[0])

def _str_from_mat(v):
    if isinstance(v, str): return v
    if isinstance(v, np.ndarray):
        f = v.flatten()
        if f.dtype.kind in ('U', 'S', 'O'):
            return str(f[0]) if len(f) else ''
        if f.dtype.kind == 'u':          # uint16 stored by h5py
            return ''.join(chr(int(c)) for c in f if c != 0)
        return str(f[0]) if len(f) else ''
    try: return str(v[0])
    except: return str(v)

def _norm_ff(ff):
    """Normalize faultFrequencies to a plain dict of floats (handles both scipy and h5py)."""
    keys = ['FTFMultiple', 'BPFMultiple', 'BPFOMultiple', 'BPFIMultiple']
    if isinstance(ff, dict):
        return {k: float(np.array(ff[k]).flat[0]) for k in keys}
    try:                          # scipy structured array (1,1)
        ff_inner = ff[0, 0]
        return {k: float(ff_inner[k].flat[0]) for k in keys}
    except Exception:
        return {k: float(np.array(ff[k]).flat[0]) for k in keys}

def _norm_sensor_h5(h5_dict):
    """Convert h5py struct dict → normalized dict compatible with process_sensor."""
    rd = h5_dict['rawData']
    if isinstance(rd, np.ndarray) and rd.dtype != object:
        rd = rd.T               # undo HDF5 column-major transposition: (L,N) → (N,L)
    return {
        'rawData':          rd,
        'label':            np.array(h5_dict['label']).flatten(),
        'RPM':              np.array(h5_dict['RPM']).flatten(),
        'samplingRate':     np.array(h5_dict['samplingRate']).flatten(),
        'faultFrequencies': _norm_ff(h5_dict['faultFrequencies']),
    }

def _norm_sensor_scipy(scipy_void):
    """Convert scipy structured-array element → normalized dict."""
    return {
        'rawData':          scipy_void['rawData'],
        'label':            np.array(scipy_void['label']).flatten(),
        'RPM':              np.array(scipy_void['RPM']).flatten(),
        'samplingRate':     np.array(scipy_void['samplingRate']).flatten(),
        'faultFrequencies': _norm_ff(scipy_void['faultFrequencies']),
    }

def load_mat_any(path):
    """Load .mat (v7.2 or v7.3/HDF5) → normalized dict with top-level scalars + sensor dicts."""
    # ── scipy (v7.2) ─────────────────────────────────────────────────────────
    try:
        raw = sio.loadmat(path)
        out = {
            'fixedSpeed':       int(_scalar(raw.get('fixedSpeed', 0))),
            'assetDescription': _str_from_mat(raw.get('assetDescription', np.array(['Unknown']))),
            'faultType':        int(_scalar(raw.get('faultType', 0))),
        }
        for sk in SENSOR_KEYS:
            if sk in raw:
                try:
                    out[sk] = _norm_sensor_scipy(raw[sk][0, 0])
                except Exception:
                    pass
        return out
    except Exception:
        pass

    # ── h5py (v7.3 / HDF5) ───────────────────────────────────────────────────
    import h5py

    def _read(item, f):
        if isinstance(item, h5py.Dataset):
            arr = item[()]
            if arr.dtype == object:             # MATLAB cell array (object refs)
                flat = arr.flatten()
                out_arr = np.empty((1, len(flat)), dtype=object)
                for i, ref in enumerate(flat):
                    try:
                        out_arr[0, i] = np.array(f[ref][()]).flatten()
                    except Exception:
                        out_arr[0, i] = None
                return out_arr
            return arr
        if isinstance(item, h5py.Group):
            return {k: _read(item[k], f) for k in item.keys()}
        return item

    with h5py.File(path, 'r') as f:
        raw = {k: _read(f[k], f) for k in f.keys() if not k.startswith('#')}

    out = {
        'fixedSpeed':       int(_scalar(raw.get('fixedSpeed', 0))),
        'assetDescription': _str_from_mat(raw.get('assetDescription', np.array([0x55]))),
        'faultType':        int(_scalar(raw.get('faultType', 0))),
    }
    for sk in SENSOR_KEYS:
        if sk in raw:
            try:
                out[sk] = _norm_sensor_h5(raw[sk])
            except Exception as e:
                pass
    return out


def extract_fault_freqs(sensor_struct):
    return _norm_ff(sensor_struct['faultFrequencies'])


def process_sensor(sensor_struct, folder_id, split_name):
    raw_data    = sensor_struct['rawData']
    labels_arr  = sensor_struct['label'].flatten()
    rpm_vals    = sensor_struct['RPM'].flatten()
    sr          = float(sensor_struct['samplingRate'].flatten()[0])
    fault_freqs = extract_fault_freqs(sensor_struct)

    # Handle ragged cell arrays (folder 9 test)
    if raw_data.dtype == object:
        n_cells      = raw_data.shape[1]
        cell_signals = []
        for ci in range(n_cells):
            elem = np.array(raw_data[0, ci]).flatten().astype(np.float64)
            cell_signals.append(elem if len(elem) >= 64 else None)
        indices = range(len(cell_signals))
    else:
        if raw_data.ndim != 2 or raw_data.shape[1] < 64:
            return []
        indices = range(raw_data.shape[0])

    samples = []
    for i in indices:
        label = int(labels_arr[i]) if i < len(labels_arr) else EXCLUDE_LABEL
        if label == EXCLUDE_LABEL:
            continue
        sig = (np.array(cell_signals[i]) if raw_data.dtype == object
               else raw_data[i].astype(np.float64))
        if sig is None or len(sig) < 64:
            continue
        rpm = float(rpm_vals[i]) if i < len(rpm_vals) else 0.0

        t_feats    = time_domain_features(sig)
        f_feats    = frequency_domain_features(sig, sr)
        stat_feats = np.concatenate([t_feats, f_feats])   # 18
        env_feats  = envelope_spectrum_features(sig, sr, fault_freqs, rpm)  # 4
        raw_norm   = raw_signal_normalized(sig)             # 4096

        asset_oh = np.zeros(len(ASSET_TYPES), dtype=np.float32)
        meta_vec = np.concatenate([
            np.array([rpm / 3000.0, np.log10(sr + 1) / 5.0, 0.0], dtype=np.float32),
            asset_oh
        ])  # 8 (fixed_speed unknown here, default 0)

        samples.append({
            'raw'   : raw_norm,
            'stat'  : stat_feats,
            'env'   : env_feats,
            'meta'  : meta_vec,
            'label' : label,
            'folder': folder_id,
            'split' : split_name,
        })
    return samples


def load_folder(folder_id):
    samples = []
    for split in ['train', 'test']:
        path = os.path.join(DATASET_PATH, str(folder_id), f"{split}.mat")
        if not os.path.exists(path):
            print(f"  [WARN] missing: {path}")
            continue
        try:
            data = load_mat_any(path)     # handles v7.2 (scipy) and v7.3 (h5py)
            fixed_speed = data.get('fixedSpeed', 0)
            asset       = data.get('assetDescription', 'Unknown')
        except Exception as e:
            print(f"  [WARN] F{folder_id}/{split}: load failed – {e}")
            continue

        for sk in SENSOR_KEYS:
            if sk not in data:
                continue
            try:
                samps  = process_sensor(data[sk], folder_id, split)
                asset_oh = np.zeros(len(ASSET_TYPES), dtype=np.float32)
                if asset in ASSET_TYPES:
                    asset_oh[ASSET_TYPES.index(asset)] = 1.0
                for s in samps:
                    old = s['meta']
                    s['meta'] = np.concatenate([
                        np.array([old[0], old[1], float(fixed_speed)], dtype=np.float32),
                        asset_oh
                    ])
                samples.extend(samps)
                print(f"  F{folder_id:02d}/{split}/{sk}: {len(samps)} samples")
            except Exception as e:
                print(f"  F{folder_id:02d}/{split}/{sk}: SKIPPED – {e}")
    return samples


def build_arrays(samples):
    raw     = np.stack([s['raw']    for s in samples])
    stat    = np.stack([s['stat']   for s in samples])
    env     = np.stack([s['env']    for s in samples])
    meta    = np.stack([s['meta']   for s in samples])
    labels  = np.array([s['label']  for s in samples])
    folders = np.array([s['folder'] for s in samples])
    return raw, stat, env, meta, labels, folders


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading training data (folders 1–7)...")
train_all = []
for fid in TRAIN_FOLDERS:
    train_all.extend(load_folder(fid))

print(f"\nTotal train+val samples from folders 1-7: {len(train_all)}")
labels_all = [s['label'] for s in train_all]
u, c = np.unique(labels_all, return_counts=True)
print("Label distribution:", dict(zip(u.tolist(), c.tolist())))

raw_all, stat_all, env_all, meta_all, y_all, fold_all = build_arrays(train_all)

# Stratified 80/20 split (fallback to random if any class is too small to stratify)
idx = np.arange(len(y_all))
try:
    idx_tr, idx_val = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=42)
except ValueError:
    print("[WARN] Stratified split failed (too few samples per class) — using random split")
    idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=42)

# Scale stat + env features
scaler_stat = StandardScaler().fit(stat_all[idx_tr])
scaler_env  = StandardScaler().fit(env_all[idx_tr])

def scale_feat(s, e):
    return np.concatenate([scaler_stat.transform(s), scaler_env.transform(e)], axis=1)

feat_tr  = scale_feat(stat_all[idx_tr],  env_all[idx_tr])
feat_val = scale_feat(stat_all[idx_val], env_all[idx_val])

with open(f"{OUTPUT_DIR}/scaler_stat.pkl", 'wb') as f: pickle.dump(scaler_stat, f)
with open(f"{OUTPUT_DIR}/scaler_env.pkl",  'wb') as f: pickle.dump(scaler_env,  f)

u, c = np.unique(y_all[idx_tr], return_counts=True)
print("\nTrain distribution:", dict(zip(u.tolist(), c.tolist())))
u, c = np.unique(y_all[idx_val], return_counts=True)
print("Val   distribution:", dict(zip(u.tolist(), c.tolist())))

print("\n" + "=" * 60)
print("Loading test data (folders 8–11)...")
test_by_folder = {}
for fid in TEST_FOLDERS:
    fsamples = load_folder(fid)
    test_by_folder[fid] = fsamples
    lbs = [s['label'] for s in fsamples]
    u2, c2 = np.unique(lbs, return_counts=True)
    print(f"  Folder {fid}: {len(fsamples)} samples | labels {dict(zip(u2.tolist(), c2.tolist()))}")

all_test = [s for fsamples in test_by_folder.values() for s in fsamples]
raw_test_all, stat_test_all, env_test_all, meta_test_all, y_test_all, fold_test_all = build_arrays(all_test)
feat_test_all = scale_feat(stat_test_all, env_test_all)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════

class BearingDataset(Dataset):
    """
    augment=True applies 4 physics-safe transforms to the RAW signal only.
    Stat/env features kept as-is (close enough for small perturbations).
    Val / test always use augment=False.
    """
    def __init__(self, raw, feat, meta, labels, augment=False):
        self.raw     = torch.tensor(raw[:, None, :], dtype=torch.float32)
        self.feat    = torch.tensor(feat,            dtype=torch.float32)
        self.meta    = torch.tensor(meta,            dtype=torch.float32)
        self.labels  = torch.tensor(labels,          dtype=torch.long)
        self.augment = augment

    def __len__(self): return len(self.labels)

    def _aug(self, sig, meta):
        """sig: (4096,) float32 tensor.  Returns (sig_aug, meta_aug)."""
        meta = meta.clone()
        rng  = torch.rand(4)

        # 1. Gaussian noise — raises noise floor, fault peak still dominant
        if rng[0] < 0.5:
            sig = sig + torch.randn_like(sig) * 0.01   # σ=0.01 (signal is unit-variance)

        # 2. Magnitude scaling 0.95–1.05 — preserves relative frequency amplitudes
        if rng[1] < 0.5:
            sig = sig * (0.95 + torch.rand(1).item() * 0.10)

        # 3. Circular time shift — FFT magnitude unchanged, only phase shifts
        if rng[2] < 0.5:
            shift = int(torch.randint(0, sig.shape[0], (1,)).item())
            sig   = torch.roll(sig, shift)

        # 4. Speed interpolation ±5% — fault freqs scale with RPM by definition
        if rng[3] < 0.5:
            r     = 0.95 + torch.rand(1).item() * 0.10        # factor ∈ [0.95, 1.05]
            n_new = max(64, int(sig.shape[0] * r))
            sig_np = sp_signal.resample(sig.numpy(), n_new)    # time-stretch/compress
            sig_np = sp_signal.resample(sig_np, sig.shape[0])  # back to 4096
            sig    = torch.tensor(sig_np, dtype=torch.float32)
            meta[0] = (meta[0] * r).clamp(0.0, 1.0)           # update normalised RPM

        return sig, meta

    def __getitem__(self, idx):
        raw, feat, meta, label = (self.raw[idx].squeeze(0), self.feat[idx],
                                   self.meta[idx], self.labels[idx])
        if self.augment:
            raw, meta = self._aug(raw, meta)
        return raw.unsqueeze(0), feat, meta, label


train_ds = BearingDataset(raw_all[idx_tr],  feat_tr,  meta_all[idx_tr],  y_all[idx_tr],  augment=True)
val_ds   = BearingDataset(raw_all[idx_val], feat_val, meta_all[idx_val], y_all[idx_val], augment=False)
test_ds  = BearingDataset(raw_test_all, feat_test_all, meta_test_all, y_test_all)

cls_counts   = np.bincount(y_all[idx_tr], minlength=N_CLASSES).astype(float)
samp_weights = 1.0 / (cls_counts[y_all[idx_tr]] + 1e-9)
sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)
print(f"Class counts (train): {dict(enumerate(cls_counts.astype(int)))}")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

# Per-folder test loaders
folder_dls = {}
for fid, fsamples in test_by_folder.items():
    if not fsamples:
        continue
    fr, fs, fe, fm, fy, _ = build_arrays(fsamples)
    ff = scale_feat(fs, fe)
    folder_dls[fid] = (DataLoader(BearingDataset(fr, ff, fm, fy),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
                       fy)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hilbert_envelope_torch(x):
    N  = x.shape[-1]
    Xf = torch.fft.fft(x.float())
    h  = torch.zeros(N, device=x.device, dtype=torch.float32)
    h[0] = 1.0
    if N % 2 == 0:
        h[1:N//2] = 2.0; h[N//2] = 1.0
    else:
        h[1:(N+1)//2] = 2.0
    return torch.fft.ifft(Xf * h).abs()


def pde_residual_loss(raw, labels, meta, ode_params):
    B, _, N_sig = raw.shape
    x_short  = raw.squeeze(1)[:, :min(1024, N_sig)]
    envelope = hilbert_envelope_torch(x_short)
    dE = (envelope[:, 2:] - envelope[:, :-2]) / (2.0 / SR_NOMINAL)
    E  = envelope[:, 1:-1]
    N_diff = E.shape[1]
    t = (torch.arange(N_diff, device=raw.device, dtype=torch.float32) / SR_NOMINAL
         ).unsqueeze(0).expand(B, -1)
    gamma = F.softplus(ode_params[:, 0]).unsqueeze(1)
    force = F.softplus(ode_params[:, 1]).unsqueeze(1)
    rpm_norm  = meta[:, 0]
    shaft_hz  = (rpm_norm * 3000.0 / 60.0).clamp(min=0.5)
    fault_order = torch.zeros(B, device=raw.device)
    for cls_idx, order in FAULT_ORDERS.items():
        fault_order[labels == cls_idx] = order
    f_fault  = (fault_order * shaft_hz).unsqueeze(1)
    residual = dE + gamma * E - force * torch.cos(2.0 * torch.pi * f_fault * t)
    fault_mask = (labels > 0).float().unsqueeze(1)
    return ((residual * (0.3 + 0.7 * fault_mask)) ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        B      = features.shape[0]
        device = features.device
        sim    = torch.mm(features, features.T) / self.T
        lc     = labels.unsqueeze(1)
        pos_mask  = (lc == lc.T).float(); pos_mask.fill_diagonal_(0.0)
        self_mask = torch.eye(B, device=device)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim       = sim - sim_max.detach()
        exp_sim   = torch.exp(sim) * (1 - self_mask)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        log_prob  = sim - log_denom
        n_pos     = pos_mask.sum(dim=1)
        valid     = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        per_anchor = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-9)
        return per_anchor[valid].mean()


class PhysicsRegularizer(nn.Module):
    MARGIN = 0.05
    def forward(self, env_features, labels):
        loss = torch.zeros(1, device=env_features.device)
        n    = 0
        for cls, freq_idx in FAULT_FREQ_MAP.items():
            mask = (labels == cls)
            if mask.sum() == 0:
                continue
            env_cls  = env_features[mask]
            dominant = env_cls[:, freq_idx]
            others   = torch.cat([env_cls[:, :freq_idx], env_cls[:, freq_idx+1:]], dim=1)
            loss     = loss + F.relu(others.max(dim=1).values - dominant + self.MARGIN).mean()
            n       += 1
        return loss / max(n, 1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma; self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        return (((1.0 - p_t) ** self.gamma) * ce).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL  (identical to best-physics.py)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleBranch(nn.Module):
    def __init__(self, kernel_size, out_ch=32, seq_len=32):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(16),     nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, out_ch, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x): return self.pool(self.conv(x))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(self.fc2(self.drop(F.gelu(self.fc1(x)))) + self.skip(x))


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq    = nn.Linear(dim, dim, bias=False)
        self.Wk    = nn.Linear(dim, dim, bias=False)
        self.Wv    = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.norm  = nn.LayerNorm(dim)

    def forward(self, target, source):
        q    = self.Wq(target); k = self.Wk(source); v = self.Wv(source)
        attn = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True) * self.scale)
        return self.norm(target + attn * v)


class PhysicsHybridNet(nn.Module):
    KERNEL_SIZES = [7, 15, 31, 63]
    SEQ_LEN      = 32
    BRANCH_CH    = 32
    TRANS_DIM    = 128

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiScaleBranch(k, out_ch=self.BRANCH_CH, seq_len=self.SEQ_LEN)
            for k in self.KERNEL_SIZES
        ])
        self.pos_enc = nn.Parameter(torch.randn(1, self.SEQ_LEN, self.TRANS_DIM) * 0.02)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=self.TRANS_DIM, nhead=4, dim_feedforward=256,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.sig_head    = nn.Sequential(
            nn.Linear(self.TRANS_DIM, 64), nn.GELU(), nn.Dropout(DROPOUT),
        )
        self.sig_proj    = nn.Linear(64, PROJ_DIM)
        self.tab_enc     = ResidualMLP(22, 64, 64)       # stat(18)+env(4) = 22
        self.tab_proj    = nn.Linear(64, PROJ_DIM)
        self.meta_enc    = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(), nn.Linear(32, PROJ_DIM), nn.GELU(),
        )
        self.attn_sig_tab  = CrossModalAttention(PROJ_DIM)
        self.attn_sig_meta = CrossModalAttention(PROJ_DIM)
        self.attn_tab_sig  = CrossModalAttention(PROJ_DIM)
        fused_dim = PROJ_DIM * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 96), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(96, N_CLASSES),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(), nn.Linear(128, 64),
        )
        self.ode_params_head = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.GELU(), nn.Linear(32, 2),
        )
        self.log_var_focal   = nn.Parameter(torch.zeros(1))
        self.log_var_supcon  = nn.Parameter(torch.zeros(1))
        self.log_var_physics = nn.Parameter(torch.zeros(1))
        self.log_var_pde     = nn.Parameter(torch.zeros(1))

    def encode(self, raw, feat, meta):
        x = torch.cat([b(raw) for b in self.branches], dim=1).permute(0, 2, 1)
        x = self.transformer(x + self.pos_enc).mean(dim=1)
        sig_emb  = self.sig_proj(self.sig_head(x))
        tab_emb  = self.tab_proj(self.tab_enc(feat))
        meta_emb = self.meta_enc(meta)
        sig2 = self.attn_sig_tab(sig_emb,  tab_emb)
        sig3 = self.attn_sig_meta(sig2,    meta_emb)
        tab2 = self.attn_tab_sig(tab_emb,  sig_emb)
        return torch.cat([sig3, tab2, meta_emb], dim=1)

    def forward(self, raw, feat, meta):
        emb        = self.encode(raw, feat, meta)
        logits     = self.classifier(emb)
        proj       = F.normalize(self.proj_head(emb), dim=1)
        ode_params = self.ode_params_head(emb)
        return logits, proj, ode_params

    def adaptive_loss(self, focal_l, supcon_l, physics_l, pde_l):
        s_f  = self.log_var_focal
        s_c  = self.log_var_supcon
        s_p  = self.log_var_physics
        s_pd = self.log_var_pde
        return (torch.exp(-s_f)  * focal_l   + s_f  +
                torch.exp(-s_c)  * supcon_l  + s_c  +
                torch.exp(-s_p)  * physics_l + s_p  +
                torch.exp(-s_pd) * pde_l     + s_pd)


model    = PhysicsHybridNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# Zero weight for classes absent from training data (avoids inf → NaN training)
w_np = np.where(cls_counts > 0, 1.0 / (cls_counts + 1e-9), 0.0)
w = torch.tensor(w_np, dtype=torch.float32)
n_present = (w > 0).float().sum()
w = (w / (w.sum() + 1e-9) * n_present).to(DEVICE)

focal_criterion   = FocalLoss(gamma=2.0, weight=w)
supcon_criterion  = SupConLoss(temperature=SUPCON_TEMP)
physics_criterion = PhysicsRegularizer()
print(f"Class weights: {w.cpu().numpy().round(3)}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
warmup    = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_focal = tot_supcon = tot_physics = tot_pde = tot_total = 0
    tot_correct = tot = 0
    preds_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for raw, feat, meta, labels in loader:
            raw, feat, meta, labels = (
                raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            )
            logits, proj, ode_params = model(raw, feat, meta)
            focal_l   = focal_criterion(logits, labels)
            supcon_l  = supcon_criterion(proj, labels)
            env_feat  = feat[:, 18:22]
            physics_l = physics_criterion(env_feat, labels)
            pde_l     = pde_residual_loss(raw, labels, meta, ode_params)
            total_l   = model.adaptive_loss(focal_l, supcon_l, physics_l, pde_l)
            if train:
                optimizer.zero_grad()
                total_l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            preds = logits.argmax(1)
            B     = len(labels)
            tot_focal   += focal_l.item()   * B
            tot_supcon  += supcon_l.item()  * B
            tot_physics += physics_l.item() * B
            tot_pde     += pde_l.item()     * B
            tot_total   += total_l.item()   * B
            tot_correct += (preds == labels).sum().item()
            tot         += B
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    losses = {'total': tot_total/tot, 'focal': tot_focal/tot,
              'supcon': tot_supcon/tot, 'physics': tot_physics/tot, 'pde': tot_pde/tot}
    return losses, tot_correct / tot, f1


print("\n" + "=" * 60)
print("Training  (folders 1-7)  →  Testing  (folders 8-11)")
print("=" * 60)

history  = {k: [] for k in ['tr_f1', 'val_f1', 'tr_total', 'val_total']}
best_f1, patience_cnt = 0.0, 0

for ep in range(1, EPOCHS + 1):
    tr_losses, tr_acc, tr_f1 = run_epoch(train_dl, train=True)
    vl_losses, vl_acc, vl_f1 = run_epoch(val_dl,   train=False)
    scheduler.step()

    history['tr_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)
    history['tr_total'].append(tr_losses['total'])
    history['val_total'].append(vl_losses['total'])

    if vl_f1 > best_f1:
        best_f1 = vl_f1
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        patience_cnt = 0
    else:
        patience_cnt += 1

    if ep % 5 == 0 or ep == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Ep {ep:3d}  tr_f1={tr_f1:.4f} vl_f1={vl_f1:.4f} best={best_f1:.4f}  "
              f"focal={tr_losses['focal']:.3f} pde={tr_losses['pde']:.4f}  lr={lr_now:.2e}")

    if patience_cnt >= PATIENCE:
        print(f"\nEarly stop at epoch {ep}")
        break

print(f"\nBest val Macro-F1: {best_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))


def predict_loader(loader):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for raw, feat, meta, labels in loader:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits, _, _ = model(raw, feat, meta)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())
            probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.array(labels_all), np.array(preds_all), np.vstack(probs_all)


def print_metrics(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Accuracy  : {acc:.4f}   Macro-F1 : {mf1:.4f}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=label_names,
                                labels=list(range(N_CLASSES)), zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    header = f"{'':14s}" + "".join(f"{n:>14s}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{label_names[i]:14s}" + "".join(f"{v:>14d}" for v in row))
    return acc, mf1, cm


# ─── Validation set ──────────────────────────────────────────────────────────
vt, vp, vprobs = predict_loader(val_dl)
val_acc, val_f1, val_cm = print_metrics(vt, vp, "VALIDATION SET (from folders 1-7)")


# ─── Per-folder test results ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST RESULTS — Folders 8-11  (raw .mat files)")
print("=" * 60)

folder_metrics = {}
for fid in TEST_FOLDERS:
    if fid not in folder_dls:
        print(f"\nFolder {fid}: NO DATA")
        continue
    floader, fy_true = folder_dls[fid]
    ft, fp, fprobs = predict_loader(floader)
    acc = accuracy_score(ft, fp)
    mf1 = f1_score(ft, fp, average='macro', zero_division=0)
    folder_metrics[fid] = {'acc': acc, 'macro_f1': mf1, 'n': len(ft)}
    print(f"\nFolder {fid}  ({len(ft)} samples)")
    print(f"  Accuracy : {acc:.4f}   Macro-F1 : {mf1:.4f}")
    u_f, c_f = np.unique(ft, return_counts=True)
    true_dist = dict(zip(u_f.tolist(), c_f.tolist()))
    print(f"  True labels: {true_dist}")
    report = classification_report(ft, fp, target_names=label_names,
                                   labels=list(range(N_CLASSES)),
                                   zero_division=0, output_dict=True)
    for ln in label_names:
        r = report[ln]
        if r['support'] > 0:
            print(f"    {ln:12s}: P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={int(r['support'])}")

# ─── Overall test (folders 8-11 combined) ────────────────────────────────────
yt, yp, yprobs = predict_loader(test_dl)
overall_acc, overall_f1, overall_cm = print_metrics(yt, yp,
    "OVERALL TEST  (folders 8-11 combined)")

print("\n" + "─" * 60)
print("SUMMARY")
print("─" * 60)
print(f"  Val   Accuracy  : {val_acc:.4f}   Val   Macro-F1 : {val_f1:.4f}")
print(f"  Test  Accuracy  : {overall_acc:.4f}   Test  Macro-F1 : {overall_f1:.4f}")
print("\n  Per-Folder Test Accuracy:")
for fid, m in folder_metrics.items():
    flag = " ← false-alarm folder" if fid == 11 else ""
    print(f"    Folder {fid}: Acc={m['acc']:.4f}  F1={m['macro_f1']:.4f}  (n={m['n']}){flag}")
print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE METRICS JSON
# ═══════════════════════════════════════════════════════════════════════════════

metrics_out = {
    'validation': {'accuracy': float(val_acc),  'macro_f1': float(val_f1)},
    'test_overall': {'accuracy': float(overall_acc), 'macro_f1': float(overall_f1)},
    'test_per_folder': {str(fid): {
        'accuracy': float(m['acc']),
        'macro_f1': float(m['macro_f1']),
        'n_samples': int(m['n'])
    } for fid, m in folder_metrics.items()},
}
with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"\nMetrics saved: {OUTPUT_DIR}/metrics.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("PhysicsHybridNet — Train on Folders 1-7 | Test on Folders 8-11",
             fontsize=13, fontweight='bold')

# 1. Training curves
ax = axes[0, 0]
ax.plot(history['tr_f1'],  label='Train F1',  color='steelblue')
ax.plot(history['val_f1'], label='Val F1',    color='orangered')
ax.set_title('Macro F1 (training)'); ax.set_xlabel('Epoch')
ax.legend(); ax.grid(alpha=0.3)

# 2. Loss curves
ax = axes[0, 1]
ax.plot(history['tr_total'],  label='Train', color='steelblue')
ax.plot(history['val_total'], label='Val',   color='orangered')
ax.set_title('Total Loss'); ax.set_xlabel('Epoch')
ax.legend(); ax.grid(alpha=0.3)

# 3. Val confusion matrix
ax = axes[0, 2]
im = ax.imshow(val_cm, cmap='Blues')
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(label_names, rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(label_names, fontsize=8)
ax.set_title(f'Val Confusion Matrix\nAcc={val_acc:.3f}  F1={val_f1:.3f}')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(val_cm[i, j]), ha='center', va='center', fontsize=9,
                color='white' if val_cm[i, j] > val_cm.max() / 2 else 'black')
plt.colorbar(im, ax=ax)

# 4. Test confusion matrix
ax = axes[0, 3]
im2 = ax.imshow(overall_cm, cmap='Oranges')
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(label_names, rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(label_names, fontsize=8)
ax.set_title(f'Test Confusion Matrix (8-11)\nAcc={overall_acc:.3f}  F1={overall_f1:.3f}')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(overall_cm[i, j]), ha='center', va='center', fontsize=9,
                color='white' if overall_cm[i, j] > overall_cm.max() / 2 else 'black')
plt.colorbar(im2, ax=ax)

# 5. Per-folder accuracy bar chart
ax = axes[1, 0]
fids = sorted(folder_metrics.keys())
accs = [folder_metrics[f]['acc'] for f in fids]
f1s  = [folder_metrics[f]['macro_f1'] for f in fids]
x    = np.arange(len(fids))
bars1 = ax.bar(x - 0.2, accs, width=0.35, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + 0.2, f1s,  width=0.35, label='Macro-F1', color='orangered')
ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in fids])
ax.set_ylim(0, 1.15); ax.set_title('Per-Folder Test Performance')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for b, v in list(zip(bars1, accs)) + list(zip(bars2, f1s)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8)

# 6. ROC curves (overall test)
ax = axes[1, 1]
y_bin = label_binarize(yt, classes=list(range(N_CLASSES)))
roc_aucs = []
for c in range(N_CLASSES):
    if y_bin[:, c].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_bin[:, c], yprobs[:, c])
    a = auc(fpr, tpr)
    roc_aucs.append(a)
    ax.plot(fpr, tpr, color=COLORS[c], label=f'{label_names[c]} ({a:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.5)
mean_auc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
ax.set_title(f'ROC Curves (Test)\nMean AUC={mean_auc:.3f}')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 7. PR curves (overall test)
ax = axes[1, 2]
for c in range(N_CLASSES):
    if y_bin[:, c].sum() == 0:
        continue
    prec, rec, _ = precision_recall_curve(y_bin[:, c], yprobs[:, c])
    ap = average_precision_score(y_bin[:, c], yprobs[:, c])
    ax.plot(rec, prec, color=COLORS[c], label=f'{label_names[c]} (AP={ap:.3f})')
ax.set_title('Precision-Recall (Test)')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 8. Classification report bar chart (per-class F1 on test)
ax = axes[1, 3]
report = classification_report(yt, yp, target_names=label_names,
                               labels=list(range(N_CLASSES)), zero_division=0, output_dict=True)
class_f1 = [report[ln]['f1-score'] for ln in label_names]
bars3 = ax.bar(label_names, class_f1, color=COLORS)
for b, v in zip(bars3, class_f1):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, 1.15); ax.set_title('Per-Class F1 (Test 8-11)')
ax.set_ylabel('F1 Score'); ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/results.png")
print(f"Model saved: {OUTPUT_DIR}/best_model.pt")
print("\nDone!")
