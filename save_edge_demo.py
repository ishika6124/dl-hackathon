"""
save_edge_demo.py — Generate demo samples pkl for Edge Model UI
================================================================
Run AFTER training the edge model:
  cd edge/
  python run_edge.py              ← trains and saves best_edge_model.pt in outputs_edge/
  cd ../e2e/
  python save_edge_demo.py        ← generates demo_samples_edge.pkl

This creates one best sample per class (by confidence) + one FA sample,
stored in a format the app.py can directly consume.
"""

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert as scipy_hilbert

# ── Add edge/ to path so we can reuse compute_order_spectrum ─────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
_EDGE   = os.path.join(_ROOT, "edge")
sys.path.insert(0, _EDGE)

DATA_DIR = os.path.join(_ROOT, "final_data")
CKPT     = os.path.join(_ROOT, "outputs_edge", "best_edge_model.pt")
OUT_PKL  = os.path.join(_HERE, "demo_samples_edge.pkl")

DEVICE     = torch.device("cpu")
DROPOUT    = 0.4
PROJ_DIM   = 48
ORDER_BINS = 64
MAX_ORDER  = 3.0
SR         = 640.0
CLASS_NAMES = ["Normal", "Inner Ring", "Ball", "Outer Ring"]


# ══════════════════════════════════════════════════════════════════════════════
# ORDER SPECTRUM (identical to edge/run_edge.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_order_spectrum(signal, rpm, sr=SR, n_bins=ORDER_BINS, max_order=MAX_ORDER):
    if rpm < 10.0:
        return np.zeros(n_bins, dtype=np.float32)
    env      = np.abs(scipy_hilbert(signal.astype(np.float64)))
    n        = len(env)
    fft_mag  = np.abs(np.fft.rfft(env))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
    shaft_hz = rpm / 60.0
    orders   = freqs / shaft_hz
    order_grid = np.linspace(0.0, max_order, n_bins)
    spectrum   = np.interp(order_grid, orders, fft_mag)
    mx = spectrum.max()
    if mx > 1e-9:
        spectrum /= mx
    return spectrum.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# EDGE MODEL ARCHITECTURE (identical to edge/run_edge.py EdgeHybridNet)
# ══════════════════════════════════════════════════════════════════════════════

class MultiScaleBranch(nn.Module):
    def __init__(self, k, out_ch=32, seq=32):
        super().__init__()
        p = k // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, k, stride=2, padding=p, bias=False),
            nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, out_ch, k, stride=2, padding=p, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(seq)
    def forward(self, x): return self.pool(self.conv(x))

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden)
        self.fc2  = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.norm(self.fc2(self.drop(F.gelu(self.fc1(x)))) + self.skip(x))

class CrossModalAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False); self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5; self.norm = nn.LayerNorm(d)
    def forward(self, tgt, src):
        a = torch.sigmoid(torch.sum(self.Wq(tgt) * self.Wk(src), -1, keepdim=True) * self.scale)
        return self.norm(tgt + a * self.Wv(src))

class EdgeHybridNet(nn.Module):
    def __init__(self, tab_dim=82):
        super().__init__()
        self.branches = nn.ModuleList([MultiScaleBranch(k) for k in [7, 15, 31, 63]])
        self.pos_enc  = nn.Parameter(torch.randn(1, 32, 128) * 0.02)
        enc = nn.TransformerEncoderLayer(128, 4, 256, DROPOUT, 'gelu',
                                         batch_first=True, norm_first=True)
        self.transformer   = nn.TransformerEncoder(enc, 3)
        self.sig_head      = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT))
        self.sig_proj      = nn.Linear(64, PROJ_DIM)
        self.tab_enc       = ResidualMLP(tab_dim, 128, 64)
        self.tab_proj      = nn.Linear(64, PROJ_DIM)
        self.meta_enc      = nn.Sequential(nn.Linear(8, 32), nn.GELU(),
                                           nn.Linear(32, PROJ_DIM), nn.GELU())
        self.attn_sig_tab  = CrossModalAttention(PROJ_DIM)
        self.attn_sig_meta = CrossModalAttention(PROJ_DIM)
        self.attn_tab_sig  = CrossModalAttention(PROJ_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(PROJ_DIM * 3, 96), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(96, 4),
        )
    def forward(self, raw, tab, meta):
        x   = torch.cat([b(raw) for b in self.branches], 1).permute(0, 2, 1) + self.pos_enc
        x   = self.transformer(x).mean(1)
        sig = self.sig_proj(self.sig_head(x))
        t   = self.tab_proj(self.tab_enc(tab))
        me  = self.meta_enc(meta)
        sig = self.attn_sig_meta(self.attn_sig_tab(sig, t), me)
        t   = self.attn_tab_sig(t, sig)
        return self.fusion(torch.cat([sig, t, me], 1))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if not os.path.isfile(CKPT):
    print(f"ERROR: Edge model checkpoint not found at {CKPT}")
    print("Train it first:  cd edge && python run_edge.py")
    sys.exit(1)

print(f"Loading checkpoint: {CKPT}")
model = EdgeHybridNet(tab_dim=82)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
model.eval()

# Load val data
print(f"Loading data from: {DATA_DIR}")
raw_vl  = np.load(f"{DATA_DIR}/X_raw_val.npy").astype(np.float32)
stat_vl = np.load(f"{DATA_DIR}/X_stat_val.npy").astype(np.float32)
meta_vl = np.load(f"{DATA_DIR}/X_meta_val.npy").astype(np.float32)
y_vl    = np.load(f"{DATA_DIR}/y_val.npy").astype(np.int64)
raw_fa  = np.load(f"{DATA_DIR}/X_raw_false_alarm.npy").astype(np.float32)
stat_fa = np.load(f"{DATA_DIR}/X_stat_false_alarm.npy").astype(np.float32)
meta_fa = np.load(f"{DATA_DIR}/X_meta_false_alarm.npy").astype(np.float32)

# Compute order spectra
print("Computing order spectra for val set...")
rpms_vl = meta_vl[:, 0] * 3000.0
ord_vl  = np.stack([compute_order_spectrum(raw_vl[i], rpms_vl[i]) for i in range(len(raw_vl))])

# Inference on val set
print("Running inference on val set...")
all_probs, all_preds = [], []
with torch.no_grad():
    for i in range(0, len(raw_vl), 64):
        r = torch.FloatTensor(raw_vl[i:i+64, None, :])
        t = torch.FloatTensor(np.concatenate([stat_vl[i:i+64], ord_vl[i:i+64]], axis=1))
        m = torch.FloatTensor(meta_vl[i:i+64])
        logits = model(r, t, m)
        probs  = logits.softmax(1).numpy()
        all_probs.append(probs)
        all_preds.extend(logits.argmax(1).numpy().tolist())

all_probs = np.concatenate(all_probs, axis=0)
all_preds = np.array(all_preds)

# Pick one best sample per class
print("Selecting best demo samples...")
demo_samples = []
for cls in range(4):
    idxs    = np.where(y_vl == cls)[0]
    correct = [i for i in idxs if all_preds[i] == cls]
    if not correct:
        correct = idxs.tolist()
    best_i = correct[int(np.argmax([all_probs[i, cls] for i in correct]))]
    demo_samples.append({
        "raw":        raw_vl[best_i],
        "stat":       stat_vl[best_i],
        "order_spec": ord_vl[best_i],
        "meta":       meta_vl[best_i],
        "true_label": int(cls),
        "label_name": CLASS_NAMES[cls],
        "confidence": float(all_probs[best_i, cls]),
    })
    print(f"  {CLASS_NAMES[cls]:12s}: idx={best_i}, conf={all_probs[best_i, cls]:.3f}")

# FA sample
rpm_fa = meta_fa[0, 0] * 3000.0
demo_samples.append({
    "raw":        raw_fa[0],
    "stat":       stat_fa[0],
    "order_spec": compute_order_spectrum(raw_fa[0], rpm_fa),
    "meta":       meta_fa[0],
    "true_label": -1,
    "label_name": "Shaft Misalignment (FA)",
    "confidence": 0.0,
})

with open(OUT_PKL, "wb") as f:
    pickle.dump(demo_samples, f)

print(f"\nSaved {len(demo_samples)} demo samples → {OUT_PKL}")
print("Now launch the app:  cd e2e && streamlit run app.py")
