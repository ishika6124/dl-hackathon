"""
Ensemble Evaluation: GAF v2 + MSConvFormer
==========================================
Averages softmax probabilities from both trained models.

Why ensemble works here:
  GAF v2       → high precision (fewer false alarms), but Ball=4/5
  MSConvFormer → perfect fault recall (Ball=5/5, IR=74/74), but 34 Normal→IR FP
  Average      → balanced: recall stays high, false alarms get suppressed by GAF

Run after both models are trained:
  outputs_gaf_v2/best_model.pt  must exist
  outputs_msct/best_model.pt    must exist

Usage: py ensemble_eval.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample as scipy_resample
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

DATA_DIR    = "final_data"
OUTPUT_DIR  = "/home/teaching/hackathon/Approach-2/outputs_ensemble"
N_CLASSES   = 4
DROPOUT     = 0.0      # no dropout at inference
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
print(f"Device: {DEVICE}")

# =============================================================================
# MODEL DEFINITIONS (copied from training scripts — inference only)
# =============================================================================

# ── GAF Model ────────────────────────────────────────────────────────────────

GAF_SIZE = 64

def compute_gasf(signal, size=GAF_SIZE):
    sig = scipy_resample(signal, size).astype(np.float64)
    mn, mx = sig.min(), sig.max()
    if mx - mn < 1e-10:
        return np.zeros((size, size), dtype=np.float32)
    sig = np.clip(2.0 * (sig - mn) / (mx - mn) - 1.0, -1.0, 1.0)
    theta = np.arccos(sig)
    return np.cos(theta[:, None] + theta[None, :]).astype(np.float32)


class ResBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class DownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.net(x) + self.skip(x))


class GAFResNet(nn.Module):
    def __init__(self, out_dim=64, dropout=0.0):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(ResBlock2D(32), ResBlock2D(32))
        self.layer2 = nn.Sequential(DownBlock2D(32, 64),  ResBlock2D(64))
        self.layer3 = nn.Sequential(DownBlock2D(64, 128), ResBlock2D(128))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.norm  = nn.LayerNorm(dim)
    def forward(self, target, source):
        q    = self.Wq(target);  k = self.Wk(source);  v = self.Wv(source)
        attn = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True) * self.scale)
        return self.norm(target + attn * v)


PROJ_DIM_GAF = 48

class MultimodalFaultNet(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.gaf_enc  = GAFResNet(out_dim=64, dropout=dropout)
        self.gaf_proj = nn.Linear(64, PROJ_DIM_GAF)
        self.feat_enc = nn.Sequential(
            nn.Linear(22, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, PROJ_DIM_GAF), nn.GELU(),
        )
        self.meta_enc = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(), nn.Linear(32, PROJ_DIM_GAF), nn.GELU(),
        )
        self.attn_gaf_feat = CrossModalAttention(PROJ_DIM_GAF)
        self.attn_gaf_meta = CrossModalAttention(PROJ_DIM_GAF)
        self.attn_feat_gaf = CrossModalAttention(PROJ_DIM_GAF)
        self.fusion = nn.Sequential(
            nn.Linear(PROJ_DIM_GAF * 3, 96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, N_CLASSES),
        )
    def forward(self, gaf, feat, meta):
        g = self.gaf_proj(self.gaf_enc(gaf))
        f = self.feat_enc(feat)
        m = self.meta_enc(meta)
        g2 = self.attn_gaf_feat(g, f)
        g3 = self.attn_gaf_meta(g2, m)
        f2 = self.attn_feat_gaf(f, g)
        return self.fusion(torch.cat([g3, f2, m], dim=1))


# ── MSConvFormer ─────────────────────────────────────────────────────────────

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
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(x)))
        return self.norm(self.fc2(h) + self.skip(x))


class MSConvFormer(nn.Module):
    KERNEL_SIZES = [7, 15, 31, 63]
    SEQ_LEN      = 32
    BRANCH_CH    = 32
    TRANS_DIM    = 128

    def __init__(self, dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiScaleBranch(k, out_ch=self.BRANCH_CH, seq_len=self.SEQ_LEN)
            for k in self.KERNEL_SIZES
        ])
        self.pos_enc = nn.Parameter(
            torch.randn(1, self.SEQ_LEN, self.TRANS_DIM) * 0.02
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.TRANS_DIM, nhead=4, dim_feedforward=256,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.sig_head = nn.Sequential(
            nn.Linear(self.TRANS_DIM, 64), nn.GELU(), nn.Dropout(dropout),
        )
        self.tab_enc  = ResidualMLP(22, 64, 64, dropout=dropout)
        self.meta_enc = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(160, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, N_CLASSES),
        )
    def forward(self, raw, feat, meta):
        x = torch.cat([b(raw) for b in self.branches], dim=1).permute(0, 2, 1) + self.pos_enc
        x = self.transformer(x).mean(dim=1)
        sig_emb  = self.sig_head(x)
        tab_emb  = self.tab_enc(feat)
        meta_emb = self.meta_enc(meta)
        return self.classifier(torch.cat([sig_emb, tab_emb, meta_emb], dim=1))


# =============================================================================
# DATASETS
# =============================================================================

class GAFDataset(Dataset):
    def __init__(self, raw, stat, env, meta, labels):
        self.stat   = torch.tensor(stat, dtype=torch.float32)
        self.env    = torch.tensor(env,  dtype=torch.float32)
        self.meta   = torch.tensor(meta, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.feat   = torch.cat([self.stat, self.env], dim=1)
        print(f"  Pre-computing {len(raw)} GAF images...", flush=True)
        gaf_np   = np.stack([compute_gasf(raw[i]) for i in range(len(raw))])[:, None, :, :]
        self.gaf = torch.tensor(gaf_np, dtype=torch.float32)
        print("  Done.", flush=True)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.gaf[idx], self.feat[idx], self.meta[idx], self.labels[idx]


class RawDataset(Dataset):
    def __init__(self, raw, stat, env, meta, labels):
        self.raw    = torch.tensor(raw[:, None, :], dtype=torch.float32)
        self.feat   = torch.tensor(np.concatenate([stat, env], axis=1), dtype=torch.float32)
        self.meta   = torch.tensor(meta,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.raw[idx], self.feat[idx], self.meta[idx], self.labels[idx]


def load_split(name):
    return (
        np.load(f"{DATA_DIR}/X_raw_{name}.npy"),
        np.load(f"{DATA_DIR}/X_stat_{name}.npy"),
        np.load(f"{DATA_DIR}/X_env_{name}.npy"),
        np.load(f"{DATA_DIR}/X_meta_{name}.npy"),
        np.load(f"{DATA_DIR}/y_{name}.npy"),
    )


print("Loading data...")
raw_val, stat_val, env_val, meta_val, y_val = load_split("val")
raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa  = load_split("false_alarm")

print("Building GAF val dataset:")
gaf_val_ds = GAFDataset(raw_val, stat_val, env_val, meta_val, y_val)
raw_val_ds = RawDataset(raw_val, stat_val, env_val, meta_val, y_val)

print("Building GAF false-alarm dataset:")
gaf_fa_ds  = GAFDataset(raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa)
raw_fa_ds  = RawDataset(raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa)

gaf_val_dl = DataLoader(gaf_val_ds, batch_size=64, shuffle=False, num_workers=0)
raw_val_dl = DataLoader(raw_val_ds, batch_size=64, shuffle=False, num_workers=0)
gaf_fa_dl  = DataLoader(gaf_fa_ds,  batch_size=64, shuffle=False, num_workers=0)
raw_fa_dl  = DataLoader(raw_fa_ds,  batch_size=64, shuffle=False, num_workers=0)


# =============================================================================
# LOAD MODELS
# =============================================================================

gaf_model  = MultimodalFaultNet(dropout=0.0).to(DEVICE)
msct_model = MSConvFormer(dropout=0.0).to(DEVICE)

gaf_path  = "outputs_gaf_v2/best_model.pt"
msct_path = "outputs_msct/best_model.pt"

assert os.path.exists(gaf_path),  f"Not found: {gaf_path}  — run train_gaf_model.py first"
assert os.path.exists(msct_path), f"Not found: {msct_path} — run train_msct_model.py first"

gaf_model.load_state_dict(torch.load(gaf_path,  map_location=DEVICE))
msct_model.load_state_dict(torch.load(msct_path, map_location=DEVICE))
gaf_model.eval()
msct_model.eval()
print(f"\nLoaded: {gaf_path}")
print(f"Loaded: {msct_path}")


# =============================================================================
# ENSEMBLE INFERENCE
# =============================================================================

def get_probs_gaf(loader):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for gaf, feat, meta, labels in loader:
            gaf, feat, meta = gaf.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits = gaf_model(gaf, feat, meta)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.concatenate(all_probs), np.array(all_labels)


def get_probs_msct(loader):
    all_probs = []
    with torch.no_grad():
        for raw, feat, meta, _ in loader:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits = msct_model(raw, feat, meta)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def ensemble_predict(gaf_probs, msct_probs, alpha=0.5):
    """Average softmax probabilities. alpha=0.5 means equal weight."""
    avg = alpha * gaf_probs + (1 - alpha) * msct_probs
    return np.argmax(avg, axis=1)


def evaluate_preds(y_true, y_pred, name):
    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    print("False Positive Rate per class:")
    for c in range(N_CLASSES):
        fp  = np.sum((y_true != c) & (y_pred == c))
        tn  = np.sum((y_true != c) & (y_pred != c))
        print(f"  {label_names[c]:12s}: FPR = {fp / (fp + tn + 1e-10):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print("\nConfusion Matrix:")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{label_names[i]:12s}" + "".join(f"{v:>12d}" for v in row))

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\nMacro F1: {f1:.4f}")
    return cm, f1


print("\n" + "="*60)
print("Running Ensemble Inference on Validation Set")
print("="*60)

gaf_probs,  y_val_true = get_probs_gaf(gaf_val_dl)
msct_probs             = get_probs_msct(raw_val_dl)

# Individual model results
y_gaf_pred  = np.argmax(gaf_probs, axis=1)
y_msct_pred = np.argmax(msct_probs, axis=1)
evaluate_preds(y_val_true, y_gaf_pred,  "GAF v2 (standalone)")
evaluate_preds(y_val_true, y_msct_pred, "MSConvFormer (standalone)")

# Ensemble with equal weights
y_ens_pred = ensemble_predict(gaf_probs, msct_probs, alpha=0.5)
cm_ens, f1_ens = evaluate_preds(y_val_true, y_ens_pred, "Ensemble (equal weights)")

# Sweep alpha to find best balance
print("\n--- Alpha sweep (GAF weight : MSCT weight) ---")
best_alpha, best_f1 = 0.5, 0.0
for alpha in np.arange(0.3, 0.8, 0.1):
    y_pred = ensemble_predict(gaf_probs, msct_probs, alpha=alpha)
    f1     = f1_score(y_val_true, y_pred, average='macro', zero_division=0)
    print(f"  alpha={alpha:.1f} (GAF {alpha:.0%} / MSCT {1-alpha:.0%}): Macro F1 = {f1:.4f}")
    if f1 > best_f1:
        best_f1, best_alpha = f1, alpha

print(f"\nBest alpha = {best_alpha:.1f} → Macro F1 = {best_f1:.4f}")
y_best_pred = ensemble_predict(gaf_probs, msct_probs, alpha=best_alpha)
cm_best, _  = evaluate_preds(y_val_true, y_best_pred, f"Ensemble (best alpha={best_alpha:.1f})")

# False alarm test
print("\n--- False Alarm Test (Folder 11) ---")
gaf_fa_probs,  _ = get_probs_gaf(gaf_fa_dl)
msct_fa_probs    = get_probs_msct(raw_fa_dl)
fa_ens_pred = ensemble_predict(gaf_fa_probs, msct_fa_probs, alpha=best_alpha)
print(f"  {np.sum(fa_ens_pred != 0)} / {len(fa_ens_pred)} predicted as fault")
print(f"  False Alarm Rate = {np.mean(fa_ens_pred != 0):.4f}  (ideal = 0)")

# --- COMPARISON PLOT ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_labels = [f"{n}\n({i})" for i, n in enumerate(label_names)]
im = axes[0].imshow(cm_best, cmap='Blues')
axes[0].set_xticks(range(N_CLASSES)); axes[0].set_xticklabels(label_names, rotation=30, ha='right')
axes[0].set_yticks(range(N_CLASSES)); axes[0].set_yticklabels(label_names)
axes[0].set_title(f'Ensemble Confusion Matrix (alpha={best_alpha:.1f})')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        axes[0].text(j, i, str(cm_best[i, j]), ha='center', va='center',
                     color='white' if cm_best[i, j] > cm_best.max() / 2 else 'black')
plt.colorbar(im, ax=axes[0])

# Alpha sweep bar chart
alphas  = np.arange(0.3, 0.8, 0.1)
f1s     = [f1_score(y_val_true, ensemble_predict(gaf_probs, msct_probs, a),
                    average='macro', zero_division=0) for a in alphas]
labels  = [f"{a:.1f}" for a in alphas]
colors  = ['orangered' if a == best_alpha else 'steelblue' for a in alphas]
axes[1].bar(labels, f1s, color=colors)
axes[1].set_title('Macro F1 vs Ensemble Alpha\n(GAF weight / orange = best)')
axes[1].set_xlabel('Alpha (GAF weight)'); axes[1].set_ylabel('Macro F1')
axes[1].set_ylim(min(f1s) - 0.01, max(f1s) + 0.01)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ensemble_results.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/ensemble_results.png")
print("\nDone!")
