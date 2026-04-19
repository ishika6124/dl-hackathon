"""
Paper-Inspired Multimodal Bearing Fault Diagnosis
===================================================
Reference: Cao & Shi (2025) - "Multimodal Joint Representation Learning
           and Residual Neural Network for Health Status Identification"

Architecture adapted for our vibration-only SCA dataset:
  Branch A : GAF image (64x64)      -> 2D ResNet-style CNN    (paper: vibration via GAF)
  Branch B : Stat + Env features    -> MLP                    (paper: feature set)
  Branch C : Metadata               -> MLP                    (paper: operating context)
  Fusion   : Cross-modal attention  -> Classifier             (paper: orthogonal proj + Transformer)

Data: final_data/  (run augment_data.py first)
Classes: 0=Normal, 1=Inner Ring, 2=Ball, 3=Outer Ring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import resample as scipy_resample
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR   = "final_data"
OUTPUT_DIR = "outputs_gaf_v2"
GAF_SIZE   = 64       # downsample signal to this, then compute 64x64 GAF image
BATCH_SIZE = 32
EPOCHS     = 80
LR         = 1e-3     # restored: 1e-3 was better than 3e-4 in run1
DROPOUT    = 0.4      # restored: 0.5 was too aggressive
PROJ_DIM   = 48       # cross-modal projection dim (was 32, slight capacity increase)
N_CLASSES  = 4
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
print(f"Device: {DEVICE}")

# ─── GAF COMPUTATION ─────────────────────────────────────────────────────────

def compute_gasf(signal, size=GAF_SIZE):
    """
    Gramian Angular Summation Field (paper Section 2.1.1).
    Steps: normalize -> polar coords -> GASF = cos(theta_i + theta_j)
    Returns float32 array of shape (size, size).
    """
    sig = scipy_resample(signal, size).astype(np.float64)
    # Normalize to [-1, 1]
    mn, mx = sig.min(), sig.max()
    if mx - mn < 1e-10:
        return np.zeros((size, size), dtype=np.float32)
    sig = 2.0 * (sig - mn) / (mx - mn) - 1.0
    sig = np.clip(sig, -1.0, 1.0)
    # Polar encoding
    theta = np.arccos(sig)           # (size,)
    # GASF[i,j] = cos(theta_i + theta_j)
    gasf = np.cos(theta[:, None] + theta[None, :])
    return gasf.astype(np.float32)  # (size, size)

# ─── DATASET ─────────────────────────────────────────────────────────────────

class BearingGAFDataset(Dataset):
    def __init__(self, raw, stat, env, meta, labels, precompute_gaf=True):
        """
        raw  : (N, 4096)  raw normalized signal
        stat : (N, 18)    statistical features (scaled)
        env  : (N, 4)     bearing fault freq amplitudes (scaled)
        meta : (N, 8)     metadata (scaled)
        """
        self.stat   = torch.tensor(stat,   dtype=torch.float32)
        self.env    = torch.tensor(env,    dtype=torch.float32)
        self.meta   = torch.tensor(meta,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Stat + Env concatenated for Branch B
        self.feat   = torch.cat([self.stat, self.env], dim=1)  # (N, 22)

        # Pre-compute GAF images for speed
        print(f"  Pre-computing {len(raw)} GAF images ({GAF_SIZE}x{GAF_SIZE})...", flush=True)
        gaf_list = [compute_gasf(raw[i]) for i in range(len(raw))]
        gaf_np   = np.stack(gaf_list)[:, None, :, :]     # (N, 1, 64, 64)
        self.gaf = torch.tensor(gaf_np, dtype=torch.float32)
        print("  Done.", flush=True)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.gaf[idx], self.feat[idx], self.meta[idx], self.labels[idx]


def load_split(name):
    raw  = np.load(f"{DATA_DIR}/X_raw_{name}.npy")
    stat = np.load(f"{DATA_DIR}/X_stat_{name}.npy")
    env  = np.load(f"{DATA_DIR}/X_env_{name}.npy")
    meta = np.load(f"{DATA_DIR}/X_meta_{name}.npy")
    y    = np.load(f"{DATA_DIR}/y_{name}.npy")
    return raw, stat, env, meta, y


print("Loading data...")
raw_tr, stat_tr, env_tr, meta_tr, y_train = load_split("train")
raw_val, stat_val, env_val, meta_val, y_val = load_split("val")
raw_fa, stat_fa, env_fa, meta_fa, y_fa     = load_split("false_alarm")

u, c = np.unique(y_train, return_counts=True)
print("Train distribution:", dict(zip(u.tolist(), c.tolist())))
u2, c2 = np.unique(y_val, return_counts=True)
print("Val   distribution:", dict(zip(u2.tolist(), c2.tolist())))

print("\nBuilding train dataset:")
train_ds = BearingGAFDataset(raw_tr,  stat_tr,  env_tr,  meta_tr,  y_train)
print("Building val dataset:")
val_ds   = BearingGAFDataset(raw_val, stat_val, env_val, meta_val, y_val)
print("Building false-alarm dataset:")
fa_ds    = BearingGAFDataset(raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa)

# Weighted sampler for class imbalance
cls_counts  = np.bincount(y_train, minlength=N_CLASSES).astype(float)
samp_weights = 1.0 / (cls_counts[y_train] + 1e-9)
sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)
fa_dl    = DataLoader(fa_ds,    batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)

# ─── MODEL ARCHITECTURE ──────────────────────────────────────────────────────

class ResBlock2D(nn.Module):
    """Identity-mapping residual block (paper Fig.4a)."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock2D(nn.Module):
    """Downsampling residual block (paper Fig.4b)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class GAFResNet(nn.Module):
    """
    ResNet-18 style 2D CNN for GAF images (paper Section 2.3).
    Input: (B, 1, 64, 64) -> Output: (B, out_dim)
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),         # -> (B, 32, 32, 32)
        )
        self.layer1 = nn.Sequential(ResBlock2D(32), ResBlock2D(32))      # 32x32
        self.layer2 = nn.Sequential(DownBlock2D(32, 64), ResBlock2D(64)) # 16x16
        self.layer3 = nn.Sequential(DownBlock2D(64,128), ResBlock2D(128))# 8x8
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


class CrossModalAttention(nn.Module):
    """
    Simplified cross-modal attention inspired by paper's
    Orthogonal Projection + Cross-model Attention (Section 2.2).
    Q from target modality, K/V from source modality.
    """
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.norm  = nn.LayerNorm(dim)

    def forward(self, target, source):
        """target attends to source."""
        q = self.Wq(target)                              # (B, dim)
        k = self.Wk(source)
        v = self.Wv(source)
        attn = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True) * self.scale)
        out  = target + attn * v                         # residual
        return self.norm(out)


class MultimodalFaultNet(nn.Module):
    """
    3-branch multimodal network with cross-modal attention fusion.

    Branch A: GAF image     -> ResNet (64-d)
    Branch B: stat+env feat -> MLP    (32-d)
    Branch C: metadata      -> MLP    (16-d)

    All branches projected to same dim (32-d), then
    cross-modal attention applied between branches,
    concatenated -> ResNet-style dense head -> 4-class output.
    """
    def __init__(self):
        super().__init__()
        # PROJ_DIM = 48 (set in CONFIG above)

        # Branch A: GAF image
        self.gaf_enc  = GAFResNet(out_dim=64)
        self.gaf_proj = nn.Linear(64, PROJ_DIM)

        # Branch B: statistical + envelope features (18+4=22)
        self.feat_enc = nn.Sequential(
            nn.Linear(22, 64), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(64, PROJ_DIM), nn.GELU(),
        )

        # Branch C: metadata (8)
        self.meta_enc = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(),
            nn.Linear(32, PROJ_DIM), nn.GELU(),
        )

        # Cross-modal attention (paper Section 2.2)
        self.attn_gaf_feat = CrossModalAttention(PROJ_DIM)  # GAF attends to feat
        self.attn_gaf_meta = CrossModalAttention(PROJ_DIM)  # GAF attends to meta
        self.attn_feat_gaf = CrossModalAttention(PROJ_DIM)  # feat attends to GAF

        # Fusion head (paper Section 2.3 + ResNet final layers)
        self.fusion = nn.Sequential(
            nn.Linear(PROJ_DIM * 3, 96),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, N_CLASSES),
        )

    def forward(self, gaf, feat, meta):
        # Encode each modality
        g = self.gaf_proj(self.gaf_enc(gaf))    # (B, 32)
        f = self.feat_enc(feat)                  # (B, 32)
        m = self.meta_enc(meta)                  # (B, 32)

        # Cross-modal attention (modality interaction, paper Eq. 14-16)
        g2 = self.attn_gaf_feat(g, f)   # GAF enhanced by feat
        g3 = self.attn_gaf_meta(g2, m)  # GAF enhanced by meta
        f2 = self.attn_feat_gaf(f, g)   # feat enhanced by GAF

        # Concatenate all three enhanced representations
        fused = torch.cat([g3, f2, m], dim=1)  # (B, 96)
        return self.fusion(fused)


model = MultimodalFaultNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# ─── LOSS & OPTIMIZER ────────────────────────────────────────────────────────
# Weighted CrossEntropy for class imbalance (paper uses sparse categorical CE)
w = torch.tensor(1.0 / (cls_counts + 1e-9), dtype=torch.float32)
w = (w / w.sum() * N_CLASSES).to(DEVICE)
# label_smoothing=0 : smoothing inflated val loss to ~1.0 and made curves misleading
criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.0)
print(f"Class weights (Normal/InnerRing/Ball/OuterRing): {w.cpu().numpy().round(3)}")

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# LR warmup (5 epochs 0.1x → 1x) then cosine annealing
warmup_sched = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)
cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - 5, eta_min=1e-5
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[5]
)

# ─── TRAINING ────────────────────────────────────────────────────────────────

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot = 0
    preds_all, labels_all = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for gaf, feat, meta, labels in loader:
            gaf, feat, meta, labels = (gaf.to(DEVICE), feat.to(DEVICE),
                                        meta.to(DEVICE), labels.to(DEVICE))
            logits = model(gaf, feat, meta)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds = logits.argmax(1)
            tot_loss    += loss.item() * len(labels)
            tot_correct += (preds == labels).sum().item()
            tot         += len(labels)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return tot_loss / tot, tot_correct / tot, f1, preds_all, labels_all


print("\n" + "="*60)
print("Training GAF Multimodal Network")
print("="*60)

history = {'tr_loss':[], 'val_loss':[], 'tr_f1':[], 'val_f1':[]}
best_f1, patience, PATIENCE = 0.0, 0, 15

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_acc, vl_f1, vp, vl = run_epoch(val_dl,   train=False)
    scheduler.step()

    history['tr_loss'].append(tr_loss);   history['val_loss'].append(vl_loss)
    history['tr_f1'].append(tr_f1);       history['val_f1'].append(vl_f1)

    if vl_f1 > best_f1:
        best_f1 = vl_f1
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        patience = 0
    else:
        patience += 1

    if ep % 5 == 0 or ep == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Ep {ep:3d}/{EPOCHS}  tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f}  "
              f"val_loss={vl_loss:.4f} val_f1={vl_f1:.4f}  best={best_f1:.4f}  lr={lr_now:.2e}")

    if patience >= PATIENCE:
        print(f"\nEarly stop at epoch {ep}")
        break

# ─── FINAL EVALUATION ────────────────────────────────────────────────────────

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))
model.eval()

def evaluate(loader, name):
    preds_all, labels_all = [], []
    with torch.no_grad():
        for gaf, feat, meta, labels in loader:
            gaf, feat, meta = gaf.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            preds = model(gaf, feat, meta).argmax(1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())

    y_true = np.array(labels_all)
    y_pred = np.array(preds_all)

    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    print("False Positive Rate per class:")
    for c in range(N_CLASSES):
        fp  = np.sum((y_true != c) & (y_pred == c))
        tn  = np.sum((y_true != c) & (y_pred != c))
        fpr = fp / (fp + tn + 1e-10)
        print(f"  {label_names[c]:12s}: FPR = {fpr:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print("\nConfusion Matrix:")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{label_names[i]:12s}" + "".join(f"{v:>12d}" for v in row))

    return y_true, y_pred, cm


vt, vp, vcm = evaluate(val_dl, "Validation Set")

# False alarm evaluation
fa_preds, fa_labels = [], []
with torch.no_grad():
    for gaf, feat, meta, labels in fa_dl:
        gaf, feat, meta = gaf.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
        p = model(gaf, feat, meta).argmax(1)
        fa_preds.extend(p.cpu().numpy())
        fa_labels.extend(labels.numpy())

fa_preds  = np.array(fa_preds)
false_alarm_rate = np.mean(fa_preds != 0)
print(f"\n{'='*60}")
print("FALSE ALARM TEST (Folder 11 - Shaft Misalignment)")
print(f"  {np.sum(fa_preds != 0)} / {len(fa_preds)} samples predicted as fault")
print(f"  False Alarm Rate = {false_alarm_rate:.4f}  (lower is better, ideal = 0)")

# ─── PLOTS ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(history['tr_f1'],  label='Train', color='steelblue')
axes[0].plot(history['val_f1'], label='Val',   color='orangered')
axes[0].set_title('Macro F1'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['tr_loss'],  label='Train', color='steelblue')
axes[1].plot(history['val_loss'], label='Val',   color='orangered')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)

im = axes[2].imshow(vcm, cmap='Blues')
axes[2].set_xticks(range(N_CLASSES)); axes[2].set_xticklabels(label_names, rotation=30, ha='right')
axes[2].set_yticks(range(N_CLASSES)); axes[2].set_yticklabels(label_names)
axes[2].set_title('Confusion Matrix (Val)')
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('True')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        axes[2].text(j, i, str(vcm[i, j]), ha='center', va='center',
                     color='white' if vcm[i, j] > vcm.max() / 2 else 'black')
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/results.png")
print(f"Model saved: {OUTPUT_DIR}/best_model.pt")
print("\nDone!")
