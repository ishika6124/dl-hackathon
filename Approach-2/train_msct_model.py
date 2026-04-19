"""
Challenger: Multi-Scale 1D Conv + Transformer (MSConvFormer)
=============================================================
Alternative approach to the paper's GAF + ResNet + CrossModalAttention.

WHY THIS CHALLENGES THE PAPER:
  1. No information loss: GAF converts 1D→2D image (lossy). We process the raw
     signal directly in 1D, preserving full temporal ordering.
  2. Multi-scale: 4 parallel CNN branches with kernel sizes [7, 15, 31, 63]
     simultaneously capture high-frequency spikes (bearing impacts) and
     low-frequency envelope patterns at the correct physical time scales.
  3. Global attention: Transformer encoder uses self-attention across the entire
     sequence. Bearing faults are PERIODIC — self-attention is ideal for
     capturing these long-range repeating patterns. A local 2D CNN on a GAF
     image cannot easily learn global periodicity.
  4. Focal Loss: automatically down-weights easy examples (abundant normals)
     and focuses on hard/rare ones (ball faults). Better than weighted CE
     for extreme imbalance without needing manual weight tuning.
  5. Pre-LN Transformer: more stable gradients than Post-LN (standard).

Architecture:
  Signal branch (4096 raw samples):
    4x MultiScaleBranch(k=7,15,31,63) each → (B, 32, 32)
    Concat channels → (B, 128, 32) → transpose → (B, 32, 128)
    + Learnable positional encoding
    TransformerEncoder(d_model=128, nhead=4, layers=3, Pre-LN)
    Global avg pool → Linear(128→64) → (B, 64)

  Tabular branch (stat=18 + env=4 = 22 features):
    ResidualMLP(22→64→64) + LayerNorm → (B, 64)

  Meta branch (8 features):
    MLP(8→32→32) → (B, 32)

  Fusion:
    Concat(64+64+32=160) → Linear(160→64) → GELU → Dropout → Linear(64→4)

Data: final_data/  (run augment_data.py first — same data as GAF model)
Output: outputs_msct/
Classes: 0=Normal, 1=Inner Ring, 2=Ball, 3=Outer Ring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# --- CONFIG ------------------------------------------------------------------
DATA_DIR      = "final_data"
OUTPUT_DIR    = "/home/teaching/hackathon/Approach-2/outputs_msct"
BATCH_SIZE    = 32
EPOCHS        = 80
LR            = 1e-3
DROPOUT       = 0.4
N_CLASSES     = 4
FOCAL_GAMMA   = 2.0   # Focal Loss gamma: 0=standard CE, 2=heavy focus on hard examples
WARMUP_EPOCHS = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
print(f"Device: {DEVICE}")

# --- FOCAL LOSS --------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., RetinaNet 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 means:
      - Easy examples (p_t > 0.9): weight ≈ 0.01  → nearly ignored
      - Hard examples (p_t = 0.5): weight ≈ 0.25  → full contribution
    This is equivalent to automatically annealing class weights per-sample.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # class weights (alpha_t in focal loss notation)

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        return (((1.0 - p_t) ** self.gamma) * ce).mean()


# --- DATASET -----------------------------------------------------------------

class BearingDataset(Dataset):
    """
    Simple dataset — no GAF pre-computation needed.
    Raw signal stored as (B, 1, 4096) for Conv1d.
    """
    def __init__(self, raw, stat, env, meta, labels):
        # Add channel dim for Conv1d: (N, 4096) → (N, 1, 4096)
        self.raw    = torch.tensor(raw[:, None, :], dtype=torch.float32)
        # Stat + Env concatenated: (N, 22)
        self.feat   = torch.tensor(
            np.concatenate([stat, env], axis=1), dtype=torch.float32
        )
        self.meta   = torch.tensor(meta,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.raw[idx], self.feat[idx], self.meta[idx], self.labels[idx]


def load_split(name):
    raw  = np.load(f"{DATA_DIR}/X_raw_{name}.npy")
    stat = np.load(f"{DATA_DIR}/X_stat_{name}.npy")
    env  = np.load(f"{DATA_DIR}/X_env_{name}.npy")
    meta = np.load(f"{DATA_DIR}/X_meta_{name}.npy")
    y    = np.load(f"{DATA_DIR}/y_{name}.npy")
    return raw, stat, env, meta, y


print("Loading data...")
raw_tr,  stat_tr,  env_tr,  meta_tr,  y_train = load_split("train")
raw_val, stat_val, env_val, meta_val, y_val    = load_split("val")
raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa     = load_split("false_alarm")

u, c = np.unique(y_train, return_counts=True)
print("Train distribution:", dict(zip(u.tolist(), c.tolist())))
u2, c2 = np.unique(y_val, return_counts=True)
print("Val   distribution:", dict(zip(u2.tolist(), c2.tolist())))

train_ds = BearingDataset(raw_tr,  stat_tr,  env_tr,  meta_tr,  y_train)
val_ds   = BearingDataset(raw_val, stat_val, env_val, meta_val, y_val)
fa_ds    = BearingDataset(raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa)

# Weighted sampler
cls_counts   = np.bincount(y_train, minlength=N_CLASSES).astype(float)
samp_weights = 1.0 / (cls_counts[y_train] + 1e-9)
sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
fa_dl    = DataLoader(fa_ds,    batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)


# --- MODEL ARCHITECTURE ------------------------------------------------------

class MultiScaleBranch(nn.Module):
    """
    Single-kernel 1D CNN branch for one frequency scale.

    Input : (B, 1, 4096)
    Output: (B, out_ch=32, seq_len=32)

    Two conv+pool stages reduce 4096 → ~256 samples,
    then AdaptiveAvgPool1d to a fixed seq_len so all branches
    can be concatenated regardless of kernel size.

    Kernel sizes map to physical time scales:
      k=7  → short window  → captures high-freq bearing impacts
      k=15 → medium window → BPFI / BPFO harmonics
      k=31 → long window   → BPF modulation envelope
      k=63 → very long     → FTF (cage) frequency patterns
    """
    def __init__(self, kernel_size, out_ch=32, seq_len=32):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            # Stage 1: 4096 → ~1024
            nn.Conv1d(1,    16,     kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(16),  nn.GELU(),
            nn.MaxPool1d(2),
            # Stage 2: ~1024 → ~256
            nn.Conv1d(16,   out_ch, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x):
        return self.pool(self.conv(x))   # (B, out_ch, seq_len)


class ResidualMLP(nn.Module):
    """
    Two-layer MLP with residual connection and LayerNorm.
    Better gradient flow than plain MLP for tabular features.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(x)))
        h = self.fc2(h)
        return self.norm(h + self.skip(x))


class MSConvFormer(nn.Module):
    """
    Multi-Scale Convolutional Transformer for bearing fault diagnosis.

    Key architectural choices vs. the paper's GAF + ResNet2D:
      - Direct 1D signal processing (no lossy 1D→2D conversion)
      - Multi-scale branches with different receptive fields
      - Transformer self-attention for global periodic patterns
      - Pre-LN (norm_first=True) for stable training
      - Focal Loss (set in training) for class imbalance
    """
    KERNEL_SIZES = [7, 15, 31, 63]
    SEQ_LEN      = 32
    BRANCH_CH    = 32
    TRANS_DIM    = 128   # 4 branches × 32 channels = 128

    def __init__(self):
        super().__init__()

        # --- Signal branch ---
        self.branches = nn.ModuleList([
            MultiScaleBranch(k, out_ch=self.BRANCH_CH, seq_len=self.SEQ_LEN)
            for k in self.KERNEL_SIZES
        ])

        # Learnable positional encoding (shape: 1, seq_len, trans_dim)
        self.pos_enc = nn.Parameter(
            torch.randn(1, self.SEQ_LEN, self.TRANS_DIM) * 0.02
        )

        # Transformer encoder with Pre-LN (more stable than Post-LN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.TRANS_DIM,
            nhead=4,
            dim_feedforward=256,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True,    # Pre-LN: normalize before attention/FFN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)

        self.sig_head = nn.Sequential(
            nn.Linear(self.TRANS_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        # --- Tabular branch: stat(18) + env(4) = 22 ---
        self.tab_enc = ResidualMLP(22, 64, 64, dropout=DROPOUT)

        # --- Meta branch: 8 features ---
        self.meta_enc = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
        )

        # --- Fusion: 64 + 64 + 32 = 160 → 4 ---
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, raw, feat, meta):
        # --- Signal branch ---
        # Each branch: (B,1,4096) → (B, BRANCH_CH=32, SEQ_LEN=32)
        branch_outs = [b(raw) for b in self.branches]
        # Concat along channel dim: (B, 4*32=128, 32)
        x = torch.cat(branch_outs, dim=1)
        # Transpose for Transformer: (B, SEQ_LEN=32, TRANS_DIM=128)
        x = x.permute(0, 2, 1)
        # Add positional encoding
        x = x + self.pos_enc
        # Self-attention across the 32-step sequence
        x = self.transformer(x)      # (B, 32, 128)
        # Global average pool over time steps
        x = x.mean(dim=1)            # (B, 128)
        sig_emb = self.sig_head(x)   # (B, 64)

        # --- Tabular branch ---
        tab_emb  = self.tab_enc(feat)    # (B, 64)

        # --- Meta branch ---
        meta_emb = self.meta_enc(meta)   # (B, 32)

        # --- Fusion ---
        fused = torch.cat([sig_emb, tab_emb, meta_emb], dim=1)  # (B, 160)
        return self.classifier(fused)


model = MSConvFormer().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# --- LOSS & OPTIMIZER --------------------------------------------------------

w = torch.tensor(1.0 / (cls_counts + 1e-9), dtype=torch.float32)
w = (w / w.sum() * N_CLASSES).to(DEVICE)
# Focal Loss + class weights: double protection against imbalance
criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=w)
print(f"Class weights (Normal/InnerRing/Ball/OuterRing): {w.cpu().numpy().round(3)}")
print(f"Focal gamma: {FOCAL_GAMMA}")

# AdamW: Adam with decoupled weight decay (better regularization than Adam + L2)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# LR warmup (5 epochs: 0.1x → 1x) then cosine annealing
warmup_sched = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
)
cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
)

# --- TRAINING ----------------------------------------------------------------

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot = 0
    preds_all, labels_all = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for raw, feat, meta, labels in loader:
            raw, feat, meta, labels = (raw.to(DEVICE), feat.to(DEVICE),
                                       meta.to(DEVICE), labels.to(DEVICE))
            logits = model(raw, feat, meta)
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
print("Training MSConvFormer (Challenger vs GAF Model)")
print("="*60)

history = {'tr_loss': [], 'val_loss': [], 'tr_f1': [], 'val_f1': []}
best_f1, patience, PATIENCE = 0.0, 0, 15

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_acc, vl_f1, _, _ = run_epoch(val_dl,   train=False)
    scheduler.step()

    history['tr_loss'].append(tr_loss);  history['val_loss'].append(vl_loss)
    history['tr_f1'].append(tr_f1);      history['val_f1'].append(vl_f1)

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

# --- FINAL EVALUATION --------------------------------------------------------

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))
model.eval()


def evaluate(loader, name):
    preds_all, labels_all = [], []
    with torch.no_grad():
        for raw, feat, meta, labels in loader:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            preds = model(raw, feat, meta).argmax(1)
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
fa_preds = []
with torch.no_grad():
    for raw, feat, meta, labels in fa_dl:
        raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
        p = model(raw, feat, meta).argmax(1)
        fa_preds.extend(p.cpu().numpy())

fa_preds         = np.array(fa_preds)
false_alarm_rate = np.mean(fa_preds != 0)
print(f"\n{'='*60}")
print("FALSE ALARM TEST (Folder 11 - Shaft Misalignment)")
print(f"  {np.sum(fa_preds != 0)} / {len(fa_preds)} samples predicted as fault")
print(f"  False Alarm Rate = {false_alarm_rate:.4f}  (lower is better, ideal = 0)")

# --- PLOTS -------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(history['tr_f1'],  label='Train', color='steelblue')
axes[0].plot(history['val_f1'], label='Val',   color='orangered')
axes[0].set_title('Macro F1 — MSConvFormer')
axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['tr_loss'],  label='Train', color='steelblue')
axes[1].plot(history['val_loss'], label='Val',   color='orangered')
axes[1].set_title('Focal Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)

im = axes[2].imshow(vcm, cmap='Blues')
axes[2].set_xticks(range(N_CLASSES))
axes[2].set_xticklabels(label_names, rotation=30, ha='right')
axes[2].set_yticks(range(N_CLASSES))
axes[2].set_yticklabels(label_names)
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
print(f"Best model saved: {OUTPUT_DIR}/best_model.pt")
print("\nDone!")
