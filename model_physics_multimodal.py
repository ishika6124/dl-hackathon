"""
Physics-Informed Multimodal Bearing Fault Diagnosis
=====================================================
Key innovations over baseline GAF+ResNet approach:

  1. EfficientNet-B0 backbone   – pretrained features, better sample efficiency
  2. FiLM conditioning          – metadata modulates stat/env feature extraction
  3. Physics-guided attention   – BPFI/BPFO/BSF amplitudes get explicit attention
  4. Supervised contrastive loss – NT-Xent pulls same-class embeds together
  5. Gated cross-modal fusion   – learnable per-sample modality weighting
  6. MC Dropout uncertainty     – epistemic confidence at inference
  7. Energy-based OOD head      – rejects shaft-misalign / external disturbances

Data: final_data/  (same format as baseline – run augment_data.py first)
Classes: 0=Normal, 1=Inner Ring, 2=Ball, 3=Outer Ring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import resample as scipy_resample
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, precision_recall_curve,
                             roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = "final_data"
OUTPUT_DIR  = "outputs_physics_v1"
GAF_SIZE    = 64
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 5e-4
DROPOUT     = 0.35
PROJ_DIM    = 64       # contrastive projection dimension (unit sphere)
EMB_DIM     = 64       # per-branch embedding dim
N_CLASSES   = 4
MC_PASSES   = 20       # Monte Carlo dropout inference passes
LAMBDA_CON  = 0.3      # contrastive loss weight
LAMBDA_PHY  = 0.1      # physics consistency loss weight
LAMBDA_OOD  = 0.05     # energy OOD regulariser weight
TEMPERATURE = 0.07     # NT-Xent temperature

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
print(f"Device: {DEVICE}")

# ─── GAF COMPUTATION ─────────────────────────────────────────────────────────

def compute_gasf(signal, size=GAF_SIZE):
    """Gramian Angular Summation Field – same as baseline."""
    sig = scipy_resample(signal, size).astype(np.float64)
    mn, mx = sig.min(), sig.max()
    if mx - mn < 1e-10:
        return np.zeros((size, size), dtype=np.float32)
    sig = np.clip(2.0 * (sig - mn) / (mx - mn) - 1.0, -1.0, 1.0)
    theta = np.arccos(sig)
    return np.cos(theta[:, None] + theta[None, :]).astype(np.float32)

# ─── DATASET ─────────────────────────────────────────────────────────────────

class BearingDataset(Dataset):
    def __init__(self, raw, stat, env, meta, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.stat   = torch.tensor(stat,   dtype=torch.float32)   # (N, 18)
        self.env    = torch.tensor(env,    dtype=torch.float32)    # (N,  4)
        self.meta   = torch.tensor(meta,   dtype=torch.float32)    # (N,  8)
        self.feat   = torch.cat([self.stat, self.env], dim=1)      # (N, 22)

        print(f"  Pre-computing {len(raw)} GAF images...", flush=True)
        gaf_np = np.stack([compute_gasf(raw[i]) for i in range(len(raw))])
        self.gaf = torch.tensor(gaf_np[:, None, :, :], dtype=torch.float32)
        print("  Done.", flush=True)

    def __len__(self):  return len(self.labels)

    def __getitem__(self, idx):
        return self.gaf[idx], self.feat[idx], self.meta[idx], self.labels[idx]


def load_split(name):
    return (np.load(f"{DATA_DIR}/X_raw_{name}.npy"),
            np.load(f"{DATA_DIR}/X_stat_{name}.npy"),
            np.load(f"{DATA_DIR}/X_env_{name}.npy"),
            np.load(f"{DATA_DIR}/X_meta_{name}.npy"),
            np.load(f"{DATA_DIR}/y_{name}.npy"))


print("Loading data...")
raw_tr, stat_tr, env_tr, meta_tr, y_train = load_split("train")
raw_val, stat_val, env_val, meta_val, y_val = load_split("val")
raw_fa, stat_fa, env_fa, meta_fa, y_fa     = load_split("false_alarm")

u, c = np.unique(y_train, return_counts=True)
print("Train distribution:", dict(zip(u.tolist(), c.tolist())))

print("\nBuilding datasets:")
train_ds = BearingDataset(raw_tr,  stat_tr,  env_tr,  meta_tr,  y_train)
val_ds   = BearingDataset(raw_val, stat_val, env_val, meta_val, y_val)
fa_ds    = BearingDataset(raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa)

cls_counts   = np.bincount(y_train, minlength=N_CLASSES).astype(float)
samp_weights = 1.0 / (cls_counts[y_train] + 1e-9)
sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)
fa_dl    = DataLoader(fa_ds,    batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)

# ─── MODEL COMPONENTS ────────────────────────────────────────────────────────

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet, adapted for 1-channel GAF input.

    Strategy: replicate 1-ch to 3-ch (no param added), freeze early layers,
    extract features from the final MBConv block, project to EMB_DIM.
    """
    def __init__(self, out_dim=EMB_DIM):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=True)

        # Keep all feature layers, drop the classifier
        self.features = backbone.features

        # Freeze first 4 blocks (low-level edges/textures already learned)
        params = list(self.features.parameters())
        for p in params[:len(params)//2]:
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256), nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, out_dim), nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 1, 64, 64) -> replicate to 3 channels
        x = x.expand(-1, 3, -1, -1)
        return self.head(self.pool(self.features(x)))


class PhysicsMLP(nn.Module):
    """
    Physics-guided feature encoder for stat+envelope features.

    Innovation: the 4 envelope features (BPFI, BPFO, BSF, FTF amplitudes)
    get a dedicated attention gate – the model learns to focus on the fault
    frequency that physically matches the predicted class.

    Also applies FiLM (Feature-wise Linear Modulation) from metadata:
    the context encoder outputs (gamma, beta) that scale/shift activations.
    """
    def __init__(self, feat_dim=22, meta_dim=8, out_dim=EMB_DIM):
        super().__init__()
        # Separate stat (18) from envelope (4)
        self.n_stat = 18
        self.n_env  = 4

        # Shared stat encoder
        self.stat_enc = nn.Sequential(
            nn.Linear(self.n_stat, 64), nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
        )

        # Envelope attention gate (physics-motivated)
        # Produces a soft attention weight for each of the 4 fault frequencies
        self.env_gate = nn.Sequential(
            nn.Linear(self.n_env, 16), nn.GELU(),
            nn.Linear(16, self.n_env), nn.Sigmoid(),  # (B, 4)
        )
        self.env_enc = nn.Sequential(
            nn.Linear(self.n_env, 32), nn.GELU(),
        )

        # FiLM conditioning: meta -> (gamma, beta) for stat features
        self.film = nn.Linear(meta_dim, 64 * 2)  # gamma + beta

        # Merge
        self.merge = nn.Sequential(
            nn.Linear(64 + 32, out_dim), nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
        )

    def forward(self, feat, meta):
        """
        feat: (B, 22)  = [stat(18) | env(4)]
        meta: (B,  8)
        """
        stat = feat[:, :self.n_stat]      # (B, 18)
        env  = feat[:, self.n_stat:]      # (B,  4)

        # Physics-guided envelope attention
        env_weights = self.env_gate(env)                      # (B, 4)
        env_attn    = env * env_weights                       # gated amplitudes
        env_emb     = self.env_enc(env_attn)                  # (B, 32)

        # Stat features with FiLM conditioning from metadata
        stat_emb = self.stat_enc(stat)                        # (B, 64)
        film_out = self.film(meta)                            # (B, 128)
        gamma, beta = film_out[:, :64], film_out[:, 64:]     # each (B, 64)
        stat_emb = gamma * stat_emb + beta                    # FiLM: scale+shift

        return self.merge(torch.cat([stat_emb, env_emb], dim=1))


class ContextEncoder(nn.Module):
    """
    Lightweight metadata encoder. Produces FiLM parameters for PhysicsMLP
    AND a direct embedding for fusion.
    """
    def __init__(self, meta_dim=8, out_dim=EMB_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.GELU(),
            nn.Linear(32, out_dim), nn.GELU(),
        )

    def forward(self, meta):
        return self.enc(meta)   # (B, EMB_DIM)


class GatedFusion(nn.Module):
    """
    Learnable gating: the network learns which modality to trust per sample.

    Gate vector g = sigmoid(W * [img_emb, feat_emb, meta_emb])
    Each of the 3 branches gets a scalar gate, then all are concatenated.

    This is interpretable: at inference, log the gates to see which
    modality drove the decision for each sample.
    """
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 3, 3),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(emb_dim * 3)

    def forward(self, g, f, m):
        concat = torch.cat([g, f, m], dim=-1)       # (B, 3*64)
        gates  = self.gate(concat)                   # (B, 3)  in [0,1]
        g_g, g_f, g_m = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        fused = torch.cat([g_g * g, g_f * f, g_m * m], dim=-1)
        return self.norm(fused), gates               # return gates for interpretability


class ProjectionHead(nn.Module):
    """Maps EMB_DIM embedding to unit sphere for contrastive learning."""
    def __init__(self, in_dim=EMB_DIM, out_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class PhysicsMultimodalNet(nn.Module):
    """
    Full model.

    Training outputs: logits, proj_g, proj_f, proj_m, energy, gates
    Inference mode:   mean/std over MC Dropout passes + OOD energy score
    """
    def __init__(self):
        super().__init__()
        # Branch encoders
        self.img_enc  = EfficientNetEncoder(out_dim=EMB_DIM)
        self.feat_enc = PhysicsMLP(feat_dim=22, meta_dim=8, out_dim=EMB_DIM)
        self.meta_enc = ContextEncoder(meta_dim=8, out_dim=EMB_DIM)

        # Contrastive projection heads (one per branch)
        self.proj_img  = ProjectionHead(EMB_DIM, PROJ_DIM)
        self.proj_feat = ProjectionHead(EMB_DIM, PROJ_DIM)
        self.proj_meta = ProjectionHead(EMB_DIM, PROJ_DIM)

        # Gated fusion
        self.fusion = GatedFusion(EMB_DIM)

        # Classifier head with MC Dropout
        self.classifier = nn.Sequential(
            nn.Linear(EMB_DIM * 3, 128), nn.GELU(),
            nn.Dropout(DROPOUT),          # KEPT active during MC inference
            nn.Linear(128, 64),  nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

        # OOD energy head: maps fused embedding to scalar energy
        # High energy = OOD / anomalous
        self.ood_head = nn.Sequential(
            nn.Linear(EMB_DIM * 3, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, gaf, feat, meta, return_gates=False):
        g = self.img_enc(gaf)
        f = self.feat_enc(feat, meta)
        m = self.meta_enc(meta)

        # Contrastive projections (for training loss only)
        pg = self.proj_img(g)
        pf = self.proj_feat(f)
        pm = self.proj_meta(m)

        # Gated fusion
        fused, gates = self.fusion(g, f, m)

        logits = self.classifier(fused)
        energy = self.ood_head(fused).squeeze(-1)    # (B,)

        if return_gates:
            return logits, pg, pf, pm, energy, gates
        return logits, pg, pf, pm, energy

    @torch.no_grad()
    def predict_with_uncertainty(self, gaf, feat, meta):
        """
        MC Dropout inference: run T forward passes with dropout active.
        Returns: mean_probs (B, 4), std_probs (B, 4), mean_energy (B,)
        """
        self.train()   # enables dropout
        all_probs  = []
        all_energy = []
        for _ in range(MC_PASSES):
            logits, _, _, _, energy = self.forward(gaf, feat, meta)
            all_probs.append(F.softmax(logits, dim=-1).unsqueeze(0))
            all_energy.append(energy.unsqueeze(0))
        self.eval()

        probs  = torch.cat(all_probs,  dim=0)   # (T, B, 4)
        energy = torch.cat(all_energy, dim=0)   # (T, B)

        mean_probs = probs.mean(0)               # (B, 4)
        std_probs  = probs.std(0)                # (B, 4) epistemic uncertainty
        mean_energy = energy.mean(0)             # (B,)
        return mean_probs, std_probs, mean_energy


model = PhysicsMultimodalNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# ─── LOSS FUNCTIONS ──────────────────────────────────────────────────────────

w = torch.tensor(1.0 / (cls_counts + 1e-9), dtype=torch.float32)
w = (w / w.sum() * N_CLASSES).to(DEVICE)
ce_loss = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
print(f"Class weights: {w.cpu().numpy().round(3)}")


def supervised_ntxent(z1, z2, labels, temperature=TEMPERATURE):
    """
    Supervised NT-Xent (contrastive) loss.
    Pulls together embeddings of the same class, pushes apart different classes.
    z1, z2: (B, PROJ_DIM) L2-normalised
    labels: (B,) class indices
    """
    B = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)          # (2B, D)
    l  = torch.cat([labels, labels], dim=0)  # (2B,)

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature   # (2B, 2B)

    # Mask: same class but not self
    mask_pos  = (l.unsqueeze(1) == l.unsqueeze(0)).float()
    mask_self = torch.eye(2 * B, device=z.device)
    mask_pos  = mask_pos - mask_self

    # Log-softmax over all negatives
    exp_sim     = torch.exp(sim)
    log_prob    = sim - torch.log(exp_sim.sum(1, keepdim=True) - exp_sim * mask_self)
    pos_pairs   = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-9)
    return -pos_pairs.mean()


def physics_consistency_loss(env_feat, logits):
    """
    Physics regulariser: for predicted Inner Ring class, BPFI amplitude
    (index 0 of env) should be high; similarly BPFO (idx 1) for Outer Ring,
    BSF (idx 2) for Ball. This soft constraint nudges the model toward
    physically interpretable decisions.

    env_feat: (B, 22) where last 4 are envelope amplitudes [BPFI, BPFO, BSF, FTF]
    logits:   (B, 4)
    """
    probs      = F.softmax(logits, dim=-1)            # (B, 4)
    env        = env_feat[:, 18:]                     # (B, 4): BPFI, BPFO, BSF, FTF
    # Normalise envelope per sample to [0,1]
    env_norm   = env / (env.max(dim=1, keepdim=True)[0] + 1e-9)

    # For each fault class, the corresponding envelope should rank highest
    # Class 1 (Inner Ring) -> env[:,0] should be max
    # Class 2 (Ball)       -> env[:,2]
    # Class 3 (Outer Ring) -> env[:,1]
    fault_probs   = probs[:, 1:]                      # (B, 3) ignore normal
    fault_env     = torch.stack([env_norm[:,0],       # BPFI for inner
                                 env_norm[:,2],       # BSF  for ball
                                 env_norm[:,1]], dim=1)  # BPFO for outer

    # We want: when model is confident about a fault class, the matching
    # envelope amplitude should be high. Loss = negative correlation.
    consistency = (fault_probs * fault_env).sum(dim=1)    # (B,)
    return -consistency.mean()


def energy_ood_loss(energy_in, energy_out_target=20.0):
    """
    Energy regulariser: push in-distribution samples to low energy.
    At test time, OOD samples will naturally have higher energy.
    """
    return torch.relu(energy_in - energy_out_target).mean()


# ─── OPTIMIZER & SCHEDULER ───────────────────────────────────────────────────

optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=2e-4
)
warmup   = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=8)
cosine   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 8, eta_min=5e-6)
sched    = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[8])

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_ce = tot_con = tot_phy = tot_ood = tot_correct = tot = 0
    preds_all, labels_all = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for gaf, feat, meta, labels in loader:
            gaf, feat, meta, labels = (gaf.to(DEVICE), feat.to(DEVICE),
                                       meta.to(DEVICE), labels.to(DEVICE))

            logits, pg, pf, pm, energy = model(gaf, feat, meta)

            # ── Classification loss
            l_ce  = ce_loss(logits, labels)

            if train:
                # ── Supervised contrastive (img vs feat projections)
                l_con = supervised_ntxent(pg, pf, labels) * LAMBDA_CON

                # ── Physics consistency
                l_phy = physics_consistency_loss(feat, logits) * LAMBDA_PHY

                # ── OOD energy (push in-dist to low energy)
                l_ood = energy_ood_loss(energy) * LAMBDA_OOD

                loss  = l_ce + l_con + l_phy + l_ood
                tot_con += l_con.item() * len(labels)
                tot_phy += l_phy.item() * len(labels)
                tot_ood += l_ood.item() * len(labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                loss = l_ce

            preds = logits.argmax(1)
            tot_loss    += loss.item() * len(labels)
            tot_ce      += l_ce.item() * len(labels)
            tot_correct += (preds == labels).sum().item()
            tot         += len(labels)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    losses = {
        'total': tot_loss / tot,
        'ce':    tot_ce   / tot,
        'con':   tot_con  / tot if train else 0,
        'phy':   tot_phy  / tot if train else 0,
        'ood':   tot_ood  / tot if train else 0,
    }
    return losses, tot_correct / tot, f1, preds_all, labels_all


print("\n" + "="*65)
print("Training Physics-Informed Multimodal Network")
print("="*65)

history = {k: [] for k in ['tr_loss','val_loss','tr_f1','val_f1',
                            'tr_ce','tr_con','tr_phy']}
best_f1, patience, PATIENCE = 0.0, 0, 18

for ep in range(1, EPOCHS + 1):
    tr_losses, tr_acc, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_losses, vl_acc, vl_f1, vp, vl = run_epoch(val_dl,   train=False)
    sched.step()

    history['tr_loss'].append(tr_losses['total'])
    history['val_loss'].append(vl_losses['total'])
    history['tr_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)
    history['tr_ce'].append(tr_losses['ce'])
    history['tr_con'].append(tr_losses['con'])
    history['tr_phy'].append(tr_losses['phy'])

    if vl_f1 > best_f1:
        best_f1 = vl_f1
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        patience = 0
    else:
        patience += 1

    if ep % 5 == 0 or ep == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Ep {ep:3d}/{EPOCHS}  "
              f"tr[tot={tr_losses['total']:.4f} ce={tr_losses['ce']:.4f} "
              f"con={tr_losses['con']:.4f} phy={tr_losses['phy']:.4f}]  "
              f"val_f1={vl_f1:.4f}  best={best_f1:.4f}  lr={lr_now:.2e}")

    if patience >= PATIENCE:
        print(f"\nEarly stop at epoch {ep}")
        break

# ─── FINAL EVALUATION ────────────────────────────────────────────────────────

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))
model.eval()


def evaluate_with_uncertainty(loader, name, ood=False):
    """Full evaluation with MC Dropout uncertainty and OOD detection."""
    all_mean_probs, all_std, all_energy, all_labels = [], [], [], []
    all_gates = []

    for gaf, feat, meta, labels in loader:
        gaf, feat, meta = gaf.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)

        mean_probs, std_probs, mean_energy = model.predict_with_uncertainty(
            gaf, feat, meta
        )

        # Also get gates for interpretability
        model.train()
        with torch.no_grad():
            _, _, _, _, _, gates = model(gaf, feat, meta, return_gates=True)
        model.eval()

        all_mean_probs.append(mean_probs.cpu())
        all_std.append(std_probs.cpu())
        all_energy.append(mean_energy.cpu())
        all_labels.append(labels)
        all_gates.append(gates.cpu())

    mean_probs = torch.cat(all_mean_probs).numpy()   # (N, 4)
    std_probs  = torch.cat(all_std).numpy()          # (N, 4)
    energies   = torch.cat(all_energy).numpy()       # (N,)
    y_true     = torch.cat(all_labels).numpy()
    gates      = torch.cat(all_gates).numpy()        # (N, 3)  img/feat/meta

    y_pred = mean_probs.argmax(1)
    uncertainty = std_probs.max(1)                   # epistemic uncertainty per sample

    print(f"\n{'='*65}")
    print(f"RESULTS: {name}")
    print('='*65)
    print(classification_report(y_true, y_pred, target_names=label_names,
                                zero_division=0))

    print("False Positive Rate per class:")
    for cls in range(N_CLASSES):
        fp  = np.sum((y_true != cls) & (y_pred == cls))
        tn  = np.sum((y_true != cls) & (y_pred != cls))
        fpr = fp / (fp + tn + 1e-10)
        print(f"  {label_names[cls]:12s}: FPR = {fpr:.4f}")

    print(f"\nEpistemic uncertainty (mean max-std): {uncertainty.mean():.4f}")
    print(f"Mean energy score: {energies.mean():.4f} ± {energies.std():.4f}")

    print("\nModal gate weights (mean per sample):")
    gate_names = ['Image (GAF)', 'Stat+Env', 'Metadata']
    for i, gn in enumerate(gate_names):
        print(f"  {gn:14s}: {gates[:, i].mean():.3f} ± {gates[:, i].std():.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print("\nConfusion Matrix:")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{label_names[i]:12s}" + "".join(f"{v:>12d}" for v in row))

    return y_true, y_pred, mean_probs, std_probs, energies, cm, gates


vt, vp, v_probs, v_std, v_energy, vcm, v_gates = evaluate_with_uncertainty(
    val_dl, "Validation Set"
)

# ─── OOD / FALSE ALARM EVALUATION ────────────────────────────────────────────

print(f"\n{'='*65}")
print("FALSE ALARM / OOD EVALUATION (Folder 11 – Shaft Misalignment)")

fa_preds_all, fa_energy_all, fa_labels_all = [], [], []
for gaf, feat, meta, labels in fa_dl:
    gaf, feat, meta = gaf.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
    mean_probs, _, mean_energy = model.predict_with_uncertainty(gaf, feat, meta)
    fa_preds_all.append(mean_probs.cpu())
    fa_energy_all.append(mean_energy.cpu())
    fa_labels_all.append(labels)

fa_probs  = torch.cat(fa_preds_all).numpy()
fa_energy = torch.cat(fa_energy_all).numpy()
fa_labels = torch.cat(fa_labels_all).numpy()
fa_preds  = fa_probs.argmax(1)

# Using energy threshold for OOD detection
# Threshold: 95th percentile of in-distribution validation energy
ood_threshold = np.percentile(v_energy, 95)
fa_ood_flags  = fa_energy > ood_threshold

false_alarm_raw    = np.mean(fa_preds != 0)
false_alarm_energy = np.mean((fa_preds != 0) & ~fa_ood_flags)  # after OOD filter

print(f"  Energy OOD threshold (95th pct of val): {ood_threshold:.4f}")
print(f"  FA samples flagged as OOD:  {fa_ood_flags.sum()} / {len(fa_ood_flags)} "
      f"({fa_ood_flags.mean():.1%})")
print(f"  Raw false alarm rate:       {false_alarm_raw:.4f}")
print(f"  After OOD filter:           {false_alarm_energy:.4f}")

# ─── ROC / PR CURVES ─────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("ROC AUC (one-vs-rest):")
y_bin = np.eye(N_CLASSES)[vt]
for i, name in enumerate(label_names):
    try:
        auc = roc_auc_score(y_bin[:, i], v_probs[:, i])
        print(f"  {name:12s}: {auc:.4f}")
    except Exception:
        print(f"  {name:12s}: N/A")

# ─── ABLATION STUDY HELPER ───────────────────────────────────────────────────

def ablation_run(disable_branch):
    """
    Quick ablation: zero out one branch's contribution to measure its impact.
    disable_branch: 'img', 'feat', or 'meta'
    """
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for gaf, feat, meta, labels in val_dl:
            gaf, feat, meta = gaf.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            # Zero out the specified branch by patching its output
            # We do a surgical forward pass
            g = model.img_enc(gaf)
            f = model.feat_enc(feat, meta)
            m = model.meta_enc(meta)

            if disable_branch == 'img':
                g = torch.zeros_like(g)
            elif disable_branch == 'feat':
                f = torch.zeros_like(f)
            elif disable_branch == 'meta':
                m = torch.zeros_like(m)

            fused, _ = model.fusion(g, f, m)
            logits   = model.classifier(fused)
            preds    = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())

    return f1_score(labels_all, preds_all, average='macro', zero_division=0)


print(f"\n{'='*65}")
print("ABLATION STUDY (val macro-F1 with each branch zeroed):")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))
model.eval()
full_f1  = f1_score(vt, vp, average='macro', zero_division=0)
print(f"  Full model:         F1 = {full_f1:.4f}")
for branch in ['img', 'feat', 'meta']:
    ab_f1 = ablation_run(branch)
    delta  = full_f1 - ab_f1
    print(f"  No {branch:6s} branch:  F1 = {ab_f1:.4f}  (Δ = -{delta:.4f})")

# ─── PLOTS ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Physics-Informed Multimodal Fault Diagnosis", fontsize=14)

# 1. F1 curves
ax = axes[0, 0]
ax.plot(history['tr_f1'],  label='Train', color='steelblue')
ax.plot(history['val_f1'], label='Val',   color='orangered')
ax.set_title('Macro F1'); ax.set_xlabel('Epoch')
ax.legend(); ax.grid(alpha=0.3)

# 2. Loss breakdown
ax = axes[0, 1]
ax.plot(history['tr_loss'], label='Total',       color='navy')
ax.plot(history['tr_ce'],   label='CE',          color='steelblue')
ax.plot(history['tr_con'],  label='Contrastive', color='darkorange')
ax.plot(history['tr_phy'],  label='Physics',     color='forestgreen')
ax.set_title('Training Loss Breakdown'); ax.set_xlabel('Epoch')
ax.legend(); ax.grid(alpha=0.3)

# 3. Confusion matrix
ax = axes[0, 2]
im = ax.imshow(vcm, cmap='Blues')
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(label_names, rotation=30, ha='right')
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(label_names)
ax.set_title('Confusion Matrix (Val)')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(vcm[i, j]), ha='center', va='center',
                color='white' if vcm[i, j] > vcm.max() / 2 else 'black')
plt.colorbar(im, ax=ax)

# 4. ROC curves
ax = axes[1, 0]
y_bin = np.eye(N_CLASSES)[vt]
colors_roc = ['#2196F3', '#F44336', '#FF9800', '#4CAF50']
for i, name in enumerate(label_names):
    try:
        fpr, tpr, _ = roc_curve(y_bin[:, i], v_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], v_probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=colors_roc[i])
    except Exception:
        pass
ax.plot([0,1],[0,1], 'k--', lw=0.8)
ax.set_title('ROC Curves (OvR)'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 5. Energy distribution: val (in-dist) vs false-alarm (OOD)
ax = axes[1, 1]
ax.hist(v_energy,  bins=30, alpha=0.7, label='Val (in-dist)',  color='steelblue')
ax.hist(fa_energy, bins=30, alpha=0.7, label='FA (OOD)',       color='orangered')
ax.axvline(ood_threshold, color='black', ls='--', lw=1.2, label=f'OOD thresh ({ood_threshold:.1f})')
ax.set_title('Energy Score Distribution'); ax.set_xlabel('Energy')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 6. Uncertainty per class
ax = axes[1, 2]
unc_per_class = [v_std[vt == i].max(1).mean() for i in range(N_CLASSES)]
bars = ax.bar(label_names, unc_per_class, color=colors_roc)
ax.set_title('Mean Epistemic Uncertainty by Class')
ax.set_ylabel('Mean max-std over MC passes')
ax.tick_params(axis='x', rotation=20)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=150, bbox_inches='tight')
print(f"\nPlots saved: {OUTPUT_DIR}/results.png")
print(f"Model saved: {OUTPUT_DIR}/best_model.pt")
print("\nDone!")
