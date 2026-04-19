# """
# Improved Physics-Informed Hybrid Model
# =======================================
# Fixes all 3 root causes of the teammate's model underperformance (~0.79 F1):

# PROBLEM 1: Physics loss went NEGATIVE (destabilized training)
# FIX: Hinge-based physics regularization — always >= 0, bounded
#      "For Inner Ring fault, BPFI amplitude should dominate over BPFO/BPF/FTF"
#      Uses F.relu() so loss can only push in the right direction.

# PROBLEM 2: Multi-task loss weights were manual (CE + Contrastive + Physics)
# FIX: Kendall & Gal (2017) adaptive weighting — model LEARNS the weights.
#      L = sum_i [ exp(-s_i) * L_i + s_i ]  where s_i = log(sigma_i^2) learnable
#      When a loss term becomes stable, its weight increases automatically.
#      Eliminates manual tuning of lambda_1, lambda_2.

# PROBLEM 3: Naive contrastive loss (pull same class together without label awareness)
# FIX: Supervised Contrastive Loss (Khosla et al. 2020) — uses ground truth labels.
#      Each sample's positives = ALL other samples of the same class in the batch.
#      Forces fault classes to form tight, well-separated clusters in feature space.
#      Much stronger than naive contrastive for known-class problems.

# KEPT FROM TEAMMATE'S MODEL (genuinely useful):
#   - MC Dropout for epistemic uncertainty (just use model.train() at inference)
#   - Energy-based OOD detection (simple, works well)
#   - ROC + uncertainty plots

# BACKBONE: Our Hybrid architecture (Multi-scale 1D CNN + Transformer + Cross-modal
#           Attention) — achieved ~0.94 Macro F1 vs teammate's ~0.79.
#           The cross-modal attention lets envelope spectrum features CORRECT
#           the signal's prediction (fixes Normal→FaultClass false alarms).

# Output: outputs_pi_hybrid/
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import os, warnings
# warnings.filterwarnings('ignore')

# torch.manual_seed(42)
# np.random.seed(42)

# # --- CONFIG ------------------------------------------------------------------
# DATA_DIR        = "final_data"
# OUTPUT_DIR      = "outputs_pi_hybrid"
# BATCH_SIZE      = 64      # larger batch → more positive pairs for SupCon
# EPOCHS          = 100
# LR              = 1e-3
# DROPOUT         = 0.4
# PROJ_DIM        = 48
# N_CLASSES       = 4
# SUPCON_TEMP     = 0.07    # SupCon temperature (standard value from paper)
# WARMUP_EPOCHS   = 5
# MC_PASSES       = 20      # MC Dropout inference passes for uncertainty
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
# print(f"Device: {DEVICE}")

# # Bearing physics: class → expected dominant envelope frequency index
# # Envelope features: [BPFI(0), BPFO(1), BPF(2), FTF(3)]
# FAULT_FREQ_MAP = {
#     1: 0,   # Inner Ring → BPFI amplitude should dominate
#     3: 1,   # Outer Ring → BPFO amplitude should dominate
#     2: 2,   # Ball       → BPF  amplitude should dominate
#     # Normal (0): no constraint
# }


# # --- LOSSES ------------------------------------------------------------------

# class SupConLoss(nn.Module):
#     """
#     Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

#     Why better than naive contrastive:
#       Naive: only one positive per anchor (augmented pair)
#       SupCon: ALL samples of the same class are positives
#       → Forces 4 tight class clusters in projection space
#       → Inner Ring, Ball, Outer Ring clusters become clearly separated
#       → Fixes the Normal→FaultClass confusion

#     math:
#       L = mean over anchors of:
#           -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p/τ) / sum_{a≠i} exp(z_i·z_a/τ) )
#       where P(i) = set of same-class samples, τ = temperature
#     """
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.T = temperature

#     def forward(self, features, labels):
#         """
#         features : (B, D) L2-normalized projection vectors
#         labels   : (B,)   integer class labels
#         """
#         B      = features.shape[0]
#         device = features.device

#         # Cosine similarity matrix (since features are L2-normalized: dot = cosine)
#         sim = torch.mm(features, features.T) / self.T    # (B, B)

#         # Positive mask: same class, exclude self
#         labels_col = labels.unsqueeze(1)
#         pos_mask   = (labels_col == labels_col.T).float()
#         pos_mask.fill_diagonal_(0.0)

#         # Self mask for denominator
#         self_mask = torch.eye(B, device=device)

#         # Numerical stability: subtract row max
#         sim_max, _ = sim.max(dim=1, keepdim=True)
#         sim        = sim - sim_max.detach()

#         # Denominator: all non-self pairs
#         exp_sim  = torch.exp(sim) * (1 - self_mask)
#         log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

#         # Log probability for each pair
#         log_prob = sim - log_denom

#         # Average over positive pairs (only include anchors with ≥1 positive)
#         n_pos   = pos_mask.sum(dim=1)
#         valid   = n_pos > 0
#         if valid.sum() == 0:
#             return torch.tensor(0.0, device=device, requires_grad=True)

#         per_anchor_loss = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-9)
#         return per_anchor_loss[valid].mean()


# class PhysicsRegularizer(nn.Module):
#     """
#     Bearing physics constraint — always NON-NEGATIVE (fixes teammate's issue).

#     Constraint: for fault class k, the corresponding envelope spectrum
#     amplitude should EXCEED all other amplitudes by a margin.

#     Implementation: hinge loss F.relu(max_other - dominant + margin)
#       → If dominant > max_other: loss = 0  (constraint satisfied, no gradient)
#       → If dominant < max_other: loss > 0  (push dominant amplitude up)
#       → Always >= 0 by construction (F.relu)

#     Normal class is excluded (no dominant fault frequency expected).
#     """
#     MARGIN = 0.05   # minimum margin dominant freq should exceed others

#     def forward(self, env_features, labels):
#         """
#         env_features : (B, 4) scaled BPFI, BPFO, BPF, FTF amplitudes
#         labels       : (B,)
#         """
#         loss = torch.zeros(1, device=env_features.device)
#         n    = 0
#         for cls, freq_idx in FAULT_FREQ_MAP.items():
#             mask = (labels == cls)
#             if mask.sum() == 0:
#                 continue
#             env_cls   = env_features[mask]          # (n_cls, 4)
#             dominant  = env_cls[:, freq_idx]        # amplitude at fault freq
#             others    = torch.cat([
#                 env_cls[:, :freq_idx], env_cls[:, freq_idx+1:]
#             ], dim=1)
#             max_other = others.max(dim=1).values
#             # Hinge: dominant must exceed max_other by MARGIN
#             loss = loss + F.relu(max_other - dominant + self.MARGIN).mean()
#             n   += 1

#         return loss / max(n, 1)


# # --- DATASET -----------------------------------------------------------------

# class BearingDataset(Dataset):
#     def __init__(self, raw, stat, env, meta, labels):
#         self.raw    = torch.tensor(raw[:, None, :], dtype=torch.float32)
#         self.feat   = torch.tensor(
#             np.concatenate([stat, env], axis=1), dtype=torch.float32   # (N, 22)
#         )
#         self.meta   = torch.tensor(meta,   dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self): return len(self.labels)

#     def __getitem__(self, idx):
#         return self.raw[idx], self.feat[idx], self.meta[idx], self.labels[idx]


# def load_split(name):
#     return (
#         np.load(f"{DATA_DIR}/X_raw_{name}.npy"),
#         np.load(f"{DATA_DIR}/X_stat_{name}.npy"),
#         np.load(f"{DATA_DIR}/X_env_{name}.npy"),
#         np.load(f"{DATA_DIR}/X_meta_{name}.npy"),
#         np.load(f"{DATA_DIR}/y_{name}.npy"),
#     )


# print("Loading data...")
# raw_tr,  stat_tr,  env_tr,  meta_tr,  y_train = load_split("train")
# raw_val, stat_val, env_val, meta_val, y_val    = load_split("val")
# raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa     = load_split("false_alarm")

# u, c = np.unique(y_train, return_counts=True)
# print("Train distribution:", dict(zip(u.tolist(), c.tolist())))
# u2, c2 = np.unique(y_val, return_counts=True)
# print("Val   distribution:", dict(zip(u2.tolist(), c2.tolist())))

# train_ds = BearingDataset(raw_tr,  stat_tr,  env_tr,  meta_tr,  y_train)
# val_ds   = BearingDataset(raw_val, stat_val, env_val, meta_val, y_val)
# fa_ds    = BearingDataset(raw_fa,  stat_fa,  env_fa,  meta_fa,  y_fa)

# cls_counts   = np.bincount(y_train, minlength=N_CLASSES).astype(float)
# samp_weights = 1.0 / (cls_counts[y_train] + 1e-9)
# sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

# train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
# val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
# fa_dl    = DataLoader(fa_ds,    batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)


# # --- MODEL BUILDING BLOCKS ---------------------------------------------------

# class MultiScaleBranch(nn.Module):
#     def __init__(self, kernel_size, out_ch=32, seq_len=32):
#         super().__init__()
#         pad = kernel_size // 2
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size, stride=2, padding=pad, bias=False),
#             nn.BatchNorm1d(16),     nn.GELU(), nn.MaxPool1d(2),
#             nn.Conv1d(16, out_ch, kernel_size, stride=2, padding=pad, bias=False),
#             nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2),
#         )
#         self.pool = nn.AdaptiveAvgPool1d(seq_len)

#     def forward(self, x): return self.pool(self.conv(x))


# class ResidualMLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
#         super().__init__()
#         self.fc1  = nn.Linear(in_dim, hidden_dim)
#         self.fc2  = nn.Linear(hidden_dim, out_dim)
#         self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
#         self.norm = nn.LayerNorm(out_dim)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.norm(self.fc2(self.drop(F.gelu(self.fc1(x)))) + self.skip(x))


# class CrossModalAttention(nn.Module):
#     """
#     Paper's cross-modal attention: target queries source.
#     Tabular (envelope spectrum) can suppress wrong signal predictions.
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.Wq = nn.Linear(dim, dim, bias=False)
#         self.Wk = nn.Linear(dim, dim, bias=False)
#         self.Wv = nn.Linear(dim, dim, bias=False)
#         self.scale = dim ** -0.5
#         self.norm  = nn.LayerNorm(dim)

#     def forward(self, target, source):
#         q    = self.Wq(target); k = self.Wk(source); v = self.Wv(source)
#         attn = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True) * self.scale)
#         return self.norm(target + attn * v)


# # --- MAIN MODEL --------------------------------------------------------------

# class PhysicsHybridNet(nn.Module):
#     """
#     Physics-Informed Hybrid Network

#     Backbone (from our Hybrid model — best classification performance):
#       4x MultiScaleBranch(k=7,15,31,63) → concat → Transformer → project
#       ResidualMLP for tabular, MLP for meta
#       Cross-modal attention (sig↔tab, sig↔meta, tab↔sig)
#       → fused embedding (B, PROJ_DIM*3)

#     Added for physics-informed learning:
#       Projection head: fused → 64-dim L2-normalized (for SupCon loss)
#       Classifier head: fused → 4 logits (for Focal loss)
#       Adaptive loss weights: 3 learnable log-variance parameters

#     Uncertainty (MC Dropout):
#       Standard Dropout is used throughout.
#       At inference: call model.train() + run N forward passes → variance = uncertainty.

#     OOD Detection (Energy Score):
#       E(x) = -T * log sum_k exp(f_k(x) / T)
#       Lower energy = in-distribution, higher = OOD.
#       No additional training needed — works on classifier logits.
#     """
#     KERNEL_SIZES = [7, 15, 31, 63]
#     SEQ_LEN      = 32
#     BRANCH_CH    = 32
#     TRANS_DIM    = 128

#     def __init__(self):
#         super().__init__()

#         # ── Signal branch (Multi-scale 1D CNN + Transformer) ──
#         self.branches = nn.ModuleList([
#             MultiScaleBranch(k, out_ch=self.BRANCH_CH, seq_len=self.SEQ_LEN)
#             for k in self.KERNEL_SIZES
#         ])
#         self.pos_enc = nn.Parameter(
#             torch.randn(1, self.SEQ_LEN, self.TRANS_DIM) * 0.02
#         )
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=self.TRANS_DIM, nhead=4, dim_feedforward=256,
#             dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
#         self.sig_head    = nn.Sequential(
#             nn.Linear(self.TRANS_DIM, 64), nn.GELU(), nn.Dropout(DROPOUT),
#         )
#         self.sig_proj = nn.Linear(64, PROJ_DIM)

#         # ── Tabular branch ──
#         self.tab_enc  = ResidualMLP(22, 64, 64)
#         self.tab_proj = nn.Linear(64, PROJ_DIM)

#         # ── Meta branch ──
#         self.meta_enc = nn.Sequential(
#             nn.Linear(8, 32), nn.GELU(), nn.Linear(32, PROJ_DIM), nn.GELU(),
#         )

#         # ── Cross-modal attention ──
#         self.attn_sig_tab  = CrossModalAttention(PROJ_DIM)
#         self.attn_sig_meta = CrossModalAttention(PROJ_DIM)
#         self.attn_tab_sig  = CrossModalAttention(PROJ_DIM)

#         fused_dim = PROJ_DIM * 3  # 144

#         # ── Classifier head → logits ──
#         self.classifier = nn.Sequential(
#             nn.Linear(fused_dim, 96), nn.GELU(), nn.Dropout(DROPOUT),
#             nn.Linear(96, N_CLASSES),
#         )

#         # ── Projection head → L2 space for SupCon ──
#         self.proj_head = nn.Sequential(
#             nn.Linear(fused_dim, 128), nn.GELU(),
#             nn.Linear(128, 64),
#         )

#         # ── Adaptive loss weights (Kendall & Gal 2017) ──
#         # L = exp(-s) * L_i + s  where s = log(sigma^2)
#         # Model learns optimal balance without manual tuning
#         self.log_var_focal   = nn.Parameter(torch.zeros(1))
#         self.log_var_supcon  = nn.Parameter(torch.zeros(1))
#         self.log_var_physics = nn.Parameter(torch.zeros(1))

#     def encode(self, raw, feat, meta):
#         """Compute fused embedding."""
#         # Signal
#         x = torch.cat([b(raw) for b in self.branches], dim=1).permute(0, 2, 1)
#         x = self.transformer(x + self.pos_enc).mean(dim=1)
#         sig_emb  = self.sig_proj(self.sig_head(x))

#         # Tabular + meta
#         tab_emb  = self.tab_proj(self.tab_enc(feat))
#         meta_emb = self.meta_enc(meta)

#         # Cross-modal attention (tabular corrects signal)
#         sig2 = self.attn_sig_tab(sig_emb,  tab_emb)
#         sig3 = self.attn_sig_meta(sig2,    meta_emb)
#         tab2 = self.attn_tab_sig(tab_emb,  sig_emb)

#         return torch.cat([sig3, tab2, meta_emb], dim=1)   # (B, 144)

#     def forward(self, raw, feat, meta):
#         emb    = self.encode(raw, feat, meta)
#         logits = self.classifier(emb)
#         proj   = F.normalize(self.proj_head(emb), dim=1)  # L2-normalized for SupCon
#         return logits, proj

#     def adaptive_loss(self, focal_l, supcon_l, physics_l):
#         """
#         Kendall & Gal (2017) multi-task loss with learned uncertainty weighting.
#         L_total = sum_i [ exp(-s_i) * L_i + s_i ]
#         where s_i = log_var_i (learnable)

#         exp(-s_i) acts as precision (inverse variance) weight.
#         s_i regularization prevents weights from collapsing.
#         No manual lambda tuning needed.
#         """
#         s_f = self.log_var_focal
#         s_c = self.log_var_supcon
#         s_p = self.log_var_physics

#         return (
#             torch.exp(-s_f) * focal_l   + s_f +
#             torch.exp(-s_c) * supcon_l  + s_c +
#             torch.exp(-s_p) * physics_l + s_p
#         )


# model = PhysicsHybridNet().to(DEVICE)
# n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Model parameters: {n_params:,}")

# # --- LOSS FUNCTIONS ----------------------------------------------------------

# # Focal loss for primary classification
# w = torch.tensor(1.0 / (cls_counts + 1e-9), dtype=torch.float32)
# w = (w / w.sum() * N_CLASSES).to(DEVICE)


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, weight=None):
#         super().__init__()
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, logits, targets):
#         ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
#         p_t = torch.exp(-ce)
#         return (((1.0 - p_t) ** self.gamma) * ce).mean()


# focal_criterion   = FocalLoss(gamma=2.0, weight=w)
# supcon_criterion  = SupConLoss(temperature=SUPCON_TEMP)
# physics_criterion = PhysicsRegularizer()

# print(f"Class weights: {w.cpu().numpy().round(3)}")

# # --- OPTIMIZER ---------------------------------------------------------------

# optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
# cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)
# scheduler = optim.lr_scheduler.SequentialLR(
#     optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
# )

# # --- TRAINING ----------------------------------------------------------------

# def run_epoch(loader, train=True):
#     model.train() if train else model.eval()
#     tot_focal = tot_supcon = tot_physics = tot_total = 0
#     tot_correct = tot = 0
#     preds_all, labels_all = [], []

#     ctx = torch.enable_grad() if train else torch.no_grad()
#     with ctx:
#         for raw, feat, meta, labels in loader:
#             raw, feat, meta, labels = (
#                 raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
#             )
#             logits, proj = model(raw, feat, meta)

#             # Individual losses
#             focal_l   = focal_criterion(logits, labels)
#             supcon_l  = supcon_criterion(proj, labels)
#             env_feat  = feat[:, 18:22]          # BPFI, BPFO, BPF, FTF (last 4 of 22)
#             physics_l = physics_criterion(env_feat, labels)

#             # Adaptive weighted total
#             total_l = model.adaptive_loss(focal_l, supcon_l, physics_l)

#             if train:
#                 optimizer.zero_grad()
#                 total_l.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()

#             preds = logits.argmax(1)
#             B = len(labels)
#             tot_focal   += focal_l.item()   * B
#             tot_supcon  += supcon_l.item()  * B
#             tot_physics += physics_l.item() * B
#             tot_total   += total_l.item()   * B
#             tot_correct += (preds == labels).sum().item()
#             tot         += B
#             preds_all.extend(preds.cpu().numpy())
#             labels_all.extend(labels.cpu().numpy())

#     f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
#     losses = {
#         'total':   tot_total   / tot,
#         'focal':   tot_focal   / tot,
#         'supcon':  tot_supcon  / tot,
#         'physics': tot_physics / tot,
#     }
#     return losses, tot_correct / tot, f1, preds_all, labels_all


# print("\n" + "="*60)
# print("Training Physics-Informed Hybrid Model")
# print("  Backbone : Multi-scale 1D CNN + Transformer + CMA")
# print("  Loss     : Focal + SupCon + Physics (adaptive weights)")
# print("="*60)

# history = {k: [] for k in ['tr_total', 'val_total', 'tr_focal', 'val_focal',
#                              'tr_supcon', 'val_supcon', 'tr_f1', 'val_f1',
#                              'lv_focal', 'lv_supcon', 'lv_physics']}
# best_f1, patience, PATIENCE = 0.0, 0, 20

# for ep in range(1, EPOCHS + 1):
#     tr_losses, tr_acc, tr_f1, _, _ = run_epoch(train_dl, train=True)
#     vl_losses, vl_acc, vl_f1, _, _ = run_epoch(val_dl,   train=False)
#     scheduler.step()

#     history['tr_total'].append(tr_losses['total'])
#     history['val_total'].append(vl_losses['total'])
#     history['tr_focal'].append(tr_losses['focal'])
#     history['val_focal'].append(vl_losses['focal'])
#     history['tr_supcon'].append(tr_losses['supcon'])
#     history['val_supcon'].append(vl_losses['supcon'])
#     history['tr_f1'].append(tr_f1)
#     history['val_f1'].append(vl_f1)
#     history['lv_focal'].append(model.log_var_focal.item())
#     history['lv_supcon'].append(model.log_var_supcon.item())
#     history['lv_physics'].append(model.log_var_physics.item())

#     if vl_f1 > best_f1:
#         best_f1 = vl_f1
#         torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
#         patience = 0
#     else:
#         patience += 1

#     if ep % 5 == 0 or ep == 1:
#         lr_now = optimizer.param_groups[0]['lr']
#         lv_f = model.log_var_focal.item()
#         lv_c = model.log_var_supcon.item()
#         lv_p = model.log_var_physics.item()
#         print(
#             f"Ep {ep:3d}  "
#             f"tr_f1={tr_f1:.4f} vl_f1={vl_f1:.4f} best={best_f1:.4f}  "
#             f"focal={tr_losses['focal']:.3f} sup={tr_losses['supcon']:.3f} "
#             f"phy={tr_losses['physics']:.3f}  "
#             f"lv=[{lv_f:.2f},{lv_c:.2f},{lv_p:.2f}]  lr={lr_now:.2e}"
#         )

#     if patience >= PATIENCE:
#         print(f"\nEarly stop at epoch {ep}")
#         break

# # --- FINAL EVALUATION --------------------------------------------------------

# model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))


# def evaluate(loader, name):
#     model.eval()
#     preds_all, labels_all, logits_all, probs_all = [], [], [], []
#     with torch.no_grad():
#         for raw, feat, meta, labels in loader:
#             raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
#             logits, _ = model(raw, feat, meta)
#             preds_all.extend(logits.argmax(1).cpu().numpy())
#             labels_all.extend(labels.numpy())
#             logits_all.append(logits.cpu())
#             probs_all.append(F.softmax(logits, dim=1).cpu())

#     y_true   = np.array(labels_all)
#     y_pred   = np.array(preds_all)
#     logits_t = torch.cat(logits_all)
#     probs_t  = torch.cat(probs_all).numpy()

#     print(f"\n{'='*60}")
#     print(f"RESULTS: {name}")
#     print('='*60)
#     print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

#     print("False Positive Rate per class:")
#     for c in range(N_CLASSES):
#         fp  = np.sum((y_true != c) & (y_pred == c))
#         tn  = np.sum((y_true != c) & (y_pred != c))
#         print(f"  {label_names[c]:12s}: FPR = {fp/(fp+tn+1e-10):.4f}")

#     cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
#     print("\nConfusion Matrix:")
#     header = f"{'':12s}" + "".join(f"{n:>12s}" for n in label_names)
#     print(header)
#     for i, row in enumerate(cm):
#         print(f"{label_names[i]:12s}" + "".join(f"{v:>12d}" for v in row))

#     # Energy scores for OOD detection
#     energy = (-1.0 * torch.logsumexp(logits_t, dim=1)).numpy()

#     return y_true, y_pred, cm, probs_t, energy


# vt, vp, vcm, vprobs, v_energy = evaluate(val_dl, "Validation Set")

# # --- MC DROPOUT UNCERTAINTY --------------------------------------------------

# print("\n--- MC Dropout Uncertainty Estimation ---")
# model.train()   # keep dropout active for MC sampling
# mc_probs_list = []
# with torch.no_grad():
#     for _ in range(MC_PASSES):
#         batch_probs = []
#         for raw, feat, meta, _ in val_dl:
#             raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
#             logits, _ = model(raw, feat, meta)
#             batch_probs.append(F.softmax(logits, dim=1).cpu())
#         mc_probs_list.append(torch.cat(batch_probs).numpy())

# mc_probs_arr = np.stack(mc_probs_list)    # (MC_PASSES, N_val, 4)
# mc_mean      = mc_probs_arr.mean(axis=0)  # (N_val, 4) — use for final prediction
# mc_std       = mc_probs_arr.std(axis=0)   # (N_val, 4) — uncertainty

# # Max-std as epistemic uncertainty per sample
# uncertainty  = mc_std.max(axis=1)         # (N_val,)

# print("Mean epistemic uncertainty by class:")
# for c in range(N_CLASSES):
#     mask = (vt == c)
#     print(f"  {label_names[c]:12s}: {uncertainty[mask].mean():.4f}")

# # --- FALSE ALARM + OOD -------------------------------------------------------

# model.eval()
# fa_preds, fa_logits = [], []
# with torch.no_grad():
#     for raw, feat, meta, _ in fa_dl:
#         raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
#         logits, _ = model(raw, feat, meta)
#         fa_preds.extend(logits.argmax(1).cpu().numpy())
#         fa_logits.append(logits.cpu())

# fa_preds  = np.array(fa_preds)
# fa_logits = torch.cat(fa_logits)
# fa_energy = (-1.0 * torch.logsumexp(fa_logits, dim=1)).numpy()

# false_alarm_rate = np.mean(fa_preds != 0)
# print(f"\n{'='*60}")
# print("FALSE ALARM TEST (Folder 11 - Shaft Misalignment)")
# print(f"  {np.sum(fa_preds != 0)} / {len(fa_preds)} predicted as fault")
# print(f"  False Alarm Rate = {false_alarm_rate:.4f}  (ideal = 0)")

# # OOD separation
# ood_threshold = np.percentile(v_energy, 95)   # 95th percentile of val energy
# ood_flagged   = np.mean(fa_energy > ood_threshold)
# print(f"  Energy OOD detection rate = {ood_flagged:.4f}  (higher = better OOD separation)")

# # --- PLOTS -------------------------------------------------------------------

# fig = plt.figure(figsize=(20, 10))
# fig.suptitle("Physics-Informed Hybrid Model — Results", fontsize=13, fontweight='bold')

# # 1. Macro F1
# ax1 = fig.add_subplot(2, 4, 1)
# ax1.plot(history['tr_f1'],  label='Train', color='steelblue')
# ax1.plot(history['val_f1'], label='Val',   color='orangered')
# ax1.set_title('Macro F1'); ax1.set_xlabel('Epoch')
# ax1.legend(); ax1.grid(alpha=0.3)

# # 2. Loss breakdown
# ax2 = fig.add_subplot(2, 4, 2)
# ax2.plot(history['tr_focal'],  label='Focal',   color='steelblue')
# ax2.plot(history['tr_supcon'], label='SupCon',  color='darkorange')
# ax2.plot(history['tr_total'],  label='Total',   color='green', linewidth=2)
# ax2.set_title('Training Loss Breakdown'); ax2.set_xlabel('Epoch')
# ax2.legend(); ax2.grid(alpha=0.3)

# # 3. Adaptive loss weights (log variance)
# ax3 = fig.add_subplot(2, 4, 3)
# ax3.plot(history['lv_focal'],   label='s_focal',   color='steelblue')
# ax3.plot(history['lv_supcon'],  label='s_supcon',  color='darkorange')
# ax3.plot(history['lv_physics'], label='s_physics', color='green')
# ax3.set_title('Learned Loss Weights (log var)\nLower = higher weight')
# ax3.set_xlabel('Epoch'); ax3.legend(); ax3.grid(alpha=0.3)

# # 4. Confusion matrix
# ax4 = fig.add_subplot(2, 4, 4)
# im = ax4.imshow(vcm, cmap='Blues')
# ax4.set_xticks(range(N_CLASSES)); ax4.set_xticklabels(label_names, rotation=30, ha='right')
# ax4.set_yticks(range(N_CLASSES)); ax4.set_yticklabels(label_names)
# ax4.set_title('Confusion Matrix (Val)')
# ax4.set_xlabel('Predicted'); ax4.set_ylabel('True')
# for i in range(N_CLASSES):
#     for j in range(N_CLASSES):
#         ax4.text(j, i, str(vcm[i, j]), ha='center', va='center',
#                  color='white' if vcm[i, j] > vcm.max() / 2 else 'black')
# plt.colorbar(im, ax=ax4)

# # 5. ROC curves (one-vs-rest)
# ax5 = fig.add_subplot(2, 4, 5)
# colors = ['steelblue', 'darkorange', 'green', 'red']
# from sklearn.preprocessing import label_binarize
# y_bin = label_binarize(vt, classes=list(range(N_CLASSES)))
# for c in range(N_CLASSES):
#     fpr, tpr, _ = roc_curve(y_bin[:, c], vprobs[:, c])
#     roc_auc = auc(fpr, tpr)
#     ax5.plot(fpr, tpr, color=colors[c], label=f'{label_names[c]} ({roc_auc:.3f})')
# ax5.plot([0,1], [0,1], 'k--', alpha=0.5)
# ax5.set_title('ROC Curves (OvR)'); ax5.set_xlabel('FPR'); ax5.set_ylabel('TPR')
# ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

# # 6. Energy score distribution (OOD detection)
# ax6 = fig.add_subplot(2, 4, 6)
# ax6.hist(v_energy,  bins=40, alpha=0.6, color='steelblue', label='Val (in-dist)')
# ax6.hist(fa_energy, bins=40, alpha=0.6, color='orangered',  label='FA (OOD)')
# ax6.axvline(ood_threshold, color='black', linestyle='--',
#             label=f'OOD thresh ({ood_threshold:.2f})')
# ax6.set_title('Energy Score Distribution'); ax6.set_xlabel('Energy')
# ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

# # 7. Epistemic uncertainty by class
# ax7 = fig.add_subplot(2, 4, 7)
# unc_by_class = [uncertainty[vt == c].mean() for c in range(N_CLASSES)]
# ax7.bar(label_names, unc_by_class, color=colors)
# ax7.set_title(f'Epistemic Uncertainty by Class\n(MC Dropout, {MC_PASSES} passes)')
# ax7.set_ylabel('Mean max-std'); ax7.grid(axis='y', alpha=0.3)

# # 8. SupCon embedding visualization (PCA)
# ax8 = fig.add_subplot(2, 4, 8)
# try:
#     from sklearn.decomposition import PCA
#     model.eval()
#     embs, lbls = [], []
#     with torch.no_grad():
#         for raw, feat, meta, labels in val_dl:
#             raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
#             logits, proj = model(raw, feat, meta)
#             embs.append(proj.cpu().numpy())
#             lbls.extend(labels.numpy())
#     embs = np.concatenate(embs)
#     lbls = np.array(lbls)
#     pca  = PCA(n_components=2).fit_transform(embs)
#     for c in range(N_CLASSES):
#         mask = (lbls == c)
#         ax8.scatter(pca[mask, 0], pca[mask, 1], s=10, alpha=0.5,
#                     color=colors[c], label=label_names[c])
#     ax8.set_title('SupCon Embedding (PCA 2D)\nTight clusters = good separation')
#     ax8.legend(fontsize=7); ax8.grid(alpha=0.3)
# except Exception as e:
#     ax8.text(0.5, 0.5, f'PCA failed:\n{e}', ha='center', va='center',
#              transform=ax8.transAxes)

# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=150, bbox_inches='tight')
# print(f"\nPlot saved: {OUTPUT_DIR}/results.png")
# print(f"Best model: {OUTPUT_DIR}/best_model.pt")
# print("\nDone!")
"""
Physics-Informed Hybrid Model — e2e version
============================================
PhysicsHybridNet: Multi-scale 1D CNN + Transformer + Cross-modal Attention
Losses: Focal + Supervised Contrastive + Physics Hinge (Kendall adaptive weights)

Evaluation outputs (./outputs_pi_hybrid/):
  best_model.pt         — best checkpoint
  results.png           — training curves + standard plots
  evaluation_report.png — confusion · ROC · PR · calibration · OOD · ablation
  metrics.json          — all numeric metrics

Model switcher:
  python train_physics_improved.py               # PhysicsHybridNet (default)
  python train_physics_improved.py --model pinn  # → runs train_pinn.py instead
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score, accuracy_score)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, warnings, argparse, json
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# --- ARGPARSE -----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='final_data')
parser.add_argument('--out_dir',  default='outputs_pi_hybrid')
parser.add_argument('--epochs',   type=int,   default=100)
parser.add_argument('--lr',       type=float, default=1e-3)
parser.add_argument('--model',    default='physics', choices=['physics', 'pinn'],
                    help='physics = PhysicsHybridNet (default) | pinn = BearingPINN')
args = parser.parse_args()

# ─── MODEL SWITCHER ───────────────────────────────────────────────────────────
if args.model == 'pinn':
    _script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_pinn.py')
    _fwd    = [a for a in sys.argv[1:] if a not in ('--model', 'pinn')]
    import subprocess; subprocess.run([sys.executable, _script] + _fwd, check=True); sys.exit()

DATA_DIR   = args.data_dir
OUTPUT_DIR = args.out_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIG -------------------------------------------------------------------
BATCH_SIZE    = 64
EPOCHS        = args.epochs
LR            = args.lr
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

FAULT_FREQ_MAP = {1: 0, 3: 1, 2: 2}   # class → dominant freq index (BPFI, BPFO, BPF)


# --- LOSSES ------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        B      = features.shape[0]
        device = features.device
        sim    = torch.mm(features, features.T) / self.T
        labels_col = labels.unsqueeze(1)
        pos_mask   = (labels_col == labels_col.T).float()
        pos_mask.fill_diagonal_(0.0)
        self_mask  = torch.eye(B, device=device)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()
        exp_sim    = torch.exp(sim) * (1 - self_mask)
        log_denom  = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        log_prob   = sim - log_denom
        n_pos      = pos_mask.sum(dim=1)
        valid      = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        per_anchor = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-9)
        return per_anchor[valid].mean()


class PhysicsRegularizer(nn.Module):
    """Hinge-based physics constraint — always non-negative."""
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


# --- DATASET -----------------------------------------------------------------

class BearingDataset(Dataset):
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

cls_counts   = np.bincount(y_train, minlength=N_CLASSES).astype(float)
samp_weights = 1.0 / (cls_counts[y_train] + 1e-9)
sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
fa_dl    = DataLoader(fa_ds,    batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)


# --- MODEL -------------------------------------------------------------------

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
        self.tab_enc     = ResidualMLP(22, 64, 64)
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
        self.log_var_focal   = nn.Parameter(torch.zeros(1))
        self.log_var_supcon  = nn.Parameter(torch.zeros(1))
        self.log_var_physics = nn.Parameter(torch.zeros(1))

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
        emb    = self.encode(raw, feat, meta)
        logits = self.classifier(emb)
        proj   = F.normalize(self.proj_head(emb), dim=1)
        return logits, proj

    def adaptive_loss(self, focal_l, supcon_l, physics_l):
        s_f = self.log_var_focal
        s_c = self.log_var_supcon
        s_p = self.log_var_physics
        return (torch.exp(-s_f) * focal_l + s_f +
                torch.exp(-s_c) * supcon_l + s_c +
                torch.exp(-s_p) * physics_l + s_p)


model    = PhysicsHybridNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

w = torch.tensor(1.0 / (cls_counts + 1e-9), dtype=torch.float32)
w = (w / w.sum() * N_CLASSES).to(DEVICE)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma; self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        return (((1.0 - p_t) ** self.gamma) * ce).mean()


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


# --- TRAINING ----------------------------------------------------------------

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_focal = tot_supcon = tot_physics = tot_total = 0
    tot_correct = tot = 0
    preds_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for raw, feat, meta, labels in loader:
            raw, feat, meta, labels = (
                raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            )
            logits, proj = model(raw, feat, meta)
            focal_l   = focal_criterion(logits, labels)
            supcon_l  = supcon_criterion(proj, labels)
            env_feat  = feat[:, 18:22]
            physics_l = physics_criterion(env_feat, labels)
            total_l   = model.adaptive_loss(focal_l, supcon_l, physics_l)
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
            tot_total   += total_l.item()   * B
            tot_correct += (preds == labels).sum().item()
            tot         += B
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    f1     = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    losses = {
        'total':   tot_total   / tot,
        'focal':   tot_focal   / tot,
        'supcon':  tot_supcon  / tot,
        'physics': tot_physics / tot,
    }
    return losses, tot_correct / tot, f1


print("\n" + "="*60)
print("Training Physics-Informed Hybrid Model")
print("  Backbone : Multi-scale 1D CNN + Transformer + CMA")
print("  Loss     : Focal + SupCon + Physics (adaptive weights)")
print("="*60)

history = {k: [] for k in ['tr_total', 'val_total', 'tr_focal', 'val_focal',
                             'tr_supcon', 'val_supcon', 'tr_f1', 'val_f1',
                             'lv_focal', 'lv_supcon', 'lv_physics']}
best_f1, patience_cnt = 0.0, 0

for ep in range(1, EPOCHS + 1):
    tr_losses, tr_acc, tr_f1 = run_epoch(train_dl, train=True)
    vl_losses, vl_acc, vl_f1 = run_epoch(val_dl,   train=False)
    scheduler.step()

    history['tr_total'].append(tr_losses['total'])
    history['val_total'].append(vl_losses['total'])
    history['tr_focal'].append(tr_losses['focal'])
    history['val_focal'].append(vl_losses['focal'])
    history['tr_supcon'].append(tr_losses['supcon'])
    history['val_supcon'].append(vl_losses['supcon'])
    history['tr_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)
    history['lv_focal'].append(model.log_var_focal.item())
    history['lv_supcon'].append(model.log_var_supcon.item())
    history['lv_physics'].append(model.log_var_physics.item())

    if vl_f1 > best_f1:
        best_f1 = vl_f1
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        patience_cnt = 0
    else:
        patience_cnt += 1

    if ep % 5 == 0 or ep == 1:
        lr_now = optimizer.param_groups[0]['lr']
        lv_f   = model.log_var_focal.item()
        lv_c   = model.log_var_supcon.item()
        lv_p   = model.log_var_physics.item()
        print(
            f"Ep {ep:3d}  tr_f1={tr_f1:.4f} vl_f1={vl_f1:.4f} best={best_f1:.4f}  "
            f"focal={tr_losses['focal']:.3f} sup={tr_losses['supcon']:.3f} "
            f"phy={tr_losses['physics']:.3f}  lv=[{lv_f:.2f},{lv_c:.2f},{lv_p:.2f}]  "
            f"lr={lr_now:.2e}"
        )

    if patience_cnt >= PATIENCE:
        print(f"\nEarly stop at epoch {ep}")
        break


# --- FINAL EVALUATION --------------------------------------------------------

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))


def evaluate(loader, name):
    model.eval()
    preds_all, labels_all, logits_all, probs_all = [], [], [], []
    with torch.no_grad():
        for raw, feat, meta, labels in loader:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits, _ = model(raw, feat, meta)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())
            logits_all.append(logits.cpu())
            probs_all.append(F.softmax(logits, dim=1).cpu())
    y_true   = np.array(labels_all)
    y_pred   = np.array(preds_all)
    logits_t = torch.cat(logits_all)
    probs_t  = torch.cat(probs_all).numpy()
    print(f"\n{'='*60}\nRESULTS: {name}\n{'='*60}")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    print("False Positive Rate per class:")
    for c in range(N_CLASSES):
        fp = np.sum((y_true != c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        print(f"  {label_names[c]:12s}: FPR = {fp/(fp+tn+1e-10):.4f}")
    cm_val = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print("\nConfusion Matrix:")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in label_names)
    print(header)
    for i, row in enumerate(cm_val):
        print(f"{label_names[i]:12s}" + "".join(f"{v:>12d}" for v in row))
    energy = (-1.0 * torch.logsumexp(logits_t, dim=1)).numpy()
    return y_true, y_pred, cm_val, probs_t, energy


vt, vp, vcm, vprobs, v_energy = evaluate(val_dl, "Validation Set")

# --- MC DROPOUT UNCERTAINTY --------------------------------------------------

print("\n--- MC Dropout Uncertainty Estimation ---")
model.train()
mc_probs_list = []
with torch.no_grad():
    for _ in range(MC_PASSES):
        batch_probs = []
        for raw, feat, meta, _ in val_dl:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits, _ = model(raw, feat, meta)
            batch_probs.append(F.softmax(logits, dim=1).cpu())
        mc_probs_list.append(torch.cat(batch_probs).numpy())

mc_probs_arr = np.stack(mc_probs_list)
mc_std       = mc_probs_arr.std(axis=0)
uncertainty  = mc_std.max(axis=1)
print("Mean epistemic uncertainty by class:")
for c in range(N_CLASSES):
    print(f"  {label_names[c]:12s}: {uncertainty[vt == c].mean():.4f}")

# --- FALSE ALARM + OOD -------------------------------------------------------

model.eval()
fa_preds_l, fa_logits_l = [], []
with torch.no_grad():
    for raw, feat, meta, _ in fa_dl:
        raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
        logits, _ = model(raw, feat, meta)
        fa_preds_l.extend(logits.argmax(1).cpu().numpy())
        fa_logits_l.append(logits.cpu())

fa_preds  = np.array(fa_preds_l)
fa_logits = torch.cat(fa_logits_l)
fa_energy = (-1.0 * torch.logsumexp(fa_logits, dim=1)).numpy()

false_alarm_rate = np.mean(fa_preds != 0)
print(f"\n{'='*60}\nFALSE ALARM TEST (Folder 11 - Shaft Misalignment)")
print(f"  {np.sum(fa_preds != 0)} / {len(fa_preds)} predicted as fault")
print(f"  False Alarm Rate = {false_alarm_rate:.4f}  (ideal = 0)")

ood_threshold = np.percentile(v_energy, 95)
ood_flagged   = np.mean(fa_energy > ood_threshold)
print(f"  Energy OOD detection rate = {ood_flagged:.4f}  (higher = better OOD separation)")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE EVALUATION — Confusion · ROC/PR · Calibration · OOD · Ablation
# ═══════════════════════════════════════════════════════════════════════════════

macro_f1 = f1_score(vt, vp, average='macro', zero_division=0)
acc      = accuracy_score(vt, vp)

# ── ROC + PR curves (one-vs-rest) ────────────────────────────────────────────
y_bin    = label_binarize(vt, classes=list(range(N_CLASSES)))
roc_data, pr_data = {}, {}
for _c in range(N_CLASSES):
    _fpr, _tpr, _  = roc_curve(y_bin[:, _c], vprobs[:, _c])
    _prec, _rec, _ = precision_recall_curve(y_bin[:, _c], vprobs[:, _c])
    roc_data[_c]   = (_fpr, _tpr, auc(_fpr, _tpr))
    pr_data[_c]    = (_prec, _rec, average_precision_score(y_bin[:, _c], vprobs[:, _c]))

# ── Calibration (ECE) ────────────────────────────────────────────────────────
_max_probs    = vprobs.max(axis=1)
_correct_mask = (vp == vt).astype(float)
_frac_pos, _mean_conf_bins = calibration_curve(_correct_mask, _max_probs, n_bins=10)
ece = float(np.mean(np.abs(_frac_pos - _mean_conf_bins)))

# ── OOD AUROC (energy-based: val in-dist, FA out-of-dist) ────────────────────
_ood_y    = np.array([0]*len(v_energy) + [1]*len(fa_energy))
_ood_s    = np.concatenate([v_energy, fa_energy])
_o_fpr, _o_tpr, _ = roc_curve(_ood_y, _ood_s)
ood_auroc = auc(_o_fpr, _o_tpr)

# ── Ablation study (zero-out each modality at test time) ─────────────────────
def _ablation_f1(zero_raw=False, zero_feat=False, zero_meta=False):
    model.eval()
    _p, _l = [], []
    with torch.no_grad():
        for raw, feat, meta, labels in val_dl:
            raw  = raw.to(DEVICE);  feat = feat.to(DEVICE); meta = meta.to(DEVICE)
            if zero_raw:  raw  = torch.zeros_like(raw)
            if zero_feat: feat = torch.zeros_like(feat)
            if zero_meta: meta = torch.zeros_like(meta)
            logits, _ = model(raw, feat, meta)
            _p.extend(logits.argmax(1).cpu().tolist())
            _l.extend(labels.tolist())
    return f1_score(_l, _p, average='macro', zero_division=0)

ablation = {
    'Full Model':   macro_f1,
    'w/o Signal':   _ablation_f1(zero_raw=True),
    'w/o Tabular':  _ablation_f1(zero_feat=True),
    'w/o Metadata': _ablation_f1(zero_meta=True),
}

_mean_roc = float(np.mean([v[2] for v in roc_data.values()]))
_mean_ap  = float(np.mean([v[2] for v in pr_data.values()]))
print(f"\n{'─'*55}")
print(f"  Accuracy     : {acc:.4f}")
print(f"  Macro F1     : {macro_f1:.4f}")
print(f"  Mean ROC-AUC : {_mean_roc:.4f}")
print(f"  Mean Avg-Prec: {_mean_ap:.4f}")
print(f"  ECE (calib)  : {ece:.4f}  (↓ better)")
print(f"  OOD AUROC    : {ood_auroc:.4f}  (energy, ↑ better)")
print(f"  FA Rate      : {false_alarm_rate:.4f}  (↓ better)")
print(f"  Per-class ROC-AUC:")
for _c in range(N_CLASSES):
    print(f"    {label_names[_c]:12s}: {roc_data[_c][2]:.4f}")
print(f"  Ablation (Macro F1):")
for _k, _v in ablation.items():
    print(f"    {_k:16s}: {_v:.4f}  ({_v - macro_f1:+.4f})")
print(f"{'─'*55}")

_metrics = {
    'accuracy':  float(acc),
    'macro_f1':  float(macro_f1),
    'roc_auc':   {label_names[_c]: float(roc_data[_c][2]) for _c in range(N_CLASSES)},
    'avg_prec':  {label_names[_c]: float(pr_data[_c][2])  for _c in range(N_CLASSES)},
    'ece':       ece,
    'ood_auroc': float(ood_auroc),
    'fa_rate':   float(false_alarm_rate),
    'ablation':  {_k: float(_v) for _k, _v in ablation.items()},
}
with open(f"{OUTPUT_DIR}/metrics.json", 'w') as _fp:
    json.dump(_metrics, _fp, indent=2)
print(f"  Metrics JSON : {OUTPUT_DIR}/metrics.json")


# --- PLOTS (results.png) -----------------------------------------------------

fig = plt.figure(figsize=(20, 10))
fig.suptitle("Physics-Informed Hybrid Model — Results", fontsize=13, fontweight='bold')

ax1 = fig.add_subplot(2, 4, 1)
ax1.plot(history['tr_f1'],  label='Train', color='steelblue')
ax1.plot(history['val_f1'], label='Val',   color='orangered')
ax1.set_title('Macro F1'); ax1.set_xlabel('Epoch')
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(2, 4, 2)
ax2.plot(history['tr_focal'],  label='Focal',  color='steelblue')
ax2.plot(history['tr_supcon'], label='SupCon', color='darkorange')
ax2.plot(history['tr_total'],  label='Total',  color='green', linewidth=2)
ax2.set_title('Training Loss Breakdown'); ax2.set_xlabel('Epoch')
ax2.legend(); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 4, 3)
ax3.plot(history['lv_focal'],   label='s_focal',   color='steelblue')
ax3.plot(history['lv_supcon'],  label='s_supcon',  color='darkorange')
ax3.plot(history['lv_physics'], label='s_physics', color='green')
ax3.set_title('Learned Loss Weights (log var)\nLower = higher weight')
ax3.set_xlabel('Epoch'); ax3.legend(); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(2, 4, 4)
_im4 = ax4.imshow(vcm, cmap='Blues')
ax4.set_xticks(range(N_CLASSES)); ax4.set_xticklabels(label_names, rotation=30, ha='right')
ax4.set_yticks(range(N_CLASSES)); ax4.set_yticklabels(label_names)
ax4.set_title(f'Confusion Matrix\nAcc={acc:.3f}  F1={macro_f1:.3f}')
ax4.set_xlabel('Predicted'); ax4.set_ylabel('True')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax4.text(j, i, str(vcm[i, j]), ha='center', va='center',
                 color='white' if vcm[i, j] > vcm.max() / 2 else 'black')
plt.colorbar(_im4, ax=ax4)

ax5 = fig.add_subplot(2, 4, 5)
for _c in range(N_CLASSES):
    _fp5, _tp5, _ra5 = roc_data[_c]
    ax5.plot(_fp5, _tp5, color=COLORS[_c], label=f'{label_names[_c]} ({_ra5:.3f})')
ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax5.set_title(f'ROC Curves (OvR)\nMean AUC={_mean_roc:.3f}')
ax5.set_xlabel('FPR'); ax5.set_ylabel('TPR')
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(2, 4, 6)
ax6.hist(v_energy,  bins=40, alpha=0.6, color='steelblue', label='Val (in-dist)')
ax6.hist(fa_energy, bins=40, alpha=0.6, color='orangered',  label='FA (OOD)')
ax6.axvline(ood_threshold, color='black', linestyle='--',
            label=f'OOD thresh ({ood_threshold:.2f})')
ax6.set_title(f'Energy Score Distribution\nOOD AUROC={ood_auroc:.3f}')
ax6.set_xlabel('Energy'); ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

ax7 = fig.add_subplot(2, 4, 7)
unc_by_class = [uncertainty[vt == c].mean() for c in range(N_CLASSES)]
ax7.bar(label_names, unc_by_class, color=COLORS)
ax7.set_title(f'Epistemic Uncertainty\n(MC Dropout, {MC_PASSES} passes)')
ax7.set_ylabel('Mean max-std'); ax7.grid(axis='y', alpha=0.3)

ax8 = fig.add_subplot(2, 4, 8)
try:
    from sklearn.decomposition import PCA
    model.eval()
    embs, lbls = [], []
    with torch.no_grad():
        for raw, feat, meta, labels in val_dl:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits, proj = model(raw, feat, meta)
            embs.append(proj.cpu().numpy())
            lbls.extend(labels.numpy())
    embs = np.concatenate(embs)
    lbls = np.array(lbls)
    pca  = PCA(n_components=2).fit_transform(embs)
    for c in range(N_CLASSES):
        mask = (lbls == c)
        ax8.scatter(pca[mask, 0], pca[mask, 1], s=10, alpha=0.5,
                    color=COLORS[c], label=label_names[c])
    ax8.set_title('SupCon Embedding (PCA 2D)\nTight clusters = good separation')
    ax8.legend(fontsize=7); ax8.grid(alpha=0.3)
except Exception as e:
    ax8.text(0.5, 0.5, f'PCA failed:\n{e}', ha='center', va='center',
             transform=ax8.transAxes)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {OUTPUT_DIR}/results.png")


# ─── EVALUATION REPORT (evaluation_report.png) ───────────────────────────────

fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
fig2.suptitle('PhysicsHybridNet — Evaluation Report', fontsize=14, fontweight='bold')

_ax = axes2[0, 0]
_im = _ax.imshow(vcm, cmap='Blues')
_ax.set_xticks(range(N_CLASSES)); _ax.set_xticklabels(label_names, rotation=30, ha='right', fontsize=8)
_ax.set_yticks(range(N_CLASSES)); _ax.set_yticklabels(label_names, fontsize=8)
_ax.set_title(f'Confusion Matrix\nAcc={acc:.3f}  F1={macro_f1:.3f}')
_ax.set_xlabel('Predicted'); _ax.set_ylabel('True')
for _i in range(N_CLASSES):
    for _j in range(N_CLASSES):
        _ax.text(_j, _i, str(vcm[_i, _j]), ha='center', va='center', fontsize=9,
                 color='white' if vcm[_i, _j] > vcm.max() / 2 else 'black')
plt.colorbar(_im, ax=_ax)

_ax = axes2[0, 1]
for _c in range(N_CLASSES):
    _fp_, _tp_, _ra_ = roc_data[_c]
    _ax.plot(_fp_, _tp_, color=COLORS[_c], lw=2, label=f'{label_names[_c]} ({_ra_:.3f})')
_ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
_ax.set_title(f'ROC Curves (OvR)\nMean AUC={_mean_roc:.3f}')
_ax.set_xlabel('FPR'); _ax.set_ylabel('TPR')
_ax.legend(fontsize=7); _ax.grid(alpha=0.3)

_ax = axes2[0, 2]
for _c in range(N_CLASSES):
    _pr_, _re_, _ap_ = pr_data[_c]
    _ax.plot(_re_, _pr_, color=COLORS[_c], lw=2, label=f'{label_names[_c]} (AP={_ap_:.3f})')
_ax.set_title(f'Precision-Recall Curves\nMean AP={_mean_ap:.3f}')
_ax.set_xlabel('Recall'); _ax.set_ylabel('Precision')
_ax.legend(fontsize=7); _ax.grid(alpha=0.3)

_ax = axes2[0, 3]
_ax.plot(_mean_conf_bins, _frac_pos, 's-', color='steelblue', lw=2,
         label=f'PhysicsHybrid (ECE={ece:.3f})')
_ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
_ax.fill_between(_mean_conf_bins, _frac_pos, _mean_conf_bins, alpha=0.1, color='steelblue')
_ax.set_title('Reliability Diagram'); _ax.set_xlabel('Mean Confidence')
_ax.set_ylabel('Fraction Correct'); _ax.legend(fontsize=8); _ax.grid(alpha=0.3)

_ax = axes2[1, 0]
_cm2 = (vp == vt)
_ax.hist(_max_probs[_cm2],  bins=30, alpha=0.65, color='green', label='Correct')
_ax.hist(_max_probs[~_cm2], bins=30, alpha=0.65, color='red',   label='Wrong')
_ax.set_title('Confidence Distribution\n(Max Softmax)'); _ax.set_xlabel('Confidence')
_ax.legend(fontsize=8); _ax.grid(alpha=0.3)

_ax = axes2[1, 1]
_ax.hist(v_energy,  bins=40, alpha=0.6, color='steelblue',
         label=f'Val in-dist (N={len(v_energy)})')
_ax.hist(fa_energy, bins=40, alpha=0.6, color='orangered',
         label=f'FA OOD (N={len(fa_energy)})')
_ax.axvline(ood_threshold, color='black', ls='--', label=f'95th={ood_threshold:.2f}')
_ax.set_title(f'OOD Energy Score\nAUROC={ood_auroc:.3f}')
_ax.set_xlabel('Energy Score'); _ax.legend(fontsize=7); _ax.grid(alpha=0.3)

_ax = axes2[1, 2]
_ab_n = list(ablation.keys()); _ab_s = list(ablation.values())
_bars = _ax.bar(_ab_n, _ab_s, color=['green', 'steelblue', 'darkorange', 'red'])
_ax.set_ylim(max(0, min(_ab_s) - 0.08), min(1.0, max(_ab_s) + 0.06))
for _b, _s in zip(_bars, _ab_s):
    _ax.text(_b.get_x() + _b.get_width() / 2, _s + 0.004,
             f'{_s:.3f}', ha='center', va='bottom', fontsize=9)
_ax.set_title('Ablation Study\n(Macro F1)'); _ax.set_ylabel('Macro F1')
_ax.tick_params(axis='x', rotation=15); _ax.grid(axis='y', alpha=0.3)

_ax = axes2[1, 3]
_fpr_cls = []
for _c in range(N_CLASSES):
    _fp2 = int(np.sum((vt != _c) & (vp == _c)))
    _tn2 = int(np.sum((vt != _c) & (vp != _c)))
    _fpr_cls.append(_fp2 / (_fp2 + _tn2 + 1e-10))
_bars2 = _ax.bar(label_names, _fpr_cls, color=COLORS)
for _b, _v in zip(_bars2, _fpr_cls):
    _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.001,
             f'{_v:.3f}', ha='center', va='bottom', fontsize=9)
_ax.set_title('False Positive Rate\nper Class'); _ax.set_ylabel('FPR')
_ax.tick_params(axis='x', rotation=20); _ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/evaluation_report.png", dpi=130, bbox_inches='tight')
plt.close()
print(f"Evaluation report: {OUTPUT_DIR}/evaluation_report.png")
print(f"Best model: {OUTPUT_DIR}/best_model.pt")
print("\nDone!")
