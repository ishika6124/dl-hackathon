"""
PINN_HybridFaultNet: Physics-Informed Neural Network for Bearing Fault Detection
================================================================================
Extends HybridFaultNet with differentiable bearing physics.

PHYSICS INGREDIENTS
───────────────────
1. BearingPhysicsLayer  (differentiable – new PINN branch)
   ┌─ From meta features → learned shaft frequency fr (Hz)
   ├─ From raw signal   → torch.fft.rfft → |F(ω)| (differentiable!)
   └─ Gaussian-weighted peak extraction at harmonics of:
        BPFO = (N/2)(1 − Bd/Pd·cosα)·fr   [outer race]
        BPFI = (N/2)(1 + Bd/Pd·cosα)·fr   [inner ring]
        BSF  = (Pd/2Bd)(1−(Bd/Pd·cosα)²)·fr  [ball]
        FTF  = ½(1 − Bd/Pd·cosα)·fr       [cage/train]
   → physics_feats (B,8): 4 normalised amplitudes + 4 normalised freq values
   → physics_evidence (B,3): [amp_bpfo, amp_bpfi, amp_bsf] for physics loss

2. PhysicsConsistencyLoss
   Soft constraint: if BPFI amplitude is high → predict Inner Ring, etc.
     Normal     → fault_total low   → score = exp(−fault_total)
     Inner Ring → BPFI dominant     → score ∝ amp_bpfi / fault_total
     Ball       → BSF dominant      → score ∝ amp_bsf  / fault_total
     Outer Ring → BPFO dominant     → score ∝ amp_bpfo / fault_total
   KL( model_probs ‖ physics_probs ) masked by physics confidence (entropy < θ)
   Only enforced when spectrum is unambiguous; silent on noisy / complex signals.

3. Combined loss
   L_total = L_focal + λ_phys(epoch) × L_phys
   λ_phys annealed 0 → LAMBDA_PHYS over WARMUP_EPOCHS (focal loss stabilises first)

ARCHITECTURE  (4 branches + cross-modal attention)
──────────────────────────────────────────────────
  Signal (4096)  → MultiScale CNN (k=7,15,31,63) → Transformer → proj(PROJ_DIM)
                 ↓ also fed into
  Physics        → BearingPhysicsLayer → PhysicsEncoder            → proj(PROJ_DIM)
  Tabular (22)   → ResidualMLP                                     → proj(PROJ_DIM)
  Meta    (8)    → MLP                                             → proj(PROJ_DIM)

  Cross-modal attention (5 attention pairs):
    sig  ← phys  (physics corrects raw-signal predictions   ← KEY fix for IR false alarms)
    sig  ← tab   (envelope spectrum confirms fault type)
    sig  ← meta  (operating conditions refine time domain)
    phys ← sig   (raw context enriches spectral interpretation)
    phys ← tab   (statistical features refine spectral interpretation)

  Fusion: concat(sig, phys, tab, meta) → Linear(4·PROJ_DIM, 128) → Linear(128, 4)

BEARING PARAMS
──────────────
Default: 6205-2RS deep groove ball bearing (CWRU-style).
Override BEARING_PARAMS if your dataset provides per-asset geometry.
Set meta_rpm_idx to the index in your 8-feature meta vector that carries
normalised shaft speed; rpm_scale converts it back to RPM.

References
──────────
Randall & Antoni (2011) – Rolling element bearing diagnostics using the
  Case Western Reserve University data: A benchmark study. Mech. Syst. Signal Process.
Cao & Shi (2025) – Cross-modal attention fusion for multimodal bearing diagnosis.
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

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR      = "final_data"
OUTPUT_DIR    = "/home/teaching/hackathon/Approach-2/outputs_pinn"
BATCH_SIZE    = 32
EPOCHS        = 100
LR            = 1e-3
DROPOUT       = 0.4
PROJ_DIM      = 48
N_CLASSES     = 4
FOCAL_GAMMA   = 2.0
WARMUP_EPOCHS = 5
LAMBDA_PHYS   = 0.3      # max weight for physics loss (annealed in)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── BEARING GEOMETRY ─────────────────────────────────────────────────────────
# Defaults: typical industrial bearing similar to CWRU 6205-2RS.
# Adjust these to match your SCA dataset bearing specifications.
# If the dataset has multiple bearing types, you can condition on meta features.
BEARING_PARAMS = dict(
    n_balls        = 9,
    ball_dia       = 0.3126,    # inches (consistent unit with pitch_dia)
    pitch_dia      = 1.748,     # inches
    contact_angle  = 15.0,      # degrees
    n_harmonics    = 3,         # number of fault-freq harmonics to check
    fs             = 25600,     # sampling frequency, Hz  ← verify with dataset
    signal_len     = 4096,
    # Which index in the 8-element meta vector holds normalised shaft speed?
    # (0-indexed). Set to None to let BearingPhysicsLayer learn fr from all 8.
    meta_rpm_idx   = 0,
    # meta[meta_rpm_idx] × rpm_scale ≈ RPM (used as initial linear bias)
    rpm_scale      = 3000.0,
)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['Normal', 'Inner Ring', 'Ball', 'Outer Ring']
print(f"Device: {DEVICE}")

# ── FOCAL LOSS ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so the model focuses on hard / rare ones.
    FL = −(1−p_t)^γ · log(p_t),  combined with class weights.
    γ=2 → correctly classified easy examples contribute <1 % of gradient.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        return (((1.0 - p_t) ** self.gamma) * ce).mean()


# ── PHYSICS CONSISTENCY LOSS ─────────────────────────────────────────────────

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft physics residual: model predictions must agree with spectral evidence.

    Physics "soft label" construction per sample:
      Normal     → total fault amplitude low → score = exp(−Σ amplitudes)
      Inner Ring → BPFI dominant             → score ∝ amp_bpfi / total
      Ball       → BSF  dominant             → score ∝ amp_bsf  / total
      Outer Ring → BPFO dominant             → score ∝ amp_bpfo / total

    L_phys = E[mask · KL(model_probs ‖ physics_probs)]

    mask = 1 only when physics entropy < entropy_threshold × log(N_CLASSES)
    i.e. physics evidence is confident; silent when spectrum is ambiguous.

    Effect on training:
      Normal signal with low all-fault amplitudes → strongly pushes toward Normal
      Inner Ring signal with high BPFI → reinforces Inner Ring prediction
      Ambiguous signal → mask=0, pure focal-loss region, no spurious physics signal
    """
    def __init__(self, temperature=5.0, entropy_threshold=0.7):
        super().__init__()
        self.temperature       = temperature
        self.entropy_threshold = entropy_threshold

    def forward(self, logits, physics_evidence):
        """
        logits:           (B, 4)
        physics_evidence: (B, 3) = [amp_bpfo, amp_bpfi, amp_bsf] (RMS-normalised)
        """
        amp_bpfo = physics_evidence[:, 0]
        amp_bpfi = physics_evidence[:, 1]
        amp_bsf  = physics_evidence[:, 2]
        fault_total = amp_bpfo + amp_bpfi + amp_bsf + 1e-8

        # Physics class scores (not gradients – detached)
        normal_score = torch.exp(-fault_total.clamp(max=10.0))
        raw_scores   = torch.stack([
            normal_score,
            amp_bpfi / fault_total,    # Inner Ring
            amp_bsf  / fault_total,    # Ball
            amp_bpfo / fault_total,    # Outer Ring
        ], dim=1)   # (B, 4)

        physics_probs = F.softmax(raw_scores * self.temperature, dim=1).detach()

        # Confidence mask: only enforce when physics is unambiguous
        ent     = -(physics_probs * (physics_probs + 1e-9).log()).sum(1)
        max_ent = float(np.log(N_CLASSES))
        mask    = (ent < self.entropy_threshold * max_ent).float()

        model_log_probs = F.log_softmax(logits, dim=1)
        kl = F.kl_div(model_log_probs, physics_probs, reduction='none').sum(1)

        return (mask * kl).mean()


# ── DATASET ──────────────────────────────────────────────────────────────────

class BearingDataset(Dataset):
    def __init__(self, raw, stat, env, meta, labels):
        self.raw    = torch.tensor(raw[:, None, :], dtype=torch.float32)
        self.feat   = torch.tensor(
            np.concatenate([stat, env], axis=1), dtype=torch.float32
        )
        self.meta   = torch.tensor(meta,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

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

cls_counts   = np.bincount(y_train, minlength=N_CLASSES).astype(float)
samp_weights = 1.0 / (cls_counts[y_train] + 1e-9)
sampler      = WeightedRandomSampler(samp_weights, len(samp_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
fa_dl    = DataLoader(fa_ds,    batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)


# ── BUILDING BLOCKS ──────────────────────────────────────────────────────────

class MultiScaleBranch(nn.Module):
    """Single-kernel 1D CNN branch from MSConvFormer. Input: (B,1,4096)."""
    def __init__(self, kernel_size, out_ch=32, seq_len=32):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(16),     nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, out_ch, kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x):
        return self.pool(self.conv(x))


class ResidualMLP(nn.Module):
    """MLP + residual + LayerNorm for tabular features."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(x)))
        return self.norm(self.fc2(h) + self.skip(x))


class CrossModalAttention(nn.Module):
    """
    Scaled dot-product cross-modal attention.
    target queries source → target is informed by source.
    Uses sigmoid gating (scalar) for stable single-head attention.
    """
    def __init__(self, dim):
        super().__init__()
        self.Wq    = nn.Linear(dim, dim, bias=False)
        self.Wk    = nn.Linear(dim, dim, bias=False)
        self.Wv    = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.norm  = nn.LayerNorm(dim)

    def forward(self, target, source):
        q    = self.Wq(target)
        k    = self.Wk(source)
        v    = self.Wv(source)
        gate = torch.sigmoid(torch.sum(q * k, dim=-1, keepdim=True) * self.scale)
        return self.norm(target + gate * v)


# ── PHYSICS LAYER (PINN CORE) ─────────────────────────────────────────────────

class BearingPhysicsLayer(nn.Module):
    """
    Differentiable bearing-physics feature extractor.

    Pipeline:
      meta (B,8) → learned linear → fr (shaft freq, Hz)
      raw  (B,1,4096) → rfft → |F(ω)|          ← differentiable FFT
      fr + |F(ω)| → Gaussian peak extraction at harmonics of BPFO/BPFI/BSF/FTF
                                                 ← differentiable
      → physics_feats (B,8)       [4 RMS-normalised amplitudes + 4 freq ratios]
      → physics_evidence (B,3)    [amp_bpfo, amp_bpfi, amp_bsf] for physics loss

    Learnable parameters:
      fr_extractor: 8→16→1 MLP that maps meta → shaft frequency (constrained >0)
      log_bw:       log of the Gaussian bandwidth σ (Hz) for peak extraction
                    starts at ~8 Hz, learned during training for robustness

    Interpretability:
      Calling forward with return_details=True also returns per-sample fr and
      the four amplitude values — useful for explaining predictions.
    """

    def __init__(self, n_balls, ball_dia, pitch_dia, contact_angle,
                 n_harmonics, fs, signal_len, meta_rpm_idx, rpm_scale, **_):
        super().__init__()

        # ── Fault-frequency multipliers (physics constants) ──────────────────
        ratio = (ball_dia / pitch_dia) * np.cos(np.radians(contact_angle))
        self.register_buffer('bpfo_mult', torch.tensor(
            (n_balls / 2) * (1 - ratio), dtype=torch.float32))
        self.register_buffer('bpfi_mult', torch.tensor(
            (n_balls / 2) * (1 + ratio), dtype=torch.float32))
        self.register_buffer('bsf_mult',  torch.tensor(
            (pitch_dia / (2 * ball_dia)) * (1 - ratio ** 2), dtype=torch.float32))
        self.register_buffer('ftf_mult',  torch.tensor(
            0.5 * (1 - ratio), dtype=torch.float32))

        print(f"\n[BearingPhysicsLayer] Fault-frequency multipliers (× fr):")
        print(f"  BPFO = {(n_balls/2)*(1-ratio):.3f} × fr")
        print(f"  BPFI = {(n_balls/2)*(1+ratio):.3f} × fr")
        print(f"  BSF  = {(pitch_dia/(2*ball_dia))*(1-ratio**2):.3f} × fr")
        print(f"  FTF  = {0.5*(1-ratio):.3f} × fr")

        # ── Spectral parameters ───────────────────────────────────────────────
        self.fs          = fs
        self.signal_len  = signal_len
        self.n_harmonics = n_harmonics
        self.n_freq_bins = signal_len // 2 + 1

        freqs = torch.linspace(0, fs / 2, self.n_freq_bins)
        self.register_buffer('freqs', freqs)

        # ── Learnable shaft-frequency extractor ──────────────────────────────
        # MLP: meta (8) → positive shaft frequency (Hz)
        # Softplus ensures fr > 0; scaled so output ~5–55 Hz (300–3300 RPM)
        self.fr_extractor = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus(),
        )
        # Bias the extractor toward a reasonable RPM using rpm_scale hint
        # (initialise last linear bias so Softplus ≈ rpm_scale/60/50)
        with torch.no_grad():
            target_fr = (rpm_scale / 60.0) / 50.0   # back-calculated from scaling below
            self.fr_extractor[-2].bias.fill_(float(np.log(np.exp(target_fr) - 1)))

        # ── Learnable Gaussian bandwidth (Hz) ────────────────────────────────
        # Starts at ~8 Hz; learns to be narrow (precise RPM) or wide (robust)
        self.log_bw = nn.Parameter(torch.tensor(np.log(8.0)))

    # ── Internal helper ───────────────────────────────────────────────────────

    def _extract_harmonic_amp(self, magnitude, f_fund, bw):
        """
        Gaussian-weighted spectral amplitude at harmonics of f_fund.

        Args:
          magnitude: (B, n_freq_bins)  ← |FFT|
          f_fund:    (B,)              ← fundamental frequency Hz
          bw:        scalar            ← Gaussian σ in Hz
        Returns:
          (B,) mean amplitude across n_harmonics harmonics
        """
        total = torch.zeros(f_fund.shape[0], device=magnitude.device)
        for h in range(1, self.n_harmonics + 1):
            f_h    = f_fund * h                                # (B,)
            diff   = self.freqs[None, :] - f_h[:, None]       # (B, n_bins)
            weight = torch.exp(-0.5 * (diff / bw) ** 2)       # (B, n_bins)
            total  = total + (weight * magnitude).sum(dim=1)
        return total / self.n_harmonics

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, raw, meta, return_details=False):
        """
        Args:
          raw:  (B, 1, 4096)
          meta: (B, 8)
        Returns:
          physics_feats    (B, 8)  – physics features for branch encoder
          physics_evidence (B, 3)  – [amp_bpfo, amp_bpfi, amp_bsf] for loss
          fr               (B,)    – estimated shaft freq Hz (if return_details)
        """
        # ── Shaft frequency ───────────────────────────────────────────────────
        # Softplus output scaled so range is approximately 5–55 Hz (300–3300 RPM)
        fr = self.fr_extractor(meta).squeeze(1) * 50.0 + 5.0   # (B,) Hz

        # ── FFT magnitude (differentiable) ────────────────────────────────────
        signal    = raw.squeeze(1)                           # (B, 4096)
        spectrum  = torch.fft.rfft(signal, n=self.signal_len)
        magnitude = torch.abs(spectrum)                      # (B, n_freq_bins)
        rms       = signal.pow(2).mean(1).sqrt() + 1e-8     # (B,)

        bw = torch.exp(self.log_bw).clamp(1.0, 50.0)

        # ── Fault frequencies ─────────────────────────────────────────────────
        f_bpfo = fr * self.bpfo_mult   # (B,)
        f_bpfi = fr * self.bpfi_mult
        f_bsf  = fr * self.bsf_mult
        f_ftf  = fr * self.ftf_mult

        # ── Spectral amplitudes (RMS-normalised) ──────────────────────────────
        amp_bpfo = self._extract_harmonic_amp(magnitude, f_bpfo, bw) / rms
        amp_bpfi = self._extract_harmonic_amp(magnitude, f_bpfi, bw) / rms
        amp_bsf  = self._extract_harmonic_amp(magnitude, f_bsf,  bw) / rms
        amp_ftf  = self._extract_harmonic_amp(magnitude, f_ftf,  bw) / rms

        fs_half = torch.tensor(self.fs / 2.0, device=raw.device)

        # Physics features: amplitudes + normalised frequency values as context
        physics_feats = torch.stack([
            amp_bpfo,          amp_bpfi,          amp_bsf,         amp_ftf,
            f_bpfo / fs_half,  f_bpfi / fs_half,  f_bsf / fs_half, f_ftf / fs_half,
        ], dim=1)   # (B, 8)

        physics_evidence = torch.stack([amp_bpfo, amp_bpfi, amp_bsf], dim=1)   # (B, 3)

        if return_details:
            return physics_feats, physics_evidence, fr
        return physics_feats, physics_evidence


# ── PINN HYBRID MODEL ─────────────────────────────────────────────────────────

class PINN_HybridFaultNet(nn.Module):
    """
    Physics-Informed Hybrid Fault Detection Network.

    Four branches:
      1. Signal  → MultiScale 1D CNN (k=7,15,31,63) → Transformer → PROJ_DIM
      2. Physics → BearingPhysicsLayer (differentiable FFT + fault-freq extraction)
                   → PhysicsEncoder → PROJ_DIM
      3. Tabular → ResidualMLP on stat+env (22 features) → PROJ_DIM
      4. Meta    → MLP on operational metadata (8 features) → PROJ_DIM

    Five cross-modal attention pairs:
      sig  ← phys   physics corrects raw-signal CNN      ← KEY: fixes IR false alarms
      sig  ← tab    envelope stats confirm fault type
      sig  ← meta   operating conditions refine time domain
      phys ← sig    raw waveform context enriches spectral interpretation
      phys ← tab    statistical features sharpen spectral interpretation

    Loss during training:
      L_total = L_focal  +  λ_phys(epoch) × L_physics

    Interpretability (call interpret(raw, feat, meta)):
      Returns estimated shaft frequency and four fault-frequency amplitudes,
      allowing human-readable explanation: "Model sees BPFI amplitude = 3.2×RMS,
      which is the dominant fault signature → predicts Inner Ring fault."
    """

    KERNEL_SIZES = [7, 15, 31, 63]
    SEQ_LEN      = 32
    BRANCH_CH    = 32
    TRANS_DIM    = 128   # 4 × BRANCH_CH

    def __init__(self, bearing_params):
        super().__init__()

        # ── Branch 1: Signal (MSConvFormer style) ────────────────────────────
        self.branches = nn.ModuleList([
            MultiScaleBranch(k, out_ch=self.BRANCH_CH, seq_len=self.SEQ_LEN)
            for k in self.KERNEL_SIZES
        ])
        self.pos_enc  = nn.Parameter(
            torch.randn(1, self.SEQ_LEN, self.TRANS_DIM) * 0.02
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.TRANS_DIM, nhead=4, dim_feedforward=256,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.sig_head    = nn.Sequential(
            nn.Linear(self.TRANS_DIM, 64), nn.GELU(), nn.Dropout(DROPOUT),
        )
        self.sig_proj    = nn.Linear(64, PROJ_DIM)

        # ── Branch 2: Physics (PINN) ──────────────────────────────────────────
        self.physics_layer = BearingPhysicsLayer(**bearing_params)
        self.phys_encoder  = nn.Sequential(
            nn.Linear(8, 32),      nn.GELU(),
            nn.Linear(32, PROJ_DIM), nn.GELU(),
        )

        # ── Branch 3: Tabular (stat 18 + env 4 = 22) ─────────────────────────
        self.tab_enc  = ResidualMLP(22, 64, 64, dropout=DROPOUT)
        self.tab_proj = nn.Linear(64, PROJ_DIM)

        # ── Branch 4: Meta (8) ────────────────────────────────────────────────
        self.meta_enc = nn.Sequential(
            nn.Linear(8, 32),      nn.GELU(),
            nn.Linear(32, PROJ_DIM), nn.GELU(),
        )

        # ── Cross-modal attention ─────────────────────────────────────────────
        self.attn_sig_phys  = CrossModalAttention(PROJ_DIM)   # physics corrects signal
        self.attn_sig_tab   = CrossModalAttention(PROJ_DIM)   # tab corrects signal
        self.attn_sig_meta  = CrossModalAttention(PROJ_DIM)   # meta refines signal
        self.attn_phys_sig  = CrossModalAttention(PROJ_DIM)   # signal informs physics
        self.attn_phys_tab  = CrossModalAttention(PROJ_DIM)   # tab informs physics

        # ── Fusion head (4 × PROJ_DIM → N_CLASSES) ───────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(PROJ_DIM * 4, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, raw, feat, meta, return_physics=False):
        # ── Branch 1: signal ──────────────────────────────────────────────────
        x = torch.cat([b(raw) for b in self.branches], dim=1).permute(0, 2, 1)
        x = (x + self.pos_enc)
        x = self.transformer(x).mean(dim=1)
        sig_emb = self.sig_proj(self.sig_head(x))          # (B, PROJ_DIM)

        # ── Branch 2: physics (PINN) ──────────────────────────────────────────
        if return_physics:
            phys_feats, phys_evidence, fr = self.physics_layer(
                raw, meta, return_details=True)
        else:
            phys_feats, phys_evidence = self.physics_layer(raw, meta)
            fr = None
        phys_emb = self.phys_encoder(phys_feats)            # (B, PROJ_DIM)

        # ── Branch 3: tabular ─────────────────────────────────────────────────
        tab_emb  = self.tab_proj(self.tab_enc(feat))        # (B, PROJ_DIM)

        # ── Branch 4: meta ────────────────────────────────────────────────────
        meta_emb = self.meta_enc(meta)                      # (B, PROJ_DIM)

        # ── Cross-modal attention ─────────────────────────────────────────────
        sig_r  = self.attn_sig_phys(sig_emb,  phys_emb)    # physics corrects signal
        sig_r  = self.attn_sig_tab( sig_r,     tab_emb)     # tab   corrects signal
        sig_r  = self.attn_sig_meta(sig_r,     meta_emb)    # meta  refines  signal
        phys_r = self.attn_phys_sig(phys_emb,  sig_emb)    # signal informs physics
        phys_r = self.attn_phys_tab(phys_r,    tab_emb)    # tab   informs physics

        fused  = torch.cat([sig_r, phys_r, tab_emb, meta_emb], dim=1)  # (B, 4·PROJ_DIM)
        logits = self.fusion(fused)

        if return_physics:
            return logits, phys_evidence, fr
        return logits, phys_evidence, None

    @torch.no_grad()
    def interpret(self, raw, feat, meta):
        """
        Human-readable physics interpretation of a prediction.

        Returns dict with:
          predicted_class   – class index and name
          confidence        – softmax probability
          shaft_freq_hz     – estimated shaft rotation frequency
          fault_amplitudes  – {BPFO, BPFI, BSF} RMS-normalised spectral amplitudes
          physics_evidence  – which fault type physics evidence points to
        """
        self.eval()
        raw, feat, meta = (raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE))
        logits, phys_ev, fr = self.forward(raw, feat, meta, return_physics=True)

        probs   = F.softmax(logits, dim=1)
        cls_idx = probs.argmax(1).item()
        conf    = probs[0, cls_idx].item()

        amp_bpfo = phys_ev[0, 0].item()
        amp_bpfi = phys_ev[0, 1].item()
        amp_bsf  = phys_ev[0, 2].item()
        fault_names = {0: 'Outer Ring (BPFO)', 1: 'Inner Ring (BPFI)', 2: 'Ball (BSF)'}
        physics_cls  = int(np.argmax([amp_bpfo, amp_bpfi, amp_bsf]))

        return {
            'predicted_class':  label_names[cls_idx],
            'confidence':       round(conf, 4),
            'shaft_freq_hz':    round(fr[0].item(), 2),
            'fault_amplitudes': {
                'BPFO': round(amp_bpfo, 4),
                'BPFI': round(amp_bpfi, 4),
                'BSF':  round(amp_bsf,  4),
            },
            'physics_evidence': fault_names[physics_cls],
            'physics_agrees':   (
                (cls_idx == 3 and physics_cls == 0) or   # Outer Ring
                (cls_idx == 1 and physics_cls == 1) or   # Inner Ring
                (cls_idx == 2 and physics_cls == 2) or   # Ball
                cls_idx == 0                             # Normal (no dominant freq)
            ),
        }


# ── INSTANTIATE ───────────────────────────────────────────────────────────────

model = PINN_HybridFaultNet(BEARING_PARAMS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# ── LOSS & OPTIMISER ─────────────────────────────────────────────────────────

w = torch.tensor(1.0 / (cls_counts + 1e-9), dtype=torch.float32)
w = (w / w.sum() * N_CLASSES).to(DEVICE)
focal_criterion  = FocalLoss(gamma=FOCAL_GAMMA, weight=w)
physics_criterion = PhysicsConsistencyLoss(temperature=5.0, entropy_threshold=0.7)
print(f"Class weights: {w.cpu().numpy().round(3)}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

warmup_sched  = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine_sched  = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])


def physics_lambda(epoch):
    """
    Anneal physics loss weight from 0 → LAMBDA_PHYS over warmup.
    Focal loss stabilises first; physics loss kicks in as training matures.
    """
    return LAMBDA_PHYS * min(1.0, (epoch - 1) / max(WARMUP_EPOCHS, 1))


# ── TRAINING ──────────────────────────────────────────────────────────────────

def run_epoch(loader, epoch=1, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_focal = tot_phys = tot_correct = tot = 0
    preds_all, labels_all = [], []
    lam = physics_lambda(epoch) if train else 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for raw, feat, meta, labels in loader:
            raw, feat, meta, labels = (
                raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            )
            logits, phys_evidence, _ = model(raw, feat, meta)

            l_focal = focal_criterion(logits, labels)
            l_phys  = physics_criterion(logits, phys_evidence)
            loss    = l_focal + lam * l_phys

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds = logits.argmax(1)
            B     = len(labels)
            tot_loss    += loss.item()    * B
            tot_focal   += l_focal.item() * B
            tot_phys    += l_phys.item()  * B
            tot_correct += (preds == labels).sum().item()
            tot         += B
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return (tot_loss/tot, tot_focal/tot, tot_phys/tot,
            tot_correct/tot, f1, preds_all, labels_all)


print("\n" + "="*65)
print("Training PINN_HybridFaultNet")
print("  = MSConvFormer signal branch")
print("  + Differentiable bearing physics branch (BearingPhysicsLayer)")
print("  + Cross-modal attention (5 pairs)")
print("  + PhysicsConsistencyLoss (annealed)")
print("="*65)

history = {k: [] for k in ['tr_loss','val_loss','tr_f1','val_f1',
                             'tr_focal','val_focal','tr_phys','val_phys']}
best_f1, patience, PATIENCE = 0.0, 0, 20

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_foc, tr_phy, tr_acc, tr_f1, _, _ = run_epoch(train_dl, ep, train=True)
    vl_loss, vl_foc, vl_phy, vl_acc, vl_f1, _, _ = run_epoch(val_dl,   ep, train=False)
    scheduler.step()

    for k, v in [('tr_loss',tr_loss),('val_loss',vl_loss),
                  ('tr_f1',tr_f1),('val_f1',vl_f1),
                  ('tr_focal',tr_foc),('val_focal',vl_foc),
                  ('tr_phys',tr_phy),('val_phys',vl_phy)]:
        history[k].append(v)

    if vl_f1 > best_f1:
        best_f1 = vl_f1
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        patience = 0
    else:
        patience += 1

    if ep % 5 == 0 or ep == 1:
        lam_now = physics_lambda(ep)
        lr_now  = optimizer.param_groups[0]['lr']
        bw_now  = float(torch.exp(model.physics_layer.log_bw).clamp(1,50).item())
        print(f"Ep {ep:3d}/{EPOCHS}  "
              f"focal={tr_foc:.4f}  phys={tr_phy:.4f}(λ={lam_now:.2f})  "
              f"val_f1={vl_f1:.4f}  best={best_f1:.4f}  "
              f"bw={bw_now:.1f}Hz  lr={lr_now:.2e}")

    if patience >= PATIENCE:
        print(f"\nEarly stop at epoch {ep}")
        break


# ── EVALUATION ────────────────────────────────────────────────────────────────

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt", map_location=DEVICE))
model.eval()


def evaluate(loader, name):
    preds_all, labels_all = [], []
    phys_ev_all = []
    with torch.no_grad():
        for raw, feat, meta, labels in loader:
            raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
            logits, phys_ev, _ = model(raw, feat, meta)
            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())
            phys_ev_all.append(phys_ev.cpu().numpy())

    y_true   = np.array(labels_all)
    y_pred   = np.array(preds_all)
    phys_ev  = np.concatenate(phys_ev_all, axis=0)

    print(f"\n{'='*65}")
    print(f"RESULTS: {name}")
    print('='*65)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    print("False Positive Rate per class:")
    for c in range(N_CLASSES):
        fp  = np.sum((y_true != c) & (y_pred == c))
        tn  = np.sum((y_true != c) & (y_pred != c))
        print(f"  {label_names[c]:12s}: FPR = {fp / (fp + tn + 1e-10):.4f}")

    # Physics consistency check
    print("\nPhysics consistency (fraction where model agrees with dominant fault amp):")
    for ci, (cls, amp_idx) in enumerate(
            [('Inner Ring',1), ('Ball',2), ('Outer Ring',0)]):
        mask      = y_pred == (ci + 1)
        if mask.sum() == 0: continue
        dominant  = np.argmax(phys_ev[mask], axis=1)
        agree     = (dominant == amp_idx).mean()
        print(f"  {cls:12s}: {agree:.3f} physics agreement")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print("\nConfusion Matrix:")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{label_names[i]:12s}" + "".join(f"{v:>12d}" for v in row))

    return y_true, y_pred, cm, phys_ev


vt, vp, vcm, v_phys = evaluate(val_dl, "Validation Set")

# False-alarm test
fa_preds, fa_phys_ev = [], []
with torch.no_grad():
    for raw, feat, meta, labels in fa_dl:
        raw, feat, meta = raw.to(DEVICE), feat.to(DEVICE), meta.to(DEVICE)
        logits, pe, _ = model(raw, feat, meta)
        fa_preds.extend(logits.argmax(1).cpu().numpy())
        fa_phys_ev.append(pe.cpu().numpy())

fa_preds        = np.array(fa_preds)
fa_phys_ev      = np.concatenate(fa_phys_ev, axis=0)
false_alarm_rate = np.mean(fa_preds != 0)
print(f"\n{'='*65}")
print("FALSE ALARM TEST (Folder 11 – Shaft Misalignment)")
print(f"  {np.sum(fa_preds != 0)} / {len(fa_preds)} predicted as fault")
print(f"  False Alarm Rate = {false_alarm_rate:.4f}  (ideal = 0)")
print(f"  Avg BPFI amp (false alarms): {fa_phys_ev[:,1].mean():.4f}  "
      f"(should be low for true normal)")


# ── EXAMPLE INTERPRETATION ───────────────────────────────────────────────────
# Show a physics interpretation for the first validation sample of each fault class
print("\n" + "="*65)
print("SAMPLE PHYSICS INTERPRETATIONS")
print("="*65)
for cls_idx in range(N_CLASSES):
    idxs = np.where(np.array(val_ds.labels) == cls_idx)[0]
    if len(idxs) == 0: continue
    idx = int(idxs[0])
    raw_s  = val_ds.raw[idx:idx+1]
    feat_s = val_ds.feat[idx:idx+1]
    meta_s = val_ds.meta[idx:idx+1]
    interp = model.interpret(raw_s, feat_s, meta_s)
    print(f"\nTrue: {label_names[cls_idx]}")
    for k, v in interp.items():
        print(f"  {k}: {v}")


# ── PLOTS ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# F1 curves
axes[0,0].plot(history['tr_f1'],  label='Train', color='steelblue')
axes[0,0].plot(history['val_f1'], label='Val',   color='orangered')
axes[0,0].set_title('Macro F1 — PINN Hybrid'); axes[0,0].set_xlabel('Epoch')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# Loss breakdown
axes[0,1].plot(history['tr_focal'], label='Train Focal',   color='steelblue',  lw=2)
axes[0,1].plot(history['val_focal'],label='Val Focal',     color='orangered',  lw=2)
axes[0,1].plot(history['tr_phys'],  label='Train Physics', color='seagreen',   lw=2, ls='--')
axes[0,1].set_title('Loss Breakdown — PINN'); axes[0,1].set_xlabel('Epoch')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# Confusion matrix
im = axes[0,2].imshow(vcm, cmap='Blues')
axes[0,2].set_xticks(range(N_CLASSES)); axes[0,2].set_xticklabels(label_names, rotation=30, ha='right')
axes[0,2].set_yticks(range(N_CLASSES)); axes[0,2].set_yticklabels(label_names)
axes[0,2].set_title('Confusion Matrix (Val) — PINN Hybrid')
axes[0,2].set_xlabel('Predicted'); axes[0,2].set_ylabel('True')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        axes[0,2].text(j, i, str(vcm[i,j]), ha='center', va='center',
                       color='white' if vcm[i,j] > vcm.max()/2 else 'black')
plt.colorbar(im, ax=axes[0,2])

# Physics amplitudes per class (interpretability plot)
fault_amp_names = ['BPFO (Outer Ring)', 'BPFI (Inner Ring)', 'BSF (Ball)']
colors = ['#e74c3c', '#3498db', '#2ecc71']
for ci in range(N_CLASSES):
    mask = vt == ci
    if mask.sum() == 0: continue
    means = v_phys[mask].mean(axis=0)
    axes[1,0].bar(
        [x + ci * 0.2 for x in range(3)],
        means, width=0.18, label=label_names[ci],
        color=colors[min(ci, len(colors)-1)], alpha=0.8
    )
axes[1,0].set_xticks([0.3, 1.3, 2.3])
axes[1,0].set_xticklabels(fault_amp_names, rotation=15, ha='right')
axes[1,0].set_title('Physics: Mean Fault-Freq Amplitudes per Class')
axes[1,0].set_ylabel('RMS-normalised amplitude')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3, axis='y')

# False alarm physics amplitudes
axes[1,1].hist(fa_phys_ev[:, 1], bins=30, color='orangered',
               alpha=0.7, label='BPFI (false alarm data)')
axes[1,1].hist(v_phys[vt==1, 1], bins=30, color='steelblue',
               alpha=0.7, label='BPFI (true Inner Ring)')
axes[1,1].set_title('Physics Separation: BPFI amplitude')
axes[1,1].set_xlabel('RMS-normalised BPFI amplitude')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

# Physics loss annealing schedule
lam_curve = [physics_lambda(ep) for ep in range(1, len(history['tr_phys'])+1)]
ax2 = axes[1,2].twinx()
axes[1,2].plot(history['tr_phys'],  color='seagreen',  lw=2, label='Physics Loss (train)')
ax2.plot(lam_curve, color='gray', ls=':', lw=1.5, label='λ_phys')
axes[1,2].set_title('Physics Loss & Annealing Schedule')
axes[1,2].set_xlabel('Epoch')
axes[1,2].set_ylabel('Physics Loss', color='seagreen')
ax2.set_ylabel('λ_phys weight', color='gray')
axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {OUTPUT_DIR}/results.png")
print(f"Best model: {OUTPUT_DIR}/best_model.pt")
print("\nDone!")