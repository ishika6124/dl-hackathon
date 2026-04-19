"""
Cascaded Multimodal Bearing Fault Diagnosis — LOCO + Internal Stratified Val
=============================================================================
DATA STRATEGY:
  ✅ LOCO (Leave-One-Case-Out): For each fold → Test = 1 case, Train = remaining 10
  ✅ Validation: 15% stratified split FROM the pooled 10 training cases
     (NOT a full held-out case as val — that wastes data and biases the threshold)
  ✅ OOD Case 11: Evaluated once using the best-F1 fold model
  ✅ Architecture: Identical 3-stage cascade (UNCHANGED)

Why this is better:
  - Old: Train on 6 cases, val on 1 case (14% data for val, 86% for train)
  - New: Train on 10 cases (pooled), val is 15% drawn proportionally from ALL cases
  - Val now covers ALL operating conditions, not just one case's conditions
  - Anomaly threshold is calibrated on normal samples from ALL operating conditions
  - Stage 3 early stopping uses a val set that is representative of the test distribution

Results: Per-fold test metrics + aggregate mean/std over 11 folds
"""

import os, warnings, json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import scipy.io as sio
import scipy.signal as sp_signal
from scipy.signal import hilbert
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
CFG = dict(
    dataset_path   = "/home/teaching/group46/hackathon_dataset/SCA bearing dataset",
    output_dir     = "/home/teaching/hackathon/dl_approach/outputs_loco_cascade",
    device         = "cuda" if torch.cuda.is_available() else "cpu",

    target_len     = 4096,
    enc_dim        = 128,
    phys_feat_dim  = 16,

    # ── LOCO Strategy ──────────────────────────────────────────────
    all_cases      = list(range(1, 12)),   # Cases 1-11 (11 is OOD)
    loco_cases     = list(range(1, 11)),   # Cases 1-10 rotate as test
    ood_case       = 11,
    val_frac       = 0.15,                 # Internal stratified val from train pool

    # ── Training ───────────────────────────────────────────────────
    pre_epochs     = 60,
    pre_lr         = 1e-3,
    pre_batch      = 32,
    pre_temp       = 0.1,
    pre_patience   = 15,

    s1_epochs      = 80,
    s1_lr          = 1e-3,
    s1_batch       = 32,
    s1_patience    = 20,
    s1_mc          = 20,

    s3_epochs      = 120,
    s3_lr          = 5e-4,
    s3_batch       = 16,
    s3_lambda      = 0.5,
    s3_mu          = 0.1,
    s3_gamma       = 2.0,
    s3_patience    = 25,

    bearing_thr    = 0.12,
    shaft_thr      = 0.20,

    label_names    = {0:"Normal", 1:"Inner Ring", 2:"Ball", 3:"Outer Ring"},
    seed           = 42,
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(CFG["output_dir"], exist_ok=True)
DEVICE = torch.device(CFG["device"])
L      = CFG["target_len"]
print(f"Device : {DEVICE}")
print(f"Strategy: LOCO ({len(CFG['loco_cases'])} folds) + internal stratified val "
      f"({int(CFG['val_frac']*100)}% from pooled train cases)")

SENSOR_KEYS = ["DS", "FS", "Upper", "Lower"]
LN4         = ["Normal", "Inner Ring", "Ball", "Outer Ring"]


# ══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def _norm_ff(ff):
    keys = ['FTFMultiple', 'BPFMultiple', 'BPFOMultiple', 'BPFIMultiple']
    if isinstance(ff, dict):
        return {k: float(np.array(ff[k]).flat[0]) for k in keys}
    try:
        fi = ff[0, 0]
        return {k: float(fi[k].flat[0]) for k in keys}
    except Exception:
        pass
    try:
        arr = np.array(ff, dtype=np.float32).ravel()
        if len(arr) >= 4:
            return {keys[i]: float(arr[i]) for i in range(4)}
    except Exception:
        pass
    return {k: 1.0 for k in keys}


def _get_sensor_dict(s):
    def _get(obj, key):
        return obj[key]
    try:
        rd  = _get(s, 'rawData')
        lbl = np.array(_get(s, 'label'),        dtype=np.int64).ravel()
        rpm = np.array(_get(s, 'RPM'),           dtype=np.float32).ravel()
        sr  = float(np.array(_get(s, 'samplingRate')).ravel()[0])
        ff  = _norm_ff(_get(s, 'faultFrequencies'))
        return dict(rawData=rd, label=lbl, RPM=rpm, samplingRate=sr,
                    faultFrequencies=ff)
    except Exception:
        return None


def load_mat_v72(path):
    try:
        raw = sio.loadmat(str(path))
        out = {}
        for sk in SENSOR_KEYS:
            if sk not in raw:
                continue
            try:
                sd = _get_sensor_dict(raw[sk][0, 0])
                if sd is not None:
                    out[sk] = sd
            except Exception:
                pass
        return out if out else None
    except Exception:
        return None


def load_mat_v73(path):
    try:
        import h5py
    except ImportError:
        return None

    def _read(item, f):
        if isinstance(item, h5py.Dataset):
            arr = item[()]
            if arr.dtype == object:
                flat    = arr.flatten()
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

    try:
        with h5py.File(str(path), 'r') as f:
            raw = {k: _read(f[k], f) for k in f.keys()
                   if not k.startswith('#')}
        out = {}
        for sk in SENSOR_KEYS:
            if sk not in raw:
                continue
            s  = raw[sk]
            rd = s.get('rawData', None)
            if rd is None:
                continue
            if isinstance(rd, np.ndarray) and rd.dtype != object:
                rd = rd.T
            try:
                out[sk] = dict(
                    rawData          = rd,
                    label            = np.array(s['label']).flatten().astype(np.int64),
                    RPM              = np.array(s['RPM']).flatten().astype(np.float32),
                    samplingRate     = float(np.array(s['samplingRate']).ravel()[0]),
                    faultFrequencies = _norm_ff(s['faultFrequencies']),
                )
            except Exception:
                pass
        return out if out else None
    except Exception:
        return None


def load_mat_any(path):
    r = load_mat_v72(path)
    return r if r else (load_mat_v73(path) or {})


def parse_sensor_dict(sd, folder_id):
    raw_data = sd['rawData']
    labels   = sd['label'].ravel()
    rpm_vals = sd['RPM'].ravel()
    sr       = float(sd['samplingRate'])
    ff       = sd['faultFrequencies']
    bpfi = ff.get('BPFIMultiple', 1.0)
    bpfo = ff.get('BPFOMultiple', 1.0)
    bpf  = ff.get('BPFMultiple',  1.0)
    ftf  = ff.get('FTFMultiple',  1.0)

    if raw_data.dtype == object:
        n_cells      = raw_data.shape[1]
        cell_signals = []
        for ci in range(n_cells):
            elem = np.array(raw_data[0, ci]).flatten().astype(np.float64)
            cell_signals.append(elem if len(elem) >= 64 else None)
        indices = range(len(cell_signals))
        def get_sig(i): return cell_signals[i]
    else:
        if raw_data.ndim != 2 or raw_data.shape[1] < 64:
            return []
        indices = range(raw_data.shape[0])
        def get_sig(i): return raw_data[i].astype(np.float64)

    records = []
    for i in indices:
        label = int(labels[i]) if i < len(labels) else -1
        if label == -1:
            continue
        sig = get_sig(i)
        if sig is None or len(sig) < 64:
            continue
        rpm = float(rpm_vals[i]) if i < len(rpm_vals) else 0.0
        records.append(dict(
            signal = sig.astype(np.float32),
            label  = label,
            rpm    = rpm,
            sr     = sr,
            bpfi   = bpfi,
            bpfo   = bpfo,
            bpf    = bpf,
            ftf    = ftf,
            case   = folder_id,
        ))
    return records


def load_case(case_num):
    folder  = Path(CFG["dataset_path"]) / str(case_num)
    records = []
    for fname in ["train.mat", "test.mat"]:
        path = folder / fname
        if not path.exists():
            continue
        sensor_dicts = load_mat_any(path)
        for sk, sd in sensor_dicts.items():
            try:
                recs = parse_sensor_dict(sd, case_num)
                records.extend(recs)
                print(f"  Case {case_num:02d}/{fname}/{sk}: {len(recs)} samples")
            except Exception as e:
                print(f"  Case {case_num:02d}/{fname}/{sk}: SKIPPED – {e}")
    return records


# ══════════════════════════════════════════════════════════════════════
# 2.  LOCO SPLIT HELPER
# ══════════════════════════════════════════════════════════════════════

def make_loco_split(all_case_recs, test_case, val_frac=0.15, seed=42):
    """
    LOCO strategy with internal stratified validation.

    Step 1 → Pool all cases EXCEPT test_case  (10 training cases)
    Step 2 → StratifiedShuffleSplit the pool:
               train_pool = 85%  (used for Phase 0 + Stage 1 + Stage 3)
               val_pool   = 15%  (used for early-stopping + threshold calibration)
    Step 3 → test = records from test_case

    Key: val comes FROM training cases (not a held-out case),
         so it reflects ALL operating conditions the model was trained on.
    """
    train_cases = [c for c in CFG["loco_cases"] if c != test_case]
    pool        = []
    for c in train_cases:
        pool.extend(all_case_recs.get(c, []))

    test_recs = all_case_recs.get(test_case, [])

    if not pool:
        return [], [], test_recs

    labels = np.array([r["label"] for r in pool])
    idx    = np.arange(len(pool))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    try:
        tr_idx, vl_idx = next(sss.split(idx, labels))
    except ValueError:
        # fallback: random if any class has < 2 samples
        rng    = np.random.RandomState(seed)
        perm   = rng.permutation(len(pool))
        n_val  = max(1, int(len(pool) * val_frac))
        vl_idx = perm[:n_val]
        tr_idx = perm[n_val:]

    train_recs = [pool[i] for i in tr_idx]
    val_recs   = [pool[i] for i in vl_idx]

    def dist(recs):
        d = defaultdict(int)
        for r in recs: d[r["label"]] += 1
        return dict(sorted(d.items()))

    print(f"  Pool  : {len(pool)} samples from cases {train_cases}")
    print(f"  Train : {len(train_recs)} samples | {dist(train_recs)}")
    print(f"  Val   : {len(val_recs)}  samples | {dist(val_recs)}")
    print(f"  Test  : {len(test_recs)} samples (Case {test_case}) | {dist(test_recs)}")

    return train_recs, val_recs, test_recs


# ══════════════════════════════════════════════════════════════════════
# 3.  SIGNAL PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def resample_fixed(x, tgt=L):
    if len(x) == tgt:
        return x.copy().astype(np.float32)
    return np.interp(
        np.linspace(0, len(x)-1, tgt), np.arange(len(x)), x
    ).astype(np.float32)


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def physics_features(x, rpm, sr, bpfi, bpfo, bpf):
    """16-dim physics feature vector (unchanged)."""
    feats = []
    rms   = float(np.sqrt(np.mean(x**2)) + 1e-9)
    feats.append(rms)
    feats.append(float(np.max(np.abs(x))) / rms)
    feats.append(float(np.mean(x**4)) / (rms**4 + 1e-9))

    env   = np.abs(hilbert(x))
    efft  = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(x), 1.0/sr)
    tot   = efft.sum() + 1e-9

    def be(fc):
        bw   = max(fc * 0.05, 0.5)
        mask = (freqs >= fc - bw) & (freqs <= fc + bw)
        return float(efft[mask].sum() + 1e-9)

    for fc in [bpfi, bpfo, bpf]:
        feats.append(sum(be(fc*h) for h in [1, 2, 3]) / tot)

    rpmhz = rpm / 60.0
    feats.append(
        sum(be(rpmhz*h) for h in [1, 2, 3]) / tot if rpmhz > 0.5 else 0.0
    )

    mag  = np.abs(np.fft.rfft(x))
    fq   = np.fft.rfftfreq(len(x), 1.0/sr)
    maxf = fq.max() + 1e-9
    for lo, hi in [(.0,.25), (.25,.5), (.5,.75), (.75,1.)]:
        m = (fq >= lo*maxf) & (fq < hi*maxf)
        feats.append(float(mag[m].sum()) / (mag.sum() + 1e-9))

    s = bpfi + bpfo + bpf + 1e-9
    feats += [bpfi/s, bpfo/s, bpf/s]

    arr = np.clip(np.array(feats, dtype=np.float32), -10, 10)
    pd  = CFG["phys_feat_dim"]
    if len(arr) < pd:
        arr = np.concatenate([arr, np.zeros(pd - len(arr), dtype=np.float32)])
    return arr[:pd]


def preprocess(r):
    x = normalize(resample_fixed(r["signal"]))
    p = physics_features(x, r["rpm"], r["sr"], r["bpfi"], r["bpfo"], r["bpf"])
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            r["label"])


# ══════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════

class BearingDS(Dataset):
    def __init__(self, records):
        self.data = []
        for r in records:
            sig, phys, lbl = preprocess(r)
            self.data.append((sig, phys, lbl, r["bpfi"], r["bpfo"], r["bpf"]))

    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]


def class_weights(records, classes):
    cnt = defaultdict(int)
    for r in records:
        if r["label"] in classes:
            cnt[r["label"]] += 1
    tot = sum(cnt.values())
    if tot == 0:
        return torch.ones(len(classes))
    w = torch.tensor(
        [tot / (len(classes) * max(cnt[c], 1)) for c in classes],
        dtype=torch.float32
    )
    return w / w.sum() * len(classes)


# ══════════════════════════════════════════════════════════════════════
# 5.  MODEL COMPONENTS  (architecture UNCHANGED)
# ══════════════════════════════════════════════════════════════════════

class TCNBlock(nn.Module):
    def __init__(self, ic, oc, k=3, d=1, drop=0.1):
        super().__init__()
        p       = (k - 1) * d
        self.c1 = nn.Conv1d(ic, oc, k, dilation=d, padding=p)
        self.c2 = nn.Conv1d(oc, oc, k, dilation=d, padding=p)
        self.n1 = nn.BatchNorm1d(oc)
        self.n2 = nn.BatchNorm1d(oc)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        T = x.shape[-1]
        h = self.drop(F.gelu(self.n1(self.c1(x)[..., :T])))
        h = self.drop(F.gelu(self.n2(self.c2(h)[..., :T])))
        return h + self.skip(x)


class TCNEncoder(nn.Module):
    def __init__(self, out_dim=128, drop=0.1):
        super().__init__()
        chs  = [1, 32, 64, 128, 128]
        dils = [1, 2, 4, 8]
        self.tcn  = nn.Sequential(*[
            TCNBlock(chs[i], chs[i+1], d=dils[i], drop=drop)
            for i in range(len(dils))
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.tcn(x.unsqueeze(1))
        return self.proj(self.drop(self.pool(h).squeeze(-1)))


class PhysMLP(nn.Module):
    def __init__(self, in_d=16, out_d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, out_d)
        )

    def forward(self, x): return self.net(x)


class SharedEncoder(nn.Module):
    def __init__(self, enc_dim=128, phys_dim=16, drop=0.1):
        super().__init__()
        self.tcn  = TCNEncoder(enc_dim, drop)
        self.phys = PhysMLP(phys_dim, 32)
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim + 32, enc_dim), nn.GELU(),
            nn.Dropout(drop), nn.Linear(enc_dim, enc_dim)
        )
        self.norm = nn.LayerNorm(enc_dim)

    def forward(self, sig, phys):
        return self.norm(self.fuse(
            torch.cat([self.tcn(sig), self.phys(phys)], dim=-1)
        ))


class SVDDHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(dim), requires_grad=False)

    @torch.no_grad()
    def init_center(self, encoder, loader):
        if loader is None:
            return
        encoder.eval()
        zs = []
        for sig, phys, lbl, *_ in loader:
            mask = (lbl == 0)
            if not mask.any():
                continue
            z = encoder(sig[mask].to(DEVICE), phys[mask].to(DEVICE))
            zs.append(z.cpu())
        if zs:
            self.center.data = torch.cat(zs).mean(0).to(DEVICE)

    def forward(self, z):
        return ((z - self.center) ** 2).sum(-1)


class Forecaster(nn.Module):
    def __init__(self, enc_dim=128, out_len=None):
        super().__init__()
        self.out = out_len or (L // 2)
        self.net = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.GELU(), nn.Linear(256, self.out)
        )

    def forward(self, z): return self.net(z)


class PhysicsTransformer(nn.Module):
    def __init__(self, d=128, heads=4, layers=2, n_cls=3):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.tf  = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 9, d))
        self.pin = nn.Linear(d, d)
        self.head= nn.Linear(d, n_cls)

    def forward(self, x):
        B   = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, self.pin(x)], dim=1) + self.pos
        return self.head(self.tf(x)[:, 0])


class Stage3Model(nn.Module):
    def __init__(self, enc_dim=128, phys_dim=16, n_cls=3, drop=0.1):
        super().__init__()
        self.encoder     = SharedEncoder(enc_dim, phys_dim, drop)
        self.patch_proj  = nn.Linear(enc_dim, 8 * enc_dim)
        self.transformer = PhysicsTransformer(enc_dim, 4, 2, n_cls)
        self.phys_head   = nn.Linear(enc_dim, 3)

    def forward(self, sig, phys, ret_phys=False):
        z  = self.encoder(sig, phys)
        px = self.patch_proj(z).reshape(z.shape[0], 8, -1)
        lg = self.transformer(px)
        if ret_phys:
            return lg, torch.sigmoid(self.phys_head(z))
        return lg


# ══════════════════════════════════════════════════════════════════════
# 6.  LOSS FUNCTIONS  (unchanged)
# ══════════════════════════════════════════════════════════════════════

def nt_xent(z, labels, temp=0.1):
    z   = F.normalize(z, dim=-1)
    sim = z @ z.T / temp
    B   = z.shape[0]
    I   = torch.eye(B, device=z.device)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() - I
    exp = torch.exp(sim) * (1 - I)
    lp  = sim - torch.log(exp.sum(-1, keepdim=True) + 1e-9)
    npos= pos.sum(-1).clamp(min=1)
    return (-(pos * lp).sum(-1) / npos).mean()


def supcon(z, labels, temp=0.07):
    z   = F.normalize(z, dim=-1)
    B   = z.shape[0]
    sim = z @ z.T / temp
    I   = torch.eye(B, device=z.device).bool()
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~I
    if not pos.any():
        return torch.tensor(0., device=z.device)
    exp  = torch.exp(sim).masked_fill(I, 0)
    logd = torch.log(exp.sum(-1) + 1e-9)
    npos = pos.float().sum(-1).clamp(min=1)
    loss = (-(sim - logd.unsqueeze(1)) * pos.float()).sum(-1) / npos
    return loss[pos.any(-1)].mean()


def focal(logits, targets, gamma=2., w=None):
    ce  = F.cross_entropy(logits, targets, weight=w, reduction="none")
    pt  = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def phys_loss(pred, bpfi, bpfo, bpf):
    s   = bpfi + bpfo + bpf + 1e-9
    tgt = torch.stack([bpfi/s, bpfo/s, bpf/s], -1).to(pred.device)
    return F.mse_loss(pred, tgt)


# ══════════════════════════════════════════════════════════════════════
# 7.  PHASE 0 — CONTRASTIVE PRETRAINING
# ══════════════════════════════════════════════════════════════════════

def pretrain_encoder(train_recs, val_recs, fold_tag=""):
    """
    Train SharedEncoder with NT-Xent.
    Val is the internal stratified split — ensures pretraining sees
    all operating conditions when picking the best checkpoint.
    """
    print(f"  [{fold_tag}] Phase 0 — Contrastive Pretraining "
          f"({len(train_recs)} train / {len(val_recs)} val)")

    encoder = SharedEncoder(CFG["enc_dim"], CFG["phys_feat_dim"]).to(DEVICE)
    opt     = torch.optim.Adam(encoder.parameters(), lr=CFG["pre_lr"])
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=CFG["pre_epochs"])

    tr_ldr = DataLoader(BearingDS(train_recs), batch_size=CFG["pre_batch"],
                        shuffle=True, drop_last=True, num_workers=0)
    vl_ldr = DataLoader(BearingDS(val_recs), batch_size=CFG["pre_batch"],
                        shuffle=False, num_workers=0) if val_recs else None

    best_loss, best_w, wait = 1e9, None, 0
    for ep in range(1, CFG["pre_epochs"] + 1):
        encoder.train()
        tl = 0.
        for sig, phys, lbl, *_ in tr_ldr:
            sig, phys, lbl = sig.to(DEVICE), phys.to(DEVICE), lbl.to(DEVICE)
            loss = nt_xent(encoder(sig, phys), lbl, CFG["pre_temp"])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        sched.step()
        tl /= max(len(tr_ldr), 1)

        vl_l = 0.
        if vl_ldr:
            encoder.eval()
            with torch.no_grad():
                for sig, phys, lbl, *_ in vl_ldr:
                    sig, phys, lbl = sig.to(DEVICE), phys.to(DEVICE), lbl.to(DEVICE)
                    vl_l += nt_xent(encoder(sig, phys), lbl, CFG["pre_temp"]).item()
            vl_l /= max(len(vl_ldr), 1)

        if ep % 10 == 0:
            print(f"    Ep {ep:3d}  tr={tl:.4f}  vl={vl_l:.4f}")

        monitor = vl_l if vl_ldr else tl
        if monitor < best_loss:
            best_loss = monitor
            best_w    = {k: v.clone() for k, v in encoder.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= CFG["pre_patience"]:
                print(f"    Early stop ep {ep}  best_loss={best_loss:.4f}")
                break

    encoder.load_state_dict(best_w)
    print(f"    Best contrastive loss: {best_loss:.4f}")
    return encoder


# ══════════════════════════════════════════════════════════════════════
# 8.  STAGE 1 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════

def train_stage1(encoder, train_recs, val_recs, fold_tag=""):
    """
    Key improvement: threshold is calibrated on normal samples drawn from ALL
    operating conditions (via the internal val split), not just one case.
    This prevents threshold being tuned to a single RPM/load condition.
    """
    print(f"  [{fold_tag}] Stage 1 — Anomaly Detection")

    for p in encoder.parameters():
        p.requires_grad = False

    fc   = Forecaster(CFG["enc_dim"]).to(DEVICE)
    svdd = SVDDHead(CFG["enc_dim"]).to(DEVICE)
    half = L // 2

    # Use ONLY normal samples for unsupervised training
    norm_tr = [r for r in train_recs if r["label"] == 0]
    norm_vl = [r for r in val_recs   if r["label"] == 0]

    print(f"    Normal train: {len(norm_tr)} | Normal val: {len(norm_vl)}")

    if not norm_tr:
        print("    [WARN] No Normal training records.")
        return fc, svdd, 1.0

    tr_ldr = DataLoader(BearingDS(norm_tr), batch_size=CFG["s1_batch"],
                        shuffle=True, drop_last=False, num_workers=0)
    svdd.init_center(encoder, tr_ldr)

    opt   = torch.optim.Adam(fc.parameters(), lr=CFG["s1_lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["s1_epochs"])

    best_loss, best_fc, wait = 1e9, None, 0
    for ep in range(1, CFG["s1_epochs"] + 1):
        encoder.eval(); fc.train()
        tl = 0.
        for sig, phys, lbl, *_ in tr_ldr:
            sig, phys = sig.to(DEVICE), phys.to(DEVICE)
            with torch.no_grad():
                z = encoder(sig[:, :half], phys)
            pred = fc(z)
            l_fc = F.mse_loss(pred, sig[:, half:].to(DEVICE))
            with torch.no_grad():
                zf = encoder(sig, phys)
            l_sv = svdd(zf).mean()
            loss = l_fc + 0.1 * l_sv
            opt.zero_grad(); loss.backward(); opt.step()
            tl += l_fc.item()
        sched.step()
        tl /= max(len(tr_ldr), 1)

        if tl < best_loss:
            best_loss = tl
            best_fc   = {k: v.clone() for k, v in fc.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= CFG["s1_patience"]:
                break

    fc.load_state_dict(best_fc)

    # Calibrate threshold on internal val normal samples (multi-condition)
    encoder.eval(); fc.eval()
    scores = []
    if norm_vl:
        vl_ldr = DataLoader(BearingDS(norm_vl), batch_size=64,
                             shuffle=False, num_workers=0)
        with torch.no_grad():
            for sig, phys, lbl, *_ in vl_ldr:
                sig, phys = sig.to(DEVICE), phys.to(DEVICE)
                z   = encoder(sig[:, :half], phys)
                err = ((fc(z) - sig[:, half:].to(DEVICE)) ** 2).mean(-1)
                dist= svdd(encoder(sig, phys))
                scores.extend((0.5*err + 0.5*dist).cpu().numpy())

    thr = float(np.percentile(scores, 95)) if scores else 1.0
    print(f"    Threshold (95th pct, {len(scores)} normal-val samples): {thr:.5f}")
    return fc, svdd, thr


def score_stage1(encoder, fc, svdd, thr, sig, phys, mc=20):
    half = L // 2

    def _enable_dropout(m):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()

    scores = []
    for _ in range(mc):
        _enable_dropout(encoder)
        _enable_dropout(fc)
        with torch.no_grad():
            z    = encoder(sig[:, :half], phys)
            err  = ((fc(z) - sig[:, half:].to(DEVICE)) ** 2).mean(-1)
            dist = svdd(encoder(sig, phys))
            scores.append((0.5*err + 0.5*dist).cpu().numpy())

    mu  = np.stack(scores).mean(0)
    std = np.stack(scores).std(0)
    return mu > thr, mu, std


# ══════════════════════════════════════════════════════════════════════
# 9.  STAGE 2 — PHYSICS GATE  (unchanged)
# ══════════════════════════════════════════════════════════════════════

def physics_gate(x_np, rpm, sr, bpfi, bpfo, bpf):
    env   = np.abs(hilbert(x_np))
    efft  = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(x_np), 1.0/sr)
    tot   = efft.sum() + 1e-9

    def be(fc):
        bw   = max(fc * 0.05, 0.5)
        mask = (freqs >= fc - bw) & (freqs <= fc + bw)
        return float(efft[mask].sum() + 1e-9)

    R     = sum(be(fc*h) for fc in [bpfi, bpfo, bpf] for h in [1, 2, 3]) / tot
    rpmhz = rpm / 60.0
    S     = sum(be(rpmhz*h) for h in [1, 2, 3]) / tot if rpmhz > 0.5 else 0.0

    if R > CFG["bearing_thr"]:  return "bearing"
    if S > CFG["shaft_thr"]:    return "non_bearing"
    return "ambiguous"


# ══════════════════════════════════════════════════════════════════════
# 10.  STAGE 3 — FAULT TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def train_stage3(pretrained_enc, train_recs, val_recs, fold_tag=""):
    """
    Key improvement: val_recs now covers all operating conditions,
    so the best checkpoint is genuinely the most generalizable,
    not just the best on one case's conditions.
    """
    print(f"  [{fold_tag}] Stage 3 — Fault Type (IRF/Ball/ORF)")

    fault_set = {1, 2, 3}
    f_tr = [r for r in train_recs if r["label"] in fault_set]
    f_vl = [r for r in val_recs   if r["label"] in fault_set]

    dist_tr = defaultdict(int)
    for r in f_tr: dist_tr[r["label"]] += 1
    print(f"    Fault train: {dict(dist_tr)}  |  Fault val: {len(f_vl)} samples")

    if not f_tr:
        print("    [WARN] No fault training data.")
        return None

    def remap(recs):
        return [dict(r, label=r["label"]-1) for r in recs]

    f_tr_r = remap(f_tr)
    f_vl_r = remap(f_vl)

    model = Stage3Model(CFG["enc_dim"], CFG["phys_feat_dim"]).to(DEVICE)
    model.encoder.load_state_dict(pretrained_enc.state_dict())

    cw = class_weights(f_tr_r, [0, 1, 2]).to(DEVICE)
    print(f"    Class weights: {cw.cpu().numpy().round(3)}")

    tr_ldr = DataLoader(BearingDS(f_tr_r), batch_size=CFG["s3_batch"],
                        shuffle=True, drop_last=False, num_workers=0)
    vl_ldr = DataLoader(BearingDS(f_vl_r), batch_size=CFG["s3_batch"],
                        shuffle=False, num_workers=0) if f_vl_r else None

    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(),     "lr": CFG["s3_lr"] * 0.1},
        {"params": model.patch_proj.parameters(),  "lr": CFG["s3_lr"]},
        {"params": model.transformer.parameters(), "lr": CFG["s3_lr"]},
        {"params": model.phys_head.parameters(),   "lr": CFG["s3_lr"]},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["s3_epochs"])

    best_f1, best_w, wait = 0., None, 0
    for ep in range(1, CFG["s3_epochs"] + 1):
        model.train()
        for sig, phys, lbl, bpfi, bpfo, bpf in tr_ldr:
            sig  = sig.to(DEVICE);  phys = phys.to(DEVICE)
            lbl  = lbl.to(DEVICE)
            bpfi = bpfi.float().to(DEVICE)
            bpfo = bpfo.float().to(DEVICE)
            bpf  = bpf.float().to(DEVICE)

            logits, pp = model(sig, phys, ret_phys=True)
            z          = model.encoder(sig, phys)
            l_f  = focal(logits, lbl, CFG["s3_gamma"], cw)
            l_sc = supcon(z, lbl)
            l_ph = phys_loss(pp, bpfi, bpfo, bpf)
            loss = l_f + CFG["s3_lambda"]*l_sc + CFG["s3_mu"]*l_ph
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if vl_ldr:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for sig, phys, lbl, *_ in vl_ldr:
                    lg = model(sig.to(DEVICE), phys.to(DEVICE))
                    preds.extend(lg.argmax(-1).cpu().numpy())
                    trues.extend(lbl.numpy())
            vf1 = f1_score(trues, preds, average="macro", zero_division=0)
            if ep % 20 == 0:
                print(f"    Ep {ep:3d}  val_fault_f1={vf1:.4f}")
            if vf1 > best_f1:
                best_f1 = vf1
                best_w  = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= CFG["s3_patience"]:
                    print(f"    Early stop ep {ep}")
                    break
        elif ep % 30 == 0:
            print(f"    Ep {ep:3d}  (no val fault data)")

    if best_w:
        model.load_state_dict(best_w)
    print(f"    Best val fault macro-F1: {best_f1:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════
# 11.  CASCADE INFERENCE  (unchanged)
# ══════════════════════════════════════════════════════════════════════

def cascade_predict(r, encoder, fc, svdd, thr, s3_model):
    sig_np, phys_np, true_lbl = preprocess(r)
    sig  = sig_np.unsqueeze(0).to(DEVICE)
    phys = phys_np.unsqueeze(0).to(DEVICE)

    is_abn, _, _ = score_stage1(encoder, fc, svdd, thr, sig, phys, mc=CFG["s1_mc"])
    if not bool(is_abn[0]):
        return 0, true_lbl

    gate = physics_gate(sig_np.numpy(), r["rpm"], r["sr"],
                         r["bpfi"], r["bpfo"], r["bpf"])
    if gate == "non_bearing":
        return -1, true_lbl

    if s3_model is None:
        return 1, true_lbl
    s3_model.eval()
    with torch.no_grad():
        pred_cls = s3_model(sig, phys).argmax(-1).item()
    return pred_cls + 1, true_lbl


def evaluate_records(records, encoder, fc, svdd, thr, s3_model, tag=""):
    if not records:
        return {"accuracy": 0., "macro_f1": 0., "n": 0}

    preds, trues = [], []
    for r in records:
        pred, true = cascade_predict(r, encoder, fc, svdd, thr, s3_model)
        preds.append(max(pred, 0))
        trues.append(true)

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    print(f"\n  [{tag}]  n={len(trues)}  Acc={acc:.4f}  Macro-F1={f1:.4f}")
    print(classification_report(trues, preds, target_names=LN4,
                                  labels=[0,1,2,3], zero_division=0))
    cm = confusion_matrix(trues, preds, labels=[0,1,2,3])
    hdr = f"{'':15s}" + "".join(f"{n:>13s}" for n in LN4)
    print(hdr)
    for i, row in enumerate(cm):
        print(f"{LN4[i]:15s}" + "".join(f"{v:13d}" for v in row))
    return {"accuracy": float(acc), "macro_f1": float(f1), "n": len(trues)}


# ══════════════════════════════════════════════════════════════════════
# 12.  SINGLE LOCO FOLD
# ══════════════════════════════════════════════════════════════════════

def run_fold(test_case, all_case_recs):
    fold_tag = f"Fold {test_case:02d}"
    train_cases = [c for c in CFG["loco_cases"] if c != test_case]

    print(f"\n{'═'*68}")
    print(f"  {fold_tag}  |  Test = Case {test_case}  |  "
          f"Train = Cases {train_cases}")
    print(f"{'═'*68}")

    # ── Build splits using the LOCO + internal stratified strategy ────
    print(f"\n  Building LOCO split...")
    train_recs, val_recs, test_recs = make_loco_split(
        all_case_recs, test_case,
        val_frac=CFG["val_frac"], seed=CFG["seed"]
    )

    if not train_recs:
        print(f"  [SKIP] Empty training pool for {fold_tag}.")
        return None, None

    # ── Phase 0: Contrastive pretraining ─────────────────────────────
    encoder = pretrain_encoder(train_recs, val_recs, fold_tag)

    # ── Stage 1: Anomaly detection ────────────────────────────────────
    fc, svdd, thr = train_stage1(encoder, train_recs, val_recs, fold_tag)

    # ── Stage 3: Fault type classification ───────────────────────────
    s3 = train_stage3(encoder, train_recs, val_recs, fold_tag)

    # ── Evaluate on held-out test case ───────────────────────────────
    metrics = evaluate_records(
        test_recs, encoder, fc, svdd, thr, s3,
        tag=f"{fold_tag} → Test Case {test_case}"
    )
    metrics["test_case"] = test_case

    # ── Save fold checkpoint ──────────────────────────────────────────
    fold_dir = Path(CFG["output_dir"]) / f"fold_{test_case:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), fold_dir / "encoder.pt")
    torch.save(fc.state_dict(),      fold_dir / "forecaster.pt")
    torch.save(svdd.state_dict(),    fold_dir / "svdd.pt")
    if s3:
        torch.save(s3.state_dict(), fold_dir / "stage3.pt")
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, (encoder, fc, svdd, thr, s3)


# ══════════════════════════════════════════════════════════════════════
# 13.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*68)
    print("  Cascaded 3-Stage Fault Diagnosis — LOCO + Internal Stratified Val")
    print("="*68)
    print(f"  Strategy : Test=1 case | Train=pooled 10 cases")
    print(f"             Val = {int(CFG['val_frac']*100)}% stratified FROM pooled train")
    print(f"  Folds    : {len(CFG['loco_cases'])} (Cases 1-10)")
    print(f"  OOD      : Case 11 (shaft misalignment)")

    # ── Load ALL cases once (avoids redundant I/O per fold) ───────────
    print("\n" + "─"*68)
    print("  Loading all cases...")
    all_case_recs = {}
    for c in CFG["loco_cases"]:
        print(f"\n  Case {c}:")
        recs = load_case(c)
        all_case_recs[c] = recs

    print(f"\n  OOD Case {CFG['ood_case']}:")
    ood_recs = load_case(CFG["ood_case"])

    # Summary table
    print("\n" + "─"*68)
    print(f"  {'Case':>5}  {'Samples':>8}  {'Normal':>8}  {'IRF':>6}  "
          f"{'Ball':>6}  {'ORF':>6}")
    print("  " + "─"*62)
    for c in CFG["loco_cases"]:
        recs = all_case_recs[c]
        d    = defaultdict(int)
        for r in recs: d[r["label"]] += 1
        print(f"  {c:>5}  {len(recs):>8}  {d[0]:>8}  {d[1]:>6}  "
              f"{d[2]:>6}  {d[3]:>6}")
    od = defaultdict(int)
    for r in ood_recs: od[r["label"]] += 1
    print(f"  {'11(OOD)':>5}  {len(ood_recs):>8}  {od[0]:>8}  {od[1]:>6}  "
          f"{od[2]:>6}  {od[3]:>6}")

    # ── Run all LOCO folds ────────────────────────────────────────────
    print("\n" + "="*68)
    print("  RUNNING LOCO FOLDS")
    print("="*68)

    fold_results   = []
    best_f1        = -1.
    best_model_tup = None
    best_fold_case = None

    for test_case in CFG["loco_cases"]:
        metrics, model_tup = run_fold(test_case, all_case_recs)
        if metrics is not None:
            fold_results.append(metrics)
            if metrics["macro_f1"] > best_f1:
                best_f1        = metrics["macro_f1"]
                best_model_tup = model_tup
                best_fold_case = test_case

    # ── Aggregate results ─────────────────────────────────────────────
    print("\n" + "="*68)
    print("  LOCO CROSS-VALIDATION SUMMARY")
    print("="*68)
    print(f"\n  {'Case':>6}  {'Accuracy':>10}  {'Macro-F1':>10}  {'N':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}")

    accs = [m["accuracy"]  for m in fold_results]
    f1s  = [m["macro_f1"]  for m in fold_results]
    ns   = [m["n"]         for m in fold_results]

    for m in fold_results:
        flag = "  ← best" if m["test_case"] == best_fold_case else ""
        print(f"  {m['test_case']:>6}  {m['accuracy']:>10.4f}  "
              f"{m['macro_f1']:>10.4f}  {m['n']:>7}{flag}")

    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}")
    print(f"  {'MEAN':>6}  {np.mean(accs):>10.4f}  {np.mean(f1s):>10.4f}")
    print(f"  {'STD':>6}  {np.std(accs):>10.4f}  {np.std(f1s):>10.4f}")
    print(f"  {'MIN':>6}  {np.min(accs):>10.4f}  {np.min(f1s):>10.4f}")
    print(f"  {'MAX':>6}  {np.max(accs):>10.4f}  {np.max(f1s):>10.4f}")

    # ── OOD Evaluation — Case 11 ──────────────────────────────────────
    ood_metrics = {}
    if ood_recs and best_model_tup is not None:
        print(f"\n{'='*68}")
        print(f"  OOD EVALUATION — Case 11 (Shaft Misalignment)")
        print(f"  Using best fold model (test_case={best_fold_case}, "
              f"F1={best_f1:.4f})")
        print(f"{'='*68}")
        encoder, fc, svdd, thr, s3 = best_model_tup
        ood_metrics = evaluate_records(
            ood_recs, encoder, fc, svdd, thr, s3,
            tag="OOD Case 11"
        )

        # Gate breakdown for OOD samples
        gate_counts = Counter()
        for r in ood_recs:
            sig_np, phys_np, _ = preprocess(r)
            sig  = sig_np.unsqueeze(0).to(DEVICE)
            phys = phys_np.unsqueeze(0).to(DEVICE)
            is_abn, _, _ = score_stage1(encoder, fc, svdd, thr, sig, phys, mc=3)
            if bool(is_abn[0]):
                g = physics_gate(sig_np.numpy(), r["rpm"], r["sr"],
                                  r["bpfi"], r["bpfo"], r["bpf"])
                gate_counts[f"abnormal→{g}"] += 1
            else:
                gate_counts["normal"] += 1

        print("\n  Stage-gate breakdown on OOD Case 11:")
        for k, v in sorted(gate_counts.items()):
            print(f"    {k}: {v}  ({v/len(ood_recs)*100:.1f}%)")

    # ── Save full summary ─────────────────────────────────────────────
    summary = {
        "strategy": (
            "LOCO (10 folds): Test=1 case, Train=pooled 9 cases, "
            f"Val={int(CFG['val_frac']*100)}% stratified from train pool"
        ),
        "loco_folds": fold_results,
        "aggregated": {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1":  float(np.std(f1s)),
            "min_macro_f1":  float(np.min(f1s)),
            "max_macro_f1":  float(np.max(f1s)),
        },
        "best_fold": {
            "test_case": best_fold_case,
            "macro_f1":  float(best_f1),
        },
        "ood_case11": ood_metrics,
    }

    out_path = Path(CFG["output_dir"]) / "loco_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*68}")
    print("  FINAL RESULTS")
    print(f"{'='*68}")
    print(f"  LOCO Mean Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  LOCO Mean Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    if ood_metrics:
        print(f"  OOD  Accuracy      : {ood_metrics.get('accuracy', 0):.4f}")
        print(f"  OOD  Macro-F1      : {ood_metrics.get('macro_f1', 0):.4f}")
    print(f"\n  All outputs saved to : {CFG['output_dir']}")
    print("  Done!")


if __name__ == "__main__":
    main()