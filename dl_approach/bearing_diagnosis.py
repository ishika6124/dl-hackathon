"""
Cascaded Multimodal Bearing Fault Diagnosis — Fixed Version
=============================================================
FIX: parse_sensor_struct now uses dict-style access (s['rawData'] etc.)
     instead of attribute access (s.rawData) which fails with scipy loadmat.
     Added robust HDF5 (v7.3 .mat) fallback loader identical to working script.

Dataset structure:
  Each case folder has train.mat + test.mat
  Sensor keys: DS, FS, Upper, Lower
  Labels: -1=off/missing  0=Normal  1=IRF  2=Ball  3=ORF

Architecture:
  Phase 0 : NT-Xent contrastive pretraining (Cases 1-6)
  Stage 1 : Normal vs Abnormal — TCN + Deep SVDD + MC-Dropout
  Stage 2 : Bearing vs Non-bearing — Pure Physics Gate
  Stage 3 : Ball/IRF/ORF — Multimodal Fusion + Compact Transformer

Train: Cases 1-6  |  Val: Case 7  |  Test: Cases 8-11
"""

import os, warnings, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.io as sio
import scipy.signal as sp_signal
from scipy.signal import hilbert

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
    output_dir     = "/home/teaching/hackathon/dl_approach/outputs_cascaded",
    device         = "cuda" if torch.cuda.is_available() else "cpu",

    target_len     = 4096,
    enc_dim        = 128,
    phys_feat_dim  = 16,

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

    train_cases    = [1,2,3,4,5,6],
    val_cases      = [7],
    test_cases     = [8,9,10,11],
    label_names    = {0:"Normal",1:"Inner Ring",2:"Ball",3:"Outer Ring"},
    seed           = 42,
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(CFG["output_dir"], exist_ok=True)
DEVICE = torch.device(CFG["device"])
L      = CFG["target_len"]
print(f"Device: {DEVICE}")

SENSOR_KEYS = ["DS", "FS", "Upper", "Lower"]


# ══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (FIXED)
# ══════════════════════════════════════════════════════════════════════

def _scalar(v):
    return float(np.array(v).flat[0])


def _norm_ff(ff):
    """Normalize faultFrequencies → dict with 4 float keys."""
    keys = ['FTFMultiple', 'BPFMultiple', 'BPFOMultiple', 'BPFIMultiple']
    # dict (from h5py or simplify_cells)
    if isinstance(ff, dict):
        return {k: float(np.array(ff[k]).flat[0]) for k in keys}
    # scipy structured array element (1,1)
    try:
        ff_inner = ff[0, 0]
        return {k: float(ff_inner[k].flat[0]) for k in keys}
    except Exception:
        pass
    # flat array fallback [bpfi, bpfo, bpf, ftf]
    try:
        arr = np.array(ff, dtype=np.float32).ravel()
        if len(arr) >= 4:
            return {keys[i]: float(arr[i]) for i in range(4)}
    except Exception:
        pass
    return {k: 1.0 for k in keys}


def _get_sensor_dict(s):
    """
    Convert sensor struct → plain dict with keys:
    rawData, label, RPM, samplingRate, faultFrequencies
    Works for:
      - scipy structured array element  (s['rawData'] etc.)
      - plain dict from simplify_cells or h5py
    """
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj[key]
        # numpy void / structured array element
        return obj[key]

    try:
        rd   = _get(s, 'rawData')
        lbl  = np.array(_get(s, 'label'),       dtype=np.int64).ravel()
        rpm  = np.array(_get(s, 'RPM'),          dtype=np.float32).ravel()
        sr   = float(np.array(_get(s, 'samplingRate')).ravel()[0])
        ff   = _norm_ff(_get(s, 'faultFrequencies'))
        return dict(rawData=rd, label=lbl, RPM=rpm,
                    samplingRate=sr, faultFrequencies=ff)
    except Exception as e:
        return None


def load_mat_v72(path):
    """Load v7.2 .mat via scipy."""
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
    """Load v7.3 HDF5 .mat via h5py."""
    try:
        import h5py
    except ImportError:
        return None

    def _read(item, f):
        if isinstance(item, h5py.Dataset):
            arr = item[()]
            if arr.dtype == object:
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
            # HDF5 stores arrays column-major → transpose
            if isinstance(rd, np.ndarray) and rd.dtype != object:
                rd = rd.T
            try:
                sd = dict(
                    rawData      = rd,
                    label        = np.array(s['label']).flatten().astype(np.int64),
                    RPM          = np.array(s['RPM']).flatten().astype(np.float32),
                    samplingRate = float(np.array(s['samplingRate']).ravel()[0]),
                    faultFrequencies = _norm_ff(s['faultFrequencies']),
                )
                out[sk] = sd
            except Exception:
                pass
        return out if out else None
    except Exception:
        return None


def load_mat_any(path):
    """Try v7.2 first, fall back to v7.3 HDF5."""
    result = load_mat_v72(path)
    if result:
        return result
    result = load_mat_v73(path)
    if result:
        return result
    print(f"  [WARN] Could not load: {path}")
    return {}


def parse_sensor_dict(sd, folder_id, split_name):
    """
    sd : plain dict with keys rawData, label, RPM, samplingRate, faultFrequencies
    Returns list of record dicts.
    """
    raw_data = sd['rawData']
    labels   = sd['label'].ravel()
    rpm_vals = sd['RPM'].ravel()
    sr       = float(sd['samplingRate'])
    ff       = sd['faultFrequencies']

    bpfi = ff.get('BPFIMultiple', 1.0)
    bpfo = ff.get('BPFOMultiple', 1.0)
    bpf  = ff.get('BPFMultiple',  1.0)
    ftf  = ff.get('FTFMultiple',  1.0)

    # Handle ragged cell arrays (folder 9 test.mat)
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
        ))
    return records


def load_case(case_num, base):
    folder  = Path(base) / str(case_num)
    records = []
    for fname in ["train.mat", "test.mat"]:
        path = folder / fname
        if not path.exists():
            print(f"  [WARN] missing: {path}")
            continue
        sensor_dicts = load_mat_any(path)
        for sk, sd in sensor_dicts.items():
            try:
                recs = parse_sensor_dict(sd, case_num, fname.replace('.mat',''))
                for r in recs:
                    r['case'] = case_num
                records.extend(recs)
                print(f"  Case {case_num:02d}/{fname}/{sk}: {len(recs)} samples")
            except Exception as e:
                print(f"  Case {case_num:02d}/{fname}/{sk}: SKIPPED – {e}")
    return records


def load_cases(case_list, base):
    all_recs = []
    for c in case_list:
        recs = load_case(c, base)
        dist = defaultdict(int)
        for r in recs:
            dist[r["label"]] += 1
        print(f"  → Case {c:2d} total: {len(recs)} | {dict(dist)}")
        all_recs.extend(recs)
    return all_recs


# ══════════════════════════════════════════════════════════════════════
# 2.  SIGNAL PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def resample_fixed(x, tgt=L):
    if len(x) == tgt:
        return x.copy().astype(np.float32)
    return np.interp(
        np.linspace(0, len(x)-1, tgt),
        np.arange(len(x)), x
    ).astype(np.float32)


def normalize(x):
    std = x.std()
    return (x - x.mean()) / (std + 1e-9)


def physics_features(x, rpm, sr, bpfi, bpfo, bpf):
    """16-dim physics feature vector."""
    feats = []
    rms   = float(np.sqrt(np.mean(x**2)) + 1e-9)

    feats.append(rms)
    feats.append(float(np.max(np.abs(x))) / rms)            # crest factor
    feats.append(float(np.mean(x**4)) / (rms**4 + 1e-9))   # kurtosis

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
        sum(be(rpmhz*h) for h in [1, 2, 3]) / tot
        if rpmhz > 0.5 else 0.0
    )

    mag  = np.abs(np.fft.rfft(x))
    fq   = np.fft.rfftfreq(len(x), 1.0/sr)
    maxf = fq.max() + 1e-9
    for lo, hi in [(.0,.25),(.25,.5),(.5,.75),(.75,1.)]:
        m = (fq >= lo*maxf) & (fq < hi*maxf)
        feats.append(float(mag[m].sum()) / (mag.sum() + 1e-9))

    s = bpfi + bpfo + bpf + 1e-9
    feats += [bpfi/s, bpfo/s, bpf/s]

    arr = np.clip(np.array(feats, dtype=np.float32), -10, 10)
    # pad/truncate to exactly phys_feat_dim
    pd = CFG["phys_feat_dim"]
    if len(arr) < pd:
        arr = np.concatenate([arr, np.zeros(pd - len(arr), dtype=np.float32)])
    return arr[:pd]


def preprocess(r):
    x    = normalize(resample_fixed(r["signal"]))
    p    = physics_features(x, r["rpm"], r["sr"], r["bpfi"], r["bpfo"], r["bpf"])
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            r["label"])


# ══════════════════════════════════════════════════════════════════════
# 3.  DATASETS
# ══════════════════════════════════════════════════════════════════════

class BearingDS(Dataset):
    def __init__(self, records):
        self.data = []
        for r in records:
            sig, phys, lbl = preprocess(r)
            self.data.append((sig, phys, lbl,
                               r["bpfi"], r["bpfo"], r["bpf"]))

    def __len__(self):            return len(self.data)
    def __getitem__(self, i):     return self.data[i]


def make_loader(records, bs, shuffle=True, label_set=None):
    recs = [r for r in records if r["label"] in label_set] \
           if label_set else records
    if not recs:
        return None
    return DataLoader(BearingDS(recs), batch_size=bs,
                      shuffle=shuffle, drop_last=False, num_workers=0)


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
# 4.  MODEL COMPONENTS
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
# 5.  LOSS FUNCTIONS
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
# 6.  PHASE 0 — CONTRASTIVE PRETRAINING
# ══════════════════════════════════════════════════════════════════════

def pretrain_encoder(train_recs, val_recs):
    print("\n" + "="*60)
    print("PHASE 0 — Contrastive Pretraining  (Cases 1-6)")
    print("="*60)

    if not train_recs:
        print("  [ERROR] No training records. Check dataset path.")
        raise RuntimeError("Empty training set in Phase 0.")

    encoder = SharedEncoder(CFG["enc_dim"], CFG["phys_feat_dim"]).to(DEVICE)
    opt     = torch.optim.Adam(encoder.parameters(), lr=CFG["pre_lr"])
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=CFG["pre_epochs"])

    tr = DataLoader(BearingDS(train_recs), batch_size=CFG["pre_batch"],
                    shuffle=True,  drop_last=True,  num_workers=0)
    vl = DataLoader(BearingDS(val_recs),   batch_size=CFG["pre_batch"],
                    shuffle=False, drop_last=False, num_workers=0) \
         if val_recs else None

    best_loss, best_w, wait = 1e9, None, 0
    for ep in range(1, CFG["pre_epochs"] + 1):
        encoder.train()
        tl = 0.
        for sig, phys, lbl, *_ in tr:
            sig, phys, lbl = sig.to(DEVICE), phys.to(DEVICE), lbl.to(DEVICE)
            z    = encoder(sig, phys)
            loss = nt_xent(z, lbl, CFG["pre_temp"])
            opt.zero_grad(); loss.backward(); opt.step()
            tl  += loss.item()
        sched.step()
        tl /= max(len(tr), 1)

        vl_l = 0.
        if vl:
            encoder.eval()
            with torch.no_grad():
                for sig, phys, lbl, *_ in vl:
                    sig, phys, lbl = sig.to(DEVICE), phys.to(DEVICE), lbl.to(DEVICE)
                    vl_l += nt_xent(encoder(sig, phys), lbl, CFG["pre_temp"]).item()
            vl_l /= max(len(vl), 1)

        if ep % 10 == 0:
            print(f"  Ep {ep:3d}  tr={tl:.4f}  vl={vl_l:.4f}")

        monitor = vl_l if vl else tl
        if monitor < best_loss:
            best_loss = monitor
            best_w    = {k: v.clone() for k, v in encoder.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= CFG["pre_patience"]:
                print(f"  Early stop ep {ep}")
                break

    encoder.load_state_dict(best_w)
    print(f"  Best contrastive loss: {best_loss:.4f}")
    return encoder


# ══════════════════════════════════════════════════════════════════════
# 7.  STAGE 1 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════

def train_stage1(encoder, train_recs, val_recs):
    print("\n" + "="*60)
    print("STAGE 1 — Normal vs Abnormal  (TCN Forecaster + SVDD)")
    print("="*60)

    for p in encoder.parameters():
        p.requires_grad = False

    fc   = Forecaster(CFG["enc_dim"]).to(DEVICE)
    svdd = SVDDHead(CFG["enc_dim"]).to(DEVICE)
    half = L // 2

    norm_tr = [r for r in train_recs if r["label"] == 0]
    norm_vl = [r for r in val_recs   if r["label"] == 0]

    if not norm_tr:
        print("  [WARN] No Normal training records for Stage 1.")
        return fc, svdd, 1.0

    tr_ldr = DataLoader(BearingDS(norm_tr), batch_size=CFG["s1_batch"],
                        shuffle=True, drop_last=False, num_workers=0)
    svdd.init_center(encoder, tr_ldr)

    opt   = torch.optim.Adam(fc.parameters(), lr=CFG["s1_lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=CFG["s1_epochs"])

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

        if ep % 20 == 0:
            print(f"  Ep {ep:3d}  fc_loss={tl:.5f}")
        if tl < best_loss:
            best_loss = tl
            best_fc   = {k: v.clone() for k, v in fc.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= CFG["s1_patience"]:
                print(f"  Early stop ep {ep}"); break

    fc.load_state_dict(best_fc)

    # Calibrate threshold on normal-val (95th percentile)
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
    print(f"  Anomaly threshold (95th pct normal val): {thr:.5f}")
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

    scores = np.stack(scores)
    mu     = scores.mean(0)
    std    = scores.std(0)
    return mu > thr, mu, std


# ══════════════════════════════════════════════════════════════════════
# 8.  STAGE 2 — PHYSICS GATE
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

    R = sum(be(fc*h) for fc in [bpfi, bpfo, bpf] for h in [1, 2, 3]) / tot
    rpmhz = rpm / 60.0
    S = sum(be(rpmhz*h) for h in [1, 2, 3]) / tot if rpmhz > 0.5 else 0.0

    if R > CFG["bearing_thr"]:  return "bearing"
    if S > CFG["shaft_thr"]:    return "non_bearing"
    return "ambiguous"


# ══════════════════════════════════════════════════════════════════════
# 9.  STAGE 3 — FAULT TYPE
# ══════════════════════════════════════════════════════════════════════

def train_stage3(pretrained_enc, train_recs, val_recs):
    print("\n" + "="*60)
    print("STAGE 3 — Fault Type  (Ball / IRF / ORF)")
    print("="*60)

    fault_set = {1, 2, 3}
    f_tr = [r for r in train_recs if r["label"] in fault_set]
    f_vl = [r for r in val_recs   if r["label"] in fault_set]

    dist = defaultdict(int)
    for r in f_tr: dist[r["label"]] += 1
    print(f"  Fault train dist: {dict(dist)}")

    if not f_tr:
        print("  [WARN] No fault training data."); return None

    def remap(recs):
        out = []
        for r in recs:
            rc = dict(r); rc["label"] = r["label"] - 1; out.append(rc)
        return out

    f_tr_r = remap(f_tr)
    f_vl_r = remap(f_vl)

    model = Stage3Model(CFG["enc_dim"], CFG["phys_feat_dim"]).to(DEVICE)
    model.encoder.load_state_dict(pretrained_enc.state_dict())

    cw = class_weights(f_tr_r, [0, 1, 2]).to(DEVICE)
    print(f"  Class weights: {cw.cpu().numpy().round(3)}")

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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=CFG["s3_epochs"])

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
                print(f"  Ep {ep:3d}  val_macro_f1={vf1:.4f}")
            if vf1 > best_f1:
                best_f1 = vf1
                best_w  = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= CFG["s3_patience"]:
                    print(f"  Early stop ep {ep}"); break
        elif ep % 20 == 0:
            print(f"  Ep {ep:3d}  (no val fault data)")

    if best_w:
        model.load_state_dict(best_w)
    print(f"  Best val fault macro-F1: {best_f1:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════
# 10.  CASCADE INFERENCE + EVALUATION
# ══════════════════════════════════════════════════════════════════════

LN4 = ["Normal", "Inner Ring", "Ball", "Outer Ring"]


def cascade_predict(r, encoder, fc, svdd, thr, s3_model):
    sig_np, phys_np, true_lbl = preprocess(r)
    sig  = sig_np.unsqueeze(0).to(DEVICE)
    phys = phys_np.unsqueeze(0).to(DEVICE)

    # Stage 1
    is_abn, _, _ = score_stage1(encoder, fc, svdd, thr, sig, phys,
                                  mc=CFG["s1_mc"])
    if not bool(is_abn[0]):
        return 0, true_lbl

    # Stage 2
    gate = physics_gate(sig_np.numpy(), r["rpm"], r["sr"],
                         r["bpfi"], r["bpfo"], r["bpf"])
    if gate == "non_bearing":
        return -1, true_lbl

    # Stage 3
    if s3_model is None:
        return 1, true_lbl
    s3_model.eval()
    with torch.no_grad():
        pred_cls = s3_model(sig, phys).argmax(-1).item()
    return pred_cls + 1, true_lbl


def evaluate(records, encoder, fc, svdd, thr, s3_model, tag=""):
    print(f"\n{'─'*60}")
    print(f"  EVALUATION: {tag}  ({len(records)} samples)")
    print(f"{'─'*60}")
    if not records:
        print("  No records."); return {}

    preds, trues = [], []
    for r in records:
        pred, true = cascade_predict(r, encoder, fc, svdd, thr, s3_model)
        pred = max(pred, 0)
        preds.append(pred)
        trues.append(true)

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    print(f"  Accuracy: {acc:.4f}   Macro-F1: {f1:.4f}")
    print(classification_report(trues, preds, target_names=LN4,
                                  labels=[0,1,2,3], zero_division=0))
    cm  = confusion_matrix(trues, preds, labels=[0,1,2,3])
    hdr = f"{'':15s}" + "".join(f"{n:>13s}" for n in LN4)
    print(hdr)
    for i, row in enumerate(cm):
        print(f"{LN4[i]:15s}" + "".join(f"{v:13d}" for v in row))
    return {"accuracy": acc, "macro_f1": f1}


# ══════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    base = CFG["dataset_path"]

    print("\nLoading Cases 1-6 (train) ...")
    train_recs = load_cases(CFG["train_cases"], base)
    print("\nLoading Case 7 (val) ...")
    val_recs   = load_cases(CFG["val_cases"], base)

    for tag, recs in [("Train", train_recs), ("Val", val_recs)]:
        dist = defaultdict(int)
        for r in recs: dist[r["label"]] += 1
        print(f"  {tag}: {len(recs)} samples | {dict(dist)}")

    if not train_recs:
        print("\n[FATAL] Training set is empty.")
        print(f"  Dataset path checked: {base}")
        print("  Please verify the path contains folders 1-11 with train.mat/test.mat")
        return

    # Phase 0
    encoder = pretrain_encoder(train_recs, val_recs)

    # Stage 1
    fc, svdd, thr = train_stage1(encoder, train_recs, val_recs)

    # Stage 3
    s3 = train_stage3(encoder, train_recs, val_recs)

    # Validation
    val_metrics = evaluate(val_recs, encoder, fc, svdd, thr, s3,
                            tag="Validation (Case 7)")

    # Per-case test
    print("\n\n" + "="*60)
    print("PER-CASE TEST RESULTS")
    print("="*60)
    all_test, per_case = [], {}
    for c in CFG["test_cases"]:
        recs = load_case(c, base)
        recs = [r for r in recs if r["label"] != -1]
        dist = defaultdict(int)
        for r in recs: dist[r["label"]] += 1
        print(f"\nCase {c}: {len(recs)} samples | {dict(dist)}")
        m = evaluate(recs, encoder, fc, svdd, thr, s3, tag=f"Case {c}")
        per_case[c] = m
        all_test.extend(recs)

    test_metrics = evaluate(all_test, encoder, fc, svdd, thr, s3,
                             tag="OVERALL TEST (Cases 8-11)")

    # Save
    out = Path(CFG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "val":      val_metrics,
        "test":     test_metrics,
        "per_case": {str(k): v for k, v in per_case.items()},
    }
    (out / "metrics_cascade.json").write_text(json.dumps(summary, indent=2))
    torch.save(encoder.state_dict(), out / "encoder.pt")
    torch.save(fc.state_dict(),      out / "forecaster.pt")
    torch.save(svdd.state_dict(),    out / "svdd.pt")
    if s3:
        torch.save(s3.state_dict(), out / "stage3.pt")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Val  Macro-F1 : {val_metrics.get('macro_f1', 0):.4f}")
    print(f"  Test Macro-F1 : {test_metrics.get('macro_f1', 0):.4f}")
    print(f"  Saved to: {out}")
    print("Done!")


if __name__ == "__main__":
    main()
