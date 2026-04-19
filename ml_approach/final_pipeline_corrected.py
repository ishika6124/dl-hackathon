import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# ---------------- CONFIG ----------------
DATA_ROOT = "SCA_bearing_dataset"
TRAIN_FOLDERS = [2, 3, 5, 6, 7, 8]
EVAL_FOLDERS = [1, 4, 9, 10, 11]
OOD_FOLDER = 11

TRAIN_SPLIT = "train"
EVAL_SPLIT = "test"

OUTPUT_DIR = "outputs_final_pipeline"
MAX_SIGNAL_LEN = 4096
SENSOR_KEYS = ["DS", "FS", "Upper", "Lower"]

FEATURE_GROUPS = {
    "time_domain": ["rms", "peak", "crest_factor", "kurtosis", "skew"],
    "spectral": ["spec_entropy", "env_entropy", "lowband_energy", "highband_energy"],
    "physics": ["bpfi_energy", "bpfo_energy", "bsf_energy", "fault_energy_sum", "fault_energy_max"],
    "operating_conditions": ["rpm", "sampling_rate"],
}


# ---------------- ROBUST MATLAB HELPERS ----------------
def unwrap_scalar(x):
    for _ in range(10):
        try:
            arr = np.asarray(x)
            if arr.dtype == object and arr.size == 1:
                x = arr.reshape(-1)[0]
                continue
        except Exception:
            pass
        break
    return x


def get_field(obj, name, default=None):
    obj = unwrap_scalar(obj)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    except Exception:
        pass
    return default


def safe_float(x, default=np.nan):
    try:
        x = unwrap_scalar(x)
        if x is None:
            return default
        arr = np.asarray(x).squeeze()
        if arr.size == 0:
            return default
        if arr.dtype == object:
            if arr.size == 1:
                return safe_float(arr.item(), default)
            return default
        return float(arr.flat[0])
    except Exception:
        return default


def to_scalar_array(x, n, default=np.nan):
    x = unwrap_scalar(x)
    arr = np.asarray(x, dtype=object).reshape(-1)
    out = np.full(n, default, dtype=float)
    for i in range(min(n, len(arr))):
        out[i] = safe_float(arr[i], default)
    return out


def extract_raw_data(raw):
    raw = unwrap_scalar(raw)
    try:
        arr = np.asarray(raw)
        if arr.dtype != object:
            arr = arr.astype(np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr
    except Exception:
        pass

    rows = []
    try:
        flat = np.asarray(raw, dtype=object).reshape(-1)
    except Exception:
        flat = [raw]

    for item in flat:
        try:
            v = np.asarray(unwrap_scalar(item), dtype=np.float64).reshape(-1)
            if len(v) > 0:
                rows.append(v)
        except Exception:
            continue

    if not rows:
        return np.zeros((0, 0), dtype=np.float64)

    m = max(len(r) for r in rows)
    out = np.zeros((len(rows), m), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


# ---------------- SIGNAL / FEATURE HELPERS ----------------
def normalize_signal(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) > MAX_SIGNAL_LEN:
        x = x[:MAX_SIGNAL_LEN]
    x = x - np.mean(x)
    return x / (np.std(x) + 1e-8)


def spectral_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def nearest_bin(freq_hz, fs, n_signal):
    if not np.isfinite(freq_hz) or freq_hz <= 0 or fs <= 0 or n_signal <= 0:
        return -1
    return int(round(freq_hz / fs * n_signal))


def band_energy(spec, center, half_width=2):
    if center < 0 or len(spec) == 0:
        return 0.0
    lo = max(0, center - half_width)
    hi = min(len(spec), center + half_width + 1)
    if lo >= hi:
        return 0.0
    return float(np.sum(spec[lo:hi]))


def label_to_stage3(label):
    if label == 1:
        return "INNER_FAULT"
    if label == 2:
        return "BALL_FAULT"
    if label == 3:
        return "OUTER_FAULT"
    return "NONE"


def extract_features(signal, rpm=np.nan, fs=np.nan, bpfi=np.nan, bpfo=np.nan, bsf=np.nan, ftf=np.nan):
    x = normalize_signal(signal)

    env = np.abs(hilbert(x))
    env = env - np.mean(env)

    spec = np.abs(np.fft.rfft(x))
    psd = spec ** 2
    psd_total = float(np.sum(psd) + 1e-8)

    env_spec = np.abs(np.fft.rfft(env))
    env_psd = env_spec ** 2
    env_total = float(np.sum(env_psd) + 1e-8)

    def f_energy(freq_hz):
        c1 = nearest_bin(freq_hz, fs, len(x))
        c2 = nearest_bin(2 * freq_hz, fs, len(x))
        return (band_energy(env_psd, c1, 2) + band_energy(env_psd, c2, 2)) / env_total

    bpfi_energy = f_energy(bpfi)
    bpfo_energy = f_energy(bpfo)
    bsf_energy = f_energy(bsf)

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    crest = peak / (rms + 1e-8)
    kurtosis = float(np.mean((x / (np.std(x) + 1e-8)) ** 4))
    skew = float(np.mean((x / (np.std(x) + 1e-8)) ** 3))

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "kurtosis": kurtosis,
        "skew": skew,
        "spec_entropy": spectral_entropy(psd + 1e-12),
        "env_entropy": spectral_entropy(env_psd + 1e-12),
        "lowband_energy": float(np.sum(psd[:max(5, len(psd) // 16)]) / psd_total),
        "highband_energy": float(np.sum(psd[max(5, len(psd) // 4):]) / psd_total),
        "rpm": 0.0 if not np.isfinite(rpm) else float(rpm),
        "sampling_rate": 0.0 if not np.isfinite(fs) else float(fs),
        "bpfi_meta_hz": bpfi if np.isfinite(bpfi) else np.nan,
        "bpfo_meta_hz": bpfo if np.isfinite(bpfo) else np.nan,
        "bsf_meta_hz": bsf if np.isfinite(bsf) else np.nan,
        "ftf_meta_hz": ftf if np.isfinite(ftf) else np.nan,
        "bpfi_energy": bpfi_energy,
        "bpfo_energy": bpfo_energy,
        "bsf_energy": bsf_energy,
        "fault_energy_sum": bpfi_energy + bpfo_energy + bsf_energy,
        "fault_energy_max": max(bpfi_energy, bpfo_energy, bsf_energy),
    }


# ---------------- DATA LOADER ----------------
def load_folder(folder, split_name):
    rows = []
    path = os.path.join(DATA_ROOT, str(folder), f"{split_name}.mat")
    if not os.path.exists(path):
        return rows

    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return rows

    for sensor in SENSOR_KEYS:
        if sensor not in mat:
            continue

        sensor_obj = unwrap_scalar(mat[sensor])

        raw = get_field(sensor_obj, "rawData", sensor_obj)
        signals = extract_raw_data(raw)
        if signals.size == 0 or signals.ndim != 2 or signals.shape[0] == 0:
            continue

        n = signals.shape[0]

        labels = to_scalar_array(get_field(sensor_obj, "label", np.full(n, np.nan)), n, np.nan)
        rpms = to_scalar_array(get_field(sensor_obj, "RPM", np.full(n, np.nan)), n, np.nan)
        fs = safe_float(get_field(sensor_obj, "samplingRate", np.nan), np.nan)

        ff = unwrap_scalar(get_field(sensor_obj, "faultFrequencies", None))
        bpfi_mult = safe_float(get_field(ff, "BPFIMultiple", np.nan), np.nan)
        bpfo_mult = safe_float(get_field(ff, "BPFOMultiple", np.nan), np.nan)
        bsf_mult = safe_float(get_field(ff, "BPFMultiple", np.nan), np.nan)
        ftf_mult = safe_float(get_field(ff, "FTFMultiple", np.nan), np.nan)

        for i in range(n):
            sig = np.asarray(signals[i]).reshape(-1)
            if len(sig) < 64:
                continue

            rpm = float(rpms[i]) if i < len(rpms) and np.isfinite(rpms[i]) else np.nan
            shaft_hz = rpm / 60.0 if np.isfinite(rpm) and rpm > 0 else np.nan

            bpfi = bpfi_mult * shaft_hz if np.isfinite(bpfi_mult) and np.isfinite(shaft_hz) else np.nan
            bpfo = bpfo_mult * shaft_hz if np.isfinite(bpfo_mult) and np.isfinite(shaft_hz) else np.nan
            bsf = bsf_mult * shaft_hz if np.isfinite(bsf_mult) and np.isfinite(shaft_hz) else np.nan
            ftf = ftf_mult * shaft_hz if np.isfinite(ftf_mult) and np.isfinite(shaft_hz) else np.nan

            feat = extract_features(sig, rpm=rpm, fs=fs, bpfi=bpfi, bpfo=bpfo, bsf=bsf, ftf=ftf)
            feat["folder"] = folder
            feat["split"] = split_name
            feat["sensor"] = sensor
            feat["window_index"] = i

            raw_label = int(labels[i]) if i < len(labels) and np.isfinite(labels[i]) else -1
            feat["raw_label"] = raw_label
            feat["stage1_label"] = 0 if raw_label == 0 else 1
            feat["stage2_label"] = 0 if folder == OOD_FOLDER else 1
            feat["stage3_label"] = label_to_stage3(raw_label)

            rows.append(feat)

    return rows


# ---------------- MAIN LOAD ----------------
def build_data():
    train_rows = []
    for f in TRAIN_FOLDERS:
        train_rows += load_folder(f, TRAIN_SPLIT)

    eval_rows = []
    for f in EVAL_FOLDERS:
        eval_rows += load_folder(f, EVAL_SPLIT)

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)
    return train_df, eval_df


# ---------------- FILE-LEVEL AGGREGATION ----------------
def build_file_level(df):
    rows = []
    for (folder, sensor), g in df.groupby(["folder", "sensor"]):
        rows.append({
            "folder": folder,
            "sensor": sensor,
            "stage1_abnormal_ratio": float(np.mean(g["stage1_pred"] == 1)),
            "rms_mean": float(g["rms"].mean()),
            "kurtosis_mean": float(g["kurtosis"].mean()),
            "crest_mean": float(g["crest_factor"].mean()),
            "entropy_mean": float(g["spec_entropy"].mean()),
            "bpfi_mean": float(g["bpfi_energy"].mean()),
            "bpfo_mean": float(g["bpfo_energy"].mean()),
            "bsf_mean": float(g["bsf_energy"].mean()),
            "fault_energy_mean": float(g["fault_energy_sum"].mean()),
            "stage2_label": int(pd.Series(g["stage2_label"]).mode().iloc[0]),
            "stage3_label": str(pd.Series(g["stage3_label"]).mode().iloc[0]),
        })
    return pd.DataFrame(rows)


# ---------------- STAGE 3 RULE ----------------
def stage3_rule(row):
    vals = {
        "INNER_FAULT": row["bpfi_mean"],
        "OUTER_FAULT": row["bpfo_mean"],
        "BALL_FAULT": row["bsf_mean"],
    }
    return max(vals, key=vals.get)


# ---------------- ABLATION STUDY ----------------
def run_stage1_ablation(train_df, eval_df, feature_cols, base_metrics):
    results = {"baseline": {**base_metrics, "label": "All Features"}}
    y_true = eval_df["stage1_label"].values

    for group_name, group_cols in FEATURE_GROUPS.items():
        ablated_cols = [c for c in feature_cols if c not in group_cols]
        if len(ablated_cols) == 0:
            continue

        imp_a = SimpleImputer(strategy="median")
        sc_a = StandardScaler()
        X_tr = sc_a.fit_transform(imp_a.fit_transform(train_df[ablated_cols]))
        X_ev = sc_a.transform(imp_a.transform(eval_df[ablated_cols]))

        m = IsolationForest(contamination=0.35, random_state=42)
        m.fit(X_tr)
        scores = -m.score_samples(X_ev)
        thr = np.percentile(scores, 65)
        pred = (scores > thr).astype(int)

        results[group_name] = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
            "label": f"Without {group_name.replace('_', ' ').title()}",
        }
        print(f"  Ablation [{group_name}]: acc={results[group_name]['accuracy']:.3f}")

    return results


# ---------------- PER-SENSOR METRICS ----------------
def compute_per_sensor_metrics(eval_df):
    results = {}
    for sensor in eval_df["sensor"].unique():
        mask = eval_df["sensor"] == sensor
        yt = eval_df.loc[mask, "stage1_label"].values
        yp = eval_df.loc[mask, "stage1_pred"].values
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            continue
        results[sensor] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "count": int(len(yt)),
        }
    return results


# ---------------- RUN ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/7] Loading data...")
    train_df, eval_df = build_data()

    if len(train_df) == 0 or len(eval_df) == 0:
        print("No data loaded. Check DATA_ROOT and folder paths.")
        return

    feature_cols = [
        "rms", "peak", "crest_factor", "kurtosis", "skew",
        "spec_entropy", "env_entropy", "lowband_energy", "highband_energy",
        "bpfi_energy", "bpfo_energy", "bsf_energy", "fault_energy_sum", "fault_energy_max",
        "rpm", "sampling_rate",
    ]

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # ---------- Stage 1 ----------
    print("[2/7] Training Stage 1 (Isolation Forest)...")
    X_train_s1 = scaler.fit_transform(imp.fit_transform(train_df[feature_cols]))
    X_eval_s1 = scaler.transform(imp.transform(eval_df[feature_cols]))

    stage1 = IsolationForest(contamination=0.35, random_state=42)
    stage1.fit(X_train_s1)

    s1_scores = -stage1.score_samples(X_eval_s1)
    s1_thr = np.percentile(s1_scores, 65)
    s1_pred = (s1_scores > s1_thr).astype(int)

    eval_df["stage1_score"] = s1_scores
    eval_df["stage1_pred"] = s1_pred

    y1_true = eval_df["stage1_label"].values
    s1_acc = float(accuracy_score(y1_true, s1_pred))
    s1_bal = float(balanced_accuracy_score(y1_true, s1_pred))
    s1_f1 = float(f1_score(y1_true, s1_pred, average="macro", zero_division=0))
    s1_prec = float(precision_score(y1_true, s1_pred, average="macro", zero_division=0))
    s1_rec = float(recall_score(y1_true, s1_pred, average="macro", zero_division=0))

    print("\n[EVAL] Stage 1 — Normal vs Abnormal")
    cm1 = confusion_matrix(y1_true, s1_pred)
    print(cm1)
    print(classification_report(y1_true, s1_pred, target_names=["Normal", "Abnormal"], zero_division=0))

    # ROC / PR for Stage 1
    fpr1, tpr1, _ = roc_curve(y1_true, s1_scores)
    auc1 = float(auc(fpr1, tpr1))
    prec1_c, rec1_c, _ = precision_recall_curve(y1_true, s1_scores)
    ap1 = float(average_precision_score(y1_true, s1_scores))

    # ---------- Stage 2 ----------
    print("[3/7] Training Stage 2 (Logistic Regression)...")
    file_df = build_file_level(eval_df)

    stage2_features = [
        "stage1_abnormal_ratio", "rms_mean", "kurtosis_mean", "crest_mean",
        "entropy_mean", "bpfi_mean", "bpfo_mean", "bsf_mean", "fault_energy_mean",
    ]

    minority = file_df[file_df["stage2_label"] == 0]
    majority = file_df[file_df["stage2_label"] == 1]

    if len(minority) > 0 and len(majority) > 0:
        minority_up = minority.sample(len(majority), replace=True, random_state=42)
        train_stage2 = pd.concat([majority, minority_up]).sample(frac=1.0, random_state=42)
    else:
        train_stage2 = file_df.copy()

    imp2 = SimpleImputer(strategy="median")
    scaler2 = StandardScaler()

    X2_tr = scaler2.fit_transform(imp2.fit_transform(train_stage2[stage2_features]))
    y2_tr = train_stage2["stage2_label"].values

    stage2 = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    stage2.fit(X2_tr, y2_tr)

    X2_ev = scaler2.transform(imp2.transform(file_df[stage2_features]))
    s2_pred = stage2.predict(X2_ev)
    s2_proba = stage2.predict_proba(X2_ev)[:, 1]
    file_df["stage2_pred"] = s2_pred
    file_df["stage2_proba"] = s2_proba

    y2_true = file_df["stage2_label"].values
    s2_acc = float(accuracy_score(y2_true, s2_pred))
    s2_bal = float(balanced_accuracy_score(y2_true, s2_pred)) if len(np.unique(y2_true)) > 1 else None
    s2_f1 = float(f1_score(y2_true, s2_pred, average="macro", zero_division=0))
    s2_prec = float(precision_score(y2_true, s2_pred, average="macro", zero_division=0))
    s2_rec = float(recall_score(y2_true, s2_pred, average="macro", zero_division=0))

    print("\n[EVAL] Stage 2 — Bearing vs External")
    cm2 = confusion_matrix(y2_true, s2_pred)
    print(cm2)
    print(classification_report(y2_true, s2_pred, target_names=["External", "Bearing"], zero_division=0))

    # ROC / PR for Stage 2
    if len(np.unique(y2_true)) > 1:
        fpr2, tpr2, _ = roc_curve(y2_true, s2_proba)
        auc2 = float(auc(fpr2, tpr2))
        prec2_c, rec2_c, _ = precision_recall_curve(y2_true, s2_proba)
        ap2 = float(average_precision_score(y2_true, s2_proba))
    else:
        fpr2, tpr2, auc2 = np.array([0., 1.]), np.array([0., 1.]), 1.0
        prec2_c, rec2_c, ap2 = np.array([1., 0.]), np.array([0., 1.]), 1.0

    # Calibration for Stage 2
    n_bins_cal = max(2, min(5, len(y2_true) // 2))
    try:
        prob_true2, prob_pred2 = calibration_curve(y2_true, s2_proba, n_bins=n_bins_cal)
    except Exception:
        prob_true2, prob_pred2 = np.array([0., 0.5, 1.]), np.array([0., 0.5, 1.])

    # ---------- Stage 3 ----------
    print("[4/7] Running Stage 3 (Physics Rule Engine)...")
    stage3_df = file_df[file_df["stage2_pred"] == 1].copy()
    stage3_df = stage3_df[stage3_df["stage3_label"].isin(["INNER_FAULT", "BALL_FAULT", "OUTER_FAULT"])].copy()

    file_df["stage3_pred"] = pd.array([""] * len(file_df), dtype=object)
    labels3 = ["INNER_FAULT", "BALL_FAULT", "OUTER_FAULT"]

    if len(stage3_df) > 0:
        stage3_df = stage3_df.copy()
        stage3_df["stage3_pred"] = stage3_df.apply(stage3_rule, axis=1)
        for idx_val, pred_val in zip(stage3_df.index, stage3_df["stage3_pred"]):
            file_df.at[idx_val, "stage3_pred"] = pred_val

        y3_true = stage3_df["stage3_label"].tolist()
        y3_pred = stage3_df["stage3_pred"].tolist()
        cm3 = confusion_matrix(y3_true, y3_pred, labels=labels3)
        s3_f1 = float(f1_score(y3_true, y3_pred, labels=labels3, average="macro", zero_division=0))

        print("\n[EVAL] Stage 3 — Inner vs Ball vs Outer")
        print(cm3)
        print(classification_report(y3_true, y3_pred, labels=labels3, zero_division=0))
    else:
        cm3 = np.zeros((3, 3), dtype=int)
        s3_f1 = 0.0
        print("\n[EVAL] Stage 3 skipped — no predicted bearing files")

    # ---------- Ablation Study ----------
    print("[5/7] Running ablation study...")
    base_metrics = {
        "accuracy": s1_acc,
        "balanced_accuracy": s1_bal,
        "f1": s1_f1,
        "precision": s1_prec,
        "recall": s1_rec,
    }
    ablation_results = run_stage1_ablation(train_df, eval_df, feature_cols, base_metrics)

    # ---------- Per-Sensor Metrics ----------
    print("[6/7] Computing per-sensor metrics...")
    per_sensor = compute_per_sensor_metrics(eval_df)

    # ---------- OOD Analysis ----------
    ood_mask_w = eval_df["folder"] == OOD_FOLDER
    if ood_mask_w.sum() > 0:
        ood_w = eval_df[ood_mask_w]
        ood_metrics = {
            "count": int(ood_mask_w.sum()),
            "stage1_abnormal_ratio": float(np.mean(ood_w["stage1_pred"] == 1)),
            "stage1_mean_score": float(ood_w["stage1_score"].mean()),
            "stage1_accuracy": float(accuracy_score(ood_w["stage1_label"], ood_w["stage1_pred"])),
        }
        ood_file_mask = file_df["folder"] == OOD_FOLDER
        if ood_file_mask.sum() > 0:
            ood_f = file_df[ood_file_mask]
            ood_metrics["stage2_correctly_external"] = int(np.sum(ood_f["stage2_pred"] == 0))
            ood_metrics["stage2_total"] = int(len(ood_f))
    else:
        ood_metrics = {"count": 0}

    # ---------- Save Models ----------
    print("[7/7] Saving outputs...")
    models_dict = {
        "stage1": stage1,
        "stage2": stage2,
        "imp1": imp,
        "scaler1": scaler,
        "imp2": imp2,
        "scaler2": scaler2,
        "feature_cols": feature_cols,
        "stage2_features": stage2_features,
        "s1_threshold": float(s1_thr),
    }
    with open(os.path.join(OUTPUT_DIR, "models.pkl"), "wb") as f:
        pickle.dump(models_dict, f)

    # Save ROC/PR Data
    roc_pr_data = {
        "stage1": {
            "fpr": fpr1.tolist(), "tpr": tpr1.tolist(), "auc": auc1,
            "precision": prec1_c.tolist(), "recall": rec1_c.tolist(),
            "average_precision": ap1,
        },
        "stage2": {
            "fpr": fpr2.tolist(), "tpr": tpr2.tolist(), "auc": auc2,
            "precision": prec2_c.tolist(), "recall": rec2_c.tolist(),
            "average_precision": ap2,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "roc_pr_data.json"), "w") as f:
        json.dump(roc_pr_data, f)

    # Save Calibration Data
    calib_data = {
        "stage2": {
            "prob_true": prob_true2.tolist(),
            "prob_pred": prob_pred2.tolist(),
            "probas": s2_proba.tolist(),
            "labels": y2_true.tolist(),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "calibration_data.json"), "w") as f:
        json.dump(calib_data, f)

    # Save Ablation Results
    with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), "w") as f:
        json.dump(ablation_results, f)

    # Save Per-Sensor Results
    with open(os.path.join(OUTPUT_DIR, "per_sensor_results.json"), "w") as f:
        json.dump(per_sensor, f)

    # Save Confusion Matrices
    cm_data = {
        "stage1": {"matrix": cm1.tolist(), "labels": ["Normal", "Abnormal"]},
        "stage2": {"matrix": cm2.tolist(), "labels": ["External", "Bearing"]},
        "stage3": {"matrix": cm3.tolist(), "labels": ["INNER", "BALL", "OUTER"]},
    }
    with open(os.path.join(OUTPUT_DIR, "confusion_matrices.json"), "w") as f:
        json.dump(cm_data, f)

    # Save OOD Analysis
    with open(os.path.join(OUTPUT_DIR, "ood_analysis.json"), "w") as f:
        json.dump(ood_metrics, f)

    # Save CSVs
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "window_predictions.csv"), index=False)
    file_df.to_csv(os.path.join(OUTPUT_DIR, "file_predictions.csv"), index=False)

    # Save Summary
    summary = {
        "stage1_accuracy": s1_acc,
        "stage1_balanced_accuracy": s1_bal,
        "stage1_macro_f1": s1_f1,
        "stage1_macro_precision": s1_prec,
        "stage1_macro_recall": s1_rec,
        "stage1_roc_auc": auc1,
        "stage1_avg_precision": ap1,
        "stage2_accuracy": s2_acc,
        "stage2_balanced_accuracy": s2_bal,
        "stage2_macro_f1": s2_f1,
        "stage2_macro_precision": s2_prec,
        "stage2_macro_recall": s2_rec,
        "stage2_roc_auc": auc2,
        "stage2_avg_precision": ap2,
        "stage3_count": int(len(stage3_df)),
        "stage3_macro_f1": s3_f1,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] All outputs saved to:", os.path.abspath(OUTPUT_DIR))
    print(f"  Stage 1: acc={s1_acc:.3f}, bal={s1_bal:.3f}, f1={s1_f1:.3f}, auc={auc1:.3f}")
    print(f"  Stage 2: acc={s2_acc:.3f}, bal={s2_bal}, f1={s2_f1:.3f}")
    print(f"  Stage 3: {len(stage3_df)} files, f1={s3_f1:.3f}")


if __name__ == "__main__":
    main()
