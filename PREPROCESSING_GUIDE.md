# Data Preprocessing Guide — SCA Bearing Dataset

## Dataset Overview

**Source:** Pulp mill vibration data (2019–2022), 11 bearing cases  
**Format:** `.mat` files (MATLAB format), 2 files per case: `train.mat` + `test.mat`  
**Sensor data:** Acceleration in m/s² — measured at Drive Side (DS), Free Side (FS), or Upper/Lower

### Label Encoding
| Label | Meaning |
|-------|---------|
| `-1` | Machine off OR shaft speed missing → **excluded** |
| `0` | Normal / Healthy |
| `1` | Inner Ring Fault (BPFI) |
| `2` | Ball Fault (BPF) |
| `3` | Outer Ring Fault (BPFO) |

### Important: Why train.mat ≠ training data
`train.mat` = healthy phase (all label 0)  
`test.mat` = fault development phase (has fault labels)

So for supervised classification, **both files must be combined** before splitting.

---

## Step 1 — Raw Feature Extraction (`preprocess.py`)

Loads all 11 folders, extracts 4 modalities from each vibration measurement.

### What it handles
- Variable sampling rates (512 Hz to 12800 Hz)
- Variable signal lengths (8192 or 16384 samples)
- MATLAB cell arrays (Folder 9 test has a ragged structure)
- Dual sensors per bearing (DS + FS processed separately → more samples)

### 4 Modalities Extracted

#### Modality A — Raw Signal `(N, 4096)`
- Every signal is **resampled to 4096 samples** using `scipy.signal.resample`
- Then **zero-mean, unit-variance normalized** per sample
- Used as input to 1D CNN

#### Modality B — Statistical Features `(N, 18)`
Time domain (10 features):
```
RMS, Peak, Mean Absolute Value, Std Dev,
Crest Factor, Shape Factor, Impulse Factor,
Kurtosis, Skewness, Peak-to-Peak
```
Frequency domain (8 features):
```
Spectral Centroid, Spectral Bandwidth,
Spectral Entropy, Spectral Kurtosis,
Band Energy Ratios × 4 (0–25%, 25–50%, 50–75%, 75–100% of Nyquist)
```

#### Modality C — Bearing Fault Frequency Amplitudes `(N, 4)`
**This is the most diagnostic feature for bearing faults.**

Process:
1. Apply Hilbert Transform → get envelope signal
2. FFT of envelope (envelope spectrum)
3. Pick amplitude at each bearing fault frequency:

```
BPFI = Ball Pass Frequency Inner Race × shaft_RPM/60
BPFO = Ball Pass Frequency Outer Race × shaft_RPM/60
BPF  = Ball Pass Frequency            × shaft_RPM/60
FTF  = Fundamental Train Frequency    × shaft_RPM/60
```

The multipliers (e.g. 0.33×, 0.21×) are unique per bearing type and stored in the dataset.

Why envelope spectrum?
- Raw FFT shows harmonics at fault freq when fault is present
- Envelope demodulates the high-frequency carrier → fault signature is clearer at low frequencies

#### Modality D — Metadata `(N, 8)`
```
RPM (normalized 0–1)
log10(Sampling Rate) / 5     (normalized)
Fixed Speed flag (0 or 1)
Asset Type one-hot × 5       (Roller, Engine, Pump, Agitator, Strainer)
```

### Output: `processed_data/`
```
X_raw_train.npy    (3345, 4096)   ← ALL normal (from train.mat)
X_raw_test.npy     (3101, 4096)   ← Normal + Faults (from test.mat)
X_stat_train.npy   (3345, 18)
X_stat_test.npy    (3101, 18)
X_env_train.npy    (3345, 4)
X_env_test.npy     (3101, 4)
X_meta_train.npy   (3345, 8)
X_meta_test.npy    (3101, 8)
y_train.npy        (3345,)        ← all zeros!
y_test.npy         (3101,)        ← 0, 1, 2, 3
folders_train.npy  (3345,)        ← folder ID per sample
folders_test.npy   (3101,)
```

---

## Step 2 — Augmentation + Final Split (`augment_data.py`)

### Class Imbalance Problem
After combining train+test and doing 80/20 split:
```
Normal     : 4245   (87%)  ← way too many
Inner Ring :  298   (6%)
Ball Fault :   18   (0.4%) ← CRITICAL: only 18 training samples!
Outer Ring :  362   (7%)
```

### Solution: 3-Layer Approach

#### Layer 1 — Signal Augmentation (raw signal level)
Applied only to fault classes, generates new synthetic signals:

| Class | Factor | Train samples after |
|-------|--------|-------------------|
| Normal | 0× | 4245 |
| Inner Ring | 3× | 298 + 894 = 1192 |
| Ball Fault | **10×** | 18 + 180 = 198 |
| Outer Ring | 2× | 362 + 724 = 1086 |

Augmentation techniques used (2 random ones per generated sample):
```
1. Gaussian Noise        → adds sensor noise (SNR ~20dB)
2. Amplitude Scaling     → ±20% scale (simulates load change)
3. Circular Time Shift   → roll signal (different start point)
4. Time Reversal         → flip signal (stationary process symmetry)
5. Random Dropout        → zero out 5% segment (sensor dropout)
```

#### Layer 2 — SMOTE on Tabular Features
After signal augmentation, applies SMOTE to `stat + env + meta` features:
- Further boosts minority classes to ≥500 samples
- Works in feature space (not raw signal)
- Uses `k_neighbors = min(5, minority_count - 1)` to be safe

#### Layer 3 — Weighted Loss (in training)
Even after augmentation, uses `CrossEntropyLoss(weight=1/class_count)`:
- Normal → weight ~1x
- Ball Fault → weight ~20x
- Forces model to not ignore rare classes

### Folder 11 — False Alarm Set (kept completely separate)
Folder 11 = shaft misalignment (NOT a bearing fault)  
All labels = 0 (normal)  
Purpose: test if model raises false alarms on non-bearing faults  
**Never used in training or validation**

### Output: `final_data/`
```
X_raw_train.npy    (N_aug, 4096)   ← augmented + SMOTE, balanced
X_stat_train.npy   (N_aug, 18)     ← scaled (StandardScaler)
X_env_train.npy    (N_aug, 4)      ← scaled
X_meta_train.npy   (N_aug, 8)      ← scaled
y_train.npy        (N_aug,)

X_raw_val.npy      (N_val, 4096)   ← NO augmentation
X_stat_val.npy     (N_val, 18)     ← scaled with train scaler
X_env_val.npy      (N_val, 4)
X_meta_val.npy     (N_val, 8)
y_val.npy          (N_val,)

X_raw_false_alarm.npy    ← folder 11 samples
X_stat_false_alarm.npy
X_env_false_alarm.npy
X_meta_false_alarm.npy
y_false_alarm.npy         ← all 0 (normal)

scaler.pkl          ← StandardScaler fitted on train, use for inference
```

---

## How to Run

```bash
# Step 1: Extract features from raw .mat files
py preprocess.py

# Step 2: Augment + create balanced final split
py augment_data.py

# Step 3: Train model (uses final_data/ by default)
py train_model.py
```

---

## For Teammates — Using the Data

Load in Python:
```python
import numpy as np, pickle

X_raw  = np.load("final_data/X_raw_train.npy")    # (N, 4096) raw signal for CNN
X_stat = np.load("final_data/X_stat_train.npy")   # (N, 18)  statistical features
X_env  = np.load("final_data/X_env_train.npy")    # (N, 4)   bearing fault freqs
X_meta = np.load("final_data/X_meta_train.npy")   # (N, 8)   metadata
y      = np.load("final_data/y_train.npy")         # (N,)     labels 0/1/2/3

# If you only want tabular features (no CNN):
X_tabular = np.concatenate([X_stat, X_env, X_meta], axis=1)  # (N, 30)

# Validation (always use this, no augmentation):
X_val = np.load("final_data/X_raw_val.npy")
y_val = np.load("final_data/y_val.npy")

# False alarm test:
X_fa = np.load("final_data/X_raw_false_alarm.npy")
y_fa = np.load("final_data/y_false_alarm.npy")   # all 0 — model should NOT alarm
```

### Feature dimensions quick reference
| Array | Shape | Description |
|-------|-------|-------------|
| `X_raw` | (N, 4096) | Resampled + normalized signal → CNN input |
| `X_stat` | (N, 18) | Time + freq domain statistics |
| `X_env` | (N, 4) | BPFI, BPFO, BPF, FTF amplitudes |
| `X_meta` | (N, 8) | RPM, SR, speed_type, asset_onehot |
| `X_tabular` | (N, 30) | All tabular features combined |

---

## Evaluation Metrics (per hackathon rubric)
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = model.predict(X_val)

print(classification_report(y_val, y_pred,
      target_names=['Normal', 'Inner Ring', 'Ball', 'Outer Ring']))

# False Positive Rate per class
for c in range(4):
    fp = np.sum((y_val != c) & (y_pred == c))
    tn = np.sum((y_val != c) & (y_pred != c))
    print(f"Class {c} FPR: {fp/(fp+tn):.4f}")

# False alarm rate on non-bearing fault (Folder 11)
fa_pred = model.predict(X_fa)
false_alarm_rate = np.mean(fa_pred != 0)
print(f"False Alarm Rate (shaft misalignment): {false_alarm_rate:.4f}")
```
