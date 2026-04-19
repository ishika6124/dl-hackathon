# 🔧 Multimodal Bearing Fault Diagnosis

## 📌 Overview

This project focuses on **bearing fault diagnosis using vibration signals**, combining:

- Raw time-series signals  
- Statistical features  
- Envelope spectrum (Hilbert transform)  
- Physics-based fault frequencies (BPFI, BPFO, BSF)  
- Operating conditions (RPM, sampling rate)  

We explored both:
- 🔴 **Deep Learning (DL)** approaches  
- 🟡 **Machine Learning (ML)** pipelines  

The project evolved from **high-accuracy but flawed setups** to a **leakage-free, realistic evaluation framework**.

---

# 🧠 Main Folders (IMPORTANT)

## 🔴 `dl_approach/` ⭐ (PRIMARY APPROACH)

This is the **main and final deep learning pipeline**.

### 🔥 Key File:
- `bearing_diagnosis_loco.py`

---

### 🚀 Core Idea

**Cascaded Multimodal Deep Learning with LOCO (Leave-One-Case-Out) and Stratified Validation**

---

### 📊 Data Strategy

- ✅ **LOCO (Leave-One-Case-Out)**  
  - Train on 10 cases  
  - Test on 1 unseen case  
  - Repeat for all cases  

- ✅ **Stratified Validation (15%)**
  - Created from pooled training cases  
  - Ensures all classes are represented  
  - Avoids biased single-case validation  

- ✅ **No Data Leakage**
  - Test case never used in training or validation  

---

### 🏗️ Architecture (3-Stage Cascade)

1. **Stage 1: Normal vs Abnormal**
2. **Stage 2: Bearing Fault vs External Disturbance**
3. **Stage 3: Fault Type Classification**
   - Inner Race  
   - Ball Fault  
   - Outer Race  

---

### 🧠 Why this approach?

- LOCO ensures **true generalization**
- Stratified validation ensures **stable training**
- Cascade improves **interpretability**

---

---

## 🟡 `ml_approach/` ⭐ (FINAL PIPELINE IMPLEMENTATION)

This contains the **final corrected ML pipeline** based on the structured approach.

---

### 🔥 Key File:
- `final_pipeline_corrected.py`

---

### ⚙️ Pipeline Overview

A **3-stage cascaded ML system**:

---

### 🔹 Stage 1 — Anomaly Detection
- Model: **Isolation Forest**
- Input: Window-level features  
- Output: Normal vs Abnormal  

---

### 🔹 Stage 2 — Fault vs Disturbance
- Model: **Logistic Regression**
- Input: File-level aggregated features  
- Output: Bearing vs External  

---

### 🔹 Stage 3 — Fault Classification
- Method: **Physics-based rule engine**

Decision based on:
- BPFI → Inner fault  
- BPFO → Outer fault  
- BSF → Ball fault  

---

### 🔬 Feature Engineering

- Time domain: RMS, kurtosis, skew, crest factor  
- Spectral: entropy, band energy  
- Physics:
  - BPFI, BPFO, BSF energies  
  - Fault energy dominance  
- Operating:
  - RPM, sampling rate  

---

### 📊 Additional Outputs

- ROC & PR curves  
- Calibration analysis  
- Ablation study  
- Per-sensor metrics  
- OOD (Out-of-Distribution) analysis  

---

---

# 🖥️ User Interface (Streamlit App)

The repository includes an interactive UI for the ML pipeline using **Streamlit**.

## 📂 Location
- `ml_approach/app.py`

---

## 🚀 How to Run the App

Navigate to the `ml_approach` folder and run:

```bash
python -m streamlit run app.py
