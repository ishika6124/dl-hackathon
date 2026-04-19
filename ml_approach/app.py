import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bearing Fault Demo",
    page_icon="⚙️",
    layout="centered",
)

OUTPUT_DIR = "outputs_final_pipeline"

# ── Load data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    file_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "file_predictions.csv"))
    window_df = pd.read_csv(os.path.join(OUTPUT_DIR, "window_predictions.csv"))
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        summary = json.load(f)
    return file_df, window_df, summary

if not os.path.exists(os.path.join(OUTPUT_DIR, "file_predictions.csv")):
    st.error("Run `python final_pipeline_corrected.py` first to generate results.")
    st.stop()

file_df, window_df, summary = load_data()

# ── Label helpers ───────────────────────────────────────────
CLASS_COLORS = {
    "Inner Race Fault":      "#E53935",
    "Outer Race Fault":      "#FB8C00",
    "Ball Fault":            "#8E24AA",
    "Healthy (No Fault)":    "#43A047",
    "External / OOD":        "#1E88E5",
    "Bearing Fault":         "#F57F17",
}

CLASS_ICONS = {
    "Inner Race Fault":   "🔴",
    "Outer Race Fault":   "🟠",
    "Ball Fault":         "🟣",
    "Healthy (No Fault)": "🟢",
    "External / OOD":     "🔵",
    "Bearing Fault":      "🟡",
}


def actual_class(row):
    if row["stage2_label"] == 0:
        return "External / OOD"
    lbl = str(row.get("stage3_label", "NONE"))
    return {
        "INNER_FAULT": "Inner Race Fault",
        "OUTER_FAULT": "Outer Race Fault",
        "BALL_FAULT":  "Ball Fault",
    }.get(lbl, "Healthy (No Fault)")


def predicted_class(row):
    if int(row["stage2_pred"]) == 0:
        return "External / OOD"
    pred = str(row.get("stage3_pred", ""))
    return {
        "INNER_FAULT": "Inner Race Fault",
        "OUTER_FAULT": "Outer Race Fault",
        "BALL_FAULT":  "Ball Fault",
    }.get(pred, "Bearing Fault")


file_df["actual"]    = file_df.apply(actual_class,    axis=1)
file_df["predicted"] = file_df.apply(predicted_class, axis=1)
file_df["correct"]   = file_df["actual"] == file_df["predicted"]


# ════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════
st.title("⚙️ Bearing Fault Classifier — Live Demo")
st.caption("Multimodal AI  •  SCA Bearing Dataset  •  3-Stage Hierarchical Pipeline")
st.divider()

# ── Session state for selected row ─────────────────────────
if "sel_idx" not in st.session_state:
    st.session_state.sel_idx = 0

# ── Controls ────────────────────────────────────────────────
col_btn, col_sel = st.columns([1, 2])

with col_btn:
    if st.button("🎲 Random Sample", type="primary", use_container_width=True):
        st.session_state.sel_idx = random.randint(0, len(file_df) - 1)

with col_sel:
    options = [
        f"Folder {int(r['folder'])}  |  Sensor {r['sensor']}"
        for _, r in file_df.iterrows()
    ]
    chosen = st.selectbox("— or pick manually", options, index=st.session_state.sel_idx,
                          label_visibility="collapsed")
    st.session_state.sel_idx = options.index(chosen)

row = file_df.iloc[st.session_state.sel_idx]
st.divider()

# ── Result card ─────────────────────────────────────────────
act  = row["actual"]
pred = row["predicted"]
correct = row["correct"]
conf = float(row.get("stage2_proba", 0.5))

result_icon = "✅ CORRECT" if correct else "❌ INCORRECT"
result_color = "#43A047" if correct else "#E53935"

st.markdown(
    f"""
    <div style="text-align:center; padding:8px 0 4px 0;">
        <span style="font-size:1.5rem; font-weight:700; color:{result_color};">
            {result_icon}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Actual vs Predicted side by side
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        f"""
        <div style="background:#f5f5f5; border-radius:12px; padding:20px; text-align:center; border:2px solid #ccc;">
            <div style="font-size:0.85rem; color:#666; margin-bottom:6px;">ACTUAL CLASS</div>
            <div style="font-size:2.2rem;">{CLASS_ICONS.get(act, '❓')}</div>
            <div style="font-size:1.3rem; font-weight:700; color:{CLASS_COLORS.get(act,'#333')};">
                {act}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    border_col = CLASS_COLORS.get(pred, "#333")
    st.markdown(
        f"""
        <div style="background:#f5f5f5; border-radius:12px; padding:20px; text-align:center; border:2px solid {border_col};">
            <div style="font-size:0.85rem; color:#666; margin-bottom:6px;">PREDICTED CLASS</div>
            <div style="font-size:2.2rem;">{CLASS_ICONS.get(pred, '❓')}</div>
            <div style="font-size:1.3rem; font-weight:700; color:{CLASS_COLORS.get(pred,'#333')};">
                {pred}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Confidence bar ──────────────────────────────────────────
st.markdown("**Model Confidence (Stage 2)**")
conf_pct = int(conf * 100)
bar_color = "#43A047" if conf > 0.7 else "#FB8C00" if conf > 0.4 else "#E53935"
st.markdown(
    f"""
    <div style="background:#e0e0e0; border-radius:8px; height:26px; width:100%; margin-bottom:4px;">
        <div style="background:{bar_color}; border-radius:8px; height:26px; width:{conf_pct}%;
                    display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
            <span style="color:white; font-weight:700; font-size:0.95rem;">{conf_pct}%</span>
        </div>
    </div>
    <div style="font-size:0.8rem; color:#666;">
        Bearing fault probability: <b>{conf:.3f}</b> &nbsp;|&nbsp;
        Folder: <b>{int(row['folder'])}</b> &nbsp;|&nbsp;
        Sensor: <b>{row['sensor']}</b> &nbsp;|&nbsp;
        Abnormal windows: <b>{row['stage1_abnormal_ratio']*100:.1f}%</b>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── 3-Stage breakdown ────────────────────────────────────────
st.markdown("**Decision Chain (3 Stages)**")

s1_ratio = float(row["stage1_abnormal_ratio"])
s1_result = "Abnormal Signal" if s1_ratio > 0.5 else "Normal Signal"
s1_icon   = "🔴" if s1_ratio > 0.5 else "🟢"

s2_result = "Bearing Fault" if int(row["stage2_pred"]) == 1 else "External Disturbance"
s2_icon   = "🔴" if int(row["stage2_pred"]) == 1 else "🔵"

s3_raw  = str(row.get("stage3_pred", ""))
s3_map  = {"INNER_FAULT": "Inner Race", "OUTER_FAULT": "Outer Race", "BALL_FAULT": "Ball Fault"}
s3_result = s3_map.get(s3_raw, "—")
s3_icon   = "🔴" if s3_raw == "INNER_FAULT" else "🟠" if s3_raw == "OUTER_FAULT" else "🟣" if s3_raw == "BALL_FAULT" else "⚪"

c1, c2, c3 = st.columns(3)
c1.metric(f"{s1_icon} Stage 1 — Anomaly",      s1_result, f"{s1_ratio*100:.0f}% windows flagged")
c2.metric(f"{s2_icon} Stage 2 — Localization", s2_result, f"P={conf:.3f}")
c3.metric(f"{s3_icon} Stage 3 — Fault Type",   s3_result)

st.divider()

# ── Anomaly score trend ──────────────────────────────────────
wf = window_df[
    (window_df["folder"] == int(row["folder"])) &
    (window_df["sensor"] == row["sensor"])
].sort_values("window_index").reset_index(drop=True)

if len(wf) > 0:
    st.markdown("**Anomaly Score Trend (this recording)**")
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(wf.index, wf["stage1_score"], color="#1976D2", lw=1.2, alpha=0.85)
    ax.fill_between(wf.index, wf["stage1_score"], alpha=0.12, color="#1976D2")

    abn = wf[wf["stage1_pred"] == 1]
    if len(abn) > 0:
        ax.scatter(abn.index, abn["stage1_score"],
                   color="#E53935", s=12, zorder=5, label="Flagged Abnormal")

    ax.set_xlabel("Window", fontsize=10)
    ax.set_ylabel("Anomaly Score", fontsize=10)
    ax.set_title(
        f"Folder {int(row['folder'])} | Sensor {row['sensor']} — "
        f"{len(abn)}/{len(wf)} windows flagged abnormal",
        fontsize=10, fontweight="bold"
    )
    if len(abn) > 0:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()

# ── Overall accuracy scorecard (small, at bottom) ───────────
st.markdown("**Overall Pipeline Performance**")
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Stage 1 Accuracy",  f"{summary.get('stage1_accuracy',0)*100:.1f}%")
mc2.metric("Stage 2 Accuracy",  f"{summary.get('stage2_accuracy',0)*100:.1f}%")
mc3.metric("Stage 3 Macro-F1",  f"{summary.get('stage3_macro_f1',0):.3f}")
