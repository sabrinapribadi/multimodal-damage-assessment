"""
BRIGHT Building Damage Assessment — Streamlit Dashboard

Reads pre-computed inference_results.parquet (no model weights or TIF files needed).
Generate the parquet first: python scripts/export_inference.py

Deploy to Streamlit Cloud: commit the parquet + this file + requirements-streamlit.txt
"""
import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score

DAMAGE_NAMES  = ["Intact", "Damaged", "Destroyed"]
CLASS_COLORS  = {"Intact": "#2ECC71", "Damaged": "#F39C12", "Destroyed": "#E74C3C"}
PARQUET_PATH  = "data/inference_results.parquet"

st.set_page_config(
    page_title="BRIGHT Damage Assessment",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["pred_multimodal_name"] = df["pred_multimodal"].map({i: n for i, n in enumerate(DAMAGE_NAMES)})
    df["pred_optical_name"]    = df["pred_optical"].map({i: n for i, n in enumerate(DAMAGE_NAMES)})
    df["mm_correct"]  = df["pred_multimodal"] == df["true_label"]
    df["opt_correct"] = df["pred_optical"]    == df["true_label"]
    return df


# ── Helper metrics ────────────────────────────────────────────────────────────

def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)

def per_class_f1(y_true, y_pred) -> list[float]:
    return f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0).tolist()

def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


# ── Chart builders ────────────────────────────────────────────────────────────

def confusion_heatmap(y_true, y_pred, title: str) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Pred {n}" for n in DAMAGE_NAMES],
        y=[f"True {n}" for n in DAMAGE_NAMES],
        colorscale=[[0, "#2C3E50"], [0.5, "#F39C12"], [1, "#E74C3C"]],
        zmin=0, zmax=1,
        text=[[str(cm[r][c]) for c in range(3)] for r in range(3)],
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
    )
    return fig


def ablation_bar(val_df: pd.DataFrame) -> go.Figure:
    f1_mm  = per_class_f1(val_df["true_label"], val_df["pred_multimodal"])
    f1_opt = per_class_f1(val_df["true_label"], val_df["pred_optical"])

    fig = go.Figure([
        go.Bar(name="Multimodal (opt+SAR)", x=DAMAGE_NAMES, y=f1_mm,
               marker_color="#3498DB", text=[f"{v:.2f}" for v in f1_mm],
               textposition="outside"),
        go.Bar(name="Optical-only",         x=DAMAGE_NAMES, y=f1_opt,
               marker_color="#95A5A6", text=[f"{v:.2f}" for v in f1_opt],
               textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="F1 Score", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── Tile gallery ──────────────────────────────────────────────────────────────

def _badge(label: str, bg: str) -> str:
    return (
        f'<span style="background:{bg};color:white;padding:2px 8px;'
        f'border-radius:8px;font-size:11px;font-weight:600">{label}</span>'
    )


def render_gallery(rows: pd.DataFrame, model_col: str, model_name_col: str):
    COLS = 4
    PAGE_SIZE = 24
    total = len(rows)

    if total == 0:
        st.info("No tiles match the current filters.")
        return

    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1,
                           label_visibility="collapsed") - 1
    st.caption(f"{total} tiles · page {page+1}/{n_pages}")

    batch = rows.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    cols  = st.columns(COLS)

    for idx, (_, row) in enumerate(batch.iterrows()):
        with cols[idx % COLS]:
            opt_bytes = base64.b64decode(row["optical_thumb"])
            sar_bytes = base64.b64decode(row["sar_thumb"])

            c1, c2 = st.columns(2)
            c1.image(opt_bytes, width='stretch', caption="Optical")
            c2.image(sar_bytes, width='stretch', caption="SAR")

            true_name = row["true_label_name"]
            pred_name = row[model_name_col]
            correct   = row[model_col] == row["true_label"]

            true_bg = CLASS_COLORS[true_name]
            pred_bg = CLASS_COLORS[pred_name]
            tick    = "✓" if correct else "✗"

            st.markdown(
                f"{tick} &nbsp; True: {_badge(true_name, true_bg)} "
                f"→ Pred: {_badge(pred_name, pred_bg)}",
                unsafe_allow_html=True,
            )
            st.caption(row["tile_id"].replace("turkey-earthquake_", ""))
            st.divider()


# ── Tile inspector ────────────────────────────────────────────────────────────

def render_inspector(df: pd.DataFrame):
    tile_ids = df["tile_id"].tolist()
    selected = st.selectbox("Select tile", tile_ids, format_func=lambda x: x.replace("turkey-earthquake_", ""))
    if not selected:
        return

    row = df[df["tile_id"] == selected].iloc[0]

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.image(base64.b64decode(row["optical_thumb"]), caption="Pre-event Optical",
                 width='stretch')
    with c2:
        st.image(base64.b64decode(row["sar_thumb"]), caption="Post-event SAR",
                 width='stretch')
    with c3:
        true_name = row["true_label_name"]
        st.markdown(f"**True label:** {_badge(true_name, CLASS_COLORS[true_name])}", unsafe_allow_html=True)
        st.markdown("")

        mm_probs  = [row["conf_mm_intact"],  row["conf_mm_damaged"],  row["conf_mm_destroyed"]]
        opt_probs = [row["conf_opt_intact"], row["conf_opt_damaged"], row["conf_opt_destroyed"]]

        fig = go.Figure([
            go.Bar(name="Multimodal", x=DAMAGE_NAMES, y=mm_probs,
                   marker_color="#3498DB", text=[f"{v:.2f}" for v in mm_probs], textposition="outside"),
            go.Bar(name="Optical",    x=DAMAGE_NAMES, y=opt_probs,
                   marker_color="#95A5A6", text=[f"{v:.2f}" for v in opt_probs], textposition="outside"),
        ])
        fig.update_layout(
            barmode="group", yaxis_range=[0, 1.1],
            yaxis_title="Confidence", height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width='stretch')


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    st.title("🛰️ BRIGHT Building Damage Assessment")
    st.caption("Turkey Earthquake · Feb 2023 · Multimodal SAR + Optical vs Optical-only ablation")

    try:
        df = load_df()
    except FileNotFoundError:
        st.error(
            f"`{PARQUET_PATH}` not found. "
            "Run `python scripts/export_inference.py` locally first, then commit the file."
        )
        st.stop()

    val_df = df[df["split"] == "val"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        st.info("Filters apply to the **Tile Gallery** and **Tile Inspector** tabs.\n\n"
                "The Ablation Results tab always shows the held-out val set.", icon="ℹ️")
        split_choice = st.radio("Split", ["Val (held-out)", "Train", "All"])
        label_filter = st.multiselect("True label", DAMAGE_NAMES, default=DAMAGE_NAMES)
        error_only   = st.checkbox("Misclassified tiles only")
        model_choice = st.radio("Model (gallery)", ["Multimodal", "Optical-only"])
        st.divider()
        st.caption("Dataset: BRIGHT · Chen et al., ESSD 2025")
        st.caption("Model: Custom dual-branch CNN · Phase 1")

    # ── KPI row (always val set) ──────────────────────────────────────────────
    f1_mm  = macro_f1(val_df["true_label"], val_df["pred_multimodal"])
    f1_opt = macro_f1(val_df["true_label"], val_df["pred_optical"])
    acc_mm  = accuracy(val_df["true_label"], val_df["pred_multimodal"])
    acc_opt = accuracy(val_df["true_label"], val_df["pred_optical"])
    delta   = f1_mm - f1_opt

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Val tiles", len(val_df))
    k2.metric("Multimodal macro F1", f"{f1_mm:.3f}")
    k3.metric("Optical macro F1",    f"{f1_opt:.3f}")
    k4.metric("SAR ΔF1",             f"{delta:+.3f}", delta=f"{delta:+.3f}")
    k5.metric("Multimodal accuracy", f"{acc_mm:.1%}", f"{acc_mm-acc_opt:+.1%}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ablation, tab_gallery, tab_inspector = st.tabs(
        ["📊 Ablation Results", "🗺️ Tile Gallery", "🔍 Tile Inspector"]
    )

    with tab_ablation:
        col_chart, col_cm1, col_cm2 = st.columns([1.2, 1, 1])

        with col_chart:
            st.subheader("Per-class F1 (val set)")
            st.plotly_chart(ablation_bar(val_df), width='stretch')

        with col_cm1:
            st.subheader("Multimodal")
            fig_mm = confusion_heatmap(
                val_df["true_label"], val_df["pred_multimodal"],
                f"Confusion — Multimodal (F1={f1_mm:.3f})"
            )
            st.plotly_chart(fig_mm, width='stretch')

        with col_cm2:
            st.subheader("Optical-only")
            fig_opt = confusion_heatmap(
                val_df["true_label"], val_df["pred_optical"],
                f"Confusion — Optical-only (F1={f1_opt:.3f})"
            )
            st.plotly_chart(fig_opt, width='stretch')

        # Label distribution
        st.subheader("Val set label distribution")
        counts = val_df["true_label_name"].value_counts().reindex(DAMAGE_NAMES, fill_value=0)
        fig_dist = go.Figure(go.Bar(
            x=DAMAGE_NAMES, y=counts.values,
            marker_color=[CLASS_COLORS[n] for n in DAMAGE_NAMES],
            text=counts.values, textposition="outside",
        ))
        fig_dist.update_layout(yaxis_title="Tiles", height=250,
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_dist, width='stretch')

    with tab_gallery:
        # Apply sidebar filters
        split_map = {"Val (held-out)": "val", "Train": "train", "All": None}
        split_val = split_map[split_choice]
        filtered  = df.copy()
        if split_val:
            filtered = filtered[filtered["split"] == split_val]
        if label_filter and set(label_filter) != set(DAMAGE_NAMES):
            filtered = filtered[filtered["true_label_name"].isin(label_filter)]

        model_col      = "pred_multimodal" if model_choice == "Multimodal" else "pred_optical"
        model_name_col = "pred_multimodal_name" if model_choice == "Multimodal" else "pred_optical_name"
        correct_col    = "mm_correct" if model_choice == "Multimodal" else "opt_correct"

        if error_only:
            filtered = filtered[~filtered[correct_col]]

        render_gallery(filtered, model_col, model_name_col)

    with tab_inspector:
        split_val_insp = split_map[split_choice]
        insp_df = df if not split_val_insp else df[df["split"] == split_val_insp]
        render_inspector(insp_df)

    # Footer
    st.divider()
    st.caption(
        "Chen et al. (2025). BRIGHT: a globally distributed multimodal building damage assessment dataset. "
        "Earth System Science Data, 17(11), 6217–6253."
    )


if __name__ == "__main__":
    main()
