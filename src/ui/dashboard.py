"""
BRIGHT Building Damage Assessment — Streamlit Dashboard

Reads pre-computed inference_results.parquet (no model weights or TIF files needed).
Generate the parquet first: python scripts/export_inference.py
"""
import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score

# ── Constants ─────────────────────────────────────────────────────────────────

DAMAGE_NAMES = ["Intact", "Damaged", "Destroyed"]
PARQUET_PATH = "data/inference_results.parquet"
PHI          = 1.618   # golden ratio for column proportions

COLORS = {
    "multimodal": "#2563EB",
    "optical":    "#64748B",
    "intact":     "#16A34A",
    "damaged":    "#D97706",
    "destroyed":  "#DC2626",
    "gate":       "#7C3AED",
    "card_bg":    "#1E293B",
    "text_muted": "#94A3B8",
}

CLASS_COLORS = {
    "Intact":    COLORS["intact"],
    "Damaged":   COLORS["damaged"],
    "Destroyed": COLORS["destroyed"],
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#CBD5E1"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BRIGHT Damage Assessment",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.3px; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; color: #CBD5E1; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="metric-container"] label {
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8 !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #F1F5F9;
    }

    /* Tabs */
    div[data-testid="stTab"] button {
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.6px !important;
        text-transform: uppercase;
        padding: 8px 20px !important;
    }

    /* Context boxes */
    .context-box {
        background: #0F172A;
        border-left: 3px solid #2563EB;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 13px;
        color: #94A3B8;
        line-height: 1.6;
    }

    /* Phase badge */
    .phase-badge {
        display: inline-block;
        background: #1E3A5F;
        color: #7EB8F7;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stRadio label { font-size: 13px; }
    .sidebar-section {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #475569;
        margin: 16px 0 6px 0;
    }

    /* Gallery cards */
    .gallery-card { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["pred_multimodal_name"] = df["pred_multimodal"].map({i: n for i, n in enumerate(DAMAGE_NAMES)})
    df["pred_optical_name"]    = df["pred_optical"].map({i: n for i, n in enumerate(DAMAGE_NAMES)})
    df["mm_correct"]           = df["pred_multimodal"] == df["true_label"]
    df["opt_correct"]          = df["pred_optical"]    == df["true_label"]
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)

def per_class_f1(y_true, y_pred) -> list[float]:
    return f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0).tolist()

def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


# ── Chart builders ────────────────────────────────────────────────────────────

def _apply_theme(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False, color="#64748B")
    fig.update_yaxes(gridcolor="#1E293B", zeroline=False, color="#64748B")
    return fig


def ablation_bar(val_df: pd.DataFrame) -> go.Figure:
    f1_mm  = per_class_f1(val_df["true_label"], val_df["pred_multimodal"])
    f1_opt = per_class_f1(val_df["true_label"], val_df["pred_optical"])

    fig = go.Figure([
        go.Bar(
            name="Multimodal  (Optical + SAR)",
            x=DAMAGE_NAMES, y=f1_mm,
            marker_color=COLORS["multimodal"],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in f1_mm],
            textposition="outside",
            textfont=dict(size=11),
        ),
        go.Bar(
            name="Optical-only  (baseline)",
            x=DAMAGE_NAMES, y=f1_opt,
            marker_color=COLORS["optical"],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in f1_opt],
            textposition="outside",
            textfont=dict(size=11),
        ),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="F1 Score", range=[0, 1.15], tickformat=".2f"),
        xaxis=dict(title="Damage Class"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, x=0,
            font=dict(size=11),
        ),
        title=dict(text="Per-class F1 Score — Validation Set", font=dict(size=13), x=0),
    )
    return _apply_theme(fig, height=340)


def confusion_heatmap(y_true, y_pred, title: str, model_color: str) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    annotations = []
    for r in range(3):
        for c in range(3):
            annotations.append(dict(
                x=c, y=r,
                text=f"<b>{cm[r][c]}</b><br><span style='font-size:10px'>{cm_norm[r][c]:.0%}</span>",
                showarrow=False,
                font=dict(color="white", size=12),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Pred {n}" for n in DAMAGE_NAMES],
        y=[f"True {n}" for n in DAMAGE_NAMES],
        colorscale=[[0, "#0F172A"], [0.5, "#1E3A5F"], [1, model_color]],
        zmin=0, zmax=1,
        showscale=False,
    ))
    fig.update_layout(
        annotations=annotations,
        title=dict(text=title, font=dict(size=12, color="#94A3B8"), x=0),
    )
    return _apply_theme(fig, height=280)


def distribution_bar(val_df: pd.DataFrame) -> go.Figure:
    counts = val_df["true_label_name"].value_counts().reindex(DAMAGE_NAMES, fill_value=0)
    total  = counts.sum()
    fig = go.Figure(go.Bar(
        x=DAMAGE_NAMES,
        y=counts.values,
        marker_color=[CLASS_COLORS[n] for n in DAMAGE_NAMES],
        marker_line_width=0,
        text=[f"{v}  ({v/total:.0%})" for v in counts.values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text="Validation Set — Label Distribution", font=dict(size=13), x=0),
        yaxis=dict(title="Tile Count", range=[0, counts.max() * 1.3]),
        xaxis=dict(title="Damage Class"),
    )
    return _apply_theme(fig, height=260)


def confidence_chart(mm_probs: list, opt_probs: list) -> go.Figure:
    fig = go.Figure([
        go.Bar(
            name="Multimodal",
            x=DAMAGE_NAMES, y=mm_probs,
            marker_color=COLORS["multimodal"],
            marker_line_width=0,
            text=[f"{v:.2f}" for v in mm_probs],
            textposition="outside",
            textfont=dict(size=11),
        ),
        go.Bar(
            name="Optical-only",
            x=DAMAGE_NAMES, y=opt_probs,
            marker_color=COLORS["optical"],
            marker_line_width=0,
            text=[f"{v:.2f}" for v in opt_probs],
            textposition="outside",
            textfont=dict(size=11),
        ),
    ])
    fig.add_hline(y=0.5, line_dash="dot", line_color="#475569",
                  annotation_text="50%", annotation_position="right",
                  annotation_font=dict(size=10, color="#475569"))
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Confidence", range=[0, 1.2], tickformat=".0%"),
        xaxis=dict(title="Damage Class"),
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        title=dict(text="Model Confidence per Class", font=dict(size=12), x=0),
    )
    return _apply_theme(fig, height=260)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _badge(label: str, bg: str) -> str:
    return (
        f'<span style="background:{bg};color:white;padding:2px 9px;'
        f'border-radius:5px;font-size:11px;font-weight:700;letter-spacing:0.3px">'
        f'{label}</span>'
    )


def _context(text: str):
    st.markdown(f'<div class="context-box">{text}</div>', unsafe_allow_html=True)


# ── Tile gallery ──────────────────────────────────────────────────────────────

def render_gallery(rows: pd.DataFrame, model_col: str, model_name_col: str):
    COLS      = 4
    PAGE_SIZE = 24
    total     = len(rows)

    if total == 0:
        st.info("No tiles match the current filters. Adjust the sidebar filters to broaden the selection.")
        return

    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    col_nav, col_info = st.columns([1, 3])
    with col_nav:
        page = st.number_input(
            "Page", min_value=1, max_value=n_pages, value=1, step=1,
            label_visibility="collapsed",
        ) - 1
    with col_info:
        st.caption(f"{total} tiles found  ·  Page {page+1} of {n_pages}  ·  {PAGE_SIZE} per page")

    batch = rows.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    cols  = st.columns(COLS)

    for idx, (_, row) in enumerate(batch.iterrows()):
        with cols[idx % COLS]:
            c1, c2 = st.columns(2)
            c1.image(base64.b64decode(row["optical_thumb"]), width="stretch", caption="Optical")
            c2.image(base64.b64decode(row["sar_thumb"]),     width="stretch", caption="SAR")

            true_name = row["true_label_name"]
            pred_name = row[model_name_col]
            correct   = row[model_col] == row["true_label"]

            marker = '<span style="color:#16A34A;font-weight:700">&#10003;</span>' if correct \
                else '<span style="color:#DC2626;font-weight:700">&#10007;</span>'
            st.markdown(
                f"{marker} &nbsp;True: {_badge(true_name, CLASS_COLORS[true_name])} "
                f"&rarr; Pred: {_badge(pred_name, CLASS_COLORS[pred_name])}",
                unsafe_allow_html=True,
            )
            tile_short = row["tile_id"].replace("turkey-earthquake_", "TUR-").replace("noto-earthquake_", "NTO-")
            st.caption(tile_short)
            st.divider()


# ── Tile inspector ────────────────────────────────────────────────────────────

def render_inspector(df: pd.DataFrame):
    _context(
        "Select any tile to examine predictions in detail. "
        "The <b>confidence chart</b> shows how certain each model was about each damage class — "
        "a well-calibrated model should assign high confidence to the correct class. "
        "The <b>Grad-CAM attention maps</b> reveal which image regions most influenced the prediction, "
        "providing a spatial explanation of model reasoning."
    )

    def _fmt_tile(tid: str) -> str:
        return tid.replace("turkey-earthquake_", "TUR ").replace("noto-earthquake_", "NTO ")

    tile_ids = df["tile_id"].tolist()
    selected = st.selectbox(
        "Select a tile to inspect",
        tile_ids,
        format_func=_fmt_tile,
        help="Use the sidebar Split filter to narrow the tile list. "
             "Search by typing a tile ID prefix.",
    )
    if not selected:
        return

    row       = df[df["tile_id"] == selected].iloc[0]
    true_name = row["true_label_name"]
    mm_name   = row["pred_multimodal_name"]
    opt_name  = row["pred_optical_name"]
    mm_probs  = [row["conf_mm_intact"],  row["conf_mm_damaged"],  row["conf_mm_destroyed"]]
    opt_probs = [row["conf_opt_intact"], row["conf_opt_damaged"], row["conf_opt_destroyed"]]

    st.markdown("---")

    # Row 1: imagery + confidence
    img_w = 1
    c_img1, c_img2, c_conf = st.columns([img_w, img_w, img_w * PHI])

    with c_img1:
        st.markdown("##### Pre-event Optical")
        st.image(base64.b64decode(row["optical_thumb"]), width="stretch")
        st.caption("RGB satellite image captured before the disaster event.")

    with c_img2:
        st.markdown("##### Post-event SAR")
        st.image(base64.b64decode(row["sar_thumb"]), width="stretch")
        st.caption("Synthetic Aperture Radar intensity image after the event. Works through cloud and darkness.")

    with c_conf:
        st.markdown("##### Prediction Summary")

        # Label row
        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown(
            f"**Ground Truth**<br>{_badge(true_name, CLASS_COLORS[true_name])}",
            unsafe_allow_html=True,
        )
        mm_correct  = mm_name  == true_name
        opt_correct = opt_name == true_name
        lc2.markdown(
            f"**Multimodal**<br>{_badge(mm_name, CLASS_COLORS[mm_name])} "
            f"{'<span style=\"color:#16A34A\">&#10003;</span>' if mm_correct else '<span style=\"color:#DC2626\">&#10007;</span>'}",
            unsafe_allow_html=True,
        )
        lc3.markdown(
            f"**Optical-only**<br>{_badge(opt_name, CLASS_COLORS[opt_name])} "
            f"{'<span style=\"color:#16A34A\">&#10003;</span>' if opt_correct else '<span style=\"color:#DC2626\">&#10007;</span>'}",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.plotly_chart(confidence_chart(mm_probs, opt_probs), width="stretch")

    # Row 2: Grad-CAM
    has_gradcam = (
        "opt_gradcam_thumb" in df.columns
        and pd.notna(row.get("opt_gradcam_thumb"))
    )
    if has_gradcam:
        st.markdown("---")
        st.markdown("##### Grad-CAM Attention Maps")
        _context(
            "<b>How to read:</b> Red regions had the strongest influence on the model's prediction for the predicted class. "
            "Blue and green regions contributed little. "
            "The two maps show what each branch of the multimodal model focused on independently — "
            "optical (pre-event structure) and SAR (post-event backscatter). "
            "For the optical-gated SAR model (Phase 3): when the gate &alpha; is near 0, "
            "the SAR branch is distrusted and its attention map will appear diffuse or uniform."
        )
        g1, g2 = st.columns(2)
        with g1:
            st.image(
                base64.b64decode(row["opt_gradcam_thumb"]),
                width="stretch",
            )
            st.caption(
                "Optical branch — ResNet-18 layer 4 activations. "
                "Should concentrate on building structure and texture changes."
            )
        with g2:
            st.image(
                base64.b64decode(row["sar_gradcam_thumb"]),
                width="stretch",
            )
            st.caption(
                "SAR branch — ResNet-18 layer 4 activations. "
                "For Destroyed tiles, expect focus on low-backscatter rubble zones. "
                "For Damaged/Intact tiles, expect diffuse attention (gate reducing SAR influence)."
            )


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        'BRIGHT Building Damage Assessment'
        '<span class="phase-badge">PHASE 3</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Multimodal SAR + Optical fusion vs. Optical-only baseline  "
        "| Turkey Earthquake & Noto Earthquake  "
        "| ResNet-18 with optical-gated SAR (Phase 3)"
    )

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
        st.markdown("## Filters")
        st.markdown(
            '<div class="context-box" style="font-size:12px">'
            'Filters apply to the <b>Tile Gallery</b> and <b>Tile Inspector</b> tabs. '
            'The <b>Performance Analysis</b> tab always uses the full held-out validation set.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">Data Split</div>', unsafe_allow_html=True)
        split_choice = st.radio(
            "split",
            ["Val (held-out)", "Train", "All"],
            label_visibility="collapsed",
            help="Val = held-out set used for all reported metrics. Train = training tiles.",
        )

        st.markdown('<div class="sidebar-section">Class Filter</div>', unsafe_allow_html=True)
        label_filter = st.multiselect(
            "True damage class",
            DAMAGE_NAMES,
            default=DAMAGE_NAMES,
            help="Filter tiles by their ground-truth damage label.",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section">Display Options</div>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Gallery model predictions",
            ["Multimodal", "Optical-only"],
            help="Choose which model's predictions to display in the Tile Gallery.",
        )
        error_only = st.checkbox(
            "Show misclassified tiles only",
            help="Filter the gallery to tiles where the selected model made an incorrect prediction.",
        )

        st.divider()
        st.markdown('<div class="sidebar-section">Dataset</div>', unsafe_allow_html=True)
        st.caption(
            "BRIGHT — Chen et al., ESSD 2025  \n"
            "Turkey Earthquake + Noto Earthquake  \n"
            "824 train · 121 val · 231 test tiles"
        )
        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
        st.caption(
            "ResNet-18 dual-branch  \n"
            "Optical-gated scalar SAR gate  \n"
            "CE + Lovász-Softmax loss  \n"
            "22.4M parameters"
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    f1_mm   = macro_f1(val_df["true_label"], val_df["pred_multimodal"])
    f1_opt  = macro_f1(val_df["true_label"], val_df["pred_optical"])
    acc_mm  = accuracy(val_df["true_label"], val_df["pred_multimodal"])
    acc_opt = accuracy(val_df["true_label"], val_df["pred_optical"])
    delta   = f1_mm - f1_opt

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Validation Tiles", len(val_df),
        help="Number of tiles in the held-out validation set, used for all reported metrics.",
    )
    k2.metric(
        "Multimodal Macro F1", f"{f1_mm:.3f}",
        help="F1 score averaged equally across Intact, Damaged, and Destroyed classes (optical + SAR model).",
    )
    k3.metric(
        "Optical Macro F1", f"{f1_opt:.3f}",
        help="F1 score for the optical-only baseline. Subtract from Multimodal F1 to get SAR delta.",
    )
    k4.metric(
        "SAR Delta (F1)", f"{delta:+.3f}", delta=f"{delta:+.3f}",
        help="Improvement from adding SAR. Positive = SAR helps. Gate requires >= +0.050 to confirm SAR value.",
    )
    k5.metric(
        "Multimodal Accuracy", f"{acc_mm:.1%}", f"{acc_mm - acc_opt:+.1%}",
        help="Percentage of tiles correctly classified by the multimodal model vs. optical-only.",
    )

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ablation, tab_gallery, tab_inspector = st.tabs(
        ["Performance Analysis", "Tile Gallery", "Tile Inspector"]
    )

    # ── Tab 1: Performance Analysis ───────────────────────────────────────────
    with tab_ablation:
        _context(
            "<b>Research question:</b> Does fusing post-event SAR imagery with pre-event optical improve "
            "building damage classification over optical alone? "
            "The ablation gate requires the multimodal model to exceed the optical-only baseline by "
            "<b>&ge; 0.050 macro F1</b> to confirm SAR adds signal. "
            "Current SAR delta: <b>"
            + f"{delta:+.3f}"
            + f"</b> &nbsp;|&nbsp; Gate status: <b>{'PASSED' if delta >= 0.05 else 'NOT YET PASSED'}</b>. "
            "All metrics are computed on the held-out validation set."
        )

        col_chart, col_cm1, col_cm2 = st.columns([PHI, 1, 1])

        with col_chart:
            st.plotly_chart(ablation_bar(val_df), width="stretch")
            st.caption(
                "Each bar shows the F1 score for one damage class. "
                "F1 = harmonic mean of precision and recall, ranging from 0 (worst) to 1 (perfect). "
                "Macro F1 averages equally across all three classes regardless of class frequency."
            )

        with col_cm1:
            st.markdown("##### Multimodal")
            st.plotly_chart(
                confusion_heatmap(
                    val_df["true_label"], val_df["pred_multimodal"],
                    f"Macro F1 = {f1_mm:.3f}", COLORS["multimodal"],
                ),
                width="stretch",
            )
            st.caption("Rows = ground truth. Columns = predicted class. Cell shows count and row-normalised rate.")

        with col_cm2:
            st.markdown("##### Optical-only")
            st.plotly_chart(
                confusion_heatmap(
                    val_df["true_label"], val_df["pred_optical"],
                    f"Macro F1 = {f1_opt:.3f}", COLORS["optical"],
                ),
                width="stretch",
            )
            st.caption("Baseline model using pre-event optical imagery only. Compare with multimodal to isolate SAR value.")

        st.markdown("")
        col_dist, col_spacer = st.columns([1, PHI - 1])
        with col_dist:
            st.plotly_chart(distribution_bar(val_df), width="stretch")
            st.caption(
                "The Damaged class is severely underrepresented, which is why macro F1 is low overall. "
                "Only tiles with >= 200 building pixels are included; sparse tiles are filtered to avoid label noise."
            )

    # ── Tab 2: Tile Gallery ───────────────────────────────────────────────────
    with tab_gallery:
        _context(
            "Browse individual tiles from the dataset. Each card shows the pre-event optical image "
            "and post-event SAR image side by side, along with the ground-truth label and model prediction. "
            "<b>Green checkmark</b> = correct prediction. <b>Red cross</b> = misclassification. "
            "Use the sidebar to filter by split, damage class, or misclassified tiles only. "
            "Select a tile ID in the <b>Tile Inspector</b> tab for a detailed per-tile analysis."
        )

        split_map  = {"Val (held-out)": "val", "Train": "train", "All": None}
        split_val  = split_map[split_choice]
        filtered   = df.copy()
        if split_val:
            filtered = filtered[filtered["split"] == split_val]
        if label_filter and set(label_filter) != set(DAMAGE_NAMES):
            filtered = filtered[filtered["true_label_name"].isin(label_filter)]

        model_col      = "pred_multimodal"      if model_choice == "Multimodal" else "pred_optical"
        model_name_col = "pred_multimodal_name" if model_choice == "Multimodal" else "pred_optical_name"
        correct_col    = "mm_correct"           if model_choice == "Multimodal" else "opt_correct"

        if error_only:
            filtered = filtered[~filtered[correct_col]]

        render_gallery(filtered, model_col, model_name_col)

    # ── Tab 3: Tile Inspector ─────────────────────────────────────────────────
    with tab_inspector:
        split_val_insp = split_map[split_choice]
        insp_df = df if not split_val_insp else df[df["split"] == split_val_insp]
        render_inspector(insp_df)

    # Footer
    st.divider()
    st.caption(
        "Chen et al. (2025). BRIGHT: a globally distributed multimodal building damage assessment dataset "
        "with very-high-resolution for all-weather disaster response. "
        "Earth System Science Data, 17(11), 6217–6253. "
        "https://doi.org/10.5194/essd-17-6217-2025"
    )


if __name__ == "__main__":
    main()
