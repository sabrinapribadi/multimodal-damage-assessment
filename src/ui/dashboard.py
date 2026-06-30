"""
Streamlit Dashboard for BRIGHT Morocco Earthquake Damage Assessment
"""
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append('.')

from src.data.brighT_loader import DAMAGE_CLASSES, NUM_CLASSES, create_dataloader
from src.models.baseline_model import create_model

DATA_DIR   = Path("data/processed/morocco-earthquake")
SPLIT_FILE = Path("BRIGHT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt")
EVENT      = "morocco-earthquake"
CKPT_PATH  = Path("outputs/best_morocco_earthquake.pt")
DAMAGE_NAMES = [DAMAGE_CLASSES[i] for i in range(NUM_CLASSES)]

st.set_page_config(
    page_title="BRIGHT Damage Assessment — Morocco",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ BRIGHT Building Damage Assessment")
st.markdown("**Event:** Morocco Earthquake (Sep 2023) · Multimodal (Pre-event Optical + Post-event SAR)")

with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Model", ["multimodal", "optical_only", "sar_only"])
    st.divider()
    st.caption("Dataset: BRIGHT (Chen et al., ESSD 2025)")
    st.caption("DOI: 10.5194/essd-17-6217-2025")

col1, col2, col3 = st.columns(3)
col1.metric("Modalities", "Optical + SAR")
col2.metric("Classes", str(NUM_CLASSES), " / ".join(DAMAGE_NAMES))
col3.metric("Event", "Morocco EQ 2023", "56 tiles / 6,269 buildings")


@st.cache_resource
def load_data():
    return create_dataloader(
        data_dir=DATA_DIR,
        split_file=SPLIT_FILE,
        event=EVENT,
        batch_size=1,
        shuffle=False,
        synthetic_fallback=True,
    )


@st.cache_resource
def load_model(mtype):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model(mtype, num_classes=NUM_CLASSES).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        st.sidebar.success(f"Loaded checkpoint (val F1={ckpt.get('val_f1', '?'):.3f})")
    else:
        st.sidebar.warning("No trained checkpoint found — using random weights.")
    model.eval()
    return model, device


try:
    loader = load_data()
    model, device = load_model(model_type)
    st.success(f"Loaded {len(loader.dataset)} val tiles.")
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

st.header("Sample Visualization")
try:
    batch = next(iter(loader))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pre-event Optical")
        img = batch["images"]["optical"][0].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        true_label = DAMAGE_NAMES[batch["label"][0].item()]
        st.image(img, caption=f"True label: {true_label}", use_container_width=True)
    with c2:
        st.subheader("Post-event SAR")
        img = batch["images"]["sar"][0].cpu().numpy().squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        st.image(img, caption=f"Tile: {batch['tile_id'][0]}", use_container_width=True)
except Exception as e:
    st.warning(f"Could not display images: {e}")

st.header("Model Inference")
if st.button("Run Inference on Sample"):
    with st.spinner("Running ..."):
        try:
            batch = next(iter(loader))
            optical = batch["images"]["optical"].to(device)
            sar     = batch["images"]["sar"].to(device)

            with torch.no_grad():
                if model_type == "multimodal":
                    logits = model(optical, sar)
                elif model_type == "optical_only":
                    logits = model(optical)
                else:
                    logits = model(sar)
                probs = torch.softmax(logits, dim=1)
                pred  = logits.argmax(dim=1).item()

            true  = batch["label"][0].item()
            p_arr = probs[0].cpu().numpy()

            st.write(f"**Tile:** `{batch['tile_id'][0]}`")
            st.write(f"**Predicted:** {DAMAGE_NAMES[pred]}  |  **True:** {DAMAGE_NAMES[true]}")
            colors = ["#4ECDC4" if j == pred else "#FF6B6B" for j in range(NUM_CLASSES)]
            fig = go.Figure(go.Bar(x=DAMAGE_NAMES, y=p_arr, marker_color=colors))
            fig.update_layout(
                title="Class Probabilities", xaxis_title="Damage Class",
                yaxis_title="Probability", height=300, template="plotly_dark",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Inference error: {e}")

st.divider()
st.caption("Chen et al. (2025). BRIGHT: a globally distributed multimodal building damage assessment dataset. "
           "Earth System Science Data, 17(11), 6217–6253. https://doi.org/10.5194/essd-17-6217-2025")
