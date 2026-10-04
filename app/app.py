import os, pathlib
 
if os.name == "nt":  # only on Windows
    # Map any PosixPath encountered during unpickling to WindowsPath
    # (needed because the model was trained on Kaggle/Linux)
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]
 
from pathlib import Path
 
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
 
# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"
ASSETS = REPO_ROOT / "app" / "assets"
 
 
@st.cache_resource(show_spinner=True)
def load_model(weights: str):
    return YOLO(weights)
 
 
# ---- SIDEBAR ----
st.sidebar.title("⚙️ Settings")
weights_path = st.sidebar.text_input("Weights (.pt)", str(DEFAULT_WEIGHTS))
model = load_model(weights_path)
 
imgsz = st.sidebar.selectbox("Image size", [640, 800, 512, 416], index=0)  # trained at 640
conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.01)
iou_thres = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.50, 0.01)
use_tta = st.sidebar.checkbox("Test-time augmentation (slower, more recall)", value=False)
 
st.sidebar.markdown(f"**Using weights:** `{weights_path}`")
st.sidebar.markdown("**Classes:** " + ", ".join(model.names.values()))
 
# ---- PAGE HEADER ----
st.title("♻️ Recyclables Detection — YOLO26l (Current Version Testing)")
st.subheader("Working on the data set of 3989 images", divider="gray")
st.write("Upload an image; the model will draw boxes and list predictions with confidence.")
 
# ---- LAYOUT ----
col1, col2 = st.columns([2, 1], gap="large")
 
# ---- LEFT: UPLOAD & RUN ----
with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  # <- defined here
 
    if uploaded:  # only run once an image exists
        img = Image.open(uploaded).convert("RGB")
 
        res = model.predict(
            source=img,
            imgsz=int(imgsz),
            conf=conf_thres,
            iou=iou_thres,
            augment=use_tta,
            device="cpu",  # or 0 to use GPU if available
            verbose=False,
        )[0]
 
        # res.plot() returns a BGR array; [:, :, ::-1] converts it to RGB for Streamlit
        st.image(res.plot()[:, :, ::-1], caption="Detections", use_container_width=True)
 
        rows = [
            {"label": model.names[int(c)], "conf": float(s)}
            for c, s in zip(res.boxes.cls, res.boxes.conf)
        ]
        if rows:
            df = pd.DataFrame(rows).sort_values("conf", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No boxes above threshold.")
 
# ---- RIGHT: EVAL VISUALS ----
with col2:
    st.subheader("Evaluation Visuals")
    for fname, caption in [
        ("BoxPR_curve.png", "Precision–Recall (per class, mAP@0.5)"),
        ("confusion_matrix.png", "Confusion Matrix (Predicted vs True + background)"),
        ("results.png", "Training Curves (loss, precision, recall, mAP)"),
    ]:
        p = ASSETS / fname
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.caption(f"Missing: {fname} (place in app/assets/)")
 
st.markdown("---")
 
