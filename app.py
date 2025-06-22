
import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("disease_best.pt")

model = load_model()

st.title("🩻 Chest X-ray Disease Classifier") # Use a visible image on top
st.markdown("Upload an X-ray image to predict the disease.")

file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    with st.spinner("Predicting..."):
        res = model(img, verbose=False)[0]
        pred_class = model.names[int(res.probs.top1)]
        confidence = float(res.probs.top1conf)

    st.success(f"**Prediction:** {pred_class}  \n**Confidence:** {confidence:.2%}")
