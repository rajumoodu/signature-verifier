import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

# Load model
model = load_model('signature_model.h5')

st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")

st.title("✍️ Signature Verification")
st.markdown("Upload a signature image to check if it's **Genuine** or **Forged**.")

uploaded_file = st.file_uploader("Choose a signature image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    # Read and preprocess
    image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image = image.reshape(1, 100, 100, 1) / 255.0

    # Predict
    prediction = model.predict(image)
    genuine_prob = prediction[0][0] * 100
    forged_prob = prediction[0][1] * 100

    if genuine_prob > forged_prob:
        result = f"**Genuine Signature** (Confidence: {genuine_prob:.1f}%)"
        color = "green"
    else:
        result = f"**Forged Signature** (Confidence: {forged_prob:.1f}%)"
        color = "red"

    st.success("Prediction Complete!")
    st.image(uploaded_file, caption="Uploaded Signature", width=300)
    st.markdown(f"<h3 style='color:{color};'>{result}</h3>", unsafe_allow_html=True)

    # Cleanup
    os.unlink(temp_path)

st.markdown("---")
st.markdown("*Made by Sujana*")