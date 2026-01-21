import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

# Load model
model = load_model('signature_model.h5')

st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    font-family: 'Arial', sans-serif;
}
.welcome-card {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    padding: 30px;
    margin: 20px auto;
    max-width: 600px;
    text-align: center;
    color: #333;
}
.welcome-card h1 {
    color: #333;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.welcome-card p {
    color: #555;
}
.stButton>button {
    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #ee5a24, #ff6b6b);
    transform: translateY(-2px);
}
.stSuccess {
    background: linear-gradient(45deg, #f39c12, #f1c40f);
    color: black;
    border: none;
    border-radius: 10px;
}
.stImage {
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.footer {
    text-align: center;
    margin-top: 50px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome-card">
    <h1 style="font-size: 3em; color: #ff6b6b;">✍️ Hi there! Welcome to Signature Verification 😊</h1>
    <p style="font-size: 1.5em; color: #333; font-weight: bold;">Hope your day is going great! 🌟 Make your day even more joyful by testing if a signature is genuine or forged. 🔍</p>
    <p style="font-size: 1.2em; color: #555;">Upload an image below and let's verify it! 📤</p>
</div>
""", unsafe_allow_html=True)

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

st.markdown("<hr style='border: none; height: 1px; background: white;'>", unsafe_allow_html=True)
st.markdown("<div class='footer'><p style='font-size: 1.5em; font-weight: bold;'>🙏 Thank you for using our service! Visit again. 👋</p><p style='font-size: 1.2em;'>*Made by Sujana* ✨</p></div>", unsafe_allow_html=True)