import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import tempfile
import os
import base64
import zipfile

# Load default model
if 'model' not in st.session_state:
    st.session_state['model'] = load_model('signature_model.h5')
model = st.session_state['model']

def load_images_from_files(genuine_files, forged_files):
    X = []
    y = []
    for label, files in [(0, genuine_files), (1, forged_files)]:
        for file in files:
            # Read image
            image = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (100, 100))
            image = img_to_array(image)
            X.append(image)
            y.append(label)
    X = np.array(X, dtype="float") / 255.0
    y = to_categorical(y, 2)
    return X, y

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

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
.stFileUploader label {
    color: white !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
}
.stFileUploader .st-bo {
    color: yellow !important;
}
.stSuccess div {
    color: yellow !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Test Signature", "Train New Model"])

with tab1:
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
        label = np.argmax(prediction)

        if label == 0:
            result = f"**Genuine Signature** (Confidence: {genuine_prob:.1f}%)"
            color = "green"
        else:
            result = f"**Forged Signature** (Confidence: {forged_prob:.1f}%)"
            color = "red"

        st.success("Prediction Complete!")
        st.markdown(f"<img src='data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}' style='border: 5px solid red; border-radius: 10px; max-width: 300px;' alt='Uploaded Signature'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: yellow; font-size: 2em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>{result}</h3>", unsafe_allow_html=True)

        # Cleanup
        os.unlink(temp_path)

with tab2:
    st.header("🧠 Train New Model")
    st.write("Upload your own genuine and forged signature images to train a custom model.")

    genuine_files = st.file_uploader("Upload Genuine Signatures", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    forged_files = st.file_uploader("Upload Forged Signatures", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("Train Model"):
        if not genuine_files or not forged_files:
            st.error("Please upload both genuine and forged images.")
        else:
            with st.spinner("Training model... This may take a few minutes."):
                X, y = load_images_from_files(genuine_files, forged_files)
                if len(X) < 10:
                    st.error("Need at least 10 images total for training.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    new_model = build_model()
                    new_model.fit(X_train, y_train, epochs=10, verbose=0)
                    loss, acc = new_model.evaluate(X_test, y_test, verbose=0)
                    st.success(f"Model trained! Accuracy: {acc*100:.1f}%")
                    st.session_state['model'] = new_model
                    st.info("New model loaded for predictions!")

st.markdown("<hr style='border: none; height: 1px; background: white;'>", unsafe_allow_html=True)
st.markdown("<div class='footer'><p style='font-size: 1.5em; font-weight: bold;'>🙏 Thank you for using our service! Visit again. 👋</p><p style='font-size: 1.2em;'>*Made by Sujana* ✨</p></div>", unsafe_allow_html=True)