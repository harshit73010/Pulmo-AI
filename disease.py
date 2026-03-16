import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="PulmoAI", page_icon="🫁")

st.title("🫁 PulmoAI")
st.write("AI-powered pneumonia detection from Chest X-ray images using Deep Learning.")

# -------------------------------
# Optional X-ray Check (soft check)
# -------------------------------
def is_xray(img):
    img_array = np.array(img)
    std = np.std(img_array)

    # Only warning, not rejection
    if std < 20 or std > 120:
        return False
    return True


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "C:/Users/ASUS/Desktop/Disease_Detection/cnn_model.h5"
    )
    return model

model = load_model()

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    return prediction


# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg","jpeg","png"]
)

# -------------------------------
# Prediction Section
# -------------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded X-ray Image", use_container_width=True)

    st.write("Analyzing Image...")

    # soft validation
    if not is_xray(img):
        st.warning("⚠️ This image may not be a chest X-ray, prediction may be unreliable.")

    prediction = predict(img)

    confidence = abs(prediction - 0.5) * 2

    if prediction > 0.5:
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal Lungs")

    st.write("Prediction Score:", float(prediction))
    st.write("Confidence:", round(confidence * 100,2), "%")

else:
    st.info("Please upload an image to start detection.")