import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Screw Anomaly Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and labels with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.savedmodel')
    return model

@st.cache_data
def load_labels():
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Image preprocessing and prediction
def predict(image_data, model, labels):
    size = (224, 224)  # Image size expected by the model
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalize the image as expected by the model
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    reshaped_image = np.expand_dims(normalized_image_array, axis=0)

    # Make prediction
    predictions = model.predict(reshaped_image)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return predicted_label, confidence

# Load model and labels
model = load_model()
labels = load_labels()

# Streamlit UI
st.title("🔩 Screw Anomaly Detection")
st.write("Upload a screw image to detect whether it's normal or defective.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image, model, labels)

    st.success(f"Prediction: **{label}** with {confidence * 100:.2f}% confidence.")
