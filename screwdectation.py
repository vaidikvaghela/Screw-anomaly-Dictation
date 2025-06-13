import streamlit as st
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image, ImageOps
import numpy as np

# Set Streamlit app layout
st.set_page_config(
    page_title="Screw Anomaly Detection",
    page_icon="🔩",
    layout="wide"
)

# Load the TFSMLayer model (SavedModel format)
@st.cache_resource
def load_model_and_key():
    layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")
    input_key = list(layer.call_kwargs_spec.keys())[0]  # Automatically get expected input name
    return layer, input_key

# Load class labels
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Predict function
def predict(image_data, model_layer, input_key, labels):
    size = (224, 224)  # Resize to model input shape
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)

    # Normalize to [-1, 1] range as expected by TM models
    normalized_image_array = (image_array / 127.5) - 1.0
    input_tensor = tf.convert_to_tensor([normalized_image_array], dtype=tf.float32)  # Add batch

    # Run prediction
    output = model_layer({input_key: input_tensor})
    prediction = list(output.values())[0].numpy()

    # Get top prediction
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    return labels[index], confidence

# App UI
st.title("🔩 Screw Anomaly Detection")
st.markdown("Upload an image of a screw to classify it as **Defective** or **Normal**.")

uploaded_file = st.file_uploader("Choose a screw image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        model_layer, input_key = load_model_and_key()
        labels = load_labels()
        label, confidence = predict(image, model_layer, input_key, labels)

    st.success(f"🧠 Prediction: **{label}**")
    st.info(f"🔍 Confidence: `{confidence:.2%}`")
