import streamlit as st
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image, ImageOps
import numpy as np

# Page setup
st.set_page_config(
    page_title="Screw Anomaly Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load TFSMLayer directly (no wrapping in Functional model)
@st.cache_resource
def load_model():
    return TFSMLayer("model.savedmodel", call_endpoint="serving_default")

# Load class labels
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Predict using raw TFSMLayer
def predict(image_data, layer, labels):
    size = (224, 224)  # Required input size for most image models
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image = (image_array / 127.5) - 1.0  # Normalize to [-1, 1]

    # Add batch dimension
    input_tensor = tf.convert_to_tensor(np.expand_dims(normalized_image, axis=0), dtype=tf.float32)

    # Get the input key name from the SavedModel signature
    input_key = list(layer.call_kwargs_spec.keys())[0]  # Usually something like 'input_1'

    # Make prediction by calling TFSMLayer directly
    result = layer({input_key: input_tensor})
    prediction = list(result.values())[0].numpy()  # Get prediction array

    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    return labels[index], confidence_score

# Streamlit UI
st.title("🔩 Screw Anomaly Detection")
st.markdown("Upload a screw image to classify it as defective or normal.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        layer = load_model()
        labels = load_labels()
        label, confidence = predict(image, layer, labels)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")
