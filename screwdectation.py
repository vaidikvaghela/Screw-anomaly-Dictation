import streamlit as st
import tensorflow as tf
from keras.layers import TFSMLayer
from keras import Model, Input
from PIL import Image, ImageOps
import numpy as np

# Streamlit page setup
st.set_page_config(
    page_title="Screw Anomaly Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the SavedModel using TFSMLayer and wrap in a Keras model
@st.cache_resource
def load_model():
    layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")
    
    # Allow any input shape; the model will resize internally
    input_tensor = Input(shape=(None, None, 3), dtype=tf.float32)
    output_tensor = layer(input_tensor)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Load class labels from a text file
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Preprocess image and make a prediction
def predict(image_data, model, labels):
    size = (224, 224)  # Expected input size for Teachable Machine models
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)

    # Normalize image to [-1, 1]
    normalized_image_array = (image_array / 127.5) - 1.0

    # Add batch dimension
    data = np.expand_dims(normalized_image_array, axis=0)

    # Run prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    return labels[index], confidence_score

# Streamlit interface
st.title("🔩 Screw Anomaly Detection")
st.markdown("Upload an image of a screw to classify it as defective or normal.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        model = load_model()
        labels = load_labels()
        label, confidence = predict(image, model, labels)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")
