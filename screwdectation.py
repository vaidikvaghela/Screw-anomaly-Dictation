import streamlit as st
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Screw Anomaly Detection",
    page_icon="🔩",
    layout="wide"
)

@st.cache_resource
def load_model_and_input_key():
    layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")
    # Use input_signature to get expected input name
    input_key = list(layer.input_signature[1].keys())[0]
    return layer, input_key

@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def predict(image_data, model_layer, input_key, labels):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    input_tensor = tf.convert_to_tensor([normalized_image_array], dtype=tf.float32)

    # Predict with the correct input key
    output = model_layer({input_key: input_tensor})
    prediction = list(output.values())[0].numpy()
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    return labels[index], confidence

# Streamlit UI
st.title("🔩 Screw Anomaly Detection")
st.markdown("Upload a screw image to detect if it's normal or defective.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        model_layer, input_key = load_model_and_input_key()
        labels = load_labels()
        label, confidence = predict(image, model_layer, input_key, labels)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")
