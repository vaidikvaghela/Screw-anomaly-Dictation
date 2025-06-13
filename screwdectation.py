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

# Function to load the model from the savedmodel directory
# Using st.cache_resource to cache the model and prevent reloading on each run
@st.cache_resource
def load_model():
    """
    Loads the trained Teachable Machine model from the 'savedmodel' directory.
    """
    model = tf.keras.models.load_model('savedmodel')
    return model

# Function to load the class labels from labels.txt
@st.cache_data
def load_labels():
    """
    Loads the class labels from the 'labels.txt' file.
    """
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Function to preprocess the uploaded image and make a prediction
def predict(image_data, model, labels):
    """
    Takes an uploaded image, preprocesses it, and returns the predicted class
    and confidence score.
    """
    # Preprocess the image to fit the model's input requirements
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
