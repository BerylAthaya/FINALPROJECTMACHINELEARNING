import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Perbaikan khusus untuk TensorFlow 2.10
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Menghilangkan warning

# Check model files
if not os.path.exists('emotion_model.h5'):
    st.error("Model file 'emotion_model.h5' not found!")
    st.stop()

if not os.path.exists('haarcascade_frontalface_default.xml'):
    st.error("Haar Cascade file not found!")
    st.stop()

# Load model and cascade
try:
    model = load_model('emotion_model.h5', compile=False)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
except Exception as e:
    st.error(f"Error loading model/files: {str(e)}")
    st.stop()

# Rest of your app code...
