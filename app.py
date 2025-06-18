import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Check model files
if not os.path.exists('emotion_model.h5'):
    st.error("Model file 'emotion_model.h5' not found!")
    st.stop()

if not os.path.exists('haarcascade_frontalface_default.xml'):
    st.error("Haar Cascade file not found!")
    st.stop()

# Load model and cascade
try:
    model = load_model('emotion_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
except Exception as e:
    st.error(f"Error loading model/files: {str(e)}")
    st.stop()

st.title("Deteksi Ekspresi Wajah")

uploaded_file = st.file_uploader("Upload gambar wajah...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Gagal memproses gambar. Pastikan format file valid.")
            st.stop()
        
        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

        if len(faces) == 0:
            st.warning("Tidak terdeteksi wajah dalam gambar.")
        else:
            for (x, y, w, h) in faces:
                # Crop face region
                face_roi = gray_img[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
                face_roi = np.expand_dims(face_roi, axis=[0, -1])
                
                # Predict emotion
                pred = model.predict(face_roi)[0]
                emotion = emotion_labels[np.argmax
