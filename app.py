import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Check model files
if not os.path.exists('emotion_model.h5'):
    st.error("Model file 'emotion_model.h5' tidak ditemukan!")
    st.stop()

if not os.path.exists('haarcascade_frontalface_default.xml'):
    st.error("File Haar Cascade tidak ditemukan!")
    st.stop()

# Load model and cascade
model = load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

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
                emotion = emotion_labels[np.argmax(pred)]
                
                # Draw rectangle and label
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            st.image(image, channels="BGR", caption="Hasil Deteksi", width=300)
            
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
