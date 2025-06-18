import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Konfigurasi awal
st.set_page_config(page_title="Deteksi Ekspresi Wajah", page_icon="😊")

# Fungsi untuk memuat model
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('emotion_model.h5')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

# Fungsi untuk memuat haar cascade
@st.cache_resource
def load_haar_cascade():
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Gagal memuat file Haar Cascade: {str(e)}")
        st.stop()

# Memeriksa file yang diperlukan
required_files = ['emotion_model.h5', 'haarcascade_frontalface_default.xml']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error(f"File berikut tidak ditemukan: {', '.join(missing_files)}")
    st.stop()

# Memuat model dan haar cascade
model = load_emotion_model()
face_cascade = load_haar_cascade()

# Label emosi (harus sesuai dengan urutan saat training)
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# Antarmuka pengguna
st.title("Deteksi Ekspresi Wajah")
st.write("Unggah gambar wajah untuk mendeteksi ekspresi emosi")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Membaca dan memproses gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Gagal memproses gambar. Pastikan format file valid.")
            st.stop()
        
        # Konversi ke grayscale untuk deteksi wajah
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        
        if len(faces) == 0:
            st.warning("Tidak terdeteksi wajah dalam gambar.")
        else:
            for (x, y, w, h) in faces:
                # Ekstrak ROI wajah
                face_roi = gray_img[y:y+h, x:x+w]
                
                # Preprocessing sama seperti saat training
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype('float32') / 255.0
                face_roi = np.expand_dims(face_roi, axis=(0, -1))  # Shape: (1, 48, 48, 1)
                
                # Prediksi emosi
                pred = model.predict(face_roi)[0]
                emotion_idx = np.argmax(pred)
                emotion = emotion_labels[emotion_idx]
                confidence = pred[emotion_idx]
                
                # Gambar bounding box dan label
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{emotion} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Tampilkan hasil
            st.image(image, channels="BGR", caption="Hasil Deteksi", use_column_width=True)
            
            # Tampilkan confidence score untuk semua emosi
            st.subheader("Probabilitas Emosi:")
            for label, prob in zip(emotion_labels, pred):
                st.write(f"{label}: {prob:.4f}")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

# Tambahkan penjelasan tentang aplikasi
st.sidebar.title("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini menggunakan model CNN untuk mendeteksi 7 ekspresi emosi wajah:
1. Marah
2. Jijik
3. Takut
4. Senang
5. Sedih
6. Terkejut
7. Netral

Pastikan gambar yang diunggah jelas dan wajah terlihat dengan baik.
""")
