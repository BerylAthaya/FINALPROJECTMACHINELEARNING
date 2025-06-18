from tensorflow.keras.models import load_model
import tf2onnx

# 1. Load Model Keras (.h5)
model = load_model('emotion_model.h5')  # Pastikan file ada di folder yang sama

# 2. Konversi ke ONNX (Sintaks Terbaru tf2onnx >=1.9.0)
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=None,  # Optional: sesuaikan dengan input model
    opset=13  # Versi ONNX opset (default=13)
)

# 3. Simpan Model ONNX
with open('emotion_model.onnx', 'wb') as f:
    f.write(model_proto.SerializeToString())

print("✅ Konversi berhasil! File ONNX disimpan sebagai 'emotion_model.onnx'")
