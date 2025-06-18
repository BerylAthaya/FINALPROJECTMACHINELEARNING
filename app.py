import tensorflow as tf
import tf2onnx

# 1. Load model TensorFlow (.h5)
model = tf.keras.models.load_model('emotion_model.h5')

# 2. Konversi ke ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    output_path='emotion_model.onnx',  # Nama output
    opset=13,                         # Versi ONNX opset
    input_signature=[tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32, name='input')]  # Sesuaikan shape input model Anda!
)

print("✅ Konversi berhasil! File ONNX disimpan sebagai 'emotion_model.onnx'")
