import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("adam_lr1e-5_7_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    for _ in range(100):
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [dummy_input]

converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized TFLite model saved as model_quantized.tflite")
