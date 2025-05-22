from flask import Flask, render_template, request, jsonify
import numpy as np
from io import BytesIO
import imghdr
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

labels = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

# Load TFLite model and allocate tensors (replace .h5 model loading)
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def is_image(file_bytes):
    file_type = imghdr.what(None, h=file_bytes)
    return file_type in ['jpeg', 'png']

def predict_tflite(img_array):
    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], img_array)
    # Run the inference
    interpreter.invoke()
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    img_bytes = img_file.read()

    if not is_image(img_bytes):
        return jsonify({"error": "Invalid image format. Only JPG/JPEG/PNG allowed."}), 400

    try:
        # Preprocess image
        img = image.load_img(BytesIO(img_bytes), target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0).astype(np.float32)

        # Predict using TFLite model
        prediction = predict_tflite(img_array)
        class_idx = np.argmax(prediction)
        result = labels[class_idx]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": "Error during prediction. Please try again.", "details": str(e)}), 500

if __name__ == '__main__':
    # debug=False for production on Render or any hosting
    app.run(debug=False)
