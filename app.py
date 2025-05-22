from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import imghdr

app = Flask(__name__)

# Load model once at startup
model = load_model("adam_lr1e-5_7_model.h5")
labels = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

def is_image(file_bytes):
    # file_bytes: bytes of image file
    file_type = imghdr.what(None, h=file_bytes)
    return file_type in ['jpeg', 'png']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    img_bytes = img_file.read()

    # Validate image format using bytes
    if not is_image(img_bytes):
        return jsonify({"error": "Invalid image format. Only JPG/JPEG/PNG allowed."}), 400

    try:
        # Load and preprocess image
        img = image.load_img(BytesIO(img_bytes), target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        result = labels[class_idx]

        return jsonify({"prediction": result})
    except Exception as e:
        # Catch any error during prediction
        return jsonify({"error": "Error during prediction. Please try again.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
