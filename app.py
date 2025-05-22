from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import imghdr

app = Flask(__name__)
model = load_model("adam_lr1e-5_7_model.h5")
labels = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

def is_image(file):
    file.seek(0)  # Go to start
    file_type = imghdr.what(file)
    file.seek(0)  # Reset pointer
    return file_type in ['jpeg', 'png']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    
    # Format validation
    if not is_image(img_file):
        return jsonify({"error": "Invalid image format. Only .jpg/.jpeg/.png allowed."}), 400

    # Preprocessing
    img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    result = labels[class_idx]

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
