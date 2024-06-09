from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Preprocess the uploaded image
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1)
    img = img / 255.0
    return img

# Define the list of emotion labels (update this with your actual labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img = preprocess_image(file)
        prediction = model.predict(img)
        emotion = emotion_labels[np.argmax(prediction)]
        return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
