from flask import Flask, request, render_template
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Initialize Flask app
app = Flask(__name__)

# Load the models
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
resnet_model = load_model('resnet50_model.h5')

Categories = ['VI-shingles', 'VI-chickenpox', 'BA-cellulitis', 'FU-athlete-foot', 
              'BA-impetigo', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans']
img_size = (224, 224)

def preprocess_image(image):
    image = cv2.resize(image, img_size)
    image = resnet_preprocess(image)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error='No file selected')

    if not file.content_type.startswith('image/'):
        return render_template('result.html', error='Invalid file type. Please upload an image.')

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)

        # Extract features
        features = resnet_model.predict(img)
        features_flat = features.reshape(1, -1)

        # Make prediction
        prediction = svm_model.predict(features_flat)
        predicted_label = Categories[prediction[0]]

        return render_template('result.html', prediction=predicted_label)

    except Exception as e:
        return render_template('result.html', error='An error occurred during prediction. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
