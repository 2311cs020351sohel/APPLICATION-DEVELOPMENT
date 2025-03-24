from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])  # Fixed typo in 'methods'
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})  # Fixed typo 'prdiction'

if __name__ == '__main__':
    app.run(debug=True)  # Fixed typo 'deburg'
