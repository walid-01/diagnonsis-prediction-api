from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = joblib.load("random_forest_model.joblib")

# Define a function to preprocess the input data
def preprocess_data(data):
    douleur = int(data['douleur'])
    gonflement = int(data['gonflement'])
    rougeur = int(data['rougeur'])
    pus = int(data['pus'])
    fivre = int(data['fivre'])
    ganglions = int(data['ganglions'])
    
    return np.array([douleur, gonflement, rougeur, pus, fivre, ganglions]).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    features = preprocess_data(data)

    # Make prediction
    prediction = model.predict(features)
    result = {'prediction': prediction[0]}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
