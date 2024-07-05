from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("random_forest_model.joblib")

# Define a function to preprocess the input data
def preprocess_data(data):
    # age = int(data['age'])
    # sexe = 1 if data['sexe'] == 'm' else 0
    # douleur_map = {'legere': 1, 'moderee': 2, 'intense': 3}
    douleur = int(data['douleur'])
    gonflement = int(data['gonflement'])
    rougeur = int(data['rougeur'])
    pus = int(data['pus'])
    fivre = int(data['fivre'])
    ganglions = int(data['ganglions'])
    # sensibilite = 1 if data['sensibilite'] == 'positive' else 0
    
    return np.array([douleur, gonflement, rougeur, pus, fivre, ganglions]).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    features = preprocess_data(data)

    # Make prediction
    prediction = model.predict(features)
    print(prediction)
    result = {'prediction': prediction[0]}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
