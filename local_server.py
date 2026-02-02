from flask import Flask, request, jsonify
import json
import numpy as np

app = Flask(__name__)

with open('heart_disease_model.json', 'r') as f:
    model = json.load(f)

w = np.array(model['weights'])
b = model['bias']
means = np.array(model['feature_means'])
stds = np.array(model['feature_stds'])

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Send JSON like:
    {"features": [60, 140, 300, 120, 2.0, 2]}
    """
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    
    X_norm = (features - means) / stds
    
    z = X_norm @ w + b
    probability = float(sigmoid(z)[0])
    
    return jsonify({
        'probability': probability,
        'risk_level': 'High' if probability > 0.5 else 'Low',
        'input': data['features']
    })

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Heart Disease Risk Prediction API</h1>
    <p>POST to /predict with JSON:</p>
    <pre>{"features": [age, bp, cholesterol, max_hr, st_depression, vessels]}</pre>
    """

if __name__ == '__main__':
    print("Starting Heart Disease Prediction Server...")
    print("Send POST requests to http://127.0.0.1:5000/predict")
    app.run(debug=True, port=5000)
