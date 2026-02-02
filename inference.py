import json
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def model_fn(model_dir):
    """Load model from model.json"""
    with open(f'{model_dir}/model.json', 'r') as f:
        data = json.load(f)
    return data

def input_fn(request_body, content_type='application/json'):
    """Parse JSON request"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        features = np.array(data['features']).reshape(1, -1)
        return features
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference: normalize and predict"""
    means = np.array(model['feature_means'])
    stds = np.array(model['feature_stds'])
    w = np.array(model['weights'])
    b = model['bias']
    
    X_norm = (input_data - means) / stds
    
    z = X_norm @ w + b
    probability = sigmoid(z)[0]
    
    return {'probability': float(probability), 'risk': 'High' if probability > 0.5 else 'Low'}

def output_fn(prediction, content_type='application/json'):
    """Format response"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {content_type}")
