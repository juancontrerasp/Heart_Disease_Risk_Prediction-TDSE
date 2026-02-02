import requests
import json

patients = [
    {
        'name': 'Patient A (High Risk)',
        'features': [65, 150, 350, 90, 3.0, 3]
    },
    {
        'name': 'Patient B (Low Risk)',
        'features': [45, 120, 200, 150, 0.5, 0]
    },
    {
        'name': 'Patient C (Medium Risk)',
        'features': [55, 130, 250, 130, 1.5, 1]
    }
]

url = 'http://127.0.0.1:5000/predict'

print("Testing Heart Disease Prediction API")
print("=" * 60)

for patient in patients:
    response = requests.post(url, json={'features': patient['features']})
    result = response.json()
    
    print(f"\n{patient['name']}")
    print(f"  Features: {patient['features']}")
    print(f"  Risk Probability: {result['probability']:.1%}")
    print(f"  Risk Level: {result['risk_level']}")

print("\n" + "=" * 60)
print("Done!")
