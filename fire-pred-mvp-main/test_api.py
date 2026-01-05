import requests
import json

# Test data
test_data = {
    "temp_max_F": 85.0,
    "humidity_pct": 45.0,
    "windspeed_mph": 10.0,
    "precip_in": 0.0,
    "ndvi": 0.3,
    "pop_density": 100.0,
    "slope": 5.0
}

print("Testing LIME explainer API...")
try:
    response = requests.post(
        "http://127.0.0.1:8000/explain",
        json=test_data,
        timeout=30
    )
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Success! LIME explanation:")
        print(f"Prediction (log): {result['prediction_log']:.4f}")
        print(f"Prediction (acres): {result['prediction_acres']:.2f}")
        print("\nFeature importance (LIME):")
        for item in result['lime_explanation']:
            print(f"  {item['feature']}: {item['weight']:.4f}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Request failed: {e}")
