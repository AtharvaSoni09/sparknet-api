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
        print("Full API response:")
        print(json.dumps(result, indent=2))
        
        print(f"\nPrediction (log): {result['prediction_log']:.4f}")
        print(f"Prediction (acres): {result['prediction_acres']:.2f}")
        print("\nFeature importance (LIME):")
        print("Type:", type(result['lime_explanation']))
        print("Length:", len(result['lime_explanation']))
        for i, item in enumerate(result['lime_explanation']):
            print(f"  {i}: {item}")
            print(f"     feature: {item['feature']} (type: {type(item['feature'])})")
            print(f"     weight: {item['weight']} (type: {type(item['weight'])})")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Request failed: {e}")
