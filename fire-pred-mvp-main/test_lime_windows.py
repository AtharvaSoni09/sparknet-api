#!/usr/bin/env python3

import sys
import dill
import numpy as np
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

print("Loading model and scalers...")
scalers = joblib.load("scalers.pkl")
xgb_model = joblib.load("xgb_model.pkl")

print("Loading LIME explainer...")
with open("lime_explainer.dill", "rb") as f:
    bundle = dill.load(f)

explainer = bundle["explainer"]
predict_fn = bundle["predict_fn"]

print("Setting up test data...")
# Test features
test_features = {
    "temp_max_F": 85.0,
    "humidity_pct": 45.0,
    "windspeed_mph": 10.0,
    "precip_in": 0.0,
    "ndvi": 0.3,
    "pop_density": 100.0,
    "slope": 5.0
}

# Setup scalers
std_scaler = scalers["standard_scaler"]
pwr_scaler = scalers["power_scaler"]

standard_cols = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi", "slope"]
power_cols = ["precip_in", "pop_density"]

# Get feature order from model
feature_order = xgb_model.get_booster().feature_names
print(f"Feature order: {feature_order}")

# Prepare input
df = pd.DataFrame([test_features])[feature_order]
print(f"Original DataFrame:\n{df}")

# Apply scalers
df[standard_cols] = std_scaler.transform(df[standard_cols])
df[power_cols] = pwr_scaler.transform(df[power_cols])
print(f"Scaled DataFrame:\n{df}")

x_instance = df.iloc[0].to_numpy()
print(f"x_instance: {x_instance}")

# Test prediction
print("\nTesting model prediction...")
pred_log = float(xgb_model.predict(df)[0])
pred_acres = float(10 ** pred_log)
print(f"pred_log: {pred_log}, pred_acres: {pred_acres}")

# Test predict function with timeout
print("\nTesting predict function...")

def test_predict_fn():
    try:
        print("Calling predict_fn...")
        pred_result = predict_fn(df.values)
        print(f"predict_fn result: {pred_result}")
        return pred_result
    except Exception as e:
        print(f"predict_fn error: {e}")
        return None

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(test_predict_fn)
        pred_result = future.result(timeout=5)  # 5 second timeout
except FutureTimeoutError:
    print("predict_fn timed out!")
except Exception as e:
    print(f"Error testing predict_fn: {e}")

# Try using the model directly as predict function
print("\nTrying to use model directly...")
try:
    def model_predict_fn(X):
        return xgb_model.predict(pd.DataFrame(X, columns=feature_order))
    
    test_pred = model_predict_fn(df.values)
    print(f"Model predict_fn result: {test_pred}")
    
    # Now try LIME with this function
    print("\nTesting LIME with model predict function...")
    
    def run_lime():
        explanation = explainer.explain_instance(
            data_row=x_instance,
            predict_fn=model_predict_fn,
            num_features=3
        )
        return explanation
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_lime)
        explanation = future.result(timeout=10)  # 10 second timeout
    
    print("LIME explanation successful!")
    print("Explanation as list:", explanation.as_list())
    
except FutureTimeoutError:
    print("LIME explanation timed out!")
except Exception as e:
    print(f"LIME error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
