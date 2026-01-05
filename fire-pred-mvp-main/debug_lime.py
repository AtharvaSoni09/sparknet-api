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

print("Explainer details:")
print(f"  Type: {type(explainer)}")
print(f"  Mode: {explainer.mode}")
print(f"  Feature names: {explainer.feature_names}")
print(f"  Has training data: {hasattr(explainer, 'training_data')}")
if hasattr(explainer, 'training_data'):
    training_data = getattr(explainer, 'training_data', None)
    print(f"  Training data shape: {training_data.shape if training_data is not None else None}")
print(f"  Has discretizer: {hasattr(explainer, 'discretizer')}")
if hasattr(explainer, 'discretizer'):
    discretizer = getattr(explainer, 'discretizer', None)
    print(f"  Discretizer: {discretizer}")

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

# Prepare input
df = pd.DataFrame([test_features])[feature_order]

# Apply scalers
df[standard_cols] = std_scaler.transform(df[standard_cols])
df[power_cols] = pwr_scaler.transform(df[power_cols])

x_instance = df.iloc[0].to_numpy()

# Create the model predict function
def model_predict_fn(X):
    """
    A proper predict function for LIME that works with the XGBoost model.
    X is expected to be a numpy array of shape (n_samples, n_features)
    """
    # Convert to DataFrame with proper feature names
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_order)
    else:
        df = X
    # Return predictions
    return xgb_model.predict(df)

print("\nTesting predict function with different inputs...")
# Test with single instance
single_input = x_instance.reshape(1, -1)
print(f"Single input shape: {single_input.shape}")
try:
    pred1 = model_predict_fn(single_input)
    print(f"Single prediction: {pred1}")
except Exception as e:
    print(f"Single prediction error: {e}")

# Test with multiple instances (what LIME might generate)
multi_input = np.tile(x_instance, (10, 1))
print(f"Multi input shape: {multi_input.shape}")
try:
    pred2 = model_predict_fn(multi_input)
    print(f"Multi prediction shape: {pred2.shape}")
    print(f"Multi prediction sample: {pred2[:3]}")
except Exception as e:
    print(f"Multi prediction error: {e}")

print("\nTrying minimal LIME explanation...")
try:
    # Try with minimal parameters
    explanation = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_fn,
        num_features=1,  # Just 1 feature
        num_samples=50   # Fewer samples for faster execution
    )
    print("Minimal LIME explanation successful!")
    print("Explanation:", explanation.as_list())
except Exception as e:
    print(f"Minimal LIME error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
