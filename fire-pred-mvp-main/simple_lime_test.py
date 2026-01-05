#!/usr/bin/env python3

import sys
import dill
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

print("Loading model and scalers...")
scalers = joblib.load("scalers.pkl")
xgb_model = joblib.load("xgb_model.pkl")

print("Loading original LIME explainer...")
with open("lime_explainer.dill", "rb") as f:
    bundle = dill.load(f)

original_explainer = bundle["explainer"]

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
df[standard_cols] = std_scaler.transform(df[standard_cols])
df[power_cols] = pwr_scaler.transform(df[power_cols])

x_instance = df.iloc[0].to_numpy()
print(f"x_instance: {x_instance}")

# Create the model predict function
def model_predict_fn(X):
    """A proper predict function for LIME that works with the XGBoost model."""
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_order)
    else:
        df = X
    return xgb_model.predict(df)

print("\nTesting model predict function...")
test_pred = model_predict_fn(x_instance.reshape(1, -1))
print(f"Test prediction: {test_pred}")

# Try creating a simple LIME explainer without training data
print("\nCreating simple LIME explainer...")
try:
    # Create explainer with minimal training data (just the instance itself)
    simple_training_data = x_instance.reshape(1, -1)
    
    simple_explainer = LimeTabularExplainer(
        training_data=simple_training_data,
        feature_names=feature_order,
        mode='regression',
        discretize_continuous=False,  # Disable discretization to avoid issues
        random_state=42
    )
    
    print("Simple explainer created!")
    
    # Test explanation
    print("Running LIME explanation...")
    explanation = simple_explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_fn,
        num_features=7
    )
    
    print("LIME explanation successful!")
    print("Explanation as list:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    # Save this working explainer
    new_bundle = {
        "explainer": simple_explainer,
        "predict_fn": model_predict_fn
    }
    
    with open("lime_explainer_working.dill", "wb") as f:
        dill.dump(new_bundle, f)
    
    print("Working explainer saved as lime_explainer_working.dill")
    
except Exception as e:
    print(f"Simple LIME error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
