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

print("Creating synthetic training data...")
# Create synthetic training data based on reasonable ranges
np.random.seed(42)
n_samples = 1000

# Generate synthetic data with reasonable ranges
synthetic_data = {
    'temp_max_F': np.random.uniform(60, 110, n_samples),
    'humidity_pct': np.random.uniform(10, 90, n_samples),
    'windspeed_mph': np.random.uniform(0, 30, n_samples),
    'precip_in': np.random.uniform(0, 2, n_samples),
    'ndvi': np.random.uniform(-0.5, 1.0, n_samples),
    'pop_density': np.random.uniform(0, 1000, n_samples),
    'slope': np.random.uniform(0, 30, n_samples)
}

synthetic_df = pd.DataFrame(synthetic_data)

# Apply the same scaling as in the service
std_scaler = scalers["standard_scaler"]
pwr_scaler = scalers["power_scaler"]

standard_cols = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi", "slope"]
power_cols = ["precip_in", "pop_density"]

synthetic_df[standard_cols] = std_scaler.transform(synthetic_df[standard_cols])
synthetic_df[power_cols] = pwr_scaler.transform(synthetic_df[power_cols])

print(f"Synthetic training data shape: {synthetic_df.shape}")
print(f"Synthetic training data sample:\n{synthetic_df.head()}")

# Get feature order from model
feature_order = xgb_model.get_booster().feature_names
print(f"Feature order: {feature_order}")

# Create new LIME explainer with proper training data
print("Creating new LIME explainer...")
explainer = LimeTabularExplainer(
    training_data=synthetic_df.values,
    feature_names=feature_order,
    mode='regression',
    discretize_continuous=True,
    random_state=42
)

print("New explainer created successfully!")

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

# Prepare input
df = pd.DataFrame([test_features])[feature_order]
df[standard_cols] = std_scaler.transform(df[standard_cols])
df[power_cols] = pwr_scaler.transform(df[power_cols])

x_instance = df.iloc[0].to_numpy()

# Create the model predict function
def model_predict_fn(X):
    """
    A proper predict function for LIME that works with the XGBoost model.
    """
    # Convert to DataFrame with proper feature names
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_order)
    else:
        df = X
    # Return predictions
    return xgb_model.predict(df)

print("\nTesting new LIME explainer...")
try:
    explanation = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_fn,
        num_features=7  # All features
    )
    print("LIME explanation successful!")
    print("Explanation as list:", explanation.as_list())
    
    # Save the new explainer
    print("\nSaving new explainer bundle...")
    new_bundle = {
        "explainer": explainer,
        "predict_fn": model_predict_fn
    }
    
    with open("lime_explainer_fixed.dill", "wb") as f:
        dill.dump(new_bundle, f)
    
    print("New explainer saved as lime_explainer_fixed.dill")
    
except Exception as e:
    print(f"LIME error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
