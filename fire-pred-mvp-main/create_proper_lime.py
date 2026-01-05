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

# Get feature order from model
feature_order = xgb_model.get_booster().feature_names
print(f"Feature order: {feature_order}")

# Create realistic training data based on scaler statistics
print("Creating realistic training data...")

# Get scaler statistics
std_scaler = scalers["standard_scaler"]
pwr_scaler = scalers["power_scaler"]

# Generate training data around the means with reasonable variance
np.random.seed(42)
n_samples = 500

# Create realistic ranges for each feature based on domain knowledge
training_data = {
    'temp_max_F': np.random.normal(0, 1, n_samples),  # Standardized
    'humidity_pct': np.random.normal(0, 1, n_samples),
    'windspeed_mph': np.random.normal(0, 1, n_samples),
    'precip_in': np.random.normal(0, 1, n_samples),
    'ndvi': np.random.normal(0, 1, n_samples),
    'pop_density': np.random.normal(0, 1, n_samples),
    'slope': np.random.normal(0, 1, n_samples)
}

# Convert to DataFrame and ensure proper scaling
training_df = pd.DataFrame(training_data)
training_df = training_df[feature_order]  # Ensure correct order

print(f"Training data shape: {training_df.shape}")
print(f"Training data sample:\n{training_df.head()}")

# Create LIME explainer with proper training data
print("Creating proper LIME explainer...")
explainer = LimeTabularExplainer(
    training_data=training_df.values,
    feature_names=feature_order,
    mode='regression',
    discretize_continuous=True,
    random_state=42
)

print("LIME explainer created successfully!")

# Test with sample data
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
standard_cols = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi", "slope"]
power_cols = ["precip_in", "pop_density"]

# Prepare input
df = pd.DataFrame([test_features])[feature_order]
df[standard_cols] = std_scaler.transform(df[standard_cols])
df[power_cols] = pwr_scaler.transform(df[power_cols])

x_instance = df.iloc[0].to_numpy()

# Create model predict function
def model_predict_fn(X):
    """A proper predict function for LIME that works with the XGBoost model."""
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_order)
    else:
        df = X
    return xgb_model.predict(df)

print("\nTesting LIME explainer...")
try:
    explanation = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_fn,
        num_features=7
    )
    
    print("LIME explanation successful!")
    print("Explanation as list:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.6f}")
    
    # Save the working explainer
    new_bundle = {
        "explainer": explainer,
        "predict_fn": model_predict_fn
    }
    
    with open("lime_explainer_proper.dill", "wb") as f:
        dill.dump(new_bundle, f)
    
    print("Proper explainer saved as lime_explainer_proper.dill")
    
except Exception as e:
    print(f"LIME error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
