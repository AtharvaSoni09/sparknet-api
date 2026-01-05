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

print("Extracting training data statistics from discretizer...")
discretizer = original_explainer.discretizer

# Get feature names
feature_names = original_explainer.feature_names
print(f"Feature names: {feature_names}")

# Extract statistics from discretizer
means = discretizer.means
stds = discretizer.stds
mins = discretizer.mins
maxs = discretizer.maxs

print(f"Extracted statistics for {len(means)} features")

# Reconstruct training data using the actual statistics
n_samples = 1000
reconstructed_data = []

for i, feature_name in enumerate(feature_names):
    if i in means:
        # Use the actual distribution from the discretizer
        feature_means = means[i]
        feature_stds = stds[i]
        
        # Create samples based on the quartile distribution
        # We'll sample from normal distributions centered at each quartile mean
        samples = []
        for j in range(n_samples):
            # Randomly choose which quartile to sample from
            quartile_idx = np.random.randint(0, len(feature_means))
            mean = feature_means[quartile_idx]
            std = feature_stds[quartile_idx] if quartile_idx < len(feature_stds) else 0.1
            
            # Sample from this quartile's distribution
            sample = np.random.normal(mean, std)
            samples.append(sample)
        
        reconstructed_data.append(samples)
    else:
        # Fallback to normal distribution if no statistics available
        reconstructed_data.append(np.random.normal(0, 1, n_samples))

# Convert to numpy array and transpose
reconstructed_data = np.array(reconstructed_data).T
print(f"Reconstructed training data shape: {reconstructed_data.shape}")

# Create new explainer with reconstructed training data
print("Creating new LIME explainer with reconstructed training data...")
new_explainer = LimeTabularExplainer(
    training_data=reconstructed_data,
    feature_names=feature_names,
    mode='regression',
    discretize_continuous=True,
    random_state=42
)

print("New explainer created successfully!")

# Test the new explainer
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
df[standard_cols] = std_scaler.transform(df[standard_cols])
df[power_cols] = pwr_scaler.transform(df[power_cols])

x_instance = df.iloc[0].to_numpy()

# Create the model predict function
def model_predict_fn(X):
    """A proper predict function for LIME that works with the XGBoost model."""
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_order)
    else:
        df = X
    return xgb_model.predict(df)

print("\nTesting new LIME explainer...")
try:
    explanation = new_explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_fn,
        num_features=7  # All features
    )
    print("LIME explanation successful!")
    print("Explanation as list:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    # Save the new explainer
    print("\nSaving new explainer bundle...")
    new_bundle = {
        "explainer": new_explainer,
        "predict_fn": model_predict_fn
    }
    
    with open("lime_explainer_fixed.dill", "wb") as f:
        dill.dump(new_bundle, f)
    
    print("New explainer saved as lime_explainer_fixed.dill")
    print("You can now replace the original lime_explainer.dill with this file.")
    
except Exception as e:
    print(f"LIME error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
