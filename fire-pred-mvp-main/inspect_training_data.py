#!/usr/bin/env python3

import sys
import dill
import numpy as np
import pandas as pd
import joblib

print("Loading model and scalers...")
scalers = joblib.load("scalers.pkl")
xgb_model = joblib.load("xgb_model.pkl")

print("Loading LIME explainer...")
with open("lime_explainer.dill", "rb") as f:
    bundle = dill.load(f)

explainer = bundle["explainer"]

print("Deep inspection of explainer...")
print(f"Type: {type(explainer)}")
print(f"All attributes: {[attr for attr in dir(explainer) if not attr.startswith('_')]}")

# Check for training data in various places
print("\nChecking for training data...")
if hasattr(explainer, 'training_data'):
    print(f"training_data: {explainer.training_data}")
    if explainer.training_data is not None:
        print(f"training_data shape: {explainer.training_data.shape}")
        print(f"training_data sample: {explainer.training_data[:3]}")

if hasattr(explainer, 'training_data_stats'):
    print(f"training_data_stats: {explainer.training_data_stats}")

if hasattr(explainer, 'data'):
    print(f"data: {explainer.data}")

# Check if there's a way to access the original training data
print("\nChecking explainer internals...")
for attr in dir(explainer):
    if 'train' in attr.lower() or 'data' in attr.lower():
        try:
            value = getattr(explainer, attr)
            print(f"{attr}: {type(value)} - {value}")
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
        except Exception as e:
            print(f"{attr}: Error accessing - {e}")

# Check if the model has any stored data
print("\nChecking XGBoost model...")
booster = xgb_model.get_booster()
print(f"Booster attributes: {[attr for attr in dir(booster) if not attr.startswith('_')]}")

# Check if we can find the original training data in the model file
print("\nChecking if there are any other files with training data...")
import os
for file in os.listdir('.'):
    if file.endswith(('.pkl', '.dill', '.csv', '.json')) and 'train' in file.lower():
        print(f"Found potential training data file: {file}")

# Let's also check if the scalers have any information about the data
print("\nChecking scalers...")
print(f"Standard scaler mean: {scalers['standard_scaler'].mean_}")
print(f"Standard scaler scale: {scalers['standard_scaler'].scale_}")
print(f"Power scaler (PowerTransformer): {scalers['power_scaler']}")

# Try to reconstruct training data from scaler statistics
print("\nAttempting to understand data distribution from scalers...")
std_scaler = scalers["standard_scaler"]
print(f"Standard scaler features: {std_scaler.feature_names_in_}")
print(f"Standard scaler mean shape: {std_scaler.mean_.shape}")
print(f"Standard scaler mean values: {std_scaler.mean_}")

pwr_scaler = scalers["power_scaler"]
if hasattr(pwr_scaler, 'lambdas_'):
    print(f"Power scaler lambdas: {pwr_scaler.lambdas_}")

print("\nInspection complete.")
