#!/usr/bin/env python3

import sys
import dill
import numpy as np
import pandas as pd
import joblib

print("Loading LIME explainer...")
with open("lime_explainer.dill", "rb") as f:
    bundle = dill.load(f)

explainer = bundle["explainer"]

print("Deep dive into explainer internals...")

# Check if there's private data stored
print("Checking private attributes...")
for attr in dir(explainer):
    if attr.startswith('_') and not attr.startswith('__'):
        try:
            value = getattr(explainer, attr)
            print(f"{attr}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                try:
                    print(f"  Length: {len(value)}")
                except:
                    pass
            # Show small samples
            if hasattr(value, 'shape') and value.shape and value.shape[0] > 0:
                try:
                    if len(value.shape) == 1:
                        print(f"  Sample: {value[:3]}")
                    elif len(value.shape) == 2:
                        print(f"  Sample shape: {value[:2].shape}")
                        print(f"  Sample data: {value[:2]}")
                except:
                    pass
        except Exception as e:
            print(f"{attr}: Error accessing - {e}")

# Check the discretizer - it might have training data
print("\nChecking discretizer...")
if hasattr(explainer, 'discretizer'):
    discretizer = explainer.discretizer
    print(f"Discretizer type: {type(discretizer)}")
    print(f"Discretizer attributes: {[attr for attr in dir(discretizer) if not attr.startswith('_')]}")
    
    # Check for training data in discretizer
    for attr in dir(discretizer):
        if 'train' in attr.lower() or 'data' in attr.lower():
            try:
                value = getattr(discretizer, attr)
                print(f"  {attr}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
            except Exception as e:
                print(f"  {attr}: Error - {e}")

# Check if we can access the original training data through the discretizer
print("\nTrying to access training data through discretizer...")
if hasattr(explainer, 'discretizer'):
    discretizer = explainer.discretizer
    if hasattr(discretizer, 'bins'):
        print(f"Bins: {discretizer.bins}")
    if hasattr(discretizer, 'means'):
        print(f"Means: {discretizer.means}")
    if hasattr(discretizer, 'data'):
        print(f"Data: {discretizer.data}")
        if hasattr(discretizer.data, 'shape'):
            print(f"Data shape: {discretizer.data.shape}")

# Let's also check if there are any other files that might contain training data
print("\nChecking all files in directory...")
import os
for file in os.listdir('.'):
    if file.endswith(('.pkl', '.dill', '.csv', '.json', '.npy')):
        print(f"Found data file: {file}")
        if 'train' in file.lower() or 'data' in file.lower():
            print(f"  -> Likely training data file")

# Try to see if the original explainer was created with training data
print("\nChecking explainer creation parameters...")
if hasattr(explainer, 'training_data_stats'):
    print(f"Training data stats: {explainer.training_data_stats}")

# Check if we can find the original data by looking at the discretizer more carefully
if hasattr(explainer, 'discretizer'):
    discretizer = explainer.discretizer
    print(f"\nDiscretizer full inspection:")
    for attr in dir(discretizer):
        if not attr.startswith('__'):
            try:
                value = getattr(discretizer, attr)
                if not callable(value):
                    print(f"  {attr}: {type(value)} = {value}")
            except Exception as e:
                print(f"  {attr}: Error - {e}")

print("\nInspection complete.")
