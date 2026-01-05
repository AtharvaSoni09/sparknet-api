print("Script started")

import dill

with open('lime_explainer.dill', 'rb') as f:
    bundle = dill.load(f)

print("Type of bundle:", type(bundle))
print("Bundle keys:", bundle.keys() if isinstance(bundle, dict) else "Not a dict")

if isinstance(bundle, dict):
    explainer = bundle.get("explainer")
    predict_fn = bundle.get("predict_fn")
    
    print("Type of explainer:", type(explainer))
    print("Type of predict_fn:", type(predict_fn))
    
    if explainer:
        print("Explainer attributes:", [attr for attr in dir(explainer) if not attr.startswith('_')])
        
        if hasattr(explainer, 'feature_names'):
            print("Feature names:", explainer.feature_names)
        if hasattr(explainer, 'training_data'):
            training_data = getattr(explainer, 'training_data', None)
            print("Training data shape:", training_data.shape if training_data is not None else None)
        if hasattr(explainer, 'mode'):
            print("Mode:", explainer.mode)